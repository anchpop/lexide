"""Reader for the packed UniDic artifact — the Python twin of `lexide/src/segment/unidic.rs`.

(Named `unidic_artifact` rather than `unidic` on purpose: tagger/ goes on sys.path, and a
module called `unidic` shadows the PyPI package of that name, which is what fugashi imports
to find its dictionary. That collision breaks fugashi with a confusing
`module 'unidic' has no attribute 'DICDIR'`.)

Three implementations have to agree on where Japanese words begin: the Rust library that
ships to users, the Modal serve that `release.sh` parity-tests it against, and whatever
generates training priors. Agreeing means reading the *same* dictionary, not merely a
similar one — fugashi with unidic-lite segments 93.5% of our validation sentences the same
way as the bundled UniDic 2.1.2, and the 6.5% that differ would show up as a parity failure
between the serve and the library, or worse, as silent train/serve skew.

So this is a deliberate reimplementation of the Rust reader rather than a wrapper around a
MeCab build. See `build_unidic_artifact.py` for the file layout and `unidic.rs` for the
algorithm; both are byte-for-byte compatible with this, and `emit-priors --spans` against
`proposal_spans` here is the check.
"""
import struct

from prior import _char_type

MAGIC = b"LXUNIDIC"
ENTRY = struct.Struct("<hHH")   # cost, left_id, right_id
ENTRY_BYTES = 6

# unk.def categories, in the order build_unidic_artifact.py writes them
_CATEGORY = {"katakana": "KATAKANA", "hiragana": "HIRAGANA", "kanji": "KANJI",
             "digit": "NUMERIC", "latin": "ALPHA", "other": "SYMBOL"}


class UniDic:
    """MeCab-compatible Viterbi over the bundled artifact.

    The cost of a path is ``sum(word cost) + sum(transition cost)`` with the transition
    read as ``matrix[prev.right_id + cur.left_id * lsize]`` (mecab's connector.cpp). Because
    one surface has several entries with different context ids, the entry that wins is not
    known until the search runs — so the frontier at each position is keyed by right-context
    id rather than collapsed to a single best cost.
    """

    def __init__(self, raw):
        if len(raw) < 40 or raw[:8] != MAGIC:
            raise ValueError("not a lexide unidic artifact (bad magic)")
        (version, self.lsize, rsize, self.max_chars, n_surf, n_ent,
         blob_len, n_cats) = struct.unpack_from("<8I", raw, 8)
        if version != 1:
            raise ValueError(f"unidic artifact version {version}, expected 1")
        self.raw = raw
        self.n_surf = n_surf

        self._surf_off = 40
        self._ent_off = self._surf_off + (n_surf + 1) * 4
        self._blob = self._ent_off + (n_surf + 1) * 4
        self._entries = self._blob + blob_len
        pos = self._entries + n_ent * ENTRY_BYTES

        self.cats = {}
        for _ in range(n_cats):
            name_len = raw[pos]
            pos += 1
            name = raw[pos:pos + name_len].decode("ascii")
            pos += name_len
            group_len, count = struct.unpack_from("<HI", raw, pos)
            pos += 6
            ents = [ENTRY.unpack_from(raw, pos + k * ENTRY_BYTES) for k in range(count)]
            pos += count * ENTRY_BYTES
            self.cats[name] = (group_len, ents)
        self._matrix = pos
        if len(raw) < self._matrix + self.lsize * rsize * 2:
            raise ValueError("unidic artifact truncated: matrix does not fit")

        # surface offsets are hot in the inner loop; unpacking once beats struct-per-probe
        self._surf = struct.unpack_from(f"<{n_surf + 1}I", raw, self._surf_off)
        self._ent = struct.unpack_from(f"<{n_surf + 1}I", raw, self._ent_off)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(f.read())

    def _surface(self, i):
        return self.raw[self._blob + self._surf[i]:self._blob + self._surf[i + 1]]

    def _find(self, cand):
        """Index of `cand` in the byte-sorted surface array, or None."""
        lo, hi = 0, self.n_surf
        while lo < hi:
            mid = (lo + hi) // 2
            if self._surface(mid) < cand:
                lo = mid + 1
            else:
                hi = mid
        if lo < self.n_surf and self._surface(lo) == cand:
            return lo
        return None

    def _lexicon_entries(self, i):
        a, b = self._ent[i], self._ent[i + 1]
        base = self._entries
        return [ENTRY.unpack_from(self.raw, base + k * ENTRY_BYTES) for k in range(a, b)]

    def _transition(self, prev_right, left):
        off = self._matrix + (prev_right + left * self.lsize) * 2
        return struct.unpack_from("<h", self.raw, off)[0]

    def segment_run(self, text):
        """Minimum-cost segmentation of one whitespace-free run, as (start, end) chars."""
        n = len(text)
        if n == 0:
            return []
        types = [_CATEGORY.get(_char_type(c), "SYMBOL") for c in text]
        data = [text[i].encode("utf-8") for i in range(n)]
        starts = [0]
        for b in data:
            starts.append(starts[-1] + len(b))
        blob = b"".join(data)

        # best[i]: right_id -> (cost, back_pos, back_right); BOS has context id 0
        best = [dict() for _ in range(n + 1)]
        best[0] = {0: (0, -1, 0)}
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_chars), i):
                if not best[j]:
                    continue
                k = self._find(blob[starts[j]:starts[i]])
                if k is not None:
                    cands = self._lexicon_entries(k)
                else:
                    # unknown word: one run of a single character class, up to that
                    # class's grouping length (char.def's rule)
                    t = types[j]
                    group_len, cands = self.cats.get(t, self.cats.get("DEFAULT", (1, [])))
                    if (i - j) > group_len or any(types[x] != t for x in range(j, i)):
                        continue
                if not cands:
                    continue
                for wcost, lid, rid in cands:
                    pick = None
                    for prid, (pc, _, _) in best[j].items():
                        c = pc + self._transition(prid, lid) + wcost
                        if pick is None or c < pick[0]:
                            pick = (c, j, prid)
                    if pick is None:
                        continue
                    cur = best[i].get(rid)
                    if cur is None or pick[0] < cur[0]:
                        best[i][rid] = pick

        if not best[n]:
            return [(0, n)]
        # close to EOS, also context id 0
        _, right = min((c + self._transition(rid, 0), rid)
                       for rid, (c, _, _) in best[n].items())
        spans, i = [], n
        while i > 0:
            _, j, prid = best[i][right]
            spans.append((j, i))
            i, right = j, prid
        return spans[::-1]


__all__ = ["UniDic"]

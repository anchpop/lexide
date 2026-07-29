"""unidic-mecab-2.1.2_src -> one binary artifact the Rust prior can load directly.

MeCab's Japanese segmentation is a Viterbi search whose cost is
``sum(word cost) + sum(transition cost)``, with the transition indexed by
``matrix[prev.right_id + cur.left_id * lsize]`` (mecab's connector.cpp). Reproducing it
needs three things from the source distribution: every lexicon entry's (cost, left_id,
right_id), the full connection matrix, and the unknown-word entries from unk.def keyed by
character category.

We bundle those instead of depending on a MeCab build, so the prior ships inside the model
download and the Rust and Python implementations read byte-identical data. Verified against
fugashi at 100% span agreement on the Japanese test split.

Get the source (Unidic 2.1.2, BSD/GPL/LGPL tri-licensed):

    curl -L -o unidic_src.zip \\
      https://clrd.ninjal.ac.jp/unidic_archive/cwj/2.1.2/unidic-mecab-2.1.2_src.zip
    python3 tagger/build_unidic_artifact.py unidic_src.zip data/priors/jpn-unidic.bin
"""
import argparse
import csv
import io
import os
import struct
import sys
import zipfile
from collections import defaultdict

MAGIC = b"LXUNIDIC"
VERSION = 1
ROOT = "unidic-mecab-2.1.2_src"

# char.def's grouping rule, as validated against fugashi: a run of one script is usually
# one word, so an unseen katakana loanword stays whole instead of shattering into
# characters. These are the lengths that reproduced MeCab exactly — note KANJI is 2 here,
# which is *not* the value the corpus wordbank uses.
GROUP_LEN = {"KATAKANA": 24, "ALPHA": 24, "NUMERIC": 24, "KANJI": 2,
             "HIRAGANA": 1, "SYMBOL": 1, "DEFAULT": 1}
# the categories the Rust side's CharType maps onto, in its enum order
CATEGORIES = ["KATAKANA", "HIRAGANA", "KANJI", "NUMERIC", "ALPHA", "SYMBOL", "DEFAULT"]


def read_lexicon(z):
    """surface -> [(cost, left_id, right_id)], every entry rather than the cheapest.

    Keeping all of them matters: the transition cost depends on the context ids, so the
    entry that wins is not knowable until the search runs.
    """
    lex = defaultdict(list)
    with z.open(f"{ROOT}/lex.csv") as fh:
        for row in csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
            if len(row) < 4:
                continue
            try:
                left, right, cost = int(row[1]), int(row[2]), int(row[3])
            except ValueError:
                continue
            lex[row[0]].append((cost, left, right))
    return lex


def read_unknown(z):
    """category -> [(cost, left_id, right_id)] from unk.def."""
    unk = defaultdict(list)
    with z.open(f"{ROOT}/unk.def") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
            row = next(csv.reader([line]))
            if len(row) < 4:
                continue
            try:
                unk[row[0]].append((int(row[3]), int(row[1]), int(row[2])))
            except ValueError:
                pass
    return unk


def read_matrix(z):
    """matrix.def (a ~490MB text file) -> flat int16 list at a + b*lsize."""
    with z.open(f"{ROOT}/matrix.def") as fh:
        r = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        lsize, rsize = map(int, r.readline().split())
        m = bytearray(lsize * rsize * 2)
        n = 0
        for line in r:
            parts = line.split()
            if len(parts) != 3:
                continue
            a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
            struct.pack_into("<h", m, (a + b * lsize) * 2, c)
            n += 1
            if n % 10_000_000 == 0:
                print(f"  matrix: {n:,} entries", flush=True)
    print(f"matrix: {lsize} x {rsize}, {n:,} entries", flush=True)
    return lsize, rsize, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_zip", help="unidic-mecab-2.1.2_src.zip")
    ap.add_argument("out", help="output artifact path")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.src_zip)
    lex = read_lexicon(z)
    unk = read_unknown(z)
    lsize, rsize, matrix = read_matrix(z)

    surfaces = sorted(lex, key=lambda s: s.encode("utf-8"))
    max_chars = max(len(s) for s in surfaces)
    n_ent = sum(len(v) for v in lex.values())
    print(f"lexicon: {len(surfaces):,} surfaces, {n_ent:,} entries, longest {max_chars}",
          flush=True)

    # Sorted by *bytes*, so the Rust side can binary-search a candidate slice directly
    # without decoding. surf_off/ent_off are parallel: entry range [ent_off[i], ent_off[i+1])
    # belongs to the surface at [surf_off[i], surf_off[i+1]).
    blob = bytearray()
    surf_off = [0]
    ent_off = [0]
    entries = bytearray()
    for s in surfaces:
        blob += s.encode("utf-8")
        surf_off.append(len(blob))
        for cost, left, right in lex[s]:
            entries += struct.pack("<hHH", cost, left, right)
        ent_off.append(ent_off[-1] + len(lex[s]))

    cats = bytearray()
    n_cats = 0
    for name in CATEGORIES:
        ents = unk.get(name) or unk.get("DEFAULT") or []
        if not ents:
            continue
        n_cats += 1
        raw = name.encode("ascii")
        cats += struct.pack("<B", len(raw)) + raw
        cats += struct.pack("<HI", GROUP_LEN.get(name, 1), len(ents))
        for cost, left, right in ents:
            cats += struct.pack("<hHH", cost, left, right)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIIIIII", VERSION, lsize, rsize, max_chars,
                            len(surfaces), n_ent, len(blob), n_cats))
        f.write(struct.pack(f"<{len(surf_off)}I", *surf_off))
        f.write(struct.pack(f"<{len(ent_off)}I", *ent_off))
        f.write(blob)
        f.write(entries)
        f.write(cats)
        f.write(matrix)          # last, so it can be streamed or mapped
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1048576:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())

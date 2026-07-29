"""Boundary priors — a cheap, exact proposal fed alongside the bytes.

The byte model has to learn where words start from raw bytes alone. For a language with
spaces that is nearly free, which is why deu/eng/fra sit at 99.x: an *unseen* German word
is still 97% correct because whitespace hands the model both boundaries. Japanese has no
such signal, so every rare word must be known rather than copied — measured on the shipped
model, a jpn token seen fewer than 500 times is 24-30% wrong, against 0% for German.

So we hand the model a proposal:

  * whitespace languages -> split on whitespace (exact, free)
  * Japanese             -> a dictionary + Viterbi analyzer (fugashi/UniDic)

The proposal is *not* the answer. UniDic follows its own segmentation policy — it splits
食べ|させ|られ where we merge, and フランス|語 where our gold has one token — so the model
still has to learn the mapping from proposal to our policy. What the prior supplies is the
part the model cannot get from 52k sentences: knowing that 翻訳 and 苛立ち are words at all.
"""
# Character classes, following MeCab's char.def: a run of one script is usually one word.
# The limits say how long an *unknown* run of that class may be grouped into one token —
# katakana loanwords and numbers run long, kanji compounds are usually 2-3, and a run of
# hiragana is grammar rather than one word, so it is never grouped.
_UNK_MAX_LEN = {"katakana": 24, "latin": 24, "digit": 24, "kanji": 3, "hiragana": 1,
                "other": 1}


def _char_type(ch):
    o = ord(ch)
    if 0x30A0 <= o <= 0x30FF or o == 0x30FC:
        return "katakana"
    if 0x3040 <= o <= 0x309F:
        return "hiragana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return "kanji"
    if ch.isdigit():
        return "digit"
    if ch.isascii() and ch.isalpha():
        return "latin"
    return "other"


# Per-byte prior ids. NONE covers BOS/EOS/padding, so 0 is a safe fill.
PRIOR_NONE, PRIOR_O, PRIOR_B, PRIOR_I = 0, 1, 2, 3
PRIOR_VOCAB = 4

_TAGGER = None


def _tagger():
    """fugashi is loaded lazily so a worker without it can still run prior-free."""
    global _TAGGER
    if _TAGGER is None:
        import fugashi
        _TAGGER = fugashi.Tagger()
    return _TAGGER


def whitespace_char_labels(text):
    """B on the first character of each whitespace-delimited run, I inside, O on space."""
    out = []
    at_start = True
    for ch in text:
        if ch.isspace():
            out.append(PRIOR_O)
            at_start = True
        else:
            out.append(PRIOR_B if at_start else PRIOR_I)
            at_start = False
    return out


def japanese_char_labels(text, tagger=None):
    """B/I from a dictionary + Viterbi analysis; characters it does not cover stay O.

    The analyzer drops whitespace, so tokens are located by walking the text rather than
    by concatenating surfaces.
    """
    tagger = tagger or _tagger()
    out = [PRIOR_O] * len(text)
    pos = 0
    for word in tagger(text):
        surface = word.surface
        if not surface:
            continue
        i = text.find(surface, pos)
        if i < 0:
            continue  # analyzer normalized something; leave those characters unproposed
        out[i] = PRIOR_B
        for k in range(i + 1, min(i + len(surface), len(text))):
            out[k] = PRIOR_I
        pos = i + len(surface)
    return out


class Wordbank:
    """Unigram wordbank + Viterbi — the same prior without the analyzer dependency.

    Measured against UniDic on jpn test, a bank built from our own 23,803 training token
    types proposes a boundary at 98.9% of gold token starts versus UniDic's 99.4%, and is
    slightly *better* on tokens the model has never seen (99.0% vs 98.5%). Knowing where a
    word begins does not need 700k dictionary entries, because Japanese words are built
    from frequent morphemes — so this ships as ~1MB beside the weights instead of a 50MB
    dictionary and a Rust morphological analyzer.
    """

    def __init__(self, counts, group_unknown=True):
        import math
        total = sum(counts.values()) or 1
        self.cost = {w: -math.log(c / total) for w, c in counts.items()}
        # also admit an unknown run at its full grouped length, or the longest
        # known entry silently caps how much unseen katakana can be one proposal
        self.max_len = max(max((len(w) for w in counts), default=1),
                           max(_UNK_MAX_LEN.values()))
        self.unk = -math.log(1 / (total * 100))
        # a longer unknown run costs a little more, but far less than one
        # unknown token per character
        self.unk_len_penalty = 0.5
        # Grouping unknown runs (MeCab's char.def idea) trades a little recall
        # for far fewer spurious proposals: shattering a run into characters
        # guarantees a boundary everywhere inside it. Which the model prefers
        # is an empirical question — hence the switch.
        self.group_unknown = group_unknown
        self.inf = float("inf")

    @classmethod
    def load(cls, path, group_unknown=True):
        counts = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, _, count = line.rstrip("\n").partition("\t")
                if word and count:
                    counts[word] = int(count)
        return cls(counts, group_unknown=group_unknown)

    def segment(self, text):
        """Minimum-cost segmentation as (start, end) character spans.

        Unknown spans follow MeCab's character-type idea: a run of one script is usually
        one word, so アイスクリーム and 2026 and "Tokyo" stay whole instead of shattering
        into single characters. Charging unknown cost per character — which is what a plain
        unigram Viterbi does — is exactly what wrecks never-seen words, and those are where
        our remaining Japanese errors live.
        """
        n = len(text)
        types = [_char_type(c) for c in text]
        best = [self.inf] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                word = text[j:i]
                c = self.cost.get(word)
                if c is None:
                    span_len = i - j
                    t = types[j]
                    limit = _UNK_MAX_LEN.get(t, 1) if self.group_unknown else 1
                    if span_len <= limit and all(types[k] == t for k in range(j, i)):
                        # one unknown word, not one per character
                        c = self.unk + self.unk_len_penalty * (span_len - 1)
                    else:
                        continue
                if best[j] + c < best[i]:
                    best[i] = best[j] + c
                    back[i] = j
        spans, i = [], n
        while i > 0:
            j = back[i]
            spans.append((j, i))
            i = j
        return spans[::-1]

    def segment_constrained(self, text):
        """Whitespace boundaries are hard; Viterbi only proposes splits *inside* a run.

        This is what lets one mechanism serve every language. Japanese has no spaces, so
        the whole sentence is a single run and this reduces to plain Viterbi. Korean gets
        the eojeol boundary free from whitespace and the wordbank proposes the split
        inside it (밥을 -> 밥|을), which whitespace alone cannot see. Spaced European
        languages mostly keep their runs whole, since the run is usually one dictionary
        word already.
        """
        spans = []
        i, n = 0, len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < n and not text[j].isspace():
                j += 1
            spans.extend((i + a, i + b) for a, b in self.segment(text[i:j]))
            i = j
        return spans

    def char_labels(self, text):
        out = [PRIOR_O] * len(text)
        for a, b in self.segment_constrained(text):
            if text[a:b].strip():
                out[a] = PRIOR_B
                for k in range(a + 1, b):
                    out[k] = PRIOR_I
        return out


def char_prior(text, lang, tagger=None, wordbanks=None):
    """Best available proposal for this language.

    Sources differ per language because the measurements do. Japanese earns a real
    analyzer: UniDic proposes 99.4% of gold boundary starts against a corpus bank's 98.8%,
    worth 1.1 F1. Korean and Hindi do not — a bank already reaches 97.8% and 100% both-edge
    coverage against whitespace's 28.7% and 59.2%, so the remaining headroom is nil and the
    file is under a megabyte. Everything else keeps plain whitespace, which is exact.
    """
    banks = wordbanks or {}
    wb = banks.get(lang) if isinstance(banks, dict) else banks
    if wb is not None:
        return wb.char_labels(text)
    if lang == "jpn":
        try:
            return japanese_char_labels(text, tagger)
        except ImportError:
            return whitespace_char_labels(text)
    return whitespace_char_labels(text)


def encode_prior_bytes(text, char_labels, max_bytes=512):
    """Expand per-character prior labels over the UTF-8 byte stream.

    Mirrors dataset.encode_bytes_and_labels exactly — one leading slot for BOS, each
    character's label on its first byte with continuation bytes marked I inside a proposed
    word, one trailing slot for EOS — so the two sequences stay aligned byte for byte.
    """
    ids = [PRIOR_NONE]
    for ch, lab in zip(text, char_labels):
        n_bytes = len(ch.encode("utf-8"))
        ids.append(lab)
        if n_bytes > 1:
            cont = PRIOR_I if lab in (PRIOR_B, PRIOR_I) else PRIOR_O
            ids.extend([cont] * (n_bytes - 1))
    ids.append(PRIOR_NONE)
    return ids[:max_bytes]


def prior_ids_for(text, lang, max_bytes=512, tagger=None, wordbanks=None):
    return encode_prior_bytes(text, char_prior(text, lang, tagger, wordbanks),
                              max_bytes)


def proposal_spans(text, lang, tagger=None, wordbanks=None):
    """The prior's own segmentation as (start, end) character spans — for diagnostics."""
    labels = char_prior(text, lang, tagger, wordbanks)
    spans, start = [], None
    for i, lab in enumerate(labels):
        if lab == PRIOR_B:
            if start is not None:
                spans.append((start, i))
            start = i
        elif lab == PRIOR_O:
            if start is not None:
                spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def describe(text, lang):
    """Human-readable proposal, for eyeballing what the model is being handed."""
    return "|".join(text[a:b] for a, b in proposal_spans(text, lang))


__all__ = ["PRIOR_NONE", "PRIOR_O", "PRIOR_B", "PRIOR_I", "PRIOR_VOCAB",
           "char_prior", "encode_prior_bytes", "prior_ids_for", "proposal_spans",
           "describe", "whitespace_char_labels", "japanese_char_labels", "Wordbank"]

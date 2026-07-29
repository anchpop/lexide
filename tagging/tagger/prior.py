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

    def __init__(self, counts):
        import math
        total = sum(counts.values()) or 1
        self.cost = {w: -math.log(c / total) for w, c in counts.items()}
        self.max_len = max((len(w) for w in counts), default=1)
        self.unk = -math.log(1 / (total * 100))
        self.inf = float("inf")

    @classmethod
    def load(cls, path):
        counts = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, _, count = line.rstrip("\n").partition("\t")
                if word and count:
                    counts[word] = int(count)
        return cls(counts)

    def segment(self, text):
        """Minimum-cost segmentation as (start, end) character spans."""
        n = len(text)
        best = [self.inf] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                c = self.cost.get(text[j:i], self.unk if i - j == 1 else self.inf)
                if best[j] + c < best[i]:
                    best[i] = best[j] + c
                    back[i] = j
        spans, i = [], n
        while i > 0:
            j = back[i]
            spans.append((j, i))
            i = j
        return spans[::-1]

    def char_labels(self, text):
        out = [PRIOR_O] * len(text)
        for a, b in self.segment(text):
            if text[a:b].strip():
                out[a] = PRIOR_B
                for k in range(a + 1, b):
                    out[k] = PRIOR_I
        return out


def char_prior(text, lang, tagger=None, wordbank=None):
    if lang == "jpn":
        if wordbank is not None:
            return wordbank.char_labels(text)
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


def prior_ids_for(text, lang, max_bytes=512, tagger=None, wordbank=None):
    return encode_prior_bytes(text, char_prior(text, lang, tagger, wordbank), max_bytes)


def proposal_spans(text, lang, tagger=None, wordbank=None):
    """The prior's own segmentation as (start, end) character spans — for diagnostics."""
    labels = char_prior(text, lang, tagger, wordbank)
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

"""Token inventory: yap's pack phonemes (g2p) plus edge, word and punctuation tokens.

Text never reaches the model. Yap stores every sentence's phonemes per gram in
its language pack (`language_utils::sentence_phonemes`, whose module docs
define the format): space-separated tokens, each a g2p phoneme with an
optional stress prefix (ˈ primary, ˌ secondary), tone number (Mandarin, Thai:
``a3``) and accent-nucleus suffix ``ꜜ`` (Japanese), or ``%`` (a Japanese
accent phrase starts) or ``#`` (a break between two words of a gram).
Training reads the same sequences from a pack dump (yap
`dump_sentence_phonemes`), so what the app feeds the model at inference is
exactly what it was trained on, including the linguistic analysis yap passes
to g2p.

* One token per g2p phoneme. Stress, tone and accent are parallel features
  (``LEXICAL``), not tokens, so they never need audio frames. So is what yap
  knows about each gram: its words' parts of speech (yap's own pedagogical
  labels, ``POS_TAGS``) and its corpus rarity (``rarity_bucket`` of its count).
* Gram boundaries and ``#`` become a word-boundary token, or the punctuation
  written there: a punctuation gram, or punctuation inside a gram ("look , I
  'm") where it falls on a ``#``. Plus utterance edges. These may last zero
  frames and carry the pauses the recording has. Their ``boundary`` feature
  says whether they are inside a gram or between grams: yap's chunking.
* Each phone belongs to a word occurrence (``Phonemized.word``), named when the
  gram's words can be matched to its ``#`` groups (``spoken_words``). A word is
  a maximal run of phones: break tokens always separate words.
* The phone inventory is data: `inventories/<language>.json`, written by
  `scripts/build_inventory.py` from the language's whole pack dump, so
  inference never meets a phone that training could not have seen.
  Append-only. Each language's models have their own; ``VOCA_LANGUAGE`` picks it.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

# 4: one inventory per language (its name in the fingerprint), more punctuation aliases (Hindi
# danda, CJK full-width marks, German quotes), French/Italian elision and Korean particles in word names. 3: part of speech, gram rarity and boundary type features. 2: punctuation inside a
# gram, clitic-aware word names and occurrences, tone and accent features, the
# inventory's always-voiced phones. 1 (unversioned): symbols only.
FRONTEND_VERSION = 4

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
WORD_BREAK = "#"            # yap's break between the words of a multi-word gram; becomes SP
ACCENT_PHRASE = "%"         # a Japanese accent phrase starts at the next phone
ACCENT_NUCLEUS = "ꜜ"        # suffix: pitch drops after this mora
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
PHRASE_BREAKS = frozenset((",", ".", "?", "!", ";", ":", "—", "…"))   # punctuation that ends an intonation phrase
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}
_PUNCTUATION_ALIASES = {"‘": "'", "’": "'", "‚": "'", "‹": "'", "›": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
                        "–": "—", "-": "—", "―": "—", "─": "—", "－": "—", "...": "…", "・・・": "…", "･･･": "…", "．．．": "…",
                        "、": ",", "，": ",", "。": ".", "．": ".", "？": "?", "！": "!", "：": ":", "；": ";", "（": "(", "）": ")",
                        "「": '"', "」": '"', "『": '"', "』": '"', "《": '"', "》": '"', "〈": '"', "〉": '"', "【": '"', "】": '"',
                        "〝": '"', "〟": '"', "〞": '"', "।": ".", "।।": "."}
_CLITIC_STARTS = ("'", "’", "n't", "n’t")   # 's, 'll, n't: spoken with the word before them

# Yap's word tags (language_utils PartOfSpeech and OtherWordType): its own
# pedagogical labels, not any treebank's. Index 0 is "none", the last "other".
POS_TAGS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
            "VERB", "X")
POS_ID = {tag: i + 1 for i, tag in enumerate(POS_TAGS)}
POS_OTHER = len(POS_TAGS) + 1
RARITY_BUCKETS = 16     # log2(count + 1), the last bucket open-ended
WITHIN_GRAM, BETWEEN_GRAMS = 1, 2

# Parallel per-token features and their sizes, 0 always meaning "none": stress
# (1 primary, 2 secondary), tone (its number), accent (bits: 1 an accent phrase
# starts here, 2 accent nucleus), part of speech (``POS_ID``) and gram rarity
# (``rarity_bucket``) on phones, and boundary type (``WITHIN_GRAM``,
# ``BETWEEN_GRAMS``) on break tokens.
LEXICAL = {"stress": 3, "tone": 10, "accent": 4, "pos": POS_OTHER + 1, "rarity": RARITY_BUCKETS + 1, "boundary": 3}
PHRASE_START, NUCLEUS = 1, 2

# Every model speaks one language, and so does every process that trains, builds data for
# or runs one: its tokens are that language's inventory (yap's code, as in voca/inventories/).
LANGUAGE = os.environ.get("VOCA_LANGUAGE", "eng")
INVENTORY_PATH = Path(__file__).with_name("inventories") / f"{LANGUAGE}.json"
_ELIDING = frozenset({"fra", "ita"})                          # see spoken_words
_ATTACHED_TAGS = {"kor": frozenset({"ADP", "AUX"})}.get(LANGUAGE, frozenset())
if not INVENTORY_PATH.exists():
    raise ValueError(f"VOCA_LANGUAGE={LANGUAGE}: no inventory {INVENTORY_PATH} (scripts/build_inventory.py makes one from the pack dump)")
_inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
PHONES = tuple(_inventory["phones"])
VOWELS = tuple(_inventory["vowels"])
# Phones whose frames are voiced by definition (the language's choice; see build_inventory).
ALWAYS_VOICED = tuple(_inventory["always_voiced"])
SONORANTS = tuple(_inventory["sonorants"])    # voiced too, but forced voiced only where the audio is (voca.data)
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOICELESS = frozenset(_inventory["voiceless"])
VOWEL_SET = frozenset(VOWELS)


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    tone: tuple
    accent: tuple
    pos: tuple
    rarity: tuple
    boundary: tuple
    word: tuple     # per token, the index in ``words`` of the word occurrence a phone belongs to; -1 for other tokens
    words: tuple    # per word occurrence, the lowercased word, or None where the gram's words did not match its phone groups
    gram: tuple     # per token, the index of the (spoken) gram a phone belongs to; -1 for other tokens

    def __len__(self):
        return len(self.ids)

    def lexical(self):
        """Per token its features in the order of ``LEXICAL``."""
        return [list(features) for features in zip(*(getattr(self, name) for name in LEXICAL))]

    def word_names(self):
        """Per token, the name of the word a phone belongs to (None elsewhere or when unresolved)."""
        return [self.words[w] if w >= 0 else None for w in self.word]


def frontend_identity():
    return {"inventory": _inventory["source"], "language": _inventory["language"], "symbols": len(SYMBOLS),
            "version": FRONTEND_VERSION}


def frontend_fingerprint():
    """What tokens mean: the language, the mapping's version, the symbols, the always-voiced phones and the features."""
    return hashlib.sha256(json.dumps({"language": LANGUAGE, "version": FRONTEND_VERSION, "symbols": SYMBOLS, "always_voiced": ALWAYS_VOICED,
                                      "lexical": LEXICAL, "pos": POS_TAGS}, ensure_ascii=False).encode()).hexdigest()


QUOTATION_MARKS = '"“”„«»‹›‚「」『』'


def sentence_type(text):
    """``statement``, ``question`` or ``exclamation``, by the text's last mark. A sentence without
    one is a statement (Thai writes none, Korean and Mandarin subtitles often omit it)."""
    last = text.rstrip().rstrip(QUOTATION_MARKS + "  ")[-1:]
    return "question" if last in "?？" else "exclamation" if last in "!！" else "statement"


def rarity_bucket(count):
    """A gram's corpus count -> 1 + floor(log2(count + 1)), capped. Count 0 is a gram
    yap has no frequency for (in English almost always a proper noun)."""
    return 1 + min((count + 1).bit_length() - 1, RARITY_BUCKETS - 1)


def bare_phones(grams):
    """Every phone of a sentence's grams, without stress, tone or accent marks."""
    return [parse_phone(t)[0] for g in grams for t in g["phonemes"].split() if t not in (WORD_BREAK, ACCENT_PHRASE)]


def parse_phone(token):
    """``ˈa3`` -> (``a``, stress 1, tone 3, nucleus False), as yap's ``PhonemeToken::parse``."""
    stress = STRESS_MARKS.get(token[0], 0)
    rest = token[1:] if stress else token
    nucleus = rest.endswith(ACCENT_NUCLEUS)
    rest = rest.removesuffix(ACCENT_NUCLEUS)
    phone = rest.rstrip("0123456789")
    return phone, stress, int(rest[len(phone):] or 0), nucleus


def punctuation_of(words):
    """The punctuation tokens a punctuation-only gram spells, in order."""
    marks = []
    for word in words:
        word = _PUNCTUATION_ALIASES.get(word, word)
        marks.extend(m for m in ([word] if word in PUNCTUATION else [_PUNCTUATION_ALIASES.get(c, c) for c in word]) if m in PUNCTUATION)
    return marks


def spoken_words(words, tags, attached=frozenset()):
    """A gram's words as they are spoken: ``(leading marks, [[name, marks after, tag], ...]``.

    A clitic (``'s``, ``'ll``, ``n't``) joins the word before it, a hyphen joins
    its neighbours (``t - shirt``), and any other word without letters or digits
    is punctuation, kept as marks after the word it follows. A joined word keeps
    its first word's tag (``POS_ID``): "don't" is the auxiliary.

    Per language, words yap splits that are written, and spoken, as one: in
    French and Italian an elided word joins the next (``l' homme``, ``qu' il``,
    ``dell' acqua``), which gives the tag. Words tagged ``attached`` join the
    word before them: ``tokenize`` reads a Korean gram that way when the plain
    reading does not fit its phone groups (a particle or the copula written on
    the word before, ``곳 은``, ``상관 이야``, but not a spaced auxiliary, ``좋지 않아``)."""
    leading, spoken, join = [], [], None     # join: what a joining word is written after ("-", "" after an elision)
    for word, tag in zip(words, tags, strict=True):
        if word == "-" and spoken:
            join = "-"
        elif not any(c.isalnum() for c in word):
            (spoken[-1][1] if spoken else leading).extend(punctuation_of([word]))
        elif spoken and not spoken[-1][1] and (join is not None or word.lower().startswith(_CLITIC_STARTS) or tag in attached):
            spoken[-1][0] += (join or "") + word.lower()
            if join == "":
                spoken[-1][2] = POS_ID.get(tag, POS_OTHER)
            join = None
        else:
            spoken.append([word.lower(), [], POS_ID.get(tag, POS_OTHER)])
            join = "" if LANGUAGE in _ELIDING and word.endswith(("'", "’")) else None
    return leading, spoken


def tokenize(grams):
    """Tokens for one sentence from its pack grams (``{"words", "pos", "count", "phonemes"}``).

    Raises ``KeyError`` for a phone outside the inventory: the inventory is
    built from the whole pack, so that means a stale inventory.
    """
    symbols, columns, word, gram_of, words = [BOS], {name: [0] for name in LEXICAL}, [-1], [-1], []

    def add(symbol, occurrence=-1, gram=-1, **features):
        symbols.append(symbol)
        for name, column in columns.items():
            column.append(features.get(name, 0))
        word.append(occurrence)
        gram_of.append(gram)

    def boundary(marks, kind):
        """Punctuation where there is some; otherwise a word break."""
        for mark in marks or [SP]:
            add(mark, boundary=kind)

    spoken_before, spoken_grams = False, 0
    for gram in grams:
        sequence = gram["phonemes"].split()
        if not sequence:
            for mark in punctuation_of(gram["words"]):
                add(mark, boundary=BETWEEN_GRAMS)
            continue
        groups = " ".join(sequence).split(f" {WORD_BREAK} ")
        leading, spoken = spoken_words(gram["words"], gram["pos"])
        if len(spoken) != len(groups) and _ATTACHED_TAGS:
            leading, spoken = spoken_words(gram["words"], gram["pos"], _ATTACHED_TAGS)
        names, tags, breaks = [s[0] for s in spoken], [s[2] for s in spoken], [s[1] for s in spoken[:-1]]
        if len(groups) == 1 and len(names) > 1:     # one group spoken for several words ("cannot")
            names, tags, breaks = ["".join(names)], tags[:1], []
        if len(names) != len(groups):               # g2p merged some words: which group is which is unknown
            names, tags, breaks = [None] * len(groups), [0] * len(groups), [[]] * (len(groups) - 1)
        if leading or spoken_before and symbols[-1] not in PUNCTUATION:
            boundary(leading, BETWEEN_GRAMS)
        rarity = rarity_bucket(gram["count"])
        for k, (group, name, tag) in enumerate(zip(groups, names, tags)):
            if k:
                boundary(breaks[k - 1], WITHIN_GRAM)
            words.append(name)
            phrase_start = 0
            for token in group.split():
                if token == ACCENT_PHRASE:
                    phrase_start = PHRASE_START
                    continue
                phone, mark, number, nucleus = parse_phone(token)
                if phone not in PHONE_SET:
                    raise KeyError(f"phone {phone!r} is not in the inventory {INVENTORY_PATH}")
                add(phone, len(words) - 1, spoken_grams, stress=mark, tone=number, accent=phrase_start | (NUCLEUS if nucleus else 0),
                    pos=tag, rarity=rarity)
                phrase_start = 0
        for mark in spoken[-1][1] if spoken else ():
            add(mark, boundary=BETWEEN_GRAMS)
        spoken_before, spoken_grams = True, spoken_grams + 1
    add(EOS)
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), *(tuple(columns[name]) for name in LEXICAL),
                      tuple(word), tuple(words), tuple(gram_of))

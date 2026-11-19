"""Token inventory: yap's pack phonemes (g2p) plus edge, word and punctuation tokens.

Text never reaches the model. Yap stores every sentence's phonemes per gram in
its language pack (`language_utils::sentence_phonemes`): space-separated
tokens, each an optional stress prefix (ˈ primary, ˌ secondary) and a g2p
phoneme. Training reads the same sequences from a pack dump
(yap `dump_sentence_phonemes`), so what the app feeds the model at inference
is exactly what it was trained on, including the linguistic analysis yap
passes to g2p.

* One token per g2p phoneme. Stress is a parallel feature (0 none,
  1 primary, 2 secondary), not a token, so it never needs audio frames.
* Gram boundaries, and yap's `#` breaks between the words of a multi-word
  gram, become a word-boundary token; punctuation grams a
  punctuation token, plus utterance edges. These may last zero frames and
  carry the pauses the recording has.
* The phone inventory is data: `inventory.json`, written by
  `scripts/build_inventory.py` from a whole pack dump, so inference never
  meets a phone that training could not have seen. Append-only.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
WORD_BREAK = "#"   # yap's break between the words of a multi-word gram; becomes SP
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}
_PUNCTUATION_ALIASES = {"‘": "'", "’": "'", "“": '"', "”": '"', "«": '"', "»": '"', "–": "—", "-": "—", "...": "…"}

INVENTORY_PATH = Path(os.environ.get("LEXIDE_TTS_INVENTORY", Path(__file__).with_name("inventory.json")))
_inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
PHONES = tuple(_inventory["phones"])
VOWELS = tuple(_inventory["vowels"])
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
    words: tuple    # per token, the lowercased word a phone belongs to (None for other tokens)

    def __len__(self):
        return len(self.ids)


def frontend_identity():
    return {"inventory": _inventory.get("source"), "language": _inventory.get("language"), "symbols": len(SYMBOLS)}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(SYMBOLS, ensure_ascii=False).encode()).hexdigest()


def bare_phones(grams):
    """Every phone of a sentence's grams, without stress marks or word breaks."""
    return [split_token(t)[0] for g in grams for t in g["phonemes"].split() if t != WORD_BREAK]


def split_token(token):
    """``ˈaɪ`` -> (``aɪ``, 1)."""
    stress = STRESS_MARKS.get(token[0], 0)
    return (token[1:] if stress else token), stress


def punctuation_of(words):
    """The punctuation tokens a punctuation-only gram spells, in order."""
    marks = []
    for word in words:
        word = _PUNCTUATION_ALIASES.get(word, word)
        marks.extend(m for m in ([word] if word in PUNCTUATION else [_PUNCTUATION_ALIASES.get(c, c) for c in word]) if m in PUNCTUATION)
    return marks


def tokenize(grams):
    """Tokens for one sentence from its pack grams (``{"words", "phonemes"}``).

    Raises ``KeyError`` for a phone outside the inventory: the inventory is
    built from the whole pack, so that means a stale inventory.
    """
    symbols, stress, words = [BOS], [0], [None]

    def add(symbol, mark=0, word=None):
        symbols.append(symbol)
        stress.append(mark)
        words.append(word)

    spoken_before = False
    for gram in grams:
        tokens = gram["phonemes"].split()
        if tokens:
            if spoken_before and symbols[-1] not in PUNCTUATION:
                add(SP)
            # One `#`-separated phone group per word; a contraction ("it" + "'s")
            # is spoken as one group and labelled with the joined word.
            names = [w.lower() for w in gram["words"]]
            if tokens.count(WORD_BREAK) + 1 != len(names):
                names = ["".join(names)] * (tokens.count(WORD_BREAK) + 1)
            group = 0
            for token in tokens:
                if token == WORD_BREAK:        # between two words of one gram
                    add(SP)
                    group += 1
                    continue
                phone, mark = split_token(token)
                if phone not in PHONE_SET:
                    raise KeyError(f"phone {phone!r} is not in the inventory {INVENTORY_PATH}")
                add(phone, mark, names[group])
            spoken_before = True
        else:
            for mark in punctuation_of(gram["words"]):
                add(mark)
    add(EOS)
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(words))

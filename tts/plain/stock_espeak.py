"""Phonemes for the public models from stock espeak-ng. Set LANGUAGE_CODE to the model language (default eng).

espeak-ng (upstream, commit ee9fa7cc of 2026-11-19 or later: earlier releases print a linking ʲ) is run as
    espeak-ng -v en-us -q --ipa --sep=_ -- "<text>"
for English on each stretch of text between punctuation marks; its words are space-separated, its phonemes `_`-separated, with
ˈ/ˌ before a stressed vowel. ``tokens`` turns that into the model's phones: ɐ and ᵻ are ə, as in the training labels.
Punctuation is kept as its own gram, so the model hears where the text pauses.

    python plain/stock_espeak.py rows.jsonl out.jsonl      # rows with "text" -> the same rows with "grams"
"""
import json, os, re, subprocess, sys, unicodedata
from concurrent.futures import ThreadPoolExecutor

ESPEAK = os.environ.get("ESPEAK", "espeak-ng")
LETTER = r"[^\W\d_]"     # any script's letter: l’école is one word
MARKS = rf"(?:[.,;:!?…—–()\"“”‘«»\[\]]|(?<!{LETTER})['’]|['’](?!{LETTER})|(?<=\s)-+(?=\s))+"
# Per language (our code: its espeak voice and how g2p's labels differ from espeak's units, see g2p's README):
# vowel units kept whole (g2p merges these; other vowel sequences are separate phones), vowels folded, and
# espeak units split in two.
LANGUAGES = {
    "eng": ("en-us", {"eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "əʊ"}, {"ɐ": "ə", "ᵻ": "ə"},
            {"əl": ("ə", "l"), "iə": ("i", "ə"), "aɪə": ("aɪ", "ə"), "aɪɚ": ("aɪ", "ɚ"), "aʊɚ": ("aʊ", "ɚ")}),
    "ita": ("it", set(), {"ɪ": "i", "ʊ": "u"}, {}),
    "fra": ("fr-fr", set(), {"uː": "u", "ʊ": "u", "ɔː": "ɔ", "ɒ": "ɔ", "ɑː": "a", "aː": "a", "ʌ": "a", "ɐ": "a", "oː": "o",
                             "iː": "i", "ɪ": "i", "yː": "y", "eː": "e", "ɜː": "œ"}, {}),
    "deu": ("de", {"aɪ", "aʊ", "ɔʏ", "ɔø"}, {}, {}),
    "spa-es": ("es", set(), {}, {}),
    "spa": ("es-419", set(), {}, {}),
    "por": ("pt-br", {"aʊ", "eɪ", "oʊ", "aɪ", "ɐ̃ʊ̃", "ɐ̃ɪ̃", "ɐ̃j", "õɪ̃", "ũɪ̃"}, {}, {}),
    "rus": ("ru", set(), {}, {}),
}
LANGUAGE = os.environ.get("LANGUAGE_CODE", "eng")
VOICE, KEEP, FOLD, SPLIT = LANGUAGES[LANGUAGE]
EN_RHOTIC = {"ɑː", "ɔː", "ɪ", "ɛ", "ʊ"}
RHOTIC = EN_RHOTIC if LANGUAGE == "eng" else set()   # English coda vowel + ɹ: one phone (more, ear, care, tour, car)
VOWELS = set("aeiouyæɑɒɔəɚɛɜɪʊʌøœɐɨʉɯɤ")
# Stock espeak's spelling of a phone our labels spell otherwise (the fork renamed them; same sound).
RENAMES = {"rus": {"y": "ɨ", "ɭ": "ɫ", "ɭʲ": "ɫʲ", "ɭː": "ɫː", 'u"': "u", "ɪ^": "ɪ"}, "por": {"æ": "ɐ", "y": "ɪ"}}
RENAME = RENAMES.get(LANGUAGE, {})
INVENTORY = set(json.load(open(os.path.join(os.path.dirname(__file__), "..", "voca", "inventories", f"{LANGUAGE}.json")))["phones"])


def parser_spec(language):
    voice, keep, fold, split = LANGUAGES[language]
    return dict(voice=voice, keep=sorted(keep), fold=fold, split=split, vowels=sorted(VOWELS),
                rhotic=sorted(EN_RHOTIC) if language == "eng" else [], rename=RENAMES.get(language, {}))


def in_inventory(p):
    """``p`` as the inventory spells it (Unicode composition, renames), or split into inventory phones, or as it is."""
    p = RENAME.get(p, p)
    for form in (p, unicodedata.normalize("NFD", p), unicodedata.normalize("NFC", p)):
        if form in INVENTORY: return [form]
    out, i = [], 0                                   # greedy longest split ("ja" -> j a)
    while i < len(p):
        for j in range(len(p), i, -1):
            if p[i:j] in INVENTORY: out.append(p[i:j]); i = j; break
        else: return [p]
    return out


def espeak(text):
    out = subprocess.run([ESPEAK, "-v", VOICE, "-q", "--ipa", "--sep=_", "--", text], capture_output=True, text=True, check=True).stdout
    return re.sub(r"\([a-z-]+\)", "", out).split()       # language-switch markers


def vowel(p):
    return bool(p) and p[0] in VOWELS


def pieces(p):
    """One espeak unit -> our phones: folded, split where g2p's labels split it, a vowel sequence not kept whole as
    separate vowels, a doubled consonant (Italian ss) as a long one."""
    p = FOLD.get(p, p)
    if p in SPLIT:
        return [FOLD.get(q, q) for q in SPLIT[p]]
    if len(p) == 2 and p[0] == p[1] and not vowel(p):
        return [p[0] + "ː"]
    if vowel(p) and p not in KEEP:
        out = []
        for ch in p:
            if ch in VOWELS or not out: out.append(FOLD.get(ch, ch))
            else: out[-1] += ch                              # a length or diacritic stays on its vowel
        return [FOLD.get(q, q) for q in out]
    return [p]


def word_phones(word):
    """One espeak word ('k_ˈɛɹ_z') -> our phone notation ('k ˈɛɹ z'), split and merged the way g2p labels are:
    a vowel sharing the stress of the vowel before it, English coda vowel + ɹ as one phone, a doubled consonant as one long one."""
    units = []                                   # [stress, phone]
    for p in word.split("_"):
        stress = "".join(c for c in p if c in "ˈˌ")
        for q in pieces(p.replace("ˈ", "").replace("ˌ", "").replace("-", "")):    # French writes ə- for a dropped schwa
            split = in_inventory(q) if q else []
            carrier = next((k for k, r in enumerate(split) if vowel(r)), 0)     # the stress goes on the split's vowel
            units.extend([stress if k == carrier else "", r] for k, r in enumerate(split))
    for i in range(1, len(units)):               # adjacent vowels share stress
        if vowel(units[i][1]) and vowel(units[i - 1][1]) and units[i - 1][0] and not units[i][0]:
            units[i][0] = units[i - 1][0]
    out = []
    for i, (s, p) in enumerate(units):
        after = units[i + 1][1] if i + 1 < len(units) else None
        if p == "ɹ" and out and out[-1][1] in RHOTIC and (after is None or not vowel(after)):
            out[-1][1] += "ɹ"; continue
        if out and p == out[-1][1] and not vowel(p) and not s:
            out[-1][1] += "ː"; continue
        out.append([s, p])
    return " ".join(s + p for s, p in out)


def grams(text):
    out, at = [], 0
    for m in list(re.finditer(MARKS, text)) + [None]:
        chunk = text[at:m.start() if m else len(text)]
        if re.search(r"\w", chunk):
            words = espeak(re.sub(rf"(?<={LETTER})’(?={LETTER})", "'", chunk))
            out.append({"words": chunk.split(), "pos": ["X"] * len(chunk.split()), "count": 0,
                        "phonemes": " # ".join(word_phones(w) for w in words)})
        if m:
            out.append({"words": [m.group()], "pos": ["PUNCT"], "count": 0, "phonemes": ""}); at = m.end()
    return out


if __name__ == "__main__":
    rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8")]
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "16"))) as pool:
        done = list(pool.map(lambda r: {**r, "grams": grams(r["text"])}, rows))
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in done)

"""Phonemize all sentences in the dataset using espeak-ng, keeping stress marks.

Writes a per-language JSONL with entries:
    {"file": "abc123.wav", "lang": "eng",
     "phonemes": ["h", "ɛ", "l", ...],
     "stress": [0, 0, 0, ...]}

Languages in OVERRIDE_LANGS get their per-record stress array replaced by
a sidecar (e.g. fra/stress_overrides.jsonl produced by relabel-french/). The
sidecar lists which words end a rhythmic group; we mark the last vowel of
each such word as primary stress and zero everything else, overriding
espeak's per-word stress for that language.

Training-time interleaving of stress into CTC token sequences happens in
dataset.py.
"""

import argparse
import hashlib
import json
import re
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

LANG_TO_ESPEAK = {
    # Languages we currently train on:
    "eng": "en-us",
    "deu": "de",
    "fra": "fr-fr",
    "ita": "it",
    "por": "pt-br",
    "spa": "es",
    "rus": "ru",
    "zho-hans": "cmn",  # Simplified-script Standard Mandarin
    # Languages added via Pimsleur — phonemes.jsonl gets regenerated for
    # these too so future training runs can opt into them just by adding
    # to --langs. espeak voice picked per language.
    "sqi": "sq",     # Albanian
    "ara": "ar",     # Arabic (Eastern + Egyptian + MSA all share)
    "hye": "hy",     # Armenian (also Western)
    "yue": "yue",    # Cantonese
    "hrv": "hr",     # Croatian
    "ces": "cs",     # Czech
    "dan": "da",     # Danish
    "fas": "fa",     # Persian (Dari + Farsi)
    "nld": "nl",     # Dutch
    "fin": "fi",     # Finnish
    "hat": "ht",     # Haitian Creole
    "heb": "he",     # Hebrew
    "hin": "hi",     # Hindi
    "hun": "hu",     # Hungarian
    "isl": "is",     # Icelandic
    "ind": "id",     # Indonesian
    "gle": "ga",     # Irish
    "jpn": "ja",     # Japanese
    "kor": "ko",     # Korean
    "ell": "el",     # Modern Greek
    "nor": "nb",     # Norwegian (Bokmål)
    "pol": "pl",     # Polish
    "pan": "pa",     # Punjabi
    "ron": "ro",     # Romanian
    "swa": "sw",     # Swahili
    "swe": "sv",     # Swedish
    "tha": "th",     # Thai
    "tur": "tr",     # Turkish
    "ukr": "uk",     # Ukrainian
    "urd": "ur",     # Urdu
    "vie": "vi",     # Vietnamese
    # No espeak voice (oji=Ojibwe, pus=Pashto, tgl=Tagalog, twi=Twi):
    # preprocess.py skips these (lang not in LANG_TO_ESPEAK so the loop
    # in main() filters them out). Audio + transcripts are still on disk
    # in their manifest.jsonl.
}

# Languages whose training labels come from a qualified external backend
# (LANGUAGE_EXPANSION.md; mirrors build_external_phoneme_sidecars.CONFIG).
# Their LANG_TO_ESPEAK entries exist for audits and tooling, but main() never
# emits phonemes.jsonl for them from eSpeak: it refreshes the G2P audit +
# sidecar chain itself, so running preprocess is the whole label pipeline.
BACKEND_REQUIRED_LANGS = {"tha", "zho-hans", "hin", "jpn"}

STRESS_NONE = 0
STRESS_PRIMARY = 1
STRESS_SECONDARY = 2

# IPA vowels (monophthongs + near-variants used by espeak-ng across our languages)
IPA_VOWELS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝᵻ")

# Combining marks and modifier letters that espeak emits AFTER a base phoneme
# and that we want folded into the previous token. Misnamed historically — the
# set also covers consonant-attaching diacritics now (dental subscript, syllabic,
# raised, pharyngealization). The parser appends each of these to phonemes[-1],
# so consecutive marks stack (e.g. r̝ then ̊ → r̝̊ for voiceless Czech ř).
VOWEL_CONTINUATIONS = set(
    "ːˑ̠̞̯̥̃̊̈"   # vowel modifiers (existing): length, half-length, retracted,
                  # lowered, non-syllabic, voiceless-below, nasalized, voiceless,
                  # centralized
    "̪̩̝"          # consonant modifiers (new): dental subscript U+032A,
                  # syllabic U+0329, raised U+031D
    "ˤ"           # pharyngealization modifier letter U+02E4 — espeak emits it as
                  # a spacing modifier after the consonant; we fold it in too
                  # so emphatic Arabic consonants (tˤ, sˤ, dˤ, ðˤ) become one token
)
# NB: palatalization ʲ (U+02B2) is handled separately in the parser, NOT here —
# it folds onto a preceding CONSONANT (Russian soft tʲ nʲ …) but stays a
# standalone token after a vowel (where espeak uses it as a glide, e.g. iʲo).

WORD_BOUNDARIES = set(" \t\n|_-")

# Languages whose espeak-emitted stress is systematically wrong and gets
# replaced from a sidecar (rhythmic-group stress for French: stress falls on
# the final syllable of each rhythmic group, not on every word). The sidecar
# is produced by train/relabel-french/ (LLM call). Languages not in this set
# keep espeak's per-word stress.
OVERRIDE_LANGS = {"fra"}

# Borrowed tokenizer that defines our IPA vocabulary. Training code loads the
# same name (see train_unified.py and articulatory.py). Phonemes we emit must
# match this vocab or appear in TOKEN_BLACKLIST — otherwise they'd silently
# become <unk> at training time and degrade the labels.
TOKENIZER_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


# Tokens to fix BEFORE the vocab check. Each entry is a deliberate remap from
# something espeak emits to its correct/intended IPA form. Currently used only
# for known encoding bugs in espeak voices.
TOKEN_REMAP: dict[str, str] = {
    # Danish voice emits Greek epsilon U+03B5 in some contexts where it means
    # IPA epsilon U+025B (open-mid front unrounded vowel). Definitely a bug
    # in the espeak data — they're different Unicode codepoints but the same
    # glyph. Remap to the IPA codepoint so it matches the rest of the vocab.
    "ε": "ɛ",
}


# Per-language phoneme remaps applied BEFORE the vocab check (like TOKEN_REMAP
# but conditioned on the dataset language, because the SAME espeak symbol is
# legitimate in one language and a frontend artifact in another). The 4-source
# audit confirmed these across tatoeba/TTS/FLEURS/Pimsleur: espeak's French
# voice leaks English/length-marked vowels (loanwords, letter-names) and its
# Italian voice emits lax high vowels Italian doesn't have. Neither language has
# phonemic vowel length or (for Italian) lax /ɪ ʊ/, so we map each to the
# nearest real vowel — the audio nativises, so this is faithful, not smoothing.
LANG_PHONEME_REMAP: dict[str, dict[str, str]] = {
    "fra": {
        # length-marked vowels -> short (French has no phonemic length)
        "uː": "u", "ɔː": "ɔ", "ɑː": "a", "oː": "o", "aː": "a",
        "iː": "i", "yː": "y", "eː": "e", "ɜː": "œ",
        # lax / English-only qualities -> nearest French vowel
        "ɪ": "i", "ʊ": "u", "ʌ": "a", "ɒ": "ɔ", "ɐ": "a",
    },
    "ita": {
        # Italian has no lax high vowels. Since fork commit 4dd31042 the it
        # voice emits i/u directly (ipa labels on the reduced I/U phonemes),
        # so for pure-Italian text this is a no-op — but it must STAY:
        # English (en)…(it) code-switch spans still emit genuine ɪ/ʊ, and the
        # corpus was generated with those normalized to i/u.
        "ɪ": "i", "ʊ": "u",
    },
}


# Phonemes we add to the borrowed vocab so they're treated as real output
# classes by training. The xls-r-2b backbone has no pretrained phoneme
# representations (it was trained on raw audio only), so extending the vocab
# is free — the CTC head's output dim grows by len(VOCAB_EXTENSIONS) and the
# new logits are learned from scratch alongside everything else.
#
# Every entry is a phoneme the patched espeak (master-232+ relative to
# 1.52.0) emits that the original xlsr-53-espeak-cv-ft vocab didn't have, and
# that we want the model to learn rather than collapse into a near-neighbor.
# Panphon validates all of these with clean feature vectors so the
# articulatory aux head also gets proper targets.
VOCAB_EXTENSIONS: set[str] = {
    # German: patched espeak (commit 9cbfd389 "fr/de/ru: pronunciation fixes
    # toward modal surface realization") correctly distinguishes short ü
    # (ʏ, near-close near-front rounded) from long ü (y, close front rounded).
    # 1.52.0 conflated both as y. ~2k occurrences/lang in deu, only in deu.
    "ʏ",
    # Danish: real long ʌ, only 8 occurrences but trivially small to add.
    "ʌː",
    # Danish: non-syllabic ɐ — the offglide of falling diphthongs ("air",
    # "fjord"). ~500 occurrences. Note this combined form is produced by the
    # parser folding the ̯ U+032F mark into the ɐ.
    "ɐ̯",
    # Arabic: long ʒ, only 1 occurrence but treated symmetrically.
    "ʒː",
    # Italian geminate consonants (real phonological distinction — pasta/pasta
    # vs cassa "cash register" / casa "house" minimal pairs). Patched espeak
    # emits these; 1.52.0 wouldn't have. ~700/600/300 occurrences in ita.
    "sː",
    "zː",
    "ʃː",
    # Italian dental d (combines with vocab's existing t̪ etc.).
    "d̪",
    # Italian dental n (rare in dataset but symmetrical with t̪/d̪).
    "n̪",
    # Portuguese nasal vowels — phonemic in Portuguese (e.g. mãe "mother",
    # bom "good"). 5,642 + 1,606 occurrences in por; the vocab already has
    # `ɔ̃` but not these two.
    "ʊ̃",
    "ɪ̃",
    # Russian soft (palatalized) consonants produced by folding ʲ onto the
    # consonant (see VOWEL_CONTINUATIONS). Most Cʲ are already vocab tokens;
    # these two frequent ones are not: soft л /lʲ/ (espeak writes ɫʲ, ~746
    # occ) and щ (espeak writes ʃʲ, ~554 occ). Palatalization is phonemic in
    # Russian (/t/ vs /tʲ/), so these must be learnable, not collapsed.
    "ɫʲ",
    "ʃʲ",
    # Stage-3 coarticulatory-nasalization narrowing (espeak_audit/narrow.py): an
    # oral vowel before a CODA nasal surfaces nasalized in every non-French
    # language (population-confirmed in the espeak audit — A1-P0 depressed vs the
    # speaker's oral vowels in deu/eng/ita/spa/rus/por). The narrowed training
    # labels (phonemes_narrowed.jsonl) carry these; espeak/preprocess itself
    # never emits them (the broad phonemes.jsonl stays as-is). Decomposed form
    # (base [+ ː length] + U+0303), matching the tokenizer's existing nasal
    # vowels (ã ɔ̃ ɛ̃ …). panphon featurizes all cleanly. Freq 40–49k in-corpus.
    "ə̃", "ʌ̃", "æ̃", "ɨ̃", "ɯ̃", "ɒ̃", "ø̃",
    "aː̃", "eː̃", "iː̃", "oː̃", "uː̃", "yː̃", "øː̃", "ɛː̃", "ɔː̃", "ɑː̃",
    # Thai TLTK/Vachana inventory. These are phonemic aspiration, length, and
    # vowel-quality distinctions; tone is kept in a separate aligned field.
    "tɕʰ", "uə", "ɯə", "əː", "ɯː",
    # Mainland Mandarin g2pM + pinyin-to-IPA inventory. Falling diphthongs and
    # syllabic apicals remain single CTC units, matching the backend's phones;
    # citation tone letters are removed into the aligned `tone` field.
    "ɤ", "ʈʂ", "ʈʂʰ", "ɻ̩", "ɹ̩", "tsʰ", "ɥ",
    "ei̯", "au̯", "ou̯", "ai̯",
    # Japanese OpenJTalk surface phones: devoiced high vowels, palatalized /r/,
    # and geminates derived from its moraic closure (`cl`) label.
    "ɯ̥ᵝ", "i̥", "ɾʲ", "ɕː", "pʲː", "tɕː", "kʲː", "ɸː", "hː", "dʑː", "ɾː",
    # Hindi schwa-classifier inventory. Aspiration, breathy voice,
    # retroflexion, and dental place are contrastive and must not be split or
    # discarded. Stress is supplied separately by the surface-weight rules.
    "ɦ", "d͡ʒ", "t͡ʃ", "t̪ʰ", "bʱ", "d̪ʱ", "t͡ʃʰ", "ɡʱ",
    "ɽʱ", "d͡ʒʱ", "ɖʱ",
}


# Tokens that aren't in the (extended) vocab and that we explicitly choose to
# drop. Every entry is a deliberate decision; anything not in this set AND not
# in vocab raises a hard error at preprocess time so the choice has to be made
# rather than silently lost.
TOKEN_BLACKLIST: set[str] = {
    # Punctuation leaking from source text into espeak's output. Stray
    # characters espeak passes through when the input contained them.
    "(",
    ")",
    '"',
    "^",
    "?",        # patched espeak's da voice leaks question marks through.
    # Syllable separators espeak emits in some voices (Arabic). Our vocab is
    # phoneme-level, not syllable-level; we don't track syllable boundaries.
    ".",
    ".ː",       # malformed period-before-length artefact.
    # Palatalized /h/ from the ʲ-fold (see phonemize). Not a phoneme of any of
    # our languages — the single occurrence is a malformed mixed-language
    # Pimsleur clip (Korean text + "Listen and repeat" read by the en-us
    # voice). Deliberate drop rather than fabricate a hʲ vocab class.
    "hʲ",
}


@cache
def _tokenizer_vocab() -> set[str]:
    """Load the borrowed tokenizer's vocab once. Cached for the whole run."""
    from transformers import Wav2Vec2CTCTokenizer
    tok = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_NAME)
    return set(tok.get_vocab().keys())


def validate_phonemes(
    phonemes: list[str], stress: list[int], lang: str | None = None,
) -> tuple[list[str], list[int], set[str]]:
    """Filter phonemes against the tokenizer vocab + TOKEN_BLACKLIST.

    Returns (kept_phonemes, kept_stress, unknown_tokens).
      - Tokens in the vocab pass through unchanged.
      - Tokens in TOKEN_BLACKLIST are dropped (along with their stress entry).
      - Tokens in neither are also dropped, AND collected into unknown_tokens.

    Caller MUST treat a non-empty unknown_tokens as an error and surface the
    finding (sentence + token) so a decision can be made — extend the
    blacklist (intentional drop), extend the phonemize() parser (treat as
    diacritic / boundary / etc.), or change the upstream text.
    """
    vocab = _tokenizer_vocab() | VOCAB_EXTENSIONS
    lang_remap = LANG_PHONEME_REMAP.get(lang or "", {})
    out_phonemes: list[str] = []
    out_stress: list[int] = []
    unknowns: set[str] = set()
    for p, s in zip(phonemes, stress):
        # Apply remaps first (language-conditional frontend fixes, then global
        # encoding-bug fixes) — these aren't "unknown", they're known-wrong and
        # we're correcting them before the vocab check.
        p = lang_remap.get(p, p)
        p = TOKEN_REMAP.get(p, p)
        if p in vocab:
            out_phonemes.append(p)
            out_stress.append(s)
        elif p in TOKEN_BLACKLIST:
            continue
        else:
            unknowns.add(p)
    return out_phonemes, out_stress, unknowns


# Split text into atomic word tokens that align with espeak's IPA word spans.
# Hyphenated compounds ("passe-temps") are kept as one word — espeak treats
# them as a single unit and emits one IPA span for the whole compound.
_ATOMIC_WORD_RE = re.compile(r"\S+")


def _atomic_words(text: str) -> list[str]:
    return _ATOMIC_WORD_RE.findall(text)


def _merge_vowelless_spans(
    phonemes: list[str], word_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge any IPA word span with no vowel into the preceding span.

    French liaison is the motivating case: espeak emits "les amis" as
    `le_z_ami`, so the floating /z/ becomes its own span and breaks
    alignment with the 2 text words. Merging the vowelless /z/ span
    back into "le" restores alignment ("lez" + "ami" = 2 spans).

    If the first span is vowelless, merge it forward into the next span
    instead.
    """
    merged: list[tuple[int, int]] = []
    for start, end in word_spans:
        has_vowel = any(phonemes[i][:1] in IPA_VOWELS for i in range(start, end))
        if has_vowel or not merged:
            merged.append((start, end))
        else:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
    # If the very first span lacked a vowel and we kept it (no `merged` yet
    # at that point), fold it forward into the next vowel-bearing span.
    if len(merged) >= 2:
        first_start, first_end = merged[0]
        if not any(phonemes[i][:1] in IPA_VOWELS for i in range(first_start, first_end)):
            next_start, next_end = merged[1]
            merged = [(first_start, next_end)] + merged[2:]
    return merged


def _strip_punct(s: str) -> str:
    """Match the Rust binary's strip_punct: trim leading/trailing non-alnum
    except apostrophes and hyphens, lowercase. So "Berlin," → "berlin" and
    "qu'il" → "qu'il".
    """
    return s.strip("""!"#$%&()*+,./:;<=>?@[\\]^_`{|}~…—–""").lower()


def apply_stress_override(
    phonemes: list[str],
    word_spans: list[tuple[int, int]],
    sentence: str,
    stressed_words: list[str],
) -> list[int] | None:
    """Build a stress array from an LLM rhythmic-group sidecar entry.

    Aligns text words (whitespace/hyphen-split) to IPA word_spans by index.
    For each text word that matches one of the LLM-flagged `stressed_words`
    (left-to-right, with repeated targets consuming successive occurrences),
    marks the LAST vowel in the corresponding IPA span as primary stress.
    Returns None if the text-word count doesn't match word_spans (espeak
    contraction or expansion broke the index alignment) — caller should fall
    back to espeak's stress.
    """
    text_words = _atomic_words(sentence)
    aligned_spans = _merge_vowelless_spans(phonemes, word_spans)
    if len(text_words) != len(aligned_spans):
        return None

    normalized = [_strip_punct(w) for w in text_words]
    stress = [STRESS_NONE] * len(phonemes)
    used = [False] * len(text_words)

    for stressed in stressed_words:
        atomics = _atomic_words(stressed)
        if not atomics:
            continue
        # LLM may return multi-token phrases or hyphenated compounds; rhythmic
        # stress falls on the final syllable of the final token.
        target = _strip_punct(atomics[-1])
        if not target:
            continue
        for i, w in enumerate(normalized):
            if used[i] or w != target:
                continue
            start, end = aligned_spans[i]
            for j in range(end - 1, start - 1, -1):
                if phonemes[j][:1] in IPA_VOWELS:
                    stress[j] = STRESS_PRIMARY
                    break
            used[i] = True
            break

    return stress


def load_stress_overrides(path: Path) -> dict[str, list[str]]:
    overrides = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            overrides[rec["file"]] = rec["stressed_words"]
    return overrides


def required_backend_provider(lang: str) -> str | None:
    """The one G2P provider whose labels this language may be trained on.

    `None` for languages labeled from eSpeak. For everything in
    [`BACKEND_REQUIRED_LANGS`] this is authoritative: see
    [`PHONEME_LABEL_SOURCES`] for why each language has one, and
    `PHONEME_BACKENDS.md` for the full rationale.
    """
    if lang not in BACKEND_REQUIRED_LANGS:
        return None
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import build_external_phoneme_sidecars as sidecars
    finally:
        sys.path.remove(str(scripts_dir))
    return sidecars.CONFIG[lang][0]


def load_phoneme_backend(path: Path, lang: str | None = None) -> dict[str, dict]:
    """Load a complete external-transcription sidecar keyed by audio file.

    Each JSONL row must contain `file` and `sentence_sha256`, plus either
    `phonemes` and `stress`, or an explicit `exclude_reason`. Exclusions make
    backend failures auditable while preserving the completeness invariant:
    every manifest row has a deliberate disposition. The text hash prevents
    a transcription generated for an older sentence from being silently
    reused. Optional suprasegmental fields are preserved in output.

    When `lang` is a backend-required language, every row's `backend` must
    name that language's required provider. Without this check a
    `--phoneme-backend jpn=…` override pointing at a sidecar built by some
    *other* G2P engine trains Japanese on labels from the wrong phoneme
    inventory, and nothing downstream can tell: the shapes are identical and
    the hashes still match. Fail closed instead — a mislabeled corpus is
    discovered epochs later, if at all.
    """
    expected_provider = required_backend_provider(lang) if lang else None
    records: dict[str, dict] = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            required = {"file", "sentence_sha256"}
            missing = required - rec.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_no}: missing fields {sorted(missing)}"
                )
            if rec["file"] in records:
                raise ValueError(f"{path}:{line_no}: duplicate file {rec['file']!r}")
            excluded = bool(rec.get("exclude_reason"))
            if excluded and ("phonemes" in rec or "stress" in rec):
                raise ValueError(
                    f"{path}:{line_no}: excluded row must not carry phonemes/stress"
                )
            if not excluded and not {"phonemes", "stress"} <= rec.keys():
                raise ValueError(
                    f"{path}:{line_no}: row needs phonemes/stress or exclude_reason"
                )
            if not excluded and len(rec["phonemes"]) != len(rec["stress"]):
                raise ValueError(
                    f"{path}:{line_no}: phonemes/stress length mismatch"
                )
            if expected_provider is not None and rec.get("backend") != expected_provider:
                raise ValueError(
                    f"{path}:{line_no}: {lang} must be labeled by "
                    f"{expected_provider!r}, but this row says "
                    f"{rec.get('backend')!r}. See PHONEME_BACKENDS.md."
                )
            records[rec["file"]] = rec
    return records


def run_narrowing(lang: str, data_dir: Path) -> None:
    """Regenerate phonemes_narrowed.jsonl for a language just labeled.

    Training reads the narrowed file (--use-narrowed is the default), so a
    stale one silently hides newly-labeled clips from training. Pass-through
    languages are a dependency-free broad copy; English and the nasal
    languages recompute their acoustic evidence from the measure cache, and
    clips the cache doesn't cover simply stay broad (narrow.py reports the
    abstain counts — re-run espeak_audit/measure_corpus.py to narrow them).
    """
    audit_dir = Path(__file__).resolve().parents[2] / "espeak_audit"
    sys.path.insert(0, str(audit_dir))
    try:
        import narrow
    finally:
        sys.path.remove(str(audit_dir))
    # narrow's module-level audio root; the acoustic measure cache resolves
    # its own paths independently, so pointing this at --data-dir is safe.
    narrow.AUDIO = data_dir
    narrow.run([lang])


def ensure_backend_sidecar(lang: str, data_dir: Path) -> Path:
    """Bring a backend-required language's label chain up to date, in place.

    Runs the incremental G2P audit (free when the manifest is unchanged; the
    language's G2P tool is only needed for new/changed rows) and rebuilds the
    hash-bound sidecar from it. This keeps preprocess self-contained: new
    Pimsleur/Tatoeba rows just work, with no separate steps to remember.
    """
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import audit_g2p_backends
        import build_external_phoneme_sidecars
    finally:
        sys.path.remove(str(scripts_dir))

    provider, _ = build_external_phoneme_sidecars.CONFIG[lang]
    lang_dir = data_dir / lang
    audit_g2p_backends.run_audit(
        lang, provider,
        manifest=lang_dir / "manifest.jsonl",
        output=lang_dir / f"g2p_audit_{provider}.jsonl",
    )
    return build_external_phoneme_sidecars.build_sidecar(lang, data_root=data_dir)


def load_accent_exclusions(path: Path) -> dict[str, str]:
    """Clips whose measured F0 contradicts their citation pitch accent.

    Written by espeak_audit/pitch_accent_audit.py. The phones are unaffected —
    only the accent factor is withheld, so the clip still trains everything
    else. Absent file means no acoustic pass has been run yet.
    """
    if not path.exists():
        return {}
    excluded = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            excluded[rec["file"]] = rec.get("reason", "f0_contradicts_citation_accent")
    return excluded


def parse_phoneme_backend_args(values: list[str] | None) -> dict[str, Path]:
    """Parse repeatable LANG=JSONL command-line values."""
    result: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected LANG=JSONL for --phoneme-backend, got {value!r}")
        lang, raw_path = value.split("=", 1)
        if not lang or not raw_path:
            raise ValueError(f"Expected LANG=JSONL for --phoneme-backend, got {value!r}")
        if lang in result:
            raise ValueError(f"Duplicate --phoneme-backend for {lang!r}")
        result[lang] = Path(raw_path)
    return result


def _parse_espeak_ipa(raw: str) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    """Parse one eSpeak IPA output line into phones, stress, and word spans.

    Stress markers ˈ and ˌ precede a syllable. The stress attaches to that
    syllable's vowel nucleus only (plus any length/nasalization diacritics).
    A new vowel after a consonant marks a new syllable → resets stress to none
    unless a fresh marker appeared.

    word_spans[i] = (start, end) indices into `phonemes` for the i-th word
    espeak emitted. Word breaks are the characters in WORD_BOUNDARIES — same
    characters the parser uses to reset stress state. Empty spans (consecutive
    boundaries) are dropped, so word_spans aligns with non-empty text words.
    """
    # espeak switches voice for foreign words (loanwords, names) and brackets
    # the switch with (lang) markers, e.g. "football" -> "(en)fˈʊtbɔːl(fr)".
    # Region-qualified markers such as (en-us) occur in mixed-script input.
    # The parens are blacklisted but the 2-letter codes are valid phonemes
    # (e/n/f/r…), so without this they'd silently inject spurious segments into
    # loanword labels — and the unknown-token safety net can't catch it. Strip
    # the markers; the switched word body stays and is parsed/remapped normally
    # (the French vowel remap nativises the English vowels espeak used). ~2.3%
    # of French sentences carry these.
    raw = re.sub(r"\([a-z]{2,4}(?:-[a-z]{2,4})?\)", "", raw,
                 flags=re.IGNORECASE)

    phonemes = []
    stress = []
    word_spans: list[tuple[int, int]] = []
    word_start = 0             # index into `phonemes` where the current word began
    pending_stress = None      # set when we see ˈ or ˌ, consumed by next vowel
    current_stress = STRESS_NONE  # active stress state for the current vowel
    in_vowel = False           # are we currently emitting the nucleus of a syllable?

    for char in raw:
        if char == "ˈ":
            pending_stress = STRESS_PRIMARY
            in_vowel = False
        elif char == "ˌ":
            pending_stress = STRESS_SECONDARY
            in_vowel = False
        elif char in WORD_BOUNDARIES:
            if len(phonemes) > word_start:
                word_spans.append((word_start, len(phonemes)))
            word_start = len(phonemes)
            pending_stress = None
            current_stress = STRESS_NONE
            in_vowel = False
        elif char in IPA_VOWELS:
            if pending_stress is not None:
                current_stress = pending_stress
                pending_stress = None
            elif not in_vowel:
                current_stress = STRESS_NONE
            in_vowel = True
            phonemes.append(char)
            stress.append(current_stress)
        elif char in VOWEL_CONTINUATIONS:
            # Combining diacritic — append to previous phoneme so the combined
            # string matches the tokenizer's precomposed vocab tokens (ɛ̃, iː).
            if phonemes:
                phonemes[-1] += char
            else:
                # Stray diacritic at start of output — no previous token to attach to
                phonemes.append(char)
                stress.append(STRESS_NONE)
        elif char == "ʲ":
            # Palatalization (U+02B2). Fold onto the preceding phoneme IFF it's a
            # CONSONANT → Russian soft consonants (tʲ nʲ sʲ …, soft л ɫʲ, щ ʃʲ):
            # one segment, not a standalone ʲ that the 4-source audit found
            # aligns to ~0 frames. After a VOWEL, espeak uses ʲ to mark a glide
            # (hiatus [j], e.g. Italian "io"=iʲo, Spanish "días"=diʲas) — keep it
            # as its own token (its prior behavior), don't fabricate a vowel+ʲ.
            if phonemes and phonemes[-1][:1] not in IPA_VOWELS:
                phonemes[-1] += char
            else:
                in_vowel = False
                current_stress = STRESS_NONE
                phonemes.append(char)
                stress.append(STRESS_NONE)
        else:
            # Consonant — not stress-bearing
            in_vowel = False
            current_stress = STRESS_NONE
            phonemes.append(char)
            stress.append(STRESS_NONE)

    if len(phonemes) > word_start:
        word_spans.append((word_start, len(phonemes)))

    return phonemes, stress, word_spans


_ESPEAK_STDOUT_DIAGNOSTIC = re.compile(r"Invalid phoneme code \d+")


def _espeak_command(espeak_lang: str, *, stdin: bool = False) -> list[str]:
    """Build the pinned/fork-aware eSpeak command used by both call paths."""
    espeak_bin = os.environ.get("ESPEAK_NG_BIN", "espeak-ng")
    data_path = os.environ.get("ESPEAK_NG_DATA_PATH")
    cmd = [espeak_bin]
    if data_path:
        cmd.append(f"--path={data_path}")
    cmd.extend(["-v", espeak_lang, "-q", "--ipa", "-x"])
    if stdin:
        cmd.append("--stdin")
    return cmd


def _espeak_run_text(espeak_lang: str, text: str) -> subprocess.CompletedProcess:
    """Run eSpeak on ONE utterance, with the text fed on stdin.

    Text never goes in argv. Passed as an argument, a leading hyphen is parsed
    as options — `-So früh?` is rejected as invalid option `S` — whereupon
    eSpeak writes the complaint to *stderr*, emits nothing on stdout, and
    still exits 0. `check=True` does not fire, and the caller receives an
    empty phoneme list indistinguishable from a punctuation-only sentence.
    That silently dropped 417 corpus rows (nearly all film clips, where
    dialogue dashes are ubiquitous) before it was found.

    `--` would also fix that, but stdin removes the option-parsing surface
    entirely rather than depending on every future call site remembering the
    separator. Verified to produce byte-identical phonemes to the argv form
    across 867 corpus sentences.

    This is deliberately ONE utterance per process. `phonemize_many`'s
    multi-line stdin batching is a different thing and is unsafe — see the
    warning there.

    Any stderr output fails the call. A healthy run writes nothing to stderr
    (measured over 840 corpus sentences), so there is no benign baseline to
    talk past; if a legitimate warning class ever appears, whitelist it
    explicitly. Defaulting to "ignore output we don't recognize" is precisely
    how the dash bug survived.
    """
    # A manifest sentence is conceptually one utterance; an embedded newline
    # would otherwise split it into two stdin records.
    flattened = re.sub(r"[\r\n]+", " ", text)
    # errors="replace": patched espeak occasionally emits non-UTF8 warnings.
    result = subprocess.run(
        _espeak_command(espeak_lang, stdin=True),
        input=flattened + "\n",
        capture_output=True, text=True, check=True, errors="replace",
    )
    if result.stderr.strip():
        raise RuntimeError(
            f"eSpeak wrote to stderr for {text!r} (voice {espeak_lang}, exit 0): "
            f"{result.stderr.strip()[:300]}"
        )
    return result


def _espeak_output_lines(stdout: str) -> list[str]:
    """Return utterance lines, excluding diagnostics eSpeak prints to stdout."""
    return [
        line.strip()
        for line in stdout.splitlines()
        if line.strip() and not _ESPEAK_STDOUT_DIAGNOSTIC.fullmatch(line.strip())
    ]


def _espeak_build_fingerprint() -> str:
    """Stat-hash of the espeak binary + every file in its data dir.

    Keys the phonemize cache: any rebuild of the fork (binary, phontab,
    dictionaries) changes some mtime and invalidates every cached entry, so
    the cache can never serve output from a different espeak build. mtime
    granularity means an identical rebuild also invalidates — a false
    invalidation is safe, a false hit is not.
    """
    h = hashlib.sha256()
    espeak_bin = os.environ.get("ESPEAK_NG_BIN", "espeak-ng")
    bin_path = shutil.which(espeak_bin) or espeak_bin
    paths = [Path(bin_path)]
    data_path = os.environ.get("ESPEAK_NG_DATA_PATH")
    if data_path:
        paths.extend(sorted(Path(data_path).rglob("*")))
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        if p.is_file():
            h.update(f"{p}\x00{st.st_size}\x00{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


_PHONEMIZE_CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "audio" / ".cache"
    / "phonemize.sqlite3"
)
_phonemize_cache: "sqlite3.Connection | None | bool" = None
_phonemize_fingerprint: str | None = None


def _phonemize_cache_conn() -> "sqlite3.Connection | None":
    """Open (once) the shared raw-output cache; None if disabled/unavailable.

    WAL + busy_timeout so the parallel per-language preprocess children can
    share one database. Stores espeak's RAW stdout, not the parsed phonemes —
    parser changes therefore never stale the cache; they just re-parse hits.
    Opt out with LEXIDE_PHONEMIZE_CACHE=0.
    """
    global _phonemize_cache, _phonemize_fingerprint
    if _phonemize_cache is False:
        return None
    if _phonemize_cache is None:
        if os.environ.get("LEXIDE_PHONEMIZE_CACHE", "1") == "0":
            _phonemize_cache = False
            return None
        try:
            _PHONEMIZE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(_PHONEMIZE_CACHE_PATH, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS phonemize (
                    fingerprint TEXT NOT NULL,
                    voice TEXT NOT NULL,
                    text_sha TEXT NOT NULL,
                    raw_stdout TEXT NOT NULL,
                    PRIMARY KEY (fingerprint, voice, text_sha)
                )
                """
            )
            _phonemize_cache = conn
            _phonemize_fingerprint = _espeak_build_fingerprint()
        except sqlite3.Error:
            _phonemize_cache = False
            return None
    return _phonemize_cache


def phonemize(text: str, espeak_lang: str) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    """Phonemize one utterance, via the raw-output cache when possible.

    A miss runs one eSpeak process (the only framing that is reliable — see
    phonemize_many's warning) and stores its raw stdout keyed by
    (build fingerprint, voice, sha256(text)). Hits skip the fork/exec and
    just re-parse, which turns full-corpus relabels with mostly-unchanged
    sentences from hours of process spawning into minutes.
    """
    conn = _phonemize_cache_conn()
    text_sha = hashlib.sha256(text.encode()).hexdigest() if conn else None
    if conn is not None:
        row = conn.execute(
            "SELECT raw_stdout FROM phonemize "
            "WHERE fingerprint=? AND voice=? AND text_sha=?",
            (_phonemize_fingerprint, espeak_lang, text_sha),
        ).fetchone()
        if row is not None:
            cached_lines = _espeak_output_lines(row[0])
            # Ignore (and delete) poisoned entries written before the stdin
            # fix, rather than requiring everyone to know to clear the cache.
            if cached_lines or not any(ch.isalpha() for ch in text):
                return _parse_espeak_ipa(" ".join(cached_lines))
            with conn:
                conn.execute(
                    "DELETE FROM phonemize "
                    "WHERE fingerprint=? AND voice=? AND text_sha=?",
                    (_phonemize_fingerprint, espeak_lang, text_sha),
                )
    result = _espeak_run_text(espeak_lang, text)
    # Never cache a suspicious empty result. A cache is only sound if what it
    # stores is what the tool would produce again — and an empty stdout for
    # text containing letters is the signature of an invocation failure, not
    # a property of the text. Storing it makes the failure PERMANENT and
    # invisible: fixing the invocation then changes nothing, because every
    # affected row is served from cache. That is exactly what happened with
    # the leading-dash bug, and it is why the fix appeared not to work.
    usable = bool(_espeak_output_lines(result.stdout)) or not any(
        ch.isalpha() for ch in text
    )
    if conn is not None and usable:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO phonemize VALUES (?, ?, ?, ?)",
                (_phonemize_fingerprint, espeak_lang, text_sha, result.stdout),
            )
    lines = _espeak_output_lines(result.stdout)
    return _parse_espeak_ipa(" ".join(lines))


def phonemize_many(
    requests: list[tuple[str, str]], *, batch_size: int = 8, desc: str = "espeak",
) -> list[tuple[list[str], list[int], list[tuple[int, int]]]]:
    """Phonemize many ``(text, voice)`` pairs with batched eSpeak processes.

    .. warning:: DO NOT USE for corpus labeling — the framing is unsound.
       eSpeak's ``--stdin`` output lines are CLAUSES, not input lines: a
       sentence with a comma emits two lines, and a line without terminal
       punctuation doesn't flush and merges into the next line's output.
       When a split and a merge land in the same chunk they compensate, the
       ``len(lines) == len(chunk)`` guard passes, and every row in the chunk
       is silently assigned a neighbor's phonemes (observed on 2-6%% of
       corpus rows, 2026-08-24; see scripts/verify_espeak_build.py). Fixing
       this needs a per-utterance delimiter that survives clause splitting —
       until then the corpus writer uses one :func:`phonemize` call per row.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    results: list[tuple[list[str], list[int], list[tuple[int, int]]] | None] = [
        None
    ] * len(requests)
    by_voice: dict[str, list[tuple[int, str]]] = {}
    for index, (text, voice) in enumerate(requests):
        by_voice.setdefault(voice, []).append((index, text))

    with tqdm(total=len(requests), desc=desc) as progress:
        for voice, voice_requests in by_voice.items():
            for start in range(0, len(voice_requests), batch_size):
                chunk = voice_requests[start:start + batch_size]
                # A manifest sentence is conceptually one utterance. Flatten
                # embedded newlines so they cannot change stdin framing.
                texts = [re.sub(r"[\r\n]+", " ", text) for _, text in chunk]
                result = subprocess.run(
                    _espeak_command(voice, stdin=True),
                    input="\n".join(texts) + "\n",
                    capture_output=True,
                    text=True,
                    check=True,
                    errors="replace",
                )
                lines = _espeak_output_lines(result.stdout)
                if len(lines) != len(chunk):
                    for index, text in chunk:
                        results[index] = phonemize(text, voice)
                else:
                    for (index, _), raw in zip(chunk, lines):
                        results[index] = _parse_espeak_ipa(raw)
                progress.update(len(chunk))

    if any(result is None for result in results):
        raise RuntimeError("internal error: missing batched eSpeak result")
    return [result for result in results if result is not None]


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAD_COMPUTE_BIN = REPO_ROOT / "vad_compare" / "target" / "release" / "vad_compute"
VAD_COMPUTE_MANIFEST = REPO_ROOT / "vad_compare" / "Cargo.toml"

# The single tarball every sky_*.yaml stages onto the GPU node. Lives under
# .work/ so it is gitignored, and matches .skyignore's `*.tar` so the workdir
# upload never carries it — file_mounts ships it deliberately instead.
DATASET_TAR = REPO_ROOT / ".work" / "pron_audio.tar"


# Some source corpora ship empty/corrupt recordings: Google FLEURS es_419
# has 490 clips (17.5% of the split) that are digital silence at the source,
# paired with valid transcripts — pure label noise if they reach training,
# and they fool speaker clustering into a fake "mega-cluster". We can't fix
# the upstream tar, so we drop any clip whose peak amplitude is below this
# floor. Real speech in our corpora peaks at 0.3-0.6; the silent clips peak
# at ~1e-4, so 1e-3 (-60 dBFS) separates them with two orders of magnitude
# of margin on each side. Source-agnostic: applies to every source/lang.
SILENCE_PEAK_FLOOR = 1e-3


def is_silent(path: Path) -> bool:
    """True if the recording is effectively digital silence (empty/corrupt
    source audio). Reads peak amplitude only — full read, since a clip can be
    silent at the head and have content later."""
    data, _ = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return len(data) == 0 or float(np.abs(data).max()) < SILENCE_PEAK_FLOOR


class SilenceCache:
    """Stat-validated cache for the full-file silence audit.

    The cache lives below ``data/audio/<lang>/.cache`` (excluded from the
    training tar). A hit is valid only while file size, nanosecond mtime, and
    the configured peak floor all match. Existing ``phonemes.jsonl`` rows can
    safely seed non-silent hits when that output is newer than the audio file:
    those rows could only have been written after passing this same guard.
    """

    def __init__(self, audio_dir: Path, phonemes_path: Path):
        cache_dir = audio_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        self.audio_dir = audio_dir
        self.conn = sqlite3.connect(cache_dir / "silence.sqlite3")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS silence (
                file TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                peak_floor REAL NOT NULL,
                silent INTEGER NOT NULL
            )
            """
        )
        self.pending = 0
        count = self.conn.execute("SELECT COUNT(*) FROM silence").fetchone()[0]
        if count == 0 and phonemes_path.exists():
            self._seed_known_nonsilent(phonemes_path)

    def _seed_known_nonsilent(self, phonemes_path: Path) -> None:
        output_mtime_ns = phonemes_path.stat().st_mtime_ns
        rows = []
        with open(phonemes_path) as source:
            for line in source:
                filename = json.loads(line)["file"]
                path = self.audio_dir / filename
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                if output_mtime_ns >= stat.st_mtime_ns:
                    rows.append((
                        filename, stat.st_size, stat.st_mtime_ns,
                        SILENCE_PEAK_FLOOR, 0,
                    ))
        self.conn.executemany(
            "INSERT OR REPLACE INTO silence VALUES (?, ?, ?, ?, ?)", rows,
        )
        self.conn.commit()
        if rows:
            print(f"seeded silence cache with {len(rows):,} known-good clips")

    def is_silent(self, path: Path) -> bool:
        stat = path.stat()
        filename = path.name
        row = self.conn.execute(
            "SELECT size, mtime_ns, peak_floor, silent FROM silence WHERE file = ?",
            (filename,),
        ).fetchone()
        if row is not None and row[:3] == (
            stat.st_size, stat.st_mtime_ns, SILENCE_PEAK_FLOOR,
        ):
            return bool(row[3])

        silent = is_silent(path)
        self.conn.execute(
            "INSERT OR REPLACE INTO silence VALUES (?, ?, ?, ?, ?)",
            (filename, stat.st_size, stat.st_mtime_ns,
             SILENCE_PEAK_FLOOR, int(silent)),
        )
        self.pending += 1
        if self.pending >= 1000:
            self.conn.commit()
            self.pending = 0
        return silent

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "SilenceCache":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def ensure_vad_compute_built() -> Path:
    """Build vad_compute if the release binary isn't there yet.

    vad_compute is a Rust binary (earshot framewise VAD). Phonemes and VAD
    are co-produced from the same audio so we build the binary lazily here
    rather than burdening the caller with a separate build step.
    """
    if VAD_COMPUTE_BIN.exists():
        return VAD_COMPUTE_BIN
    print("vad_compute binary not present; building (cargo build --release)...")
    subprocess.run(
        ["cargo", "build", "--release",
         "--manifest-path", str(VAD_COMPUTE_MANIFEST),
         "--bin", "vad_compute", "--quiet"],
        check=True,
    )
    if not VAD_COMPUTE_BIN.exists():
        raise RuntimeError(f"vad_compute build succeeded but binary not at {VAD_COMPUTE_BIN}")
    return VAD_COMPUTE_BIN


def regenerate_vad(phonemes_path: Path, audio_dir: Path) -> None:
    """Run vad_compute over a lang's phonemes.jsonl to (re)build vad.jsonl.

    vad_compute reads every file referenced by phonemes.jsonl and rewrites
    vad.jsonl from scratch, so this is safe to re-run after any extraction
    pass. Keeps vad coverage in lockstep with phonemes — otherwise newly-
    added clips silently train without the nonblank-head soft-regularizer.
    """
    vad_bin = ensure_vad_compute_built()
    vad_path = audio_dir / "vad.jsonl"
    subprocess.run(
        [str(vad_bin), str(phonemes_path), str(audio_dir), str(vad_path)],
        check=True,
    )


def newest_mtime(data_dir: Path) -> float:
    """Newest mtime under data_dir, ignoring .cache. Used to decide whether the
    staging tar is stale. Walks ~450k entries, which costs a second or two on
    SSD — cheap next to repacking 34 GB we already packed."""
    newest = 0.0
    stack = [data_dir]
    while stack:
        with os.scandir(stack.pop()) as it:
            for entry in it:
                if entry.name == ".cache":
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
    return newest


def build_dataset_tar(data_dir: Path, output: Path = DATASET_TAR) -> Path:
    """Pack data/audio into the one tarball the training launchers stage.

    Half a million loose wavs defeat every transport we've tried: rsync crawls,
    and the Hub caps repository commits at 256/hour, which turns a loose-file
    dataset repo into a multi-day upload that stalls out. So the dataset moves
    as a single file — sky's file_mounts rsyncs it to the node, the run: block
    untars it into ~/data, and the loader still opens loose wavs from there.

    Always packs the WHOLE data_dir, even under --langs: training reads every
    language, so a tar of one language would be a footgun.

    Writes to a .partial path and renames, so an interrupted run leaves the
    previous good tar in place rather than a truncated one a launcher would
    cheerfully upload.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_mtime >= newest_mtime(data_dir):
        size_gb = output.stat().st_size / 1e9
        print(f"staging tar up to date: {output} ({size_gb:.1f} GB)")
        return output

    # Named ".partial.tar", not ".tar.partial", so it still matches .skyignore's
    # `*.tar` — a leftover partial must never ride along in the workdir upload.
    tmp = output.with_name(output.stem + ".partial.tar")
    tmp.unlink(missing_ok=True)
    print(f"packing {data_dir} -> {output} ...")
    subprocess.run(
        ["tar", "-cf", str(tmp), "-C", str(data_dir), "--exclude=.cache", "."],
        check=True,
    )
    tmp.replace(output)
    size_gb = output.stat().st_size / 1e9
    print(f"wrote staging tar: {output} ({size_gb:.1f} GB)")
    return output


def refresh_mixed_script_exclusions() -> None:
    """Regenerate train/mixed_script_exclusions.jsonl from the manifests.

    Deterministic, manifest-derived, no API calls — so preprocess keeps its
    "self-contained" contract: the sidecar can never go stale relative to the
    manifests a run just processed. train.sh passes it to the trainer via
    --audit-path alongside the ASR-audit sidecars.
    """
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import build_mixed_script_exclusions
    finally:
        sys.path.remove(str(scripts_dir))
    build_mixed_script_exclusions.main()


def _eligible_languages(
    data_dir: Path, requested: list[str] | None, backend_paths: dict[str, Path],
) -> list[str]:
    """Resolve processable language directories in deterministic order."""
    languages = []
    for lang_dir in sorted(data_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name == ".cache":
            continue
        lang = lang_dir.name
        if requested and lang not in requested:
            continue
        if lang not in LANG_TO_ESPEAK and lang not in backend_paths:
            continue
        if not (lang_dir / "manifest.jsonl").exists():
            continue
        languages.append(lang)
    return languages


def _run_parallel_languages(
    args: argparse.Namespace, languages: list[str], backend_paths: dict[str, Path],
) -> None:
    """Run isolated per-language children, then pack once in the parent.

    Language outputs and caches are disjoint. Children always receive
    ``--no-pack``, preventing concurrent writes to the shared staging tar.
    Per-language logs keep concurrent tqdm output readable.
    """
    log_dir = REPO_ROOT / ".work" / "preprocess_parallel"
    log_dir.mkdir(parents=True, exist_ok=True)
    queued = list(languages)
    running: dict[str, tuple[subprocess.Popen, object, Path]] = {}
    failures: list[str] = []

    print(f"processing {len(languages)} languages with {args.jobs} workers")
    print(f"per-language logs: {log_dir}")
    while queued or running:
        while queued and len(running) < args.jobs:
            lang = queued.pop(0)
            log_path = log_dir / f"{lang}.log"
            log_file = open(log_path, "w")
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--data-dir", str(args.data_dir),
                "--langs", lang,
                "--jobs", "1",
                "--no-pack",
                "--espeak-batch-size", str(args.espeak_batch_size),
            ]
            if args.skip_vad:
                cmd.append("--skip-vad")
            if args.allow_noncommercial:
                cmd.append("--allow-noncommercial")
            if lang in backend_paths:
                cmd.extend([
                    "--phoneme-backend", f"{lang}={backend_paths[lang]}",
                ])
            process = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT,
            )
            running[lang] = (process, log_file, log_path)
            print(f"started {lang}: pid {process.pid} -> {log_path}", flush=True)

        completed = []
        for lang, (process, log_file, log_path) in running.items():
            returncode = process.poll()
            if returncode is None:
                continue
            log_file.close()
            completed.append(lang)
            if returncode == 0:
                print(f"completed {lang}", flush=True)
            else:
                failures.append(lang)
                print(
                    f"FAILED {lang} (exit {returncode}); see {log_path}",
                    flush=True,
                )
        for lang in completed:
            del running[lang]
        if running and not completed:
            time.sleep(1)

    if failures:
        raise RuntimeError(
            "language preprocessing failed: " + ", ".join(failures)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "data" / "audio")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Restrict to these lang codes (default: all dirs). "
                             "Lets independent languages run as parallel jobs.")
    parser.add_argument("--skip-vad", action="store_true",
                        help="Don't regenerate vad.jsonl after phonemization. "
                             "Use only when you're certain vad coverage is "
                             "current — by default we keep vad in lockstep "
                             "with phonemes.")
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Process this many languages concurrently. Each language writes "
             "an isolated log and the parent packs the dataset exactly once.",
    )
    parser.add_argument(
        "--espeak-batch-size", type=int, default=8,
        help="Utterances per eSpeak --stdin invocation (default: 8; larger "
             "batches are slower in eSpeak).",
    )
    parser.add_argument(
        "--no-pack", action="store_true", help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-noncommercial", action="store_true",
        help="Include CC BY-NC/noncommercial source rows. By default these "
             "remain in the auditable manifest/sidecars but are excluded from "
             "phonemes.jsonl so normal training output is commercially usable.",
    )
    parser.add_argument(
        "--phoneme-backend", action="append", default=None, metavar="LANG=JSONL",
        help="Override the sidecar for LANG with a specific JSONL instead of "
             "the auto-refreshed canonical one. Rarely needed: backend "
             "languages regenerate their audit + sidecar automatically. Rows "
             "must be hash-bound to the manifest sentence; missing/stale rows "
             "fail closed.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.espeak_batch_size < 1:
        parser.error("--espeak-batch-size must be at least 1")
    backend_paths = parse_phoneme_backend_args(args.phoneme_backend)

    languages = _eligible_languages(args.data_dir, args.langs, backend_paths)
    if args.jobs > 1 and len(languages) > 1:
        _run_parallel_languages(args, languages, backend_paths)
        if not args.no_pack:
            refresh_mixed_script_exclusions()
            build_dataset_tar(args.data_dir)
        return

    langs_with_unknowns: list[str] = []
    for lang_dir in sorted(args.data_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if args.langs and lang not in args.langs:
            continue
        if lang not in LANG_TO_ESPEAK and lang not in backend_paths:
            print(f"Skipping {lang} (no espeak mapping or external backend)")
            continue

        manifest_path = lang_dir / "manifest.jsonl"
        phonemes_path = lang_dir / "phonemes.jsonl"

        # download_pimsleur.py creates the lang dir before it knows whether
        # any clips will match the target language; some dirs may end up
        # holding only `pimsleur_processed.txt` and no manifest.
        if not manifest_path.exists():
            print(f"Skipping {lang} (no manifest.jsonl)")
            continue

        records = []
        with open(manifest_path) as f:
            for line in f:
                records.append(json.loads(line))

        backend_path = backend_paths.get(lang)
        if backend_path is None and lang in BACKEND_REQUIRED_LANGS:
            # eSpeak is never the label source for these languages. Refresh
            # the audit + sidecar chain right here so a preprocess run after
            # new data lands is complete on its own.
            backend_path = ensure_backend_sidecar(lang, args.data_dir)
        backend_records = (
            load_phoneme_backend(backend_path, lang) if backend_path else None
        )
        if backend_records is None and lang in BACKEND_REQUIRED_LANGS:
            # Unreachable via the branch above, which always builds a sidecar.
            # Asserted anyway: this is the invariant that keeps eSpeak's
            # phoneme inventory out of a language it cannot represent, and it
            # must not become false by some later refactor of that branch.
            raise ValueError(
                f"{lang} requires the {required_backend_provider(lang)!r} "
                f"phoneme backend; refusing to fall back to eSpeak. "
                f"See PHONEME_BACKENDS.md."
            )
        if backend_records is not None:
            print(f"{lang}: using external phoneme backend {backend_path} "
                  f"({len(backend_records)} rows)")

        # Acoustics get the last word on the accent factor: the sidecar's
        # accent is what the dictionary says, and this file lists the clips
        # where measured F0 says otherwise.
        accent_exclusions = load_accent_exclusions(
            Path(__file__).resolve().parents[1] / f"{lang}_pitch_accent_exclusions.jsonl"
        )
        if accent_exclusions:
            print(f"{lang}: withholding pitch accent on {len(accent_exclusions)} "
                  f"clips the acoustic audit rejected")

        # Load rhythmic-group sidecar for override languages. Missing sidecar
        # just means no overrides apply — espeak stress is used as-is.
        stress_overrides: dict[str, list[str]] = {}
        if lang in OVERRIDE_LANGS:
            override_path = lang_dir / "stress_overrides.jsonl"
            if override_path.exists():
                stress_overrides = load_stress_overrides(override_path)
                print(f"{lang}: loaded {len(stress_overrides)} stress overrides "
                      f"from {override_path.name}")
            else:
                print(f"{lang}: WARNING — no stress_overrides.jsonl; espeak's "
                      f"per-word stress will be used (systematically wrong for "
                      f"this language). Run train/relabel-french/ to generate it.")

        override_applied = 0
        override_align_failures = 0
        backend_excluded = 0
        license_excluded = 0
        silent_dropped = 0
        prepared_records: list[dict] = []
        with SilenceCache(lang_dir, phonemes_path) as silence_cache:
            for rec in tqdm(records, desc=f"{lang} silence"):
                license_name = str(rec.get("license") or "")
                if not args.allow_noncommercial and (
                        "BY-NC" in license_name.upper()
                        or "NONCOMMERCIAL" in license_name.upper()):
                    license_excluded += 1
                    continue
                # Drop empty/corrupt source recordings. The stat-validated
                # cache avoids decoding unchanged WAVs on every label rebuild.
                if silence_cache.is_silent(lang_dir / rec["file"]):
                    silent_dropped += 1
                    continue
                prepared_records.append(rec)

        phonemized_results: list[
            tuple[list[str], list[int], list[tuple[int, int]]] | None
        ]
        if backend_records is None:
            # One espeak invocation per utterance — the only framing that is
            # actually reliable. phonemize_many's stdin batching mislabels
            # rows: espeak's output lines are CLAUSES, not input lines (a
            # comma sentence emits two lines; a punctuation-less line doesn't
            # flush and merges into the next), and when splits and merges
            # compensate, the line-count guard passes and every row in the
            # chunk silently gets a neighbor's phonemes. Caught 2026-08-24 by
            # scripts/verify_espeak_build.py (2-6%% of rows misassigned).
            phonemized_results = [
                phonemize(rec["sentence"],
                          rec.get("espeak_voice") or LANG_TO_ESPEAK[lang])
                for rec in tqdm(prepared_records, desc=f"{lang} phonemize")
            ]
        else:
            phonemized_results = [None] * len(prepared_records)

        # token -> (count, first-example sentence). Buffered per-lang so we
        # can report all unknowns and skip writing the file if any are found
        # — partial output would silently train on a vocab-mismatched corpus.
        unknown_examples: dict[str, tuple[int, str]] = {}
        # Sentences that have letters but phonemized to nothing — see the
        # check further down where these are collected.
        empty_phoneme_examples: list[str] = []
        entries: list[dict] = []
        for rec, phonemized_result in zip(prepared_records, phonemized_results):
            # Prefer the per-record espeak voice if the manifest stored
            # one (Pimsleur does, since "por" covers both Brazilian and
            # European Portuguese, "spa" covers Castilian + Latin
            # American, etc — these dialects share a lang code but
            # need different espeak voices for faithful phoneme labels).
            # Fall back to the canonical voice for FLEURS / Tatoeba rows.
            # Note: Tatoeba uses `voice` for the uploader's display
            # name (a human, not an espeak voice), so we deliberately
            # read `espeak_voice` only.
            backend_rec = backend_records.get(rec["file"]) if backend_records is not None else None
            if backend_records is not None:
                if backend_rec is None:
                    raise ValueError(
                        f"{backend_path}: no transcription for {lang}/{rec['file']}"
                    )
                sentence_hash = hashlib.sha256(rec["sentence"].encode()).hexdigest()
                if backend_rec["sentence_sha256"] != sentence_hash:
                    raise ValueError(
                        f"{backend_path}: stale transcription for {lang}/{rec['file']}"
                    )
                if backend_rec.get("exclude_reason"):
                    backend_excluded += 1
                    continue
                phonemes = list(backend_rec["phonemes"])
                stress = list(backend_rec["stress"])
                word_spans = []
            else:
                if phonemized_result is None:
                    raise RuntimeError("missing batched eSpeak result")
                phonemes, stress, word_spans = phonemized_result
            if backend_rec is None and rec["file"] in stress_overrides:
                new_stress = apply_stress_override(
                    phonemes, word_spans, rec["sentence"],
                    stress_overrides[rec["file"]],
                )
                if new_stress is not None:
                    stress = new_stress
                    override_applied += 1
                else:
                    override_align_failures += 1
            phonemes, stress, unknowns = validate_phonemes(phonemes, stress, lang)
            for u in unknowns:
                if u in unknown_examples:
                    count, example = unknown_examples[u]
                    unknown_examples[u] = (count + 1, example)
                else:
                    unknown_examples[u] = (1, rec["sentence"])
            # An utterance with letters in it must produce phonemes. When it
            # doesn't, that is a labeling failure, not a property of the
            # sentence — and it is invisible downstream, because dataset.py
            # drops empty-target rows as `no_phonemes` and training simply
            # proceeds with slightly less data. Count them here and report at
            # the end of the language, so the failure is attributable to its
            # cause instead of showing up as an unexplained row-count drift.
            if not phonemes and any(ch.isalpha() for ch in rec["sentence"]):
                empty_phoneme_examples.append(rec["sentence"])
            entry = {
                "file": rec["file"],
                "lang": lang,
                "sentence": rec["sentence"],
                "phonemes": phonemes,
                "stress": stress,
                "source": rec.get("source"),
                "license": rec.get("license"),
                "phoneme_backend": (
                    backend_rec.get("backend", "external")
                    if backend_rec is not None else "espeak"
                ),
            }
            if backend_rec is not None:
                for k in (
                    "tone", "pitch_accent", "pitch_accent_exclude_reason",
                    "syllables", "stress_source",
                ):
                    if k in backend_rec:
                        entry[k] = backend_rec[k]
                acoustic_reason = accent_exclusions.get(rec["file"])
                if acoustic_reason is not None:
                    entry.pop("pitch_accent", None)
                    entry["pitch_accent_exclude_reason"] = acoustic_reason
            # Propagate Whisper signal fields from the manifest. Only
            # present for Pimsleur (extracted with download_pimsleur.py).
            # FLEURS / Tatoeba rows lack these and pass them through as None.
            for k in ("whisper_avg_logprob", "whisper_no_speech_prob",
                     "whisper_compression_ratio", "duration_sec"):
                if k in rec:
                    entry[k] = rec[k]
            entries.append(entry)

        if empty_phoneme_examples:
            # Loud, but not fatal: unlike an unknown token (which would train
            # a vocab mismatch), an empty row is merely lost. Refusing to write
            # the file would block a whole language over a handful of rows, so
            # report precisely instead and let the operator judge.
            print(f"\nWARNING: {lang} has {len(empty_phoneme_examples):,} "
                  f"sentence(s) with letters that phonemized to NOTHING. "
                  f"These rows are written but dataset.py will drop them as "
                  f"`no_phonemes`, so they are silently absent from training.")
            for example in empty_phoneme_examples[:5]:
                print(f"    {example[:100]!r}")
            if len(empty_phoneme_examples) > 5:
                print(f"    ... and {len(empty_phoneme_examples) - 5:,} more")
            print("  Usually an eSpeak invocation problem rather than a "
                  "property of the text — check the voice and that the text "
                  "reaches eSpeak after `--`.")

        if unknown_examples:
            total = sum(c for c, _ in unknown_examples.values())
            print(f"\nERROR: {lang} has {len(unknown_examples)} unknown token "
                  f"type(s) ({total:,} occurrences) not in vocab "
                  f"and not in TOKEN_BLACKLIST:")
            for tok, (count, example) in sorted(
                unknown_examples.items(), key=lambda kv: -kv[1][0]
            ):
                codepoints = " ".join(f"U+{ord(c):04X}" for c in tok)
                print(f"  {tok!r:>10}  {count:>6,}x   {codepoints}")
                print(f"             example sentence: {example[:100]!r}")
            print(f"\n  Fix by either:")
            print(f"    (a) adding the token to TOKEN_BLACKLIST (intentional drop), or")
            print(f"    (b) extending phonemize() to fold it into the preceding phoneme")
            print(f"        (treat as combining diacritic) or split it further, or")
            print(f"    (c) transforming it upstream (espeak voice change, text cleaning).")
            print(f"  NOT writing {phonemes_path} — fix the above and re-run.")
            langs_with_unknowns.append(lang)
            continue

        with open(phonemes_path, "w") as out:
            for entry in entries:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"{lang}: wrote {len(entries)} entries to {phonemes_path}")
        if silent_dropped:
            print(f"{lang}: dropped {silent_dropped} silent/empty recording(s) "
                  f"(peak < {SILENCE_PEAK_FLOOR:g}, e.g. corrupt FLEURS source audio)")
        if backend_excluded:
            print(f"{lang}: external backend explicitly excluded "
                  f"{backend_excluded} recording(s)")
        if license_excluded:
            print(f"{lang}: excluded {license_excluded} noncommercial recording(s); "
                  f"pass --allow-noncommercial only for an explicitly NC run")
        if lang in OVERRIDE_LANGS and stress_overrides:
            print(f"{lang}: applied stress override to {override_applied} records, "
                  f"{override_align_failures} alignment failures "
                  f"(fell back to espeak stress)")

        run_narrowing(lang, args.data_dir)

        if not args.skip_vad:
            print(f"{lang}: regenerating vad.jsonl ...")
            regenerate_vad(phonemes_path, lang_dir)

    if langs_with_unknowns:
        print(f"\n{len(langs_with_unknowns)} language(s) had unknown tokens "
              f"and were NOT written: {', '.join(langs_with_unknowns)}")
        sys.exit(1)

    # Last steps, unconditionally: refresh the manifest-derived exclusion
    # sidecar and pack the dataset. Doing this here is the whole point of
    # preprocess being self-contained — finishing this script means there is
    # nothing left to do before `sky launch`.
    if not args.no_pack:
        refresh_mixed_script_exclusions()
        build_dataset_tar(args.data_dir)


if __name__ == "__main__":
    main()

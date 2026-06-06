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
import json
import re
import os
import subprocess
import sys
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
        # Italian has no lax high vowels; espeak emits them spuriously
        # (ɪ ×1230, ʊ ×804 in the audit corpus).
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


def phonemize(text: str, espeak_lang: str) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    """Run espeak-ng and parse IPA output into (phonemes, stress_labels, word_spans).

    Stress markers ˈ and ˌ precede a syllable. The stress attaches to that
    syllable's vowel nucleus only (plus any length/nasalization diacritics).
    A new vowel after a consonant marks a new syllable → resets stress to none
    unless a fresh marker appeared.

    word_spans[i] = (start, end) indices into `phonemes` for the i-th word
    espeak emitted. Word breaks are the characters in WORD_BOUNDARIES — same
    characters the parser uses to reset stress state. Empty spans (consecutive
    boundaries) are dropped, so word_spans aligns with non-empty text words.
    """
    # ESPEAK_NG_BIN / ESPEAK_NG_DATA_PATH let you point at a non-system build
    # (e.g. a custom patched espeak-ng under ~/coding/tmp/espeak-ng/build).
    # Defaults fall back to the system `espeak-ng` on PATH.
    espeak_bin = os.environ.get("ESPEAK_NG_BIN", "espeak-ng")
    data_path = os.environ.get("ESPEAK_NG_DATA_PATH")
    cmd = [espeak_bin]
    if data_path:
        cmd.append(f"--path={data_path}")
    cmd.extend(["-v", espeak_lang, "-q", "--ipa", "-x", text])
    # errors="replace": patched espeak (master-232) occasionally emits non-UTF8
    # debug warnings to stderr for certain inputs (e.g. some Russian/Spanish
    # sentences). text=True with default 'strict' decoding would crash the whole
    # pipeline on those bytes even though stdout (the IPA we care about) is fine.
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, errors="replace",
    )
    raw = result.stdout.strip()

    # espeak switches voice for foreign words (loanwords, names) and brackets
    # the switch with (lang) markers, e.g. "football" -> "(en)fˈʊtbɔːl(fr)".
    # The parens are blacklisted but the 2-letter codes are valid phonemes
    # (e/n/f/r…), so without this they'd silently inject spurious segments into
    # loanword labels — and the unknown-token safety net can't catch it. Strip
    # the markers; the switched word body stays and is parsed/remapped normally
    # (the French vowel remap nativises the English vowels espeak used). ~2.3%
    # of French sentences carry these.
    raw = re.sub(r"\([a-z]{2,4}\)", "", raw)

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


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAD_COMPUTE_BIN = REPO_ROOT / "vad_compare" / "target" / "release" / "vad_compute"
VAD_COMPUTE_MANIFEST = REPO_ROOT / "vad_compare" / "Cargo.toml"


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
    args = parser.parse_args()

    langs_with_unknowns: list[str] = []
    for lang_dir in sorted(args.data_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if args.langs and lang not in args.langs:
            continue
        if lang not in LANG_TO_ESPEAK:
            print(f"Skipping {lang} (no espeak mapping)")
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
        silent_dropped = 0
        # token -> (count, first-example sentence). Buffered per-lang so we
        # can report all unknowns and skip writing the file if any are found
        # — partial output would silently train on a vocab-mismatched corpus.
        unknown_examples: dict[str, tuple[int, str]] = {}
        entries: list[dict] = []
        for rec in tqdm(records, desc=lang):
            # Drop empty/corrupt source recordings before doing any work —
            # silent audio paired with a transcript is pure label noise.
            if is_silent(lang_dir / rec["file"]):
                silent_dropped += 1
                continue
            # Prefer the per-record espeak voice if the manifest stored
            # one (Pimsleur does, since "por" covers both Brazilian and
            # European Portuguese, "spa" covers Castilian + Latin
            # American, etc — these dialects share a lang code but
            # need different espeak voices for faithful phoneme labels).
            # Fall back to the canonical voice for FLEURS / Tatoeba rows.
            # Note: Tatoeba uses `voice` for the uploader's display
            # name (a human, not an espeak voice), so we deliberately
            # read `espeak_voice` only.
            espeak_voice = rec.get("espeak_voice") or LANG_TO_ESPEAK[lang]
            phonemes, stress, word_spans = phonemize(rec["sentence"], espeak_voice)
            if rec["file"] in stress_overrides:
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
            entry = {
                "file": rec["file"],
                "lang": lang,
                "sentence": rec["sentence"],
                "phonemes": phonemes,
                "stress": stress,
                "source": rec.get("source"),
            }
            # Propagate Whisper signal fields from the manifest. Only
            # present for Pimsleur (extracted with download_pimsleur.py).
            # FLEURS / Tatoeba rows lack these and pass them through as None.
            for k in ("whisper_avg_logprob", "whisper_no_speech_prob",
                     "whisper_compression_ratio", "duration_sec"):
                if k in rec:
                    entry[k] = rec[k]
            entries.append(entry)

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
        if lang in OVERRIDE_LANGS and stress_overrides:
            print(f"{lang}: applied stress override to {override_applied} records, "
                  f"{override_align_failures} alignment failures "
                  f"(fell back to espeak stress)")

        if not args.skip_vad:
            print(f"{lang}: regenerating vad.jsonl ...")
            regenerate_vad(phonemes_path, lang_dir)

    if langs_with_unknowns:
        print(f"\n{len(langs_with_unknowns)} language(s) had unknown tokens "
              f"and were NOT written: {', '.join(langs_with_unknowns)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

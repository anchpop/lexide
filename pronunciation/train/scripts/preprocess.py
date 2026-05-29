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
import subprocess
import sys
from functools import cache
from pathlib import Path

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

# Diacritics that continue the preceding vowel (length, nasalization, etc.).
# Appended to the previous phoneme so the combined string (e.g. "ɛ̃", "iː") matches
# the tokenizer's precomposed vocab — emitting them standalone made them UNK and
# silently stripped nasalization/length from training labels.
VOWEL_CONTINUATIONS = set("ːˑ̠̞̯̥̃̊̈")

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


# Tokens that aren't in the vocab but that we explicitly choose to drop.
# Every entry is a deliberate decision; anything not in this set AND not in
# vocab raises a hard error at preprocess time so the choice has to be made
# rather than silently lost.
TOKEN_BLACKLIST: set[str] = {
    # Punctuation leaking from the source text into espeak's output. These
    # aren't phonemes — they're stray characters espeak passes through when
    # the input contained them. Safe to drop.
    "(",
    ")",
    '"',
    "^",
}


@cache
def _tokenizer_vocab() -> set[str]:
    """Load the borrowed tokenizer's vocab once. Cached for the whole run."""
    from transformers import Wav2Vec2CTCTokenizer
    tok = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_NAME)
    return set(tok.get_vocab().keys())


def validate_phonemes(
    phonemes: list[str], stress: list[int],
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
    vocab = _tokenizer_vocab()
    out_phonemes: list[str] = []
    out_stress: list[int] = []
    unknowns: set[str] = set()
    for p, s in zip(phonemes, stress):
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
    result = subprocess.run(
        ["espeak-ng", "-v", espeak_lang, "-q", "--ipa", "-x", text],
        capture_output=True, text=True, check=True,
    )
    raw = result.stdout.strip()

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
        # token -> (count, first-example sentence). Buffered per-lang so we
        # can report all unknowns and skip writing the file if any are found
        # — partial output would silently train on a vocab-mismatched corpus.
        unknown_examples: dict[str, tuple[int, str]] = {}
        entries: list[dict] = []
        for rec in tqdm(records, desc=lang):
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
            phonemes, stress, unknowns = validate_phonemes(phonemes, stress)
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

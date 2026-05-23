"""Phonemize all sentences in the dataset using espeak-ng, keeping stress marks.

Writes a per-language JSONL with entries:
    {"file": "abc123.wav", "lang": "eng",
     "phonemes": ["h", "ɛ", "l", ...],
     "stress": [0, 0, 0, ...]}

These pre-computed phoneme sequences are used at training time to avoid
subprocess overhead and ensure determinism.
"""

import argparse
import json
import subprocess
from pathlib import Path

from tqdm import tqdm

LANG_TO_ESPEAK = {
    "eng": "en-us",
    "deu": "de",
    "fra": "fr-fr",
    "ita": "it",
    "por": "pt-br",
    "spa": "es",
    "rus": "ru",
}

STRESS_NONE = 0
STRESS_PRIMARY = 1
STRESS_SECONDARY = 2

# IPA vowels (monophthongs + near-variants used by espeak-ng across our languages)
IPA_VOWELS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝᵻ")

# Diacritics that continue the preceding vowel (length, nasalization, etc.)
VOWEL_CONTINUATIONS = set("ːˑ̠̞̯̥̃̊̈")

WORD_BOUNDARIES = set(" \t\n|_-")


def phonemize(text: str, espeak_lang: str) -> tuple[list[str], list[int]]:
    """Run espeak-ng and parse IPA output into (phonemes, stress_labels).

    Stress markers ˈ and ˌ precede a syllable. The stress attaches to that
    syllable's vowel nucleus only (plus any length/nasalization diacritics).
    A new vowel after a consonant marks a new syllable → resets stress to none
    unless a fresh marker appeared.
    """
    result = subprocess.run(
        ["espeak-ng", "-v", espeak_lang, "-q", "--ipa", "-x", text],
        capture_output=True, text=True, check=True,
    )
    raw = result.stdout.strip()

    phonemes = []
    stress = []
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
            pending_stress = None
            current_stress = STRESS_NONE
            in_vowel = False
        elif char in IPA_VOWELS:
            if pending_stress is not None:
                # Stressed vowel: consume the pending marker
                current_stress = pending_stress
                pending_stress = None
            elif not in_vowel:
                # New syllable after a consonant → unstressed
                current_stress = STRESS_NONE
            # else: continuation of diphthong (vowel right after vowel) → inherit
            in_vowel = True
            phonemes.append(char)
            stress.append(current_stress)
        elif char in VOWEL_CONTINUATIONS:
            # Length mark, nasalization, etc. — combining diacritics in espeak's
            # output. Append to the previous phoneme so the combined string (e.g.
            # "ɛ̃", "iː", "ɐ̃") matches the tokenizer's *precomposed* vocab tokens.
            # Emitting them as standalone tokens caused them to be looked up
            # individually, which made the diacritic UNK and silently stripped
            # nasalization / length / etc. from training labels — the dominant
            # cause of confident nasal-vowel denasalization in xls-r-2b-full.
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

    return phonemes, stress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "data" / "audio")
    args = parser.parse_args()

    for lang_dir in sorted(args.data_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if lang not in LANG_TO_ESPEAK:
            print(f"Skipping {lang} (no espeak mapping)")
            continue

        manifest_path = lang_dir / "manifest.jsonl"
        phonemes_path = lang_dir / "phonemes.jsonl"

        records = []
        with open(manifest_path) as f:
            for line in f:
                records.append(json.loads(line))

        with open(phonemes_path, "w") as out:
            for rec in tqdm(records, desc=lang):
                phonemes, stress = phonemize(rec["sentence"], LANG_TO_ESPEAK[lang])
                entry = {
                    "file": rec["file"],
                    "lang": lang,
                    "sentence": rec["sentence"],
                    "phonemes": phonemes,
                    "stress": stress,
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"{lang}: wrote {len(records)} entries to {phonemes_path}")


if __name__ == "__main__":
    main()

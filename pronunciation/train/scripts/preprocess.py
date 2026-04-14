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


def phonemize(text: str, espeak_lang: str) -> tuple[list[str], list[int]]:
    """Run espeak-ng and parse IPA output into (phonemes, stress_labels).

    Stress markers ˈ and ˌ attach to all following phonemes until the next
    word boundary or stress mark.
    """
    result = subprocess.run(
        ["espeak-ng", "-v", espeak_lang, "-q", "--ipa", "-x", text],
        capture_output=True, text=True, check=True,
    )
    raw = result.stdout.strip()

    phonemes = []
    stress = []
    current_stress = STRESS_NONE

    for char in raw:
        if char == "ˈ":
            current_stress = STRESS_PRIMARY
        elif char == "ˌ":
            current_stress = STRESS_SECONDARY
        elif char in " \t\n|_":
            # Word/syllable boundary — reset stress
            current_stress = STRESS_NONE
        else:
            phonemes.append(char)
            stress.append(current_stress)

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

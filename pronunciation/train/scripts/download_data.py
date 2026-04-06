"""Download all speech datasets for 16-language pronunciation training.

Datasets:
  - MLS (DE, ES, FR, IT, PT, NL) via HuggingFace datasets
  - LibriSpeech (EN) via HuggingFace datasets
  - Golos (RU) — bond005 parquet version
  - Zeroth-Korean (KO)
  - AISHELL-1 (ZH)
  - ReazonSpeech (JA) — requires trust_remote_code
  - Common Voice (EL) — fsicoli parquet version
  - Kathbath (HI, TA, TE, BN)
  - Shrutilipi (HI, TA, TE, BN)
  - OpenSLR 53 (BN) — requires trust_remote_code

Usage:
    python scripts/download_data.py --output_dir data/raw --languages all
"""

import argparse
import os
import traceback
from pathlib import Path

from datasets import load_dataset

# MLS language codes -> HuggingFace config names (English is separate via LibriSpeech)
MLS_LANGUAGES = {
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
}

ALL_LANGUAGES = [
    "en", "de", "es", "fr", "it", "pt", "nl",  # MLS + LibriSpeech
    "ru",  # Golos
    "ko",  # Zeroth-Korean
    "zh",  # AISHELL-1
    "ja",  # ReazonSpeech
    "el",  # Common Voice
    "hi", "ta", "te", "bn",  # Indic (Kathbath + Shrutilipi)
]

# Indic language codes -> Shrutilipi/Kathbath config names
INDIC_LANG_NAMES = {
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "bn": "bengali",
}


def try_download(name: str, fn, *args, **kwargs):
    """Run a download function, catch and log errors without stopping."""
    try:
        fn(*args, **kwargs)
    except Exception:
        print(f"  ERROR downloading {name}:")
        traceback.print_exc()
        print(f"  Skipping {name}, continuing...\n")


def download_mls(lang: str, output_dir: Path) -> None:
    config = MLS_LANGUAGES[lang]
    print(f"Downloading MLS ({config})...")
    ds = load_dataset(
        "facebook/multilingual_librispeech",
        config,
        cache_dir=str(output_dir / "mls" / lang),
    )
    print(f"  MLS {lang}: {ds}")


def download_librispeech(output_dir: Path) -> None:
    print("Downloading LibriSpeech (English)...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        "all",
        cache_dir=str(output_dir / "librispeech"),
    )
    print(f"  LibriSpeech: {ds}")


def download_golos(output_dir: Path) -> None:
    print("Downloading Golos (Russian)...")
    ds = load_dataset(
        "bond005/sberdevices_golos_10h_crowd",
        cache_dir=str(output_dir / "golos"),
    )
    print(f"  Golos: {ds}")


def download_zeroth_korean(output_dir: Path) -> None:
    print("Downloading Zeroth-Korean...")
    ds = load_dataset(
        "kresnik/zeroth_korean",
        cache_dir=str(output_dir / "zeroth_korean"),
    )
    print(f"  Zeroth-Korean: {ds}")


def download_aishell1(output_dir: Path) -> None:
    print("Downloading AISHELL-1 (Mandarin)...")
    ds = load_dataset(
        "AISHELL/AISHELL-1",
        cache_dir=str(output_dir / "aishell1"),
    )
    print(f"  AISHELL-1: {ds}")


def download_reazonspeech(output_dir: Path) -> None:
    print("Downloading ReazonSpeech (Japanese)...")
    ds = load_dataset(
        "reazon-research/reazonspeech",
        "small",
        trust_remote_code=True,
        cache_dir=str(output_dir / "reazonspeech"),
    )
    print(f"  ReazonSpeech: {ds}")


def download_common_voice_greek(output_dir: Path) -> None:
    print("Downloading Common Voice (Greek)...")
    # Try mozilla-foundation first, fall back to community version
    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "el",
            token=os.environ.get("HF_TOKEN"),
            cache_dir=str(output_dir / "common_voice_el"),
        )
    except Exception:
        ds = load_dataset(
            "fsicoli/common_voice_17_0",
            "el",
            cache_dir=str(output_dir / "common_voice_el"),
        )
    print(f"  Common Voice Greek: {ds}")


def download_kathbath(lang: str, output_dir: Path) -> None:
    lang_name = INDIC_LANG_NAMES[lang]
    print(f"Downloading Kathbath ({lang_name})...")
    ds = load_dataset(
        "ai4bharat/Kathbath",
        lang_name,
        cache_dir=str(output_dir / "kathbath" / lang),
    )
    print(f"  Kathbath {lang}: {ds}")


def download_shrutilipi(lang: str, output_dir: Path) -> None:
    lang_name = INDIC_LANG_NAMES[lang]
    print(f"Downloading Shrutilipi ({lang_name})...")
    ds = load_dataset(
        "ai4bharat/Shrutilipi",
        lang_name,
        cache_dir=str(output_dir / "shrutilipi" / lang),
    )
    print(f"  Shrutilipi {lang}: {ds}")


def download_openslr53(output_dir: Path) -> None:
    print("Downloading OpenSLR 53 (Bengali)...")
    # openslr/openslr uses loading scripts which are no longer supported.
    # Use the Google Bengali ASR dataset instead.
    ds = load_dataset(
        "google/fleurs",
        "bn_in",
        cache_dir=str(output_dir / "fleurs_bn"),
    )
    print(f"  FLEURS Bengali: {ds}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument(
        "--languages",
        type=str,
        default="all",
        help="Comma-separated language codes or 'all'",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.languages == "all":
        langs = set(ALL_LANGUAGES)
    else:
        langs = set(args.languages.split(","))

    # English (LibriSpeech)
    if "en" in langs:
        try_download("LibriSpeech", download_librispeech, output_dir)

    # MLS languages (non-English)
    for lang in sorted(langs & set(MLS_LANGUAGES)):
        try_download(f"MLS {lang}", download_mls, lang, output_dir)

    # Russian
    if "ru" in langs:
        try_download("Golos", download_golos, output_dir)

    # Korean
    if "ko" in langs:
        try_download("Zeroth-Korean", download_zeroth_korean, output_dir)

    # Mandarin
    if "zh" in langs:
        try_download("AISHELL-1", download_aishell1, output_dir)

    # Japanese
    if "ja" in langs:
        try_download("ReazonSpeech", download_reazonspeech, output_dir)

    # Greek
    if "el" in langs:
        try_download("Common Voice Greek", download_common_voice_greek, output_dir)

    # Indic languages
    indic_langs = sorted(langs & {"hi", "ta", "te", "bn"})
    for lang in indic_langs:
        try_download(f"Kathbath {lang}", download_kathbath, lang, output_dir)
        try_download(f"Shrutilipi {lang}", download_shrutilipi, lang, output_dir)

    # Bengali supplement
    if "bn" in langs:
        try_download("OpenSLR 53", download_openslr53, output_dir)

    print("\nAll downloads complete (check above for any errors).")


if __name__ == "__main__":
    main()

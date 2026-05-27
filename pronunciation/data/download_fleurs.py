#!/usr/bin/env python3
"""Download FLEURS dataset and save as WAV files with manifest."""

import argparse
import csv
import hashlib
import json
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

LANG_CONFIG = {
    "eng": "en_us",
    "deu": "de_de",
    "fra": "fr_fr",
    "ita": "it_it",
    "por": "pt_br",
    "spa": "es_419",
    "rus": "ru_ru",
}


def clean_tsv_field(value: str) -> str:
    """Normalize literal FLEURS TSV field quoting without CSV quote parsing."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    elif value.startswith('"') and value.count('"') == 1:
        value = value[1:]
    elif value.endswith('"') and value.count('"') == 1:
        value = value[:-1]
    return value


def sentence_hash(sentence: str) -> str:
    """First 16 hex chars of SHA256."""
    return hashlib.sha256(sentence.encode()).hexdigest()[:16]


def download_for_language(lang: str, output_root: Path):
    fleurs_code = LANG_CONFIG[lang]

    # Download TSV and audio tar
    print(f"Downloading FLEURS {fleurs_code} metadata...")
    tsv_path = hf_hub_download("google/fleurs", f"data/{fleurs_code}/train.tsv", repo_type="dataset")
    print(f"Downloading FLEURS {fleurs_code} audio...")
    tar_path = hf_hub_download("google/fleurs", f"data/{fleurs_code}/audio/train.tar.gz", repo_type="dataset")

    # Parse TSV: columns are id, filename, transcription, normalized, chars, num_samples, gender
    sentences_by_filename = {}
    with open(tsv_path, newline="") as f:
        for line_no, row in enumerate(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE), start=1):
            if len(row) != 7:
                raise ValueError(f"Malformed FLEURS TSV row {line_no}: expected 7 fields, got {len(row)}")
            sentence = clean_tsv_field(row[2])
            if "\t" in sentence or "\n" in sentence:
                raise ValueError(f"Malformed FLEURS sentence at row {line_no}: contains row separators")
            sentences_by_filename[row[1]] = sentence

    out_dir = output_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Load existing hashes to allow resuming
    existing = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                existing.add(json.loads(line)["file"].removesuffix(".wav"))

    with open(manifest_path, "a") as manifest:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".wav")]
            for member in tqdm(members, desc=f"fleurs-{lang}"):
                audio_filename = Path(member.name).name
                sentence = sentences_by_filename.get(audio_filename)
                if sentence is None:
                    continue

                h = sentence_hash(sentence)
                if h in existing:
                    continue

                audio_bytes = tar.extractfile(member).read()
                data, sr = sf.read(BytesIO(audio_bytes))

                wav_path = out_dir / f"{h}.wav"
                sf.write(str(wav_path), data, sr)

                record = {
                    "file": f"{h}.wav",
                    "sentence": sentence,
                    "source": "fleurs",
                    "voice": None,
                    "lang": lang,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                existing.add(h)


def main():
    parser = argparse.ArgumentParser(description="Download FLEURS dataset")
    parser.add_argument("--langs", nargs="+", default=list(LANG_CONFIG.keys()),
                        choices=list(LANG_CONFIG.keys()),
                        help="Languages to download (default: all)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "audio",
                        help="Output root directory")
    args = parser.parse_args()

    for lang in args.langs:
        try:
            download_for_language(lang, args.output)
        except Exception as e:
            print(f"Error downloading {lang}: {e}")
            print("Continuing with next language...")
            continue


if __name__ == "__main__":
    main()

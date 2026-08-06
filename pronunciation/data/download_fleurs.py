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
    # New learner languages. Keep the repository's language identifiers here;
    # the values are FLEURS' config/directory identifiers.
    "tha": "th_th",
    "zho-hans": "cmn_hans_cn",
    "hin": "hi_in",
    "jpn": "ja_jp",
}

# FLEURS audio dialects that differ from the canonical espeak voice in
# preprocess.LANG_TO_ESPEAK, written into each record's `espeak_voice` so
# preprocess phonemizes with the right dialect. Spanish FLEURS is the es_419
# (Latin American) split — seseo/yeísmo — so the Castilian "es" voice would
# mislabel c/z as θ. Langs not listed match the canonical voice (left as None).
FLEURS_ESPEAK_VOICE = {
    "spa": "es-419",
}

# Source metadata repairs verified against the corresponding recordings.
# Keyed by original FLEURS audio id so a re-download remains deterministic.
TEXT_CORRECTIONS = {
    ("hin", "1485690589703092330.wav"): "300 और गाड़ियों से संख्या बनती हैं, 1,300 गाडियाँ जो मंगवाई गई हैं ताकि भीड़भाड़ से निजात मिल सके।",
    ("hin", "960460417485172369.wav"): "300 और गाड़ियों से संख्या बनती हैं, 1,300 गाडियाँ जो मंगवाई गई हैं ताकि भीड़भाड़ से निजात मिल सके।",
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


def recording_id(lang: str, fleurs_filename: str) -> str:
    """Stable 16-hex id per FLEURS RECORDING (not per sentence).

    FLEURS records the same sentence by multiple speakers; the old code keyed
    the filename by sentence hash and so kept only ONE recording per sentence,
    throwing away the rest (~45% of sentences had duplicates) plus the per-clip
    gender/speaker link. Keying by the recording's own FLEURS filename keeps
    every recording — more training data, and gender per clip."""
    return hashlib.sha256(f"{lang}/{fleurs_filename}".encode()).hexdigest()[:16]


def download_for_language(lang: str, output_root: Path):
    fleurs_code = LANG_CONFIG[lang]

    # Download TSV and audio tar (cached after first fetch)
    print(f"Downloading FLEURS {fleurs_code} metadata...")
    tsv_path = hf_hub_download("google/fleurs", f"data/{fleurs_code}/train.tsv", repo_type="dataset")
    print(f"Downloading FLEURS {fleurs_code} audio...")
    tar_path = hf_hub_download("google/fleurs", f"data/{fleurs_code}/audio/train.tar.gz", repo_type="dataset")

    # Parse TSV keyed by RECORDING filename → (sentence, gender). Columns:
    # id, filename, transcription, normalized, chars, num_samples, gender.
    meta_by_filename: dict[str, tuple[str, str]] = {}
    with open(tsv_path, newline="") as f:
        for line_no, row in enumerate(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE), start=1):
            if len(row) != 7:
                raise ValueError(f"Malformed FLEURS TSV row {line_no}: expected 7 fields, got {len(row)}")
            sentence = clean_tsv_field(row[2])
            if "\t" in sentence or "\n" in sentence:
                raise ValueError(f"Malformed FLEURS sentence at row {line_no}: contains row separators")
            meta_by_filename[row[1]] = (sentence, clean_tsv_field(row[6]))

    out_dir = output_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Idempotent + non-destructive to other sources: load any existing manifest,
    # split off the FLEURS rows (we rebuild those), keep the rest untouched.
    other_rows: list[dict] = []
    old_fleurs_files: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                (old_fleurs_files.add(r["file"]) if r.get("source") == "fleurs"
                 else other_rows.append(r))

    # Extract EVERY recording (no sentence dedup). Extract before touching the
    # manifest so an interruption leaves the old data intact.
    new_fleurs_rows: list[dict] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".wav")]
        for member in tqdm(members, desc=f"fleurs-{lang}"):
            audio_filename = Path(member.name).name
            meta = meta_by_filename.get(audio_filename)
            if meta is None:
                continue
            sentence, gender = meta
            sentence = TEXT_CORRECTIONS.get((lang, audio_filename), sentence)
            rid = recording_id(lang, audio_filename)
            wav_path = out_dir / f"{rid}.wav"
            if not wav_path.exists():
                data, sr = sf.read(BytesIO(tar.extractfile(member).read()))
                sf.write(str(wav_path), data, sr)
            new_fleurs_rows.append({
                "file": f"{rid}.wav",
                "sentence": sentence,
                "source": "fleurs",
                "license": "CC BY 4.0",
                "attribution_url": "https://huggingface.co/datasets/google/fleurs",
                "voice": None,
                "espeak_voice": FLEURS_ESPEAK_VOICE.get(lang),
                "gender": gender or None,
                "fleurs_id": audio_filename,
                "lang": lang,
            })

    # Atomic manifest rewrite: other sources + rebuilt FLEURS rows.
    tmp = manifest_path.parent / (manifest_path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in other_rows + new_fleurs_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(manifest_path)

    # Drop orphaned old FLEURS wavs (old sentence-hash names no longer used).
    new_files = {r["file"] for r in new_fleurs_rows}
    removed = 0
    for f in old_fleurs_files:
        if f in new_files:
            continue
        p = out_dir / f
        if p.exists():
            p.unlink()
            removed += 1
    print(f"{lang}: {len(new_fleurs_rows)} FLEURS recordings "
          f"(old had {len(old_fleurs_files)}); removed {removed} orphaned wavs")


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

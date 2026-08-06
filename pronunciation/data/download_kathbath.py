#!/usr/bin/env python3
"""Import Hindi audio from an AI4Bharat Kathbath Parquet shard.

Kathbath is a CC BY 4.0 multi-speaker human corpus. This importer is intentionally
shard-oriented: a small validation shard can be audited before deciding whether
the much larger training split is useful. Embedded audio is normalized to the
repository's 16 kHz mono WAV contract with ffmpeg.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


# Kathbath validation-shard text repairs verified against the audio. These are
# detached/reordered Devanagari matras in the source metadata, not editorial
# pronunciation changes.
TEXT_CORRECTIONS = {
    "844424931069710-229-f.m4a": "कहा कि वह पार्टी की बेहतरी के लिए और मेहनत करने की जरूरत है",
    "844424932918936-888-f.m4a": "फिल्म में चंकी पांडे अहम भूमिका निभाते हुए नजर आएंगे",
}


def convert_audio(audio_bytes: bytes, output: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output),
        ],
        input=audio_bytes, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace")[:500])


def import_shard(parquet_path: Path, output_root: Path, limit: int | None) -> None:
    lang = "hin"
    out_dir = output_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    existing_other: list[dict] = []
    existing_kathbath: dict[str, dict] = {}
    if manifest_path.exists():
        with manifest_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("source") == "kathbath":
                    existing_kathbath[rec["file"]] = rec
                else:
                    existing_other.append(rec)

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]

    imported: list[dict] = []
    failures: list[str] = []
    for index, row in enumerate(rows, 1):
        source_text = row["text"].strip()
        text = TEXT_CORRECTIONS.get(row["fname"], source_text)
        if not text:
            continue
        stable_id = hashlib.sha256(
            f"{row['speaker_id']}/{row['fname']}/{source_text}".encode()
        ).hexdigest()[:16]
        filename = f"kathbath_{stable_id}.wav"
        wav_path = out_dir / filename
        if not wav_path.exists():
            try:
                convert_audio(row["audio_filepath"]["bytes"], wav_path)
            except Exception as exc:
                failures.append(f"{row['fname']}: {exc}")
                continue
        imported.append({
            "file": filename,
            "sentence": text,
            "source": "kathbath",
            "voice": f"kathbath_{row['speaker_id']}",
            "gender": row.get("gender") or None,
            "lang": lang,
            "duration_sec": row.get("duration"),
            "license": "CC-BY-4.0",
            "attribution_url": "https://huggingface.co/datasets/ai4bharat/Kathbath",
            "original_file": row["fname"],
            "kathbath_speaker_id": row["speaker_id"],
        })
        if index % 250 == 0 or index == len(rows):
            print(f"kathbath: {index}/{len(rows)} imported={len(imported)} "
                  f"failed={len(failures)}")

    # Keep Kathbath rows imported from other shards, replacing only rows whose
    # stable filename occurs in this shard. This makes adding a second shard
    # append-safe while remaining idempotent.
    merged_kathbath = dict(existing_kathbath)
    merged_kathbath.update({r["file"]: r for r in imported})
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as out:
        for rec in existing_other + list(merged_kathbath.values()):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(manifest_path)

    print(f"hin: manifest now has {len(existing_other)} non-Kathbath + "
          f"{len(merged_kathbath)} Kathbath rows")
    if failures:
        print(f"WARNING: {len(failures)} conversions failed; first five:")
        for failure in failures[:5]:
            print(f"  {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("parquet", type=Path)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(__file__).resolve().parent / "audio",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    import_shard(args.parquet, args.output_root, args.limit)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report auditable corpus coverage for the four expansion languages."""

from __future__ import annotations

import argparse
import collections
import json
import wave
from pathlib import Path


LANGS = ("tha", "zho-hans", "hin", "jpn")
ROOT = Path(__file__).resolve().parents[1]


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return None


def report_language(data_root: Path, lang: str) -> dict:
    lang_dir = data_root / lang
    manifest = lang_dir / "manifest.jsonl"
    records = []
    if manifest.exists():
        with manifest.open() as source:
            records = [json.loads(line) for line in source if line.strip()]

    sources = collections.Counter(r.get("source") or "unknown" for r in records)
    voices = collections.defaultdict(set)
    licenses = collections.defaultdict(collections.Counter)
    duration = collections.Counter()
    missing = 0
    unreadable = 0
    noncommercial = 0
    for rec in records:
        source = rec.get("source") or "unknown"
        voice = rec.get("voice") or rec.get("speaker_cluster")
        if voice:
            voices[source].add(str(voice))
        licenses[source][rec.get("license") or "unspecified"] += 1
        license_name = str(rec.get("license") or "").upper()
        noncommercial += int("BY-NC" in license_name or "NONCOMMERCIAL" in license_name)
        wav = lang_dir / rec["file"]
        if not wav.exists():
            missing += 1
            continue
        seconds = rec.get("duration_sec")
        if seconds is None:
            seconds = wav_duration(wav)
        if seconds is None:
            unreadable += 1
        else:
            duration[source] += float(seconds)

    backend_audits = {}
    for audit in sorted(lang_dir.glob("g2p_audit_*.jsonl")):
        rows = [json.loads(line) for line in audit.open() if line.strip()]
        backend_audits[audit.stem.removeprefix("g2p_audit_")] = {
            "rows": len(rows),
            "errors": sum(bool(row.get("error")) for row in rows),
            "coverage_fraction": round(len(rows) / len(records), 4) if records else 0.0,
            "complete_manifest": len(rows) == len(records),
        }

    phoneme_sidecars = {}
    for sidecar in sorted(lang_dir.glob("phoneme_backend_*.jsonl")):
        rows = [json.loads(line) for line in sidecar.open() if line.strip()]
        trainable = [row for row in rows if "phonemes" in row]
        metadata = collections.Counter()
        stress_sources = collections.Counter(
            row["stress_source"] for row in trainable if row.get("stress_source")
        )
        invalid_lengths = 0
        for row in trainable:
            phones = row["phonemes"]
            for field in ("stress", "tone", "pitch_accent"):
                if field in row:
                    metadata[field] += 1
                    if len(row[field]) != len(phones):
                        invalid_lengths += 1
            if "syllables" in row:
                metadata["syllables"] += 1
        phoneme_sidecars[sidecar.stem.removeprefix("phoneme_backend_")] = {
            "rows": len(rows),
            "trainable": len(trainable),
            "excluded": sum("exclude_reason" in row for row in rows),
            "metadata_rows": dict(sorted(metadata.items())),
            "stress_sources": dict(sorted(stress_sources.items())),
            "invalid_aligned_lengths": invalid_lengths,
        }

    preprocessed = None
    preprocessed_files: set[str] = set()
    phonemes_path = lang_dir / "phonemes.jsonl"
    if phonemes_path.exists():
        rows = [json.loads(line) for line in phonemes_path.open() if line.strip()]
        preprocessed_files = {row["file"] for row in rows}
        preprocessed = {
            "rows": len(rows),
            "backends": dict(sorted(collections.Counter(
                row.get("phoneme_backend") or "unspecified" for row in rows
            ).items())),
            "noncommercial_rows": sum(
                "BY-NC" in str(row.get("license") or "").upper()
                or "NONCOMMERCIAL" in str(row.get("license") or "").upper()
                for row in rows
            ),
            "bad_aligned_lengths": sum(
                len(row.get("phonemes", [])) != len(row.get("stress", []))
                for row in rows
            ),
        }

    vad = None
    vad_path = lang_dir / "vad.jsonl"
    if vad_path.exists():
        vad_files: set[str] = set()
        empty_prob_rows = 0
        rows = 0
        with vad_path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                vad_files.add(row["file"])
                empty_prob_rows += int(not row.get("vad_probs"))
        vad = {
            "rows": rows,
            "duplicate_files": rows - len(vad_files),
            "missing_preprocessed_files": len(preprocessed_files - vad_files),
            "extra_files": len(vad_files - preprocessed_files),
            "empty_probability_rows": empty_prob_rows,
        }

    return {
        "language": lang,
        "clips": len(records),
        "hours": round(sum(duration.values()) / 3600, 2),
        "sources": {
            source: {
                "clips": count,
                "hours": round(duration[source] / 3600, 2),
                "voices": len(voices[source]),
                "licenses": dict(sorted(licenses[source].items())),
            }
            for source, count in sorted(sources.items())
        },
        "missing_audio": missing,
        "unreadable_audio": unreadable,
        "noncommercial_clips": noncommercial,
        "backend_audits": backend_audits,
        "phoneme_sidecars": phoneme_sidecars,
        "preprocessed": preprocessed,
        "vad": vad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "audio",
    )
    parser.add_argument("--langs", nargs="+", default=list(LANGS))
    args = parser.parse_args()
    print(json.dumps(
        [report_language(args.data_root, lang) for lang in args.langs],
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()

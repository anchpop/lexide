#!/usr/bin/env python3
"""Backfill per-language pimsleur_processed.txt from existing manifest.jsonl.

The marker file (download_pimsleur.py's resume mechanism) only started
being written after commit 000facc. Manifests written by earlier runs
have no markers, so a naive re-run would re-pay Whisper for every
lesson already in the manifest.

This script reads each `data/audio/<lang>/manifest.jsonl`, extracts the
lesson_ids from any `pimsleur_*.wav` filenames, and writes them to
`data/audio/<lang>/pimsleur_processed.txt`. After running this, a
download_pimsleur.py re-run safely skips work already on disk.

Note: a lesson where Whisper rejected every segment (zero clips kept)
won't appear in the manifest, so this backfill misses those. Re-running
will re-process them. The cost is bounded by the number of "empty"
lessons, which should be small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "audio")
    args = parser.parse_args()

    for lang_dir in sorted(args.audio_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        manifest = lang_dir / "manifest.jsonl"
        if not manifest.exists():
            continue

        lessons: set[str] = set()
        with manifest.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("source") != "pimsleur":
                    continue
                fname = rec.get("file", "")
                # pimsleur_<lesson_id>_<seg_idx>.wav
                if not (fname.startswith("pimsleur_") and fname.endswith(".wav")):
                    continue
                stem = fname[len("pimsleur_"):-len(".wav")]
                # Strip the _NNNN segment-index suffix.
                lesson_id = stem.rsplit("_", 1)[0]
                lessons.add(lesson_id)

        if not lessons:
            continue
        marker = lang_dir / "pimsleur_processed.txt"
        # Merge with anything already in the marker file (resume safety).
        if marker.exists():
            for line in marker.read_text().splitlines():
                line = line.strip()
                if line:
                    lessons.add(line)
        marker.write_text("\n".join(sorted(lessons)) + "\n")
        print(f"{lang_dir.name}: marked {len(lessons)} lessons done → {marker}")


if __name__ == "__main__":
    main()

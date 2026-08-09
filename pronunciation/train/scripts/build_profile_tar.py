"""Build a small, representative single-language tarball for GH200 profiling."""

import argparse
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/audio/eng"))
    parser.add_argument("--output", type=Path, default=Path(".work/profile_audio.tar"))
    parser.add_argument("--samples", type=int, default=2048)
    args = parser.parse_args()

    labels_path = args.source / "phonemes_narrowed.jsonl"
    records = []
    with labels_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if (args.source / record["file"]).exists():
                records.append(record)
                if len(records) == args.samples:
                    break
    if len(records) < args.samples:
        raise SystemExit(f"found only {len(records)} usable records")

    wanted = {record["file"] for record in records}
    vad_records = []
    vad_path = args.source / "vad.jsonl"
    if vad_path.exists():
        with vad_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["file"] in wanted:
                    vad_records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    vad_bytes = "".join(json.dumps(r) + "\n" for r in vad_records).encode()
    with tarfile.open(args.output, "w") as archive:
        # The trainer discovers languages through phonemes.jsonl before it
        # switches to the narrowed filename. Keep both discovery and selected
        # label paths in the benchmark archive.
        for name, payload in (("eng/phonemes.jsonl", labels_bytes),
                              ("eng/phonemes_narrowed.jsonl", labels_bytes),
                              ("eng/vad.jsonl", vad_bytes)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            import io
            archive.addfile(info, io.BytesIO(payload))
        for record in records:
            archive.add(args.source / record["file"], arcname=f"eng/{record['file']}")
    print(f"wrote {args.output} with {len(records)} clips and {len(vad_records)} VAD rows")


if __name__ == "__main__":
    main()

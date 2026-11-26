"""Replace pitch in an aligned feature set with full CREPE. Publish metadata last.

    python scripts/crepe_features.py ALIGNED_SET NEW_SET [--tracks DIR]

Optional precomputed tracks are [frames, 2] log(F0/200), gated periodicity,
keyed by clip ID. Their waveforms must be the same recordings as ALIGNED_SET.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from voca.config import Config
from voca.crepe_pitch import track


def convert(stored, measured):
    if measured.shape != (len(stored["prosody"]), 2) or not measured.isfinite().all():
        raise ValueError("invalid CREPE track")
    log_pitch, confidence = measured.T
    del stored["pitch_check"]
    stored["prosody"][:, 0] = log_pitch
    stored["prosody"][:, 1] = confidence >= 0.30
    stored["periodicity"] = confidence.contiguous()
    return stored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tracks", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.source / "annotations.json").read_text())
    if not metadata["complete"]:
        raise ValueError("source build is incomplete")
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config(f0_max=1000.0, feature_version=4)
    digest = hashlib.sha256(config.acoustic_fingerprint.encode())
    for split in ("train", "validation"):
        with (args.output / f"{split}.jsonl").open("w") as stream:
            for line in (args.source / f"{split}.jsonl").open():
                row = json.loads(line)
                stored = torch.load(args.source / metadata["acoustic_root"] / row["path"], weights_only=True)
                if args.tracks:
                    measured = torch.load(args.tracks / (row["id"] + ".pt"), weights_only=True)
                else:
                    hz, confidence = track(stored["wave"].cuda(), config.sample_rate, config.hop)
                    measured = torch.stack(((hz / config.pitch_reference_hz).log(), confidence), 1).cpu()
                path = args.output / row["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(convert(stored, measured), path)
                row["acoustic"] = config.acoustic_fingerprint
                digest.update(line.encode())
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    annotation = digest.hexdigest()
    for split in ("train", "validation"):
        manifest = args.output / f"{split}.jsonl"
        with manifest.with_suffix(".tmp").open("w") as stream:
            for row in map(json.loads, manifest.open()):
                stream.write(json.dumps({**row, "annotation": annotation}, ensure_ascii=False) + "\n")
        manifest.with_suffix(".tmp").replace(manifest)
    (args.output / "rejected.jsonl").write_bytes((args.source / "rejected.jsonl").read_bytes())
    (args.output / "annotations.json").write_text(json.dumps(dict(fingerprint=annotation, complete=True, acoustic_root=".")))


if __name__ == "__main__":
    main()

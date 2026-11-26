"""Pack exported ONNX stages into one immutable browser asset. Run with VOCA_LANGUAGE and PYTHONPATH=."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from voca.text import frontend_fingerprint, frontend_spec


def bundle(stages, destination):
    config = json.loads((stages / "model.json").read_text())
    if config["fingerprint"] != frontend_fingerprint():
        raise ValueError("Exported model and current tokenizer have different fingerprints")
    graphs = {name: (stages / f"{name}.onnx").read_bytes() for name in ("encode", "predict", "melody", "blend", "render")}
    config.update(frontend=frontend_spec(), graphs={name: len(data) for name, data in graphs.items()})
    header = json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode()
    data = struct.pack("<I", len(header)) + header + b"".join(graphs.values())
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (hashlib.sha256(data).hexdigest() + ".voca")
    path.write_bytes(data)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(bundle(args.stages, args.destination))

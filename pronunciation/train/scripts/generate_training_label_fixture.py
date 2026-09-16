"""Replay the shared conformance inputs through the actual Python validator.

The cases and expected outputs were captured before table extraction at c4c8647,
using the real cached tokenizer. Their provenance and separately tested output
SHA preserve that independent baseline; this generator is a reproducible replay,
not a replacement oracle or a reason to refresh goldens after a label change.
No g2p, inference, network or corpus access is needed.

From the repo root, use --check for a no-write byte comparison, --write to emit
canonical JSON, or no flag to print it. Review any changed expected outputs as
label changes, not formatting updates.
"""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import preprocess

FIXTURE = (Path(__file__).resolve().parents[3]
           / "tagging/lexide/data/training_labels_conformance.json")
GENERATED_BY = "python pronunciation/train/scripts/generate_training_label_fixture.py --write"


def render_fixture() -> bytes:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    base = preprocess._tokenizer_vocab()
    for case in fixture["cases"]:
        # Only the acceptance-precedence case augments the real inventory.
        with patch.object(preprocess, "_tokenizer_vocab",
                          return_value=base | set(case["extra_vocab"])):
            phones, stress, unknowns = preprocess.validate_phonemes(
                case["phones"], case["stress"], case["lang"],
            )
        case["expected"] = [phones, stress, sorted(unknowns)]
    # One readable case per line, avoiding thousands of one-element array lines.
    text = ('{\n  "generated_by": ' + json.dumps(GENERATED_BY)
            + ',\n  "provenance": ' + json.dumps(fixture["provenance"], ensure_ascii=False)
            + ',\n  "cases": [\n'
            + ',\n'.join('    ' + json.dumps(c, ensure_ascii=False) for c in fixture["cases"])
            + '\n  ]\n}\n')
    return text.encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = render_fixture()
    if args.check:
        if output != FIXTURE.read_bytes():
            raise SystemExit("Training-label fixture differs from Python validator; review label drift.")
        print("Training-label fixture is byte-identical to Python validator replay.")
    elif args.write:
        FIXTURE.write_bytes(output)
    else:
        sys.stdout.buffer.write(output)


if __name__ == "__main__":
    main()

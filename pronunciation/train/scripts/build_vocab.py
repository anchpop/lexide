"""Build IPA vocabulary from preprocessed manifest files.

Scans all IPA strings in manifest JSONL files, collects unique Unicode
codepoints, and writes vocab.json.

Usage:
    python scripts/build_vocab.py --manifest_dir data/manifests --output data/vocab.json
"""

import argparse
import json
import unicodedata
from pathlib import Path

BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
SPACE_TOKEN = " "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/vocab.json")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    manifests = sorted(manifest_dir.glob("manifest_*_train.jsonl"))
    assert manifests, f"No train manifests found in {manifest_dir}"

    # Collect all unique IPA codepoints
    codepoints: set[str] = set()
    total_entries = 0

    for path in manifests:
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                ipa = entry["ipa"]
                for ch in ipa:
                    if ch != " ":  # space is handled separately
                        codepoints.add(ch)
                total_entries += 1

    # Sort: base characters first, then combining marks, then symbols
    def sort_key(ch: str) -> tuple[int, int]:
        cat = unicodedata.category(ch)
        # Combining marks (Mn, Mc, Me) sort after base characters
        if cat.startswith("M"):
            priority = 1
        else:
            priority = 0
        return (priority, ord(ch))

    sorted_codepoints = sorted(codepoints, key=sort_key)

    # Build token list: reserved tokens first
    tokens = [BLANK_TOKEN, UNK_TOKEN, SPACE_TOKEN] + sorted_codepoints

    # Write vocab
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"tokens": tokens}, f, ensure_ascii=False, indent=2)

    # Stats
    n_base = sum(1 for ch in sorted_codepoints if not unicodedata.category(ch).startswith("M"))
    n_combining = sum(1 for ch in sorted_codepoints if unicodedata.category(ch).startswith("M"))

    print(f"Scanned {total_entries} entries from {len(manifests)} manifests")
    print(f"Vocab size: {len(tokens)} tokens")
    print(f"  Reserved: 3 (blank, unk, space)")
    print(f"  Base characters: {n_base}")
    print(f"  Combining marks: {n_combining}")

    # Show a sample of tokens
    print(f"\nFirst 30 IPA tokens: {sorted_codepoints[:30]}")
    if n_combining > 0:
        combining = [ch for ch in sorted_codepoints if unicodedata.category(ch).startswith("M")]
        print(f"Combining marks: {combining}")


if __name__ == "__main__":
    main()

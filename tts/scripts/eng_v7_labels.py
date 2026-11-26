"""Prepare corrected input manifests for voca.build --reuse from the accepted English corpus.

    python scripts/eng_v7_labels.py data features/eng_gemini_multi_v6 /path/to/sets
"""
import argparse
import json
from pathlib import Path
from edge_grams import initialisms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "features", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    corrections = json.loads((Path(__file__).resolve().parents[1] / "data/eng_word_corrections.json").read_text())
    ids = {row["id"] for split in ("train", "validation")
           for row in map(json.loads, (args.features / f"{split}.jsonl").open())}
    for name in sorted({i.split("/")[0] for i in ids}):
        src, dest = args.data / name, args.output / name
        dest.mkdir(parents=True)
        for subdir in ("wav", "frames"):
            (dest / subdir).symlink_to((src / subdir).resolve())
        with (dest / "manifest.jsonl").open("w") as stream:
            for row in map(json.loads, (src / "manifest.jsonl").open()):
                key = name + "/" + row["id"]
                if key not in ids:
                    continue
                if key in corrections:
                    before, after = corrections.pop(key)
                    gram, = [g for g in row["grams"] if g["phonemes"]]
                    if gram["phonemes"] != before:
                        raise ValueError(f"{key} no longer has the audited pronunciation")
                    gram["phonemes"] = after
                if row.get("kind") == "rare/acronyms":
                    initialisms(row["grams"], row["written"])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if corrections:
        raise ValueError(f"missing recordings: {list(corrections)}")


if __name__ == "__main__":
    main()

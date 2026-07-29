"""Emit the Japanese wordbank the Viterbi prior uses: word<TAB>count, train split only.

Train-split only so the bank never sees evaluation data. Rebuild whenever data_prep runs:

    python3 tagger/build_jpn_wordbank.py --data-dir data/processed --out data/jpn_wordbank.tsv
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import maybe_merge_inflection  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/processed")
ap.add_argument("--out", default="data/jpn_wordbank.tsv")
ap.add_argument("--lang", default="jpn")
ap.add_argument("--min-count", type=int, default=1)
ap.add_argument("--no-merge-inflection", dest="merge", action="store_false", default=True)
args = ap.parse_args()

counts = collections.Counter()
with open(os.path.join(args.data_dir, "train.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("lang") != args.lang:
            continue
        text = r["text"]
        for t in maybe_merge_inflection(r, args.merge):
            w = text[t["start"]:t["end"]]
            if w.strip():
                counts[w] += 1

kept = {w: c for w, c in counts.items() if c >= args.min_count}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    for w, c in sorted(kept.items(), key=lambda kv: (-kv[1], kv[0])):
        f.write(f"{w}\t{c}\n")
print(f"{len(kept):,} types, {sum(kept.values()):,} tokens -> {args.out} "
      f"({os.path.getsize(args.out)/1e6:.2f} MB)")

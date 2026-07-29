"""Emit the per-language wordbanks the Viterbi prior uses: word<TAB>count, train split only.

Train-split only so the bank never sees evaluation data. Rebuild whenever data_prep runs:

    python3 tagger/build_jpn_wordbank.py --data-dir data/processed --out data/jpn_wordbank.tsv
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import maybe_merge_inflection  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/processed")
ap.add_argument("--out-dir", default="data/wordbanks")
ap.add_argument("--langs", default="jpn,kor,hin",
                help="comma-separated; the no-whitespace and low-resource ones "
                     "are where a wordbank can propose splits whitespace cannot")
ap.add_argument("--min-count", type=int, default=1)
ap.add_argument("--no-merge-inflection", dest="merge", action="store_false", default=True)
args = ap.parse_args()

banks = collections.defaultdict(collections.Counter)
wanted = set(args.langs.split(","))
with open(os.path.join(args.data_dir, "train.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        lang = r.get("lang")
        if lang not in wanted:
            continue
        text = r["text"]
        for t in maybe_merge_inflection(r, args.merge):
            w = text[t["start"]:t["end"]]
            if w.strip():
                banks[lang][w] += 1

os.makedirs(args.out_dir, exist_ok=True)
for lang, counts in sorted(banks.items()):
    kept = {w: c for w, c in counts.items() if c >= args.min_count}
    path = os.path.join(args.out_dir, f"{lang}.tsv")
    with open(path, "w", encoding="utf-8") as f:
        for w, c in sorted(kept.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{w}\t{c}\n")
    print(f"{lang}: {len(kept):,} types, {sum(kept.values()):,} tokens -> {path} "
          f"({os.path.getsize(path)/1e6:.2f} MB)")

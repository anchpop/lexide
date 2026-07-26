"""Targeted segmenter eval on hand-built hard-pattern cases (eval-patterns.jsonl).

Each record is {id, lang, category, text, tokens:[{start,end}]} — the same span schema
as processed/*.jsonl, but every case is a known failure family (abbreviation + dash
continuation, dash parentheticals with inner punctuation, jpn 「…」だ copula tails,
sentence-final abbreviations, multi-sentence quotes, decimals/URLs, hard wraps).
Cases are worded differently from the mechanical training templates, so this is a
(small) held-out check that the pattern generalized, not that templates memorized.

    python3 sentence-labeller/eval_patterns.py \
        --ckpt output/segmenter/segmenter.pt \
        --patterns sentence-labeller/eval-patterns.jsonl

Prints per-category exact-span-match rates and every failing case with the predicted
vs gold spans (rendered as `|`-delimited text).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tagger"))
from dataset import encode_bytes_and_labels  # noqa: E402
from predict import load_char_model  # noqa: E402
from train_segmenter import spans_from_labels  # noqa: E402


def byte_spans_to_text(text, spans):
    data = text.encode("utf-8")
    return " | ".join(data[s:e].decode("utf-8", "replace") for s, e in sorted(spans))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--patterns", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval-patterns.jsonl"))
    ap.add_argument("--max-bytes", type=int, default=4096)
    ap.add_argument("--json-out", default=None,
                    help="also write {pass, total, by_category} as JSON (for sweep selection)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_char_model(args.ckpt, device)
    model.eval()

    by_cat = defaultdict(lambda: [0, 0])  # category -> [passed, total]
    failures = []
    with open(args.patterns, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            byte_ids, labels = encode_bytes_and_labels(
                r["text"], r["tokens"], args.max_bytes, r.get("lang"))
            with torch.no_grad():
                pred = model(torch.tensor([byte_ids], device=device))[0].argmax(-1).tolist()
            gold, got = spans_from_labels(labels), spans_from_labels(pred)
            ok = gold == got
            by_cat[r["category"]][1] += 1
            by_cat[r["category"]][0] += int(ok)
            if not ok:
                failures.append((r, gold, got))

    total_pass = sum(p for p, _ in by_cat.values())
    total = sum(t for _, t in by_cat.values())
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"pass": total_pass, "total": total,
                       "by_category": {k: {"pass": p, "total": t}
                                       for k, (p, t) in sorted(by_cat.items())}}, f, indent=2)
    print(f"exact-case pass rate: {total_pass}/{total}")
    for cat, (p, t) in sorted(by_cat.items()):
        print(f"  {cat:14s} {p}/{t}")
    for r, gold, got in failures:
        print(f"\nFAIL {r['id']} [{r['lang']}/{r['category']}]")
        print(f"  text: {r['text']!r}")
        print(f"  gold: {byte_spans_to_text(r['text'], gold)}")
        print(f"  pred: {byte_spans_to_text(r['text'], got)}")


if __name__ == "__main__":
    main()

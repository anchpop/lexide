#!/usr/bin/env python3
"""Benchmark joint HTTP batching; optionally compare the same checkpoint on CPU.

First-request latency is cold ONLY after the deployment has scaled to zero.
L4 GPU-only list price checked at https://modal.com/pricing on 2026-10-04.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import time
from urllib.request import Request, urlopen


def post(url, records):
    payload = {"sentences": [r["text"] for r in records], "lang": records[0]["lang"]}
    req = Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as response:
        results = json.load(response)["results"]
    assert len(results) == len(records)
    return results


def sample(path, count):
    # Per-language reservoir avoids benchmarking only the first (German) corpus block.
    rng = random.Random(42)
    pools, seen = defaultdict(list), defaultdict(int)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            lang = r["lang"]
            seen[lang] += 1
            if len(pools[lang]) < count:
                pools[lang].append({"text": r["text"], "lang": lang})
            else:
                j = rng.randrange(seen[lang])
                if j < count:
                    pools[lang][j] = {"text": r["text"], "lang": lang}
    records = []
    for i in range(count):
        lang = sorted(pools)[i % len(pools)]
        records.append(pools[lang][i // len(pools) % len(pools[lang])])
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "data/processed-joint/test.jsonl")
    ap.add_argument("--sentences", type=int, default=1200)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--l4-price-per-second", type=float, default=0.000222)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--parity-count", type=int, default=50)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if min(args.sentences, args.clients, args.batch_size, args.parity_count) <= 0:
        ap.error("counts must be positive")
    records = sample(args.input, args.sentences)
    start = time.perf_counter()
    post(args.url, records[:1])
    first = time.perf_counter() - start
    grouped = defaultdict(list)
    for r in records:
        grouped[r["lang"]].append(r)
    batches = [rs[i:i + args.batch_size] for rs in grouped.values()
               for i in range(0, len(rs), args.batch_size)]
    random.Random(42).shuffle(batches)
    start = time.perf_counter()
    with ThreadPoolExecutor(args.clients) as pool:
        list(pool.map(lambda batch: post(args.url, batch), batches))
    seconds = time.perf_counter() - start
    out = {"sentences": len(records), "clients": args.clients, "batch_size": args.batch_size,
           "first_request_seconds": first, "warm_seconds": seconds,
           "sentences_per_second": len(records) / seconds,
           "l4_price_per_second": args.l4_price_per_second,
           "gpu_only_dollars_per_million": args.l4_price_per_second * seconds / len(records) * 1e6,
           "cost_caveat": "One GPU; excludes CPU/RAM, cold start and 300s idle tail"}
    if args.checkpoint:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tagger"))
        import torch
        from predict_joint import load_checkpoint, predict_records
        torch.set_num_threads(4)
        model, tokenizer, vocab = load_checkpoint(args.checkpoint, "cpu")
        subset = records[:args.parity_count]
        predict_records(model, tokenizer, vocab, subset[:1], device="cpu")
        start = time.perf_counter()
        local = predict_records(model, tokenizer, vocab, subset, batch_size=16, device="cpu")
        cpu_seconds = time.perf_counter() - start
        mismatches = []
        equal_tokens = total_tokens = 0
        for i, (r, prediction) in enumerate(zip(subset, local)):
            remote = post(args.url, [r])[0]
            expected = [{**t, "text": r["text"][t["start"]:t["end"]]} for t in prediction["tokens"]]
            total_tokens += max(len(remote), len(expected))
            equal_tokens += sum(a == b for a, b in zip(remote, expected))
            if remote != expected:
                mismatches.append({"index": i, "lang": r["lang"], "text": r["text"],
                                   "local": expected, "remote": remote})
        out.update(cpu_sentences=len(subset), cpu_seconds=cpu_seconds,
                   cpu_sentences_per_second=len(subset) / cpu_seconds,
                   parity_sentences=len(subset), parity_exact_sentences=len(subset) - len(mismatches),
                   parity_exact_tokens=equal_tokens, parity_total_tokens=total_tokens,
                   parity_mismatches=mismatches)
    encoded = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()

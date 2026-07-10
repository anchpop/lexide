"""Train the char/byte-level boundary tagger (the tokenizer) with a bidirectional minGRU.

Predicts per-byte O/B/I; token spans are recovered from B...I runs. Reports token-level
boundary F1 (a token is correct iff its exact [start,end) span is predicted).
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (CharBoundaryDataset, char_collate, encode_bytes_and_labels,
                     read_jsonl)
from model import CharBoundaryTagger


def spans_from_labels(labels):
    """labels: list[int] over bytes (0=O,1=B,2=I) -> set of (start,end) byte spans."""
    spans = []
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] == 1:  # B
            j = i + 1
            while j < n and labels[j] == 2:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return set(spans)


@torch.no_grad()
def evaluate(model, records, device, max_items=2000, max_bytes=512):
    model.eval()
    tp = fp = fn = 0
    byte_correct = byte_total = 0
    for r in records[:max_items]:
        byte_ids, labels = encode_bytes_and_labels(r["text"], r["tokens"], max_bytes)
        x = torch.tensor([byte_ids], device=device)
        logits = model(x)[0]
        pred = logits.argmax(-1).tolist()
        gold_spans = spans_from_labels(labels)
        pred_spans = spans_from_labels(pred)
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
        for a, b in zip(pred, labels):
            if b == -100:
                continue
            byte_total += 1
            byte_correct += int(a == b)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    model.train()
    return {"token_f1": round(f1 * 100, 2), "prec": round(prec * 100, 2),
            "rec": round(rec * 100, 2), "byte_acc": round(byte_correct / max(1, byte_total) * 100, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="output/tokenizer")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--max-bytes", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    train_records = read_jsonl(os.path.join(args.data_dir, "train.jsonl"), args.train_limit)
    val_records = read_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    print(f"train={len(train_records)} val={len(val_records)}")

    model = CharBoundaryTagger(emb_dim=args.emb_dim, hidden_dim=args.hidden, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"char tokenizer params: {n_params/1e6:.2f}M")

    ds = CharBoundaryDataset(train_records, args.max_bytes)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=char_collate, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = int(len(loader) * args.epochs)
    if args.smoke:
        total_steps = 60
    print(f"total_steps={total_steps}")

    if args.wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "lexide-parsley"),
                   name="char-tokenizer", config=vars(args))

    step = 0
    t0 = time.time()
    run = 0.0
    best = -1.0
    model.train()
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for batch in loader:
            if step >= total_steps:
                done = True
                break
            byte_ids = batch["byte_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(byte_ids)
            loss = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
            step += 1
            if step % args.log_every == 0:
                sps = step / (time.time() - t0)
                print(f"[tok] step {step}/{total_steps} loss={run/args.log_every:.4f} ({sps:.1f} it/s)", flush=True)
                run = 0.0
            if step % args.eval_every == 0 or (args.smoke and step == total_steps):
                m = evaluate(model, val_records, device)
                print(f"[tok] eval@{step} {m}", flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({f"tok/{k}": v for k, v in m.items()}, step=step)
                if m["token_f1"] > best:
                    best = m["token_f1"]
                    torch.save(model.state_dict(), os.path.join(args.out_dir, "tokenizer.pt"))
                    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
                        json.dump({"metrics": m, "step": step, "config": vars(args)}, f, indent=2)

    m = evaluate(model, val_records, device)
    print(f"[tok] final {m}", flush=True)
    if m["token_f1"] >= best:
        torch.save(model.state_dict(), os.path.join(args.out_dir, "tokenizer.pt"))
    print("done")


if __name__ == "__main__":
    main()

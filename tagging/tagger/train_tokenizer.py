"""Train the char/byte-level boundary tagger (the tokenizer) with a bidirectional minGRU.

Predicts per-byte O/B/I; token spans are recovered from B...I runs. Reports token-level
boundary F1 (a token is correct iff its exact [start,end) span is predicted).
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (CHAR_VOCAB_SIZE, CharBoundaryDataset, char_collate,
                     encode_bytes_and_labels, read_jsonl)
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


def stratified(records, per_lang):
    """Up to `per_lang` records of each language, in file order.

    The val file is written language-by-language, so `records[:n]` is not a sample of the
    corpus — it is whichever languages happen to come first alphabetically. Scoring that
    prefix reported 99.8 token F1 for years while Japanese sat at 86.6, unmeasured.
    """
    seen = Counter()
    out = []
    for r in records:
        lang = r.get("lang", "und")
        if seen[lang] >= per_lang:
            continue
        seen[lang] += 1
        out.append(r)
    return out


def fmt_metrics(m):
    langs = " ".join(f"{lang} {d['token_f1']}" for lang, d in m["per_lang"].items())
    return (f"F1={m['token_f1']} prec={m['prec']} rec={m['rec']} "
            f"byte_acc={m['byte_acc']} | {langs}")


@torch.no_grad()
def evaluate(model, records, device, per_lang=400, max_bytes=512, use_lang=True):
    """Micro-averaged token-span F1 overall and per language."""
    model.eval()
    agg = dict(tp=0, fp=0, fn=0, bc=0, bt=0)
    per = {}
    for r in stratified(records, per_lang):
        lang = r.get("lang") if use_lang else None
        byte_ids, labels = encode_bytes_and_labels(r["text"], r["tokens"], max_bytes, lang)
        x = torch.tensor([byte_ids], device=device)
        pred = model(x)[0].argmax(-1).tolist()
        gold_spans, pred_spans = spans_from_labels(labels), spans_from_labels(pred)
        counts = dict(
            tp=len(gold_spans & pred_spans),
            fp=len(pred_spans - gold_spans),
            fn=len(gold_spans - pred_spans),
            bc=sum(int(a == b) for a, b in zip(pred, labels) if b != -100),
            bt=sum(1 for b in labels if b != -100),
        )
        d = per.setdefault(r.get("lang", "und"), dict(tp=0, fp=0, fn=0, bc=0, bt=0))
        for k, v in counts.items():
            d[k] += v
            agg[k] += v

    def score(d):
        prec = d["tp"] / max(1, d["tp"] + d["fp"])
        rec = d["tp"] / max(1, d["tp"] + d["fn"])
        return {"token_f1": round(200 * prec * rec / max(1e-9, prec + rec), 2),
                "prec": round(prec * 100, 2), "rec": round(rec * 100, 2),
                "byte_acc": round(d["bc"] / max(1, d["bt"]) * 100, 2)}

    model.train()
    out = score(agg)
    out["per_lang"] = {lang: score(d) for lang, d in sorted(per.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="output/tokenizer")
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=96)
    ap.add_argument("--lang-dropout", type=float, default=0.15,
                    help="fraction of training examples encoded with the generic BOS "
                         "instead of their language token (keeps lang-free use working)")
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

    model = CharBoundaryTagger(vocab_size=CHAR_VOCAB_SIZE, emb_dim=args.emb_dim,
                               hidden_dim=args.hidden, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"char tokenizer params: {n_params/1e6:.2f}M")

    ds = CharBoundaryDataset(train_records, args.max_bytes, lang_dropout=args.lang_dropout)
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
                print(f"[tok] eval@{step} {fmt_metrics(m)}", flush=True)
                if args.wandb:
                    import wandb
                    flat = {f"tok/{k}": v for k, v in m.items() if k != "per_lang"}
                    flat.update({f"tok/{lang}_f1": d["token_f1"]
                                 for lang, d in m["per_lang"].items()})
                    wandb.log(flat, step=step)
                if m["token_f1"] > best:
                    best = m["token_f1"]
                    torch.save(model.state_dict(), os.path.join(args.out_dir, "tokenizer.pt"))
                    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
                        json.dump({"metrics": m, "step": step, "config": vars(args)}, f, indent=2)

    m = evaluate(model, val_records, device)
    print(f"[tok] final {fmt_metrics(m)}", flush=True)
    if m["token_f1"] >= best:
        torch.save(model.state_dict(), os.path.join(args.out_dir, "tokenizer.pt"))
    nl = evaluate(model, val_records, device, use_lang=False)
    print(f"[tok] final lang-free {fmt_metrics(nl)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

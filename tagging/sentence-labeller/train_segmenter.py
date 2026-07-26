"""Train the byte-level sentence segmenter — a bidirectional minGRU over UTF-8 bytes
predicting per-byte O/B/I, where B = a sentence begins here, I = inside a sentence,
O = a gap (whitespace / headings / separators between sentences). Sentence spans are
recovered from each B...I run, exactly like the token boundary tagger recovers tokens.

This is the *same* `CharBoundaryTagger` architecture as `tagger/train_tokenizer.py`
(so the Rust byte-minGRU reimplementation and the safetensors export path are reused
verbatim) — only the training labels change (sentence spans instead of token spans).

    LD_LIBRARY_PATH=<gcc-lib>:/run/opengl-driver/lib \
      .venv-seg/bin/python sentence-labeller/train_segmenter.py \
      --data-dir sentence-labeller/processed --out-dir sentence-labeller/output

Reports sentence-span F1 (a sentence is correct iff its exact [start,end) char span is
predicted) overall and per language.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tagger"))
from dataset import (CHAR_VOCAB_SIZE, CharBoundaryDataset, char_collate,  # noqa: E402
                     encode_bytes_and_labels)
from model import CharBoundaryTagger  # noqa: E402


def spans_from_labels(labels):
    """labels over bytes (0=O,1=B,2=I) -> set of (start,end) byte spans (each B + trailing I)."""
    spans = []
    i, n = 0, len(labels)
    while i < n:
        if labels[i] == 1:
            j = i + 1
            while j < n and labels[j] == 2:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return set(spans)


@torch.no_grad()
def evaluate(model, records, device, max_bytes=4096, use_lang=True):
    """Micro-averaged sentence-span F1 overall and per language."""
    model.eval()
    agg = dict(tp=0, fp=0, fn=0, bc=0, bt=0)
    per = {}
    for r in records:
        lang = r.get("lang") if use_lang else None
        byte_ids, labels = encode_bytes_and_labels(r["text"], r["tokens"], max_bytes, lang)
        x = torch.tensor([byte_ids], device=device)
        pred = model(x)[0].argmax(-1).tolist()
        gold, got = spans_from_labels(labels), spans_from_labels(pred)
        tp, fp, fn = len(gold & got), len(got - gold), len(gold - got)
        bc = sum(int(a == b) for a, b in zip(pred, labels) if b != -100)
        bt = sum(1 for b in labels if b != -100)
        lang = r.get("lang", "und")
        p = per.setdefault(lang, dict(tp=0, fp=0, fn=0, bc=0, bt=0))
        for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("bc", bc), ("bt", bt)):
            p[k] += v
            agg[k] += v

    def f1(d):
        prec = d["tp"] / max(1, d["tp"] + d["fp"])
        rec = d["tp"] / max(1, d["tp"] + d["fn"])
        return dict(
            f1=round(200 * prec * rec / max(1e-9, prec + rec), 2),
            prec=round(prec * 100, 2), rec=round(rec * 100, 2),
            byte_acc=round(d["bc"] / max(1, d["bt"]) * 100, 2),
        )

    model.train()
    return f1(agg), {lang: f1(d) for lang, d in sorted(per.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="processed")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=96)
    ap.add_argument("--lang-dropout", type=float, default=0.15,
                    help="fraction of training examples encoded with the generic BOS "
                         "instead of their language token (keeps lang-free use working)")
    ap.add_argument("--max-bytes", type=int, default=768)
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed torch RNG (init + batch order); at this scale seed variance "
                         "dominates marginal cases, so sweeps train several and pick by "
                         "eval_patterns.py score")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    def read(split, limit=None):
        path = os.path.join(args.data_dir, f"{split}.jsonl")
        out = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                out.append(json.loads(line))
        return out

    train_records = read("train", args.train_limit)
    val_records = read("val")
    print(f"train={len(train_records)} val={len(val_records)}", flush=True)

    model = CharBoundaryTagger(vocab_size=CHAR_VOCAB_SIZE, emb_dim=args.emb_dim,
                               hidden_dim=args.hidden, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"segmenter params: {n_params/1e6:.3f}M", flush=True)

    ds = CharBoundaryDataset(train_records, args.max_bytes, lang_dropout=args.lang_dropout)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=char_collate, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = 60 if args.smoke else int(len(loader) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)
    print(f"total_steps={total_steps} steps/epoch={len(loader)}", flush=True)

    step, run, best, t0, done = 0, 0.0, -1.0, time.time(), False
    model.train()
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
            sched.step()
            run += loss.item()
            step += 1
            if step % args.log_every == 0:
                sps = step / (time.time() - t0)
                print(f"[seg] step {step}/{total_steps} ep{epoch} loss={run/args.log_every:.4f} "
                      f"({sps:.1f} it/s)", flush=True)
                run = 0.0
            if step % args.eval_every == 0 or (args.smoke and step == total_steps):
                overall, per = evaluate(model, val_records, device)
                print(f"[seg] eval@{step} overall={overall}", flush=True)
                print(f"[seg] per-lang={json.dumps(per)}", flush=True)
                if overall["f1"] > best:
                    best = overall["f1"]
                    torch.save(model.state_dict(), os.path.join(args.out_dir, "segmenter.pt"))
                    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
                        json.dump({"metrics": overall, "per_lang": per, "step": step,
                                   "config": vars(args)}, f, indent=2, ensure_ascii=False)
                    print(f"[seg] new best f1={best} -> saved", flush=True)

    # Final: reload best, eval on val + held-out test.
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "segmenter.pt"), map_location=device))
    v_overall, v_per = evaluate(model, val_records, device)
    print(f"[final/val] overall={v_overall}")
    print(f"[final/val] per-lang={json.dumps(v_per, indent=2)}")
    nl_overall, _ = evaluate(model, val_records, device, use_lang=False)
    print(f"[final/val/lang-free] overall={nl_overall}")
    test_path = os.path.join(args.data_dir, "test.jsonl")
    if os.path.exists(test_path):
        t_overall, t_per = evaluate(model, read("test"), device)
        print(f"[final/test] overall={t_overall}")
        print(f"[final/test] per-lang={json.dumps(t_per, indent=2)}")
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump({"overall": t_overall, "per_lang": t_per}, f, indent=2, ensure_ascii=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()

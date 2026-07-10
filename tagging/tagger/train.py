"""Train the multi-task tagger (POS + lemma + biaffine dependency) on the unified data.

Runs a short smoke phase first (unless disabled) so a broken stack fails in minutes
rather than hours. Metrics: POS accuracy, lemma-script accuracy, UAS, LAS, per language
and overall.
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import TaggerDataset, load_vocab, pad_collate, read_jsonl
from model import MultiTaskTagger


def make_loader(records, tok, vocab, bs, shuffle, args):
    ds = TaggerDataset(records, tok, vocab, args.max_subwords, args.max_words)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=args.workers,
                      collate_fn=partial(pad_collate, pad_id=tok.pad_token_id),
                      pin_memory=True, drop_last=shuffle)


@torch.no_grad()
def evaluate(model, records, tok, vocab, args, device, max_batches=None):
    model.eval()
    loader = make_loader(records, tok, vocab, args.eval_batch_size, False, args)
    # accumulate per-language counts
    per = defaultdict(lambda: dict(pos=0, lemma=0, uas=0, las=0, n=0))
    lang_of = [r["lang"] for r in records]
    # We iterate without shuffle so batch order matches record order for lang lookup.
    idx = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        m = batch["word_mask"].bool()
        pos_pred = out.pos_logits.argmax(-1)
        lemma_pred = out.lemma_logits.argmax(-1)
        arc_pred = out.arc_scores.argmax(-1)
        # relation predicted at predicted head
        B, W = pos_pred.shape
        rel_at_pred = out.rel_scores[torch.arange(B).unsqueeze(1), torch.arange(W).unsqueeze(0), arc_pred]
        rel_pred = rel_at_pred.argmax(-1)
        bs_now = batch["input_ids"].size(0)
        for r in range(bs_now):
            lang = lang_of[idx] if idx < len(lang_of) else "?"
            idx += 1
            wm = m[r]
            n = wm.sum().item()
            if n == 0:
                continue
            p = per[lang]
            p["n"] += n
            p["pos"] += (pos_pred[r][wm] == batch["pos"][r][wm]).sum().item()
            p["lemma"] += (lemma_pred[r][wm] == batch["lemma"][r][wm]).sum().item()
            arc_ok = (arc_pred[r][wm] == batch["head"][r][wm])
            rel_ok = (rel_pred[r][wm] == batch["rel"][r][wm])
            p["uas"] += arc_ok.sum().item()
            p["las"] += (arc_ok & rel_ok).sum().item()
    # aggregate
    agg = dict(pos=0, lemma=0, uas=0, las=0, n=0)
    report = {}
    for lang, p in per.items():
        if p["n"] == 0:
            continue
        report[lang] = {k: round(p[k] / p["n"] * 100, 2) for k in ("pos", "lemma", "uas", "las")}
        for k in ("pos", "lemma", "uas", "las", "n"):
            agg[k] += p[k]
    overall = {k: round(agg[k] / max(1, agg["n"]) * 100, 2) for k in ("pos", "lemma", "uas", "las")}
    model.train()
    return overall, report


def build_optimizer(model, args):
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if n.startswith("encoder.") else head_params).append(p)
    return torch.optim.AdamW([
        {"params": enc_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)


def run_phase(model, opt, sched, loader, device, args, max_steps, log_prefix,
              eval_fn=None, eval_every=None):
    # bf16 autocast needs no gradient scaling (unlike fp16), so we skip GradScaler.
    model.train()
    step = 0
    t0 = time.time()
    running = defaultdict(float)
    best = -1.0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {out.parts}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if sched:
                sched.step()
            for k, v in out.parts.items():
                running[k] += v
            running["loss"] += loss.item()
            step += 1
            if step % args.log_every == 0:
                sps = step / (time.time() - t0)
                msg = " ".join(f"{k}={running[k]/args.log_every:.3f}" for k in
                               ("loss", "pos", "lemma", "arc", "rel"))
                print(f"[{log_prefix}] step {step}/{max_steps} {msg} ({sps:.1f} it/s)", flush=True)
                running = defaultdict(float)
            if eval_fn and eval_every and step % eval_every == 0:
                overall, report = eval_fn()
                print(f"[{log_prefix}] eval@{step} overall={overall}", flush=True)
                score = overall["las"] + overall["pos"]
                if score > best:
                    best = score
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="output/tagger")
    ap.add_argument("--encoder", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-subwords", type=int, default=192)
    ap.add_argument("--max-words", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--allow-cpu", action="store_true", help="permit CPU training (default: refuse)")
    ap.add_argument("--smoke", action="store_true", help="run a tiny smoke phase before full training")
    ap.add_argument("--smoke-only", action="store_true", help="run only the smoke phase, then exit")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")
    if device != "cuda" and not args.allow_cpu:
        raise SystemExit(
            f"Refusing to train on CPU: torch.cuda.is_available() is False (torch={torch.__version__}). "
            "This usually means a torch build incompatible with the driver. "
            "Pass --allow-cpu to override.")

    vocab = load_vocab(os.path.join(args.data_dir, "vocab.json"))
    tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    n_pos, n_dep, n_lemma = len(vocab["pos"]), len(vocab["dep"]), len(vocab["lemma_scripts"])
    print(f"labels: POS={n_pos} DEP={n_dep} lemma={n_lemma}")

    train_records = read_jsonl(os.path.join(args.data_dir, "train.jsonl"), args.train_limit)
    val_records = read_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    print(f"train={len(train_records)} val={len(val_records)}")

    model = MultiTaskTagger(args.encoder, n_pos, n_dep, n_lemma).to(device)

    if args.wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "lexide-parsley"),
                   name=f"tagger-{args.encoder.split('/')[-1]}", config=vars(args))

    # ---- smoke phase on GPU: few steps on a small subset, one eval, assert sanity ----
    if args.smoke or args.smoke_only:
        print("=== SMOKE PHASE ===", flush=True)
        sm_records = train_records[:2000]
        sm_loader = make_loader(sm_records, tok, vocab, min(8, args.batch_size), True, args)
        opt = build_optimizer(model, args)
        run_phase(model, opt, None, sm_loader, device, args, max_steps=60, log_prefix="smoke")
        overall, _ = evaluate(model, val_records[:500], tok, vocab, args, device, max_batches=8)
        print(f"[smoke] eval overall={overall}", flush=True)
        assert all(math.isfinite(v) for v in overall.values()), "smoke eval produced non-finite metrics"
        print("=== SMOKE PASSED ===", flush=True)
        if args.smoke_only:
            return

    # ---- full training ----
    train_loader = make_loader(train_records, tok, vocab, args.batch_size, True, args)
    steps_per_epoch = len(train_loader)
    total_steps = int(steps_per_epoch * args.epochs)
    opt = build_optimizer(model, args)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup_ratio), total_steps)
    print(f"total_steps={total_steps} steps/epoch={steps_per_epoch}")

    def eval_fn():
        return evaluate(model, val_records, tok, vocab, args, device)

    best = -1.0
    step = 0
    t0 = time.time()
    running = defaultdict(float)
    model.train()
    for epoch in range(math.ceil(args.epochs)):
        for batch in train_loader:
            if step >= total_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {out.parts}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            for k, v in out.parts.items():
                running[k] += v
            running["loss"] += loss.item()
            step += 1
            if step % args.log_every == 0:
                sps = step / (time.time() - t0)
                msg = " ".join(f"{k}={running[k]/args.log_every:.3f}" for k in
                               ("loss", "pos", "lemma", "arc", "rel"))
                print(f"[train] step {step}/{total_steps} ep{epoch} {msg} ({sps:.1f} it/s)", flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({f"train/{k}": running[k] / args.log_every for k in running}, step=step)
                running = defaultdict(float)
            if step % args.eval_every == 0:
                overall, report = eval_fn()
                print(f"[eval] step {step} overall={overall}", flush=True)
                print(f"[eval] per-lang={json.dumps(report)}", flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({f"val/{k}": v for k, v in overall.items()}, step=step)
                score = overall["las"] + overall["pos"]
                if score > best:
                    best = score
                    save_checkpoint(model, tok, vocab, args, overall, step)
                    print(f"[eval] new best (las+pos={score:.2f}) -> saved", flush=True)
        if step >= total_steps:
            break

    # final eval on val + save
    overall, report = eval_fn()
    print(f"[final/val] overall={overall}")
    print(f"[final/val] per-lang={json.dumps(report, indent=2)}")
    save_checkpoint(model, tok, vocab, args, overall, step, final=True)

    # held-out test eval
    test_path = os.path.join(args.data_dir, "test.jsonl")
    if os.path.exists(test_path):
        test_records = read_jsonl(test_path)
        t_overall, t_report = evaluate(model, test_records, tok, vocab, args, device)
        print(f"[final/test] overall={t_overall}")
        print(f"[final/test] per-lang={json.dumps(t_report, indent=2)}")
        if args.wandb:
            import wandb
            wandb.log({f"test/{k}": v for k, v in t_overall.items()}, step=step)
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump({"overall": t_overall, "per_lang": t_report}, f, indent=2)


def save_checkpoint(model, tok, vocab, args, metrics, step, final=False):
    name = "final" if final else "best"
    d = os.path.join(args.out_dir, name)
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, "model.pt"))
    tok.save_pretrained(d)
    with open(os.path.join(d, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"pos": vocab["pos_list"], "dep": vocab["dep_list"],
                   "lemma_scripts": vocab["lemma_scripts"]}, f, ensure_ascii=False)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"encoder": args.encoder, "metrics": metrics, "step": step}, f, indent=2)


if __name__ == "__main__":
    main()

"""Train parsley-joint. CUDA is mandatory unless explicitly running a CPU smoke."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from dataset_joint import IndexedCorpus, budget_batches, collate
from eval_e2e import read_jsonl, score, write_scores
from model import JointTagger
from mst import single_root_mst
from predict_joint import predict_batch, save_checkpoint


def balanced(records, per_lang):
    counts, selected = Counter(), []
    for record in records:
        if not per_lang or counts[record["lang"]] < per_lang:
            selected.append(record)
            counts[record["lang"]] += 1
    return selected


@torch.inference_mode()
def evaluate(model, tokenizer, vocab, records, device, batch_size):
    predictions, correct = [], Counter()
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        predicted, encoded, batch = predict_batch(model, tokenizer, vocab, chunk, device)
        predictions.extend(predicted)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.word_heads(encoded, batch["starts"], batch["ends"], batch["word_mask"])
        pos = out["pos_logits"].argmax(-1).cpu().tolist()
        rel = out["rel_scores"].argmax(-1).cpu().tolist()
        arcs = out["arc_scores"].float().cpu().numpy()
        for i, record in enumerate(chunk):
            n = len(record["tokens"])
            heads = single_root_mst(arcs[i, :n, :n + 1])
            for j, token in enumerate(record["tokens"]):
                correct["words"] += 1
                correct["pos"] += vocab["pos"][pos[i][j]] == token["pos"]
                correct["las"] += heads[j] == token["head"] and vocab["dep"][rel[i][j][heads[j]]] == token["dep"]
    metrics = score(records, predictions)
    metrics["teacher_forced"] = {k: correct[k] / max(1, correct["words"]) for k in ("pos", "las")}
    metrics["selection_score"] = sum(metrics["macro"][m]["f1"] for m in ("token", "pos", "las")) / 3
    return metrics


def upload(args):
    if not args.hf_repo:
        return
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=args.hf_repo, folder_path=str(args.output),
                         path_in_repo=args.hf_path,
                         ignore_patterns=["*.tmp", "smoke/*"],
                         commit_message="parsley-joint best checkpoint and evaluation")


def train(args):
    from transformers import AutoTokenizer
    assert torch.cuda.is_available() or args.cpu_ok, "CUDA unavailable; pass --cpu-ok only for intentional CPU tests"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_args.json").write_text(json.dumps(vars(args), default=str, indent=2))
    vocab = json.loads((args.data / "vocab.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, revision=args.encoder_revision, use_fast=True)
    config = {"encoder_name": args.encoder, "encoder_revision": args.encoder_revision,
              "encoder_layers": args.encoder_layers, "n_pos": len(vocab["pos"]),
              "n_dep": len(vocab["dep"]), "n_lemma": len(vocab["lemma_scripts"]),
              "n_langs": len(vocab["langs"]), "char_dim": args.char_dim,
              "char_hidden": args.char_hidden, "word_dim": args.word_dim,
              "char_buckets": 65536, "arc_dim": 256, "rel_dim": 128,
              "dropout": 0.2, "lang_dropout": 0.15,
              "loss_weights": {k: getattr(args, k + "_weight") for k in ("boundary", "pos", "lemma", "arc", "rel")}}
    model = JointTagger(**config).to(device)
    # Store the resolved encoder/tokenizer revision, not a floating main alias.
    config["encoder_revision"] = getattr(model.encoder.config, "_commit_hash", None) or args.encoder_revision
    (args.output / "tokenizer_source.json").write_text(json.dumps({"name": args.encoder, "revision": config["encoder_revision"]}, indent=2))
    parameters = [{"params": model.encoder.parameters(), "lr": args.encoder_lr},
                  {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")], "lr": args.head_lr}]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    corpus = IndexedCorpus(args.data / "train.jsonl")
    validation = balanced(read_jsonl(args.data / "val.jsonl"), args.val_per_lang)
    if not validation:
        raise ValueError("No validation rows")
    sample, counts = corpus.sample(args.lang_cap, args.seed, args.limit_per_lang)
    planned_examples = max(1, len(sample) * args.epochs)
    # Linear warmup/decay by examples seen, rather than a guessed number of
    # variable-sized token-budget batches. Fresh samples have the same counts.
    base_lrs = [args.encoder_lr, args.head_lr]
    best, step, seen, last_upload = -1., 0, 0, -math.inf
    pending_upload = False
    started = time.monotonic()
    wandb = None
    if args.wandb_project:
        import wandb as wb
        wandb = wb.init(project=args.wandb_project, config=vars(args))
    history = (args.output / "history.jsonl").open("w")

    def run_eval():
        nonlocal best, last_upload, pending_upload
        metrics = evaluate(model, tokenizer, vocab, validation, device, args.eval_batch_size)
        metrics.update(step=step, elapsed_seconds=time.monotonic() - started)
        write_scores(metrics, args.output / "val_latest")
        history.write(json.dumps(metrics) + "\n")
        history.flush()
        improved = metrics["selection_score"] > best
        if improved:
            best = metrics["selection_score"]
            save_checkpoint(args.output / "best", model, tokenizer, vocab, config)
            write_scores(metrics, args.output / "val_best")
            pending_upload = True
        if pending_upload and time.monotonic() - last_upload >= args.upload_interval:
            upload(args)
            last_upload = time.monotonic()
            pending_upload = False
        print(json.dumps({"eval_step": step, "score": metrics["selection_score"],
                          "teacher_forced": metrics["teacher_forced"], "best": best}), flush=True)
        if wandb:
            wandb.log({"val/score": metrics["selection_score"], "val/teacher_pos": metrics["teacher_forced"]["pos"],
                       "val/teacher_las": metrics["teacher_forced"]["las"]}, step=step)
        model.train()

    for epoch in range(args.epochs):
        if args.limit_per_lang:
            # A smoke/debug limit defines one tiny fixed corpus. Only its order
            # changes; normal training still draws a fresh capped sample below.
            indices = sample.copy()
            random.Random(args.seed + epoch).shuffle(indices)
        else:
            indices, counts = corpus.sample(args.lang_cap, args.seed + epoch)
        print(json.dumps({"epoch": epoch + 1, "sampled": counts}), flush=True)
        model.train()
        for items in budget_batches(corpus, indices, tokenizer, vocab, args.token_budget,
                                    args.max_subwords, model.char_buckets, args.seed + epoch):
            batch = collate(items, tokenizer.pad_token_id, device)
            progress = min(1., (step + 1) / args.max_steps if args.max_steps else (seen + len(items)) / planned_examples)
            factor = min(progress / max(args.warmup_fraction, 1e-9),
                         (1 - progress) / max(1 - args.warmup_fraction, 1e-9))
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * max(0., factor)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(batch)
            if not torch.isfinite(out["loss"]):
                raise FloatingPointError(f"Nonfinite loss at step {step}: {out['parts']}")
            out["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            seen += len(items)
            if step % args.log_every == 0 or step == 1:
                log = {"step": step, "loss": float(out["loss"].detach()),
                       "grad_norm": float(norm), "sentences_per_second": seen / (time.monotonic() - started),
                       **{k: float(v) for k, v in out["parts"].items()}}
                print(json.dumps(log), flush=True)
                if wandb:
                    wandb.log(log, step=step)
            del out
            if step % args.eval_every == 0:
                run_eval()
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    if step % args.eval_every or not step:
        run_eval()
    history.close()
    upload(args)  # Force final pending best upload before any node can disappear.
    if wandb:
        wandb.finish()
    print(json.dumps({"finished_steps": step, "seconds": time.monotonic() - started, "best": best}), flush=True)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/processed-joint"))
    ap.add_argument("--output", type=Path, default=Path("tagger/output/joint-v1"))
    ap.add_argument("--encoder", default="BAAI/bge-m3")
    ap.add_argument("--encoder-revision", default=None)
    ap.add_argument("--encoder-layers", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lang-cap", type=int, default=300000)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--max-subwords", type=int, default=256)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup-fraction", type=float, default=0.06)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--val-per-lang", type=int, default=100)
    ap.add_argument("--limit-per-lang", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--char-dim", type=int, default=256)
    ap.add_argument("--char-hidden", type=int, default=256)
    ap.add_argument("--word-dim", type=int, default=768)
    ap.add_argument("--cpu-ok", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--wandb-project")
    ap.add_argument("--hf-repo")
    ap.add_argument("--hf-path", default="training-runs/joint-v1")
    ap.add_argument("--upload-interval", type=float, default=1800)
    ap.add_argument("--smoke", action="store_true", help="200 tiny-data steps, then restart the real run from pretrained weights")
    ap.add_argument("--smoke-steps", type=int, default=200)
    for name in ("boundary", "pos", "lemma", "arc", "rel"):
        ap.add_argument(f"--{name}-weight", type=float, default=1.)
    return ap


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = parser().parse_args()
    if args.smoke:
        # Fresh process releases optimizer/encoder allocations before the full run.
        import subprocess
        import sys
        command = [a for a in sys.argv[1:] if a != "--smoke"]
        command += ["--output", str(args.output / "smoke"), "--max-steps", str(args.smoke_steps),
                    "--epochs", "1000", "--limit-per-lang", "17", "--val-per-lang", "8",
                    "--eval-every", str(args.smoke_steps), "--hf-repo", "", "--wandb-project", ""]
        subprocess.run([sys.executable, __file__, *command], check=True)
    train(args)

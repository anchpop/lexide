#!/usr/bin/env python3
"""Stage 2: ByT5-small fine-tuning for lemmatization.

Seq2seq model: given "POS : token_text", predict "lemma_text".
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

LANG_MAP = {
    "eng": "English", "deu": "German", "fra": "French", "spa": "Spanish",
    "kor": "Korean", "por": "Portuguese", "ita": "Italian", "rus": "Russian",
    "jpn": "Japanese", "hin": "Hindi",
}


def load_lemma_pairs(data_dir):
    """Extract (token_text, pos, lemma, lang) tuples from all JSONL files."""
    pairs = []
    data_dir = Path(data_dir)
    for lang_code in LANG_MAP:
        path = data_dir / f"cleaned_{lang_code}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                for tok in obj["tokens"]:
                    text = tok["text"]
                    pos = tok.get("pos", "X")
                    lemma = tok.get("lemma", text)
                    pairs.append((text, pos, lemma, lang_code))
    print(f"Loaded {len(pairs)} lemma pairs")
    return pairs


def deduplicate_pairs(pairs):
    """Deduplicate (text, pos) -> lemma. Keep most common lemma for conflicts."""
    from collections import Counter

    counts = {}
    for text, pos, lemma, lang in pairs:
        key = (text, pos, lang)
        if key not in counts:
            counts[key] = Counter()
        counts[key][lemma] += 1

    deduped = []
    for (text, pos, lang), lemma_counts in counts.items():
        best_lemma = lemma_counts.most_common(1)[0][0]
        deduped.append((text, pos, best_lemma, lang))

    print(f"Deduplicated: {len(pairs)} -> {len(deduped)} unique (text, pos, lang)")
    return deduped


class LemmaDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_input_len=64, max_target_len=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text, pos, lemma, lang = self.pairs[idx]
        input_text = f"{pos} : {text}"

        input_enc = self.tokenizer(
            input_text, max_length=self.max_input_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        target_enc = self.tokenizer(
            lemma, max_length=self.max_target_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_len=64):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0
    n_batches = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        total_loss += out.loss.item()
        n_batches += 1

        # Generate and check exact match
        gen = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=32, num_beams=1,
        )

        gold_labels = labels.clone()
        gold_labels[gold_labels == -100] = tokenizer.pad_token_id
        pred_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        gold_texts = tokenizer.batch_decode(gold_labels, skip_special_tokens=True)

        for p, g in zip(pred_texts, gold_texts):
            total_correct += int(p.strip() == g.strip())
            total_count += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "lemma_acc": total_correct / max(total_count, 1),
    }


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if HAS_WANDB and args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    pairs = load_lemma_pairs(args.data_dir)
    if args.deduplicate:
        pairs = deduplicate_pairs(pairs)

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_train = int(0.9 * len(pairs))
    n_val = int(0.05 * len(pairs))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/byt5-small").to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = LemmaDataset(train_pairs, tokenizer, args.max_input_len, args.max_target_len)
    val_ds = LemmaDataset(val_pairs, tokenizer, args.max_input_len, args.max_target_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs // args.grad_accum
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    step = 0

    for epoch in range(args.num_epochs):
        model.train()
        ep_loss = 0
        ep_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=ids, attention_mask=mask, labels=labels)
                    loss = out.loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = out.loss / args.grad_accum
                loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            ep_loss += out.loss.item()
            ep_steps += 1
            pbar.set_postfix(loss=f"{out.loss.item():.4f}")

            if HAS_WANDB and args.wandb and step > 0 and step % 50 == 0:
                wandb.log({"train/loss": out.loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=step)

        print(f"\nEpoch {epoch + 1} train loss: {ep_loss / ep_steps:.4f}")

        val = evaluate(model, val_loader, tokenizer, device, args.max_target_len)
        print(f"Epoch {epoch + 1} val: " + ", ".join(f"{k}={v:.4f}" for k, v in val.items()))

        if HAS_WANDB and args.wandb:
            wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            print(f"  -> New best val loss: {best_val_loss:.4f}")
            model.save_pretrained(out_dir / "best_model")
            tokenizer.save_pretrained(out_dir / "best_model")

        model.save_pretrained(out_dir / "latest")
        tokenizer.save_pretrained(out_dir / "latest")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {out_dir}")

    # Upload to HuggingFace Hub
    hf_username = os.getenv("HF_USERNAME")
    if hf_username:
        hf_repo_name = f"{hf_username}/lexide-byt5-lemma"
        print(f"Uploading Stage 2 model to HuggingFace Hub: {hf_repo_name}")
        try:
            best_model = T5ForConditionalGeneration.from_pretrained(out_dir / "best_model")
            best_tokenizer = AutoTokenizer.from_pretrained(out_dir / "best_model")
            best_model.push_to_hub(hf_repo_name, private=True, create_pr=False)
            best_tokenizer.push_to_hub(hf_repo_name, private=True, create_pr=False)
            print(f"Model uploaded to: https://huggingface.co/{hf_repo_name}")
        except Exception as e:
            print(f"Failed to upload to HuggingFace: {e}")
            print("Model saved locally only.")

    if HAS_WANDB and args.wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="output/stage2")
    p.add_argument("--max_input_len", type=int, default=64)
    p.add_argument("--max_target_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--deduplicate", action="store_true", default=True)
    p.add_argument("--no_deduplicate", dest="deduplicate", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="lexide-pipeline")
    p.add_argument("--run_name", default="stage2-byt5-lemma")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()

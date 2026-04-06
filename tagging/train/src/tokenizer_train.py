#!/usr/bin/env python3
"""Character-level BIO tokenizer with CRF.

Sliding window classifier with CRF: for each character, predict B (token start),
I (inside token), or O (outside/whitespace). 31-character window (15 left, target,
15 right) + language ID. CRF enforces valid transitions (e.g. I cannot follow O).
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ============================================================
# Constants
# ============================================================
LANG_MAP = {
    "eng": "English", "deu": "German", "fra": "French", "spa": "Spanish",
    "kor": "Korean", "por": "Portuguese", "ita": "Italian", "rus": "Russian",
    "jpn": "Japanese", "hin": "Hindi",
}
LANG2ID = {lang: i for i, lang in enumerate(LANG_MAP)}
NUM_LANGS = len(LANG2ID)

WINDOW = 15  # characters on each side
WINDOW_SIZE = 2 * WINDOW + 1  # 31
PAD_CHAR = 0
NUM_TAGS = 3  # B=0, I=1, O=2


# ============================================================
# CRF
# ============================================================
class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.02)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.02)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.02)
        self._init_constraints()

    def _init_constraints(self):
        with torch.no_grad():
            # I (1) cannot follow O (2) — can't be inside a token without starting one
            self.transitions[2, 1] = -10000.0
            # Can't start a sequence with I
            self.start_transitions[1] = -10000.0

    def forward(self, emissions, labels, mask):
        """Negative log-likelihood loss. labels: (batch, seq_len), mask: (batch, seq_len) bool."""
        labels = labels.clamp(min=0)
        log_Z = self._forward_alg(emissions, mask)
        gold_score = self._score_sentence(emissions, labels, mask)
        return (log_Z - gold_score).mean()

    def _forward_alg(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            next_score = score.unsqueeze(2) + trans + emit
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, emissions, labels, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[labels[:, 0]]
        score = score + emissions[torch.arange(batch_size, device=emissions.device), 0, labels[:, 0]]
        for t in range(1, seq_len):
            m = mask[:, t].float()
            trans = self.transitions[labels[:, t - 1], labels[:, t]]
            emit = emissions[torch.arange(batch_size, device=emissions.device), t, labels[:, t]]
            score = score + (trans + emit) * m
        last_idx = mask.long().sum(dim=1) - 1
        last_tags = labels[torch.arange(batch_size, device=emissions.device), last_idx]
        score = score + self.end_transitions[last_tags]
        return score

    def decode(self, emissions, mask):
        """Viterbi decoding."""
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        for t in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            next_score, indices = next_score.max(dim=1)
            next_score = next_score + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
            history.append(indices)
        score = score + self.end_transitions
        _, best_last = score.max(dim=1)
        best_tags = torch.zeros(batch_size, seq_len, dtype=torch.long, device=emissions.device)
        best_tags[:, -1] = best_last
        for t in range(len(history) - 1, -1, -1):
            best_tags[:, t] = history[t][torch.arange(batch_size, device=emissions.device), best_tags[:, t + 1]]
        return best_tags


# ============================================================
# Data
# ============================================================
def load_all_sentences(data_dir, include_generated=True):
    sentences = []
    data_dir = Path(data_dir)

    # Main annotated data
    for lang_code in LANG_MAP:
        path = data_dir / f"cleaned_{lang_code}.jsonl"
        if not path.exists():
            continue
        count = 0
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                obj["lang"] = lang_code
                sentences.append(obj)
                count += 1
        print(f"  {lang_code}: {count} sentences")

    # Generated data
    if include_generated:
        gen_dir = data_dir / "generated-by-big-model"
        if gen_dir.exists():
            for lang_code in LANG_MAP:
                for suffix in ["target_language_sentences_tokenization.jsonl",
                               "target_language_multiword_terms_tokenization.jsonl"]:
                    path = gen_dir / f"{lang_code}_{suffix}"
                    if not path.exists():
                        continue
                    count = 0
                    with open(path) as f:
                        for line in f:
                            obj = json.loads(line)
                            obj["lang"] = lang_code
                            # Normalize nested text/lemma format
                            for tok in obj.get("tokens", []):
                                if isinstance(tok.get("text"), dict):
                                    tok["text"] = tok["text"]["text"]
                                if isinstance(tok.get("lemma"), dict):
                                    tok["lemma"] = tok["lemma"]["lemma"]
                            sentences.append(obj)
                            count += 1
                    print(f"  {lang_code} (gen {suffix.split('_')[2]}): {count} sentences")

    print(f"Total: {len(sentences)} sentences")
    return sentences


def sentence_to_chars(tokens):
    """Convert tokens to character sequence with BIO labels."""
    chars = []
    labels = []
    for tok in tokens:
        text = tok["text"]
        for ci, ch in enumerate(text):
            chars.append(min(ord(ch), 0xFFFF))  # Clamp to BMP
            labels.append(0 if ci == 0 else 1)  # B=0, I=1
        ws = tok.get("whitespace", "")
        for ch in ws:
            chars.append(min(ord(ch), 0xFFFF))
            labels.append(2)  # O=2
    return chars, labels


class TokenizerDataset(Dataset):
    """Returns full sentences for CRF training."""
    def __init__(self, sentences, max_len=2048):
        self.sentences = sentences
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        tokens = sent["tokens"]
        lang_id = LANG2ID.get(sent.get("lang", "eng"), 0)
        chars, labels = sentence_to_chars(tokens)

        n = min(len(chars), self.max_len)
        chars = chars[:n]
        labels = labels[:n]

        # Build windows for each character
        windows = []
        for i in range(n):
            window = []
            for j in range(i - WINDOW, i + WINDOW + 1):
                if 0 <= j < n:
                    window.append(chars[j])
                else:
                    window.append(PAD_CHAR)
            windows.append(window)

        return {
            "windows": torch.tensor(windows, dtype=torch.long),  # (n, 31)
            "labels": torch.tensor(labels, dtype=torch.long),     # (n,)
            "lang_id": lang_id,
            "n_chars": n,
        }


def collate_fn(batch):
    max_n = max(b["n_chars"] for b in batch)
    batch_size = len(batch)

    windows = torch.zeros(batch_size, max_n, WINDOW_SIZE, dtype=torch.long)
    labels = torch.full((batch_size, max_n), -100, dtype=torch.long)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    lang_ids = torch.tensor([b["lang_id"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        n = b["n_chars"]
        windows[i, :n] = b["windows"]
        labels[i, :n] = b["labels"]
        mask[i, :n] = True

    return {
        "windows": windows,
        "labels": labels,
        "mask": mask,
        "lang_ids": lang_ids,
    }


# ============================================================
# Model
# ============================================================
class CharTokenizer(nn.Module):
    def __init__(self, num_langs=NUM_LANGS, char_emb_dim=128, lang_emb_dim=32,
                 hidden_dim=256, num_layers=3, window_size=WINDOW_SIZE):
        super().__init__()
        self.char_emb = nn.Embedding(0x10000, char_emb_dim, padding_idx=PAD_CHAR)
        self.lang_emb = nn.Embedding(num_langs, lang_emb_dim)

        input_dim = char_emb_dim * window_size + lang_emb_dim
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, NUM_TAGS))
        self.feature_net = nn.Sequential(*layers)
        self.crf = CRF(NUM_TAGS)

    def get_emissions(self, windows, lang_ids):
        """windows: (batch, seq_len, window_size), lang_ids: (batch,)"""
        batch_size, seq_len, ws = windows.shape
        char_embs = self.char_emb(windows)  # (batch, seq_len, ws, emb)
        char_flat = char_embs.view(batch_size, seq_len, -1)  # (batch, seq_len, ws*emb)
        lang = self.lang_emb(lang_ids).unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, lang_emb)
        x = torch.cat([char_flat, lang], dim=-1)  # (batch, seq_len, input_dim)
        emissions = self.feature_net(x)  # (batch, seq_len, NUM_TAGS)
        return emissions.clamp(-5, 5)  # Prevent CRF forward algorithm overflow

    def forward(self, windows, lang_ids, labels, mask):
        """Training: returns CRF loss."""
        emissions = self.get_emissions(windows, lang_ids)
        return self.crf(emissions, labels, mask)

    def decode(self, windows, lang_ids, mask):
        """Inference: Viterbi decoding."""
        emissions = self.get_emissions(windows, lang_ids)
        return self.crf.decode(emissions, mask)


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    stats = defaultdict(float)
    for batch in tqdm(loader, desc="Eval", leave=False):
        windows = batch["windows"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        lang_ids = batch["lang_ids"].to(device)

        loss = model(windows, lang_ids, labels, mask)
        preds = model.decode(windows, lang_ids, mask)

        valid = mask
        stats["loss"] += loss.item() * mask.sum().item()
        stats["correct"] += (preds[valid] == labels[valid]).sum().item()
        stats["total"] += valid.sum().item()

        pred_b = (preds == 0) & valid
        gold_b = (labels == 0) & valid
        stats["b_tp"] += (pred_b & gold_b).sum().item()
        stats["b_fp"] += (pred_b & ~gold_b).sum().item()
        stats["b_fn"] += (~pred_b & gold_b).sum().item()

    acc = stats["correct"] / max(stats["total"], 1)
    loss = stats["loss"] / max(stats["total"], 1)
    prec = stats["b_tp"] / max(stats["b_tp"] + stats["b_fp"], 1)
    rec = stats["b_tp"] / max(stats["b_tp"] + stats["b_fn"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"loss": loss, "acc": acc, "b_prec": prec, "b_rec": rec, "b_f1": f1}


@torch.no_grad()
def evaluate_sentence_accuracy(model, sentences, device):
    """Full-sentence accuracy: % of sentences where every character is predicted correctly."""
    model.eval()
    per_lang = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total = 0

    for sent in tqdm(sentences, desc="SentAcc"):
        tokens = sent["tokens"]
        lang_id = LANG2ID.get(sent.get("lang", "eng"), 0)
        lang_code = sent.get("lang", "eng")
        chars, labels = sentence_to_chars(tokens)
        n = len(chars)
        if n == 0:
            continue

        # Build windows
        windows = []
        for i in range(n):
            window = []
            for j in range(i - WINDOW, i + WINDOW + 1):
                if 0 <= j < n:
                    window.append(chars[j])
                else:
                    window.append(PAD_CHAR)
            windows.append(window)

        w_tensor = torch.tensor([windows], dtype=torch.long, device=device)
        l_tensor = torch.tensor([lang_id], dtype=torch.long, device=device)
        m_tensor = torch.ones(1, n, dtype=torch.bool, device=device)
        preds = model.decode(w_tensor, l_tensor, m_tensor)[0].cpu().tolist()

        all_correct = all(p == g for p, g in zip(preds, labels))
        total += 1
        per_lang[lang_code]["total"] += 1
        if all_correct:
            total_correct += 1
            per_lang[lang_code]["correct"] += 1

    return {
        "overall": total_correct / max(total, 1),
        "correct": total_correct,
        "total": total,
        "per_lang": {lang: d["correct"] / max(d["total"], 1) for lang, d in per_lang.items()},
    }


# ============================================================
# Training
# ============================================================
def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if HAS_WANDB and args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    print("Loading data...")
    sentences = load_all_sentences(args.data_dir)

    rng = random.Random(args.seed)
    rng.shuffle(sentences)
    n_train = int(0.9 * len(sentences))
    n_val = int(0.05 * len(sentences))
    train_s = sentences[:n_train]
    val_s = sentences[n_train:n_train + n_val]

    train_ds = TokenizerDataset(train_s, args.max_len)
    val_ds = TokenizerDataset(val_s, args.max_len)
    print(f"Train: {len(train_ds):,} sentences, Val: {len(val_ds):,} sentences")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    model = CharTokenizer(
        char_emb_dim=args.char_emb_dim,
        lang_emb_dim=args.lang_emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    warmup = int(total_steps * args.warmup_ratio)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    step = 0

    for epoch in range(args.num_epochs):
        model.train()
        ep_loss = 0.0
        ep_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in pbar:
            windows = batch["windows"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            lang_ids = batch["lang_ids"].to(device)

            loss = model(windows, lang_ids, labels, mask)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            n_chars = mask.sum().item()
            ep_loss += loss.item() * n_chars
            ep_total += n_chars

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if HAS_WANDB and args.wandb and step % 100 == 0:
                wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=step)

        train_loss = ep_loss / max(ep_total, 1)
        print(f"\nEpoch {epoch + 1} train: loss={train_loss:.4f}")

        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1} val: " + ", ".join(f"{k}={v:.4f}" for k, v in val.items()))

        if HAS_WANDB and args.wandb:
            wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)

        if val["b_f1"] > best_val_f1:
            best_val_f1 = val["b_f1"]
            print(f"  -> New best val B-F1: {best_val_f1:.4f}")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_metrics": val, "args": vars(args),
            }, out_dir / "best_model.pt")

    print(f"\nDone! Best val B-F1: {best_val_f1:.4f}")
    print(f"Model saved to {out_dir}")

    # Full-sentence accuracy on test set
    test_s = sentences[n_train + n_val:]
    print(f"\n--- Full-sentence accuracy on {len(test_s)} test sentences ---")
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    sent_eval = evaluate_sentence_accuracy(model, test_s, device)
    print(f"Overall: {sent_eval['overall']:.4f} ({sent_eval['correct']}/{sent_eval['total']})")
    for lang, acc in sorted(sent_eval["per_lang"].items()):
        print(f"  {LANG_MAP.get(lang, lang)}: {acc:.4f}")
    if HAS_WANDB and args.wandb:
        wandb.log({"test/sentence_acc": sent_eval["overall"]})
        for lang, acc in sent_eval["per_lang"].items():
            wandb.log({f"test/sentence_acc_{lang}": acc})

    # Upload to HuggingFace Hub
    hf_username = os.getenv("HF_USERNAME")
    if hf_username:
        from huggingface_hub import HfApi
        hf_repo_name = f"{hf_username}/lexide-char-tokenizer"
        print(f"Uploading model to HuggingFace Hub: {hf_repo_name}")
        try:
            api = HfApi()
            api.create_repo(hf_repo_name, private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(out_dir / "best_model.pt"),
                path_in_repo="best_model.pt",
                repo_id=hf_repo_name,
            )
            print(f"Model uploaded to: https://huggingface.co/{hf_repo_name}")
        except Exception as e:
            print(f"Failed to upload to HuggingFace: {e}")

    if HAS_WANDB and args.wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="output/tokenizer")
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--char_emb_dim", type=int, default=192)
    p.add_argument("--lang_emb_dim", type=int, default=48)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="lexide-pipeline")
    p.add_argument("--run_name", default="char-tokenizer-crf")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()

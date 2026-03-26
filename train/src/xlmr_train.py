#!/usr/bin/env python3
"""XLM-R Large for POS tagging + biaffine dependency parsing.

Single forward pass through XLM-R encoder, word-level POS classification
and biaffine arc/label prediction in parallel.
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
from transformers import XLMRobertaModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ============================================================
# Constants
# ============================================================
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
UPOS2ID = {t: i for i, t in enumerate(UPOS_TAGS)}
NUM_POS = len(UPOS_TAGS)

DEP_RELS = [
    "acl", "acl:relcl", "advcl", "advcl:relcl", "advmod", "advmod:emph",
    "advmod:lmod", "amod", "appos", "aux", "aux:pass", "case", "cc",
    "cc:preconj", "ccomp", "clf", "compound", "compound:lvc", "compound:prt",
    "compound:redup", "compound:svc", "conj", "cop", "csubj", "csubj:outer",
    "csubj:pass", "dep", "det", "det:nummod", "det:poss", "discourse",
    "dislocated", "expl", "expl:impers", "expl:pass", "expl:pv", "fixed",
    "flat", "flat:foreign", "flat:name", "goeswith", "iobj", "list", "mark",
    "nmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubj:outer", "nsubj:pass",
    "nummod", "obj", "obl", "obl:agent", "obl:arg", "obl:lmod", "obl:tmod",
    "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp",
]
DEP2ID = {d: i for i, d in enumerate(DEP_RELS)}
ID2DEP = {i: d for d, i in DEP2ID.items()}
NUM_DEP_RELS = len(DEP_RELS)

LANG_MAP = {
    "eng": "English", "deu": "German", "fra": "French", "spa": "Spanish",
    "kor": "Korean", "por": "Portuguese", "ita": "Italian", "rus": "Russian",
}
LANG2ID = {lang: i for i, lang in enumerate(LANG_MAP)}
NUM_LANGS = len(LANG2ID)


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
            print(f"Warning: {path} not found")
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
                            for tok in obj.get("tokens", []):
                                if isinstance(tok.get("text"), dict):
                                    tok["text"] = tok["text"]["text"]
                                if isinstance(tok.get("lemma"), dict):
                                    tok["lemma"] = tok["lemma"]["lemma"]
                            sentences.append(obj)
                            count += 1
                    print(f"  {lang_code} (gen): {count} sentences")

    # Filter out bad sentences
    clean = []
    bad_heads = 0
    empty_tokens = 0
    for sent in sentences:
        tokens = sent["tokens"]
        n = len(tokens)
        if any(t.get("head", 0) > n or t.get("head", 0) < 0 for t in tokens):
            bad_heads += 1
            continue
        if any(t.get("text", "") == "" for t in tokens):
            empty_tokens += 1
            continue
        clean.append(sent)
    if bad_heads:
        print(f"Filtered {bad_heads} sentences with out-of-bounds head indices")
    if empty_tokens:
        print(f"Filtered {empty_tokens} sentences with empty tokens")

    print(f"Total: {len(clean)} sentences")
    return clean


class DepParseDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        tokens = sent["tokens"]
        words = [t["text"] for t in tokens]

        # Tokenize with word-level alignment
        enc = self.tokenizer(
            words, is_split_into_words=True,
            max_length=self.max_length, truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        wids = enc.word_ids()

        # Find first subword index for each word
        word_starts = []
        prev_wid = None
        for i, wid in enumerate(wids):
            if wid is not None and wid != prev_wid:
                word_starts.append(i)
            prev_wid = wid

        n_words = len(word_starts)
        tokens = tokens[:n_words]  # truncation may drop trailing words

        pos_labels = [UPOS2ID.get(t.get("pos", "X"), UPOS2ID["X"]) for t in tokens]
        head_labels = [t.get("head", 0) for t in tokens]
        rel_labels = [DEP2ID.get(t.get("dep", "dep"), DEP2ID.get("dep", 0)) for t in tokens]
        lang_id = LANG2ID.get(sent.get("lang", "eng"), 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_starts": word_starts,
            "pos_labels": pos_labels,
            "head_labels": head_labels,
            "rel_labels": rel_labels,
            "n_words": n_words,
            "lang_id": lang_id,
        }


def collate_fn(batch):
    # Pad subword sequences
    max_subwords = max(b["input_ids"].shape[0] for b in batch)
    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_subwords, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_subwords, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]

    # Pad word-level sequences
    max_words = max(b["n_words"] for b in batch)
    word_starts = torch.zeros(batch_size, max_words, dtype=torch.long)
    pos_labels = torch.full((batch_size, max_words), -100, dtype=torch.long)
    head_labels = torch.full((batch_size, max_words), -100, dtype=torch.long)
    rel_labels = torch.full((batch_size, max_words), -100, dtype=torch.long)
    word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool)

    lang_ids = torch.tensor([b["lang_id"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        n = b["n_words"]
        word_starts[i, :n] = torch.tensor(b["word_starts"])
        pos_labels[i, :n] = torch.tensor(b["pos_labels"])
        head_labels[i, :n] = torch.tensor(b["head_labels"])
        rel_labels[i, :n] = torch.tensor(b["rel_labels"])
        word_mask[i, :n] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_starts": word_starts,
        "pos_labels": pos_labels,
        "head_labels": head_labels,
        "rel_labels": rel_labels,
        "word_mask": word_mask,
        "lang_ids": lang_ids,
    }


# ============================================================
# Model
# ============================================================
class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.zeros(out_dim, in_dim + 1, in_dim + 1))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], -1)
        y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], -1)
        s = torch.einsum("bnd,odh,bmh->bonm", x, self.W, y)
        if self.out_dim == 1:
            return s.squeeze(1)
        return s.permute(0, 2, 3, 1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.33):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class XLMRForNLP(nn.Module):
    def __init__(self, num_pos=NUM_POS, num_dep_rels=NUM_DEP_RELS, num_langs=NUM_LANGS,
                 hidden=1024, arc_dim=256, rel_dim=128, mlp_drop=0.33):
        super().__init__()
        self.xlmr = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        self.hidden = hidden

        # Language embedding
        self.lang_emb = nn.Embedding(num_langs, hidden)

        # POS head
        self.pos_head = nn.Linear(hidden, num_pos)

        # Biaffine dependency parsing
        self.root_emb = nn.Parameter(torch.randn(hidden) * 0.02)
        self.arc_dep_mlp = MLP(hidden, arc_dim, mlp_drop)
        self.arc_head_mlp = MLP(hidden, arc_dim, mlp_drop)
        self.rel_dep_mlp = MLP(hidden, rel_dim, mlp_drop)
        self.rel_head_mlp = MLP(hidden, rel_dim, mlp_drop)
        self.arc_biaffine = Biaffine(arc_dim, 1)
        self.rel_biaffine = Biaffine(rel_dim, num_dep_rels)

    def forward(self, input_ids, attention_mask, word_starts, word_mask, lang_ids=None):
        h = self.xlmr(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if lang_ids is not None:
            h = h + self.lang_emb(lang_ids).unsqueeze(1)

        batch_size = h.shape[0]
        max_words = word_starts.shape[1]

        # Extract first-subword representations per word
        idx = word_starts.unsqueeze(-1).expand(-1, -1, self.hidden)
        word_reps = h.gather(1, idx)
        word_reps = word_reps * word_mask.unsqueeze(-1).float()

        # POS
        pos_logits = self.pos_head(word_reps)

        # Deps: prepend root embedding
        root = self.root_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        head_reps = torch.cat([root, word_reps], dim=1)  # (B, max_words+1, H)
        dep_reps = word_reps  # (B, max_words, H)

        arc_scores = self.arc_biaffine(self.arc_dep_mlp(dep_reps), self.arc_head_mlp(head_reps))
        rel_scores = self.rel_biaffine(self.rel_dep_mlp(dep_reps), self.rel_head_mlp(head_reps))

        return pos_logits, arc_scores, rel_scores


# ============================================================
# Loss
# ============================================================
def compute_loss(pos_logits, arc_scores, rel_scores,
                 pos_labels, head_labels, rel_labels, word_mask, device):
    valid = word_mask

    # Assertions to catch OOB before CUDA hides the source
    assert pos_labels[valid].min() >= 0 and pos_labels[valid].max() < pos_logits.shape[-1], \
        f"POS OOB: min={pos_labels[valid].min()}, max={pos_labels[valid].max()}, n_classes={pos_logits.shape[-1]}"
    assert head_labels[valid].min() >= 0 and head_labels[valid].max() < arc_scores.shape[-1], \
        f"Head OOB: min={head_labels[valid].min()}, max={head_labels[valid].max()}, n_classes={arc_scores.shape[-1]}"
    assert rel_labels[valid].min() >= 0 and rel_labels[valid].max() < rel_scores.shape[-1], \
        f"Rel OOB: min={rel_labels[valid].min()}, max={rel_labels[valid].max()}, n_classes={rel_scores.shape[-1]}"

    pos_loss = F.cross_entropy(pos_logits[valid], pos_labels[valid])
    arc_loss = F.cross_entropy(arc_scores[valid], head_labels[valid])

    # Rel loss at gold heads
    n_rels = rel_scores.shape[-1]
    gold_idx = head_labels.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n_rels)
    rel_at_head = rel_scores.gather(2, gold_idx).squeeze(2)
    rel_loss = F.cross_entropy(rel_at_head[valid], rel_labels[valid])

    return {
        "total": pos_loss + arc_loss + rel_loss,
        "pos": pos_loss,
        "arc": arc_loss,
        "rel": rel_loss,
    }


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    stats = defaultdict(float)
    n_batches = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        ws = batch["word_starts"].to(device)
        wm = batch["word_mask"].to(device)
        pos_lab = batch["pos_labels"].to(device)
        head_lab = batch["head_labels"].to(device)
        rel_lab = batch["rel_labels"].to(device)

        lang = batch["lang_ids"].to(device)

        pos_log, arc_sc, rel_sc = model(ids, mask, ws, wm, lang)
        losses = compute_loss(pos_log, arc_sc, rel_sc, pos_lab, head_lab, rel_lab, wm, device)

        for k, v in losses.items():
            stats[f"loss_{k}"] += v.item()
        n_batches += 1

        valid = wm

        # POS accuracy
        pos_pred = pos_log.argmax(-1)
        stats["pos_correct"] += (pos_pred[valid] == pos_lab[valid]).sum().item()
        stats["pos_total"] += valid.sum().item()

        # UAS
        arc_pred = arc_sc.argmax(-1)
        correct_arc = (arc_pred[valid] == head_lab[valid])
        stats["arc_correct"] += correct_arc.sum().item()
        stats["arc_total"] += valid.sum().item()

        # LAS
        batch_size, max_words = arc_pred.shape
        bi = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(arc_pred)
        wi = torch.arange(max_words, device=device).unsqueeze(0).expand_as(arc_pred)
        pred_rel = rel_sc[bi, wi, arc_pred].argmax(-1)
        correct_rel = (pred_rel[valid] == rel_lab[valid])
        stats["las_correct"] += (correct_arc & correct_rel).sum().item()

    m = {}
    m["pos_acc"] = stats["pos_correct"] / max(stats["pos_total"], 1)
    m["uas"] = stats["arc_correct"] / max(stats["arc_total"], 1)
    m["las"] = stats["las_correct"] / max(stats["arc_total"], 1)
    for k in ["loss_total", "loss_pos", "loss_arc", "loss_rel"]:
        m[k] = stats[k] / max(n_batches, 1)
    return m


@torch.no_grad()
def evaluate_sentence_accuracy(model, sentences, tokenizer, device, max_length=512):
    """Full-sentence accuracy: % of sentences where every token is correct."""
    model.eval()
    metrics = {}
    for metric in ["pos", "uas", "las", "all"]:
        metrics[metric] = {"correct": 0, "total": 0, "per_lang": defaultdict(lambda: {"correct": 0, "total": 0})}

    for sent in tqdm(sentences, desc="SentAcc"):
        tokens = sent["tokens"]
        lang_code = sent.get("lang", "eng")
        lang_id = LANG2ID.get(lang_code, 0)
        words = [t["text"] for t in tokens]
        if not words:
            continue

        enc = tokenizer(words, is_split_into_words=True, max_length=max_length,
                        truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        wids = enc.word_ids()

        word_starts = []
        prev_wid = None
        for i, wid in enumerate(wids):
            if wid is not None and wid != prev_wid:
                word_starts.append(i)
            prev_wid = wid

        n_words = len(word_starts)
        tokens = tokens[:n_words]
        if n_words == 0:
            continue

        ws = torch.tensor([word_starts], dtype=torch.long, device=device)
        wm = torch.ones(1, n_words, dtype=torch.bool, device=device)
        lang = torch.tensor([lang_id], dtype=torch.long, device=device)

        pos_log, arc_sc, rel_sc = model(input_ids, attention_mask, ws, wm, lang)
        pos_pred = pos_log[0, :n_words].argmax(-1).cpu().tolist()
        arc_pred = arc_sc[0, :n_words].argmax(-1).cpu().tolist()
        rel_pred_logits = rel_sc[0, :n_words]  # (n_words, n_heads+1, n_rels)
        rel_pred = [rel_pred_logits[w, arc_pred[w]].argmax(-1).item() for w in range(n_words)]

        gold_pos = [UPOS2ID.get(t.get("pos", "X"), UPOS2ID["X"]) for t in tokens]
        gold_head = [t.get("head", 0) for t in tokens]
        gold_rel = [DEP2ID.get(t.get("dep", "dep"), DEP2ID.get("dep", 0)) for t in tokens]

        pos_ok = all(p == g for p, g in zip(pos_pred, gold_pos))
        uas_ok = all(p == g for p, g in zip(arc_pred, gold_head))
        las_ok = uas_ok and all(p == g for p, g in zip(rel_pred, gold_rel))
        all_ok = pos_ok and las_ok

        for metric, ok in [("pos", pos_ok), ("uas", uas_ok), ("las", las_ok), ("all", all_ok)]:
            metrics[metric]["total"] += 1
            metrics[metric]["per_lang"][lang_code]["total"] += 1
            if ok:
                metrics[metric]["correct"] += 1
                metrics[metric]["per_lang"][lang_code]["correct"] += 1

    return metrics


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
    train_s, val_s = sentences[:n_train], sentences[n_train:n_train + n_val]
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    train_ds = DepParseDataset(train_s, tokenizer, args.max_length)
    val_ds = DepParseDataset(val_s, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    model = XLMRForNLP(
        arc_dim=args.arc_dim, rel_dim=args.rel_dim,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    backbone = list(model.xlmr.parameters())
    backbone_ids = {id(p) for p in backbone}
    heads = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": heads, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

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
        ep_loss = defaultdict(float)
        ep_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            ws = batch["word_starts"].to(device)
            wm = batch["word_mask"].to(device)
            pos_lab = batch["pos_labels"].to(device)
            head_lab = batch["head_labels"].to(device)
            rel_lab = batch["rel_labels"].to(device)
            lang = batch["lang_ids"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pos_log, arc_sc, rel_sc = model(ids, mask, ws, wm, lang)
                    losses = compute_loss(pos_log, arc_sc, rel_sc, pos_lab, head_lab, rel_lab, wm, device)
                scaler.scale(losses["total"] / args.grad_accum).backward()
            else:
                pos_log, arc_sc, rel_sc = model(ids, mask, ws, wm, lang)
                losses = compute_loss(pos_log, arc_sc, rel_sc, pos_lab, head_lab, rel_lab, wm, device)
                (losses["total"] / args.grad_accum).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            for k, v in losses.items():
                ep_loss[k] += v.item()
            ep_steps += 1

            pbar.set_postfix(loss=f"{losses['total'].item():.3f}",
                             pos=f"{losses['pos'].item():.3f}",
                             arc=f"{losses['arc'].item():.3f}",
                             rel=f"{losses['rel'].item():.3f}")

            if HAS_WANDB and args.wandb and step > 0 and step % 10 == 0:
                wandb.log({f"train/{k}": v.item() for k, v in losses.items()}, step=step)
                wandb.log({"train/lr": scheduler.get_last_lr()[0]}, step=step)

        avg = {k: v / ep_steps for k, v in ep_loss.items()}
        print(f"\nEpoch {epoch + 1} train: " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))

        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1} val: " + ", ".join(f"{k}={v:.4f}" for k, v in val.items()))

        if HAS_WANDB and args.wandb:
            wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)

        if val["loss_total"] < best_val_loss:
            best_val_loss = val["loss_total"]
            print(f"  -> New best val loss: {best_val_loss:.4f}")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_metrics": val, "args": vars(args),
            }, out_dir / "best_model.pt")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "step": step,
        }, out_dir / "latest_checkpoint.pt")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {out_dir}")

    # Full-sentence accuracy on test set
    test_s = sentences[n_train + n_val:]
    print(f"\n--- Full-sentence accuracy on {len(test_s)} test sentences ---")
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    sent_eval = evaluate_sentence_accuracy(model, test_s, tokenizer, device, args.max_length)
    for metric in ["pos", "uas", "las", "all"]:
        overall = sent_eval[metric]["correct"] / max(sent_eval[metric]["total"], 1)
        print(f"  {metric}: {overall:.4f} ({sent_eval[metric]['correct']}/{sent_eval[metric]['total']})")
        for lang in sorted(sent_eval[metric].get("per_lang", {})):
            d = sent_eval[metric]["per_lang"][lang]
            acc = d["correct"] / max(d["total"], 1)
            print(f"    {LANG_MAP.get(lang, lang)}: {acc:.4f}")
        if HAS_WANDB and args.wandb:
            wandb.log({f"test/sent_acc_{metric}": overall})
            for lang, d in sent_eval[metric].get("per_lang", {}).items():
                wandb.log({f"test/sent_acc_{metric}_{lang}": d["correct"] / max(d["total"], 1)})

    # Upload to HuggingFace Hub
    hf_username = os.getenv("HF_USERNAME")
    if hf_username:
        from huggingface_hub import HfApi
        hf_repo_name = f"{hf_username}/lexide-xlmr-nlp"
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
    p.add_argument("--output_dir", default="output/xlmr")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--backbone_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--arc_dim", type=int, default=256)
    p.add_argument("--rel_dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="lexide-pipeline")
    p.add_argument("--run_name", default="xlmr-biaffine")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ByT5-XXL fine-tuning for unified NLP pipeline with LoRA.

LoRA on ByT5-XXL encoder+decoder, full finetune on task heads.
Joint model for:
- Token segmentation (byte-level BIO tagging)
- POS tagging (joint with BIO)
- Sentence boundary detection
- Dependency parsing (biaffine)
- Lemmatization (ByT5 decoder with cross-attention)
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
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from stage1_train import (
    UPOS_TAGS, UPOS2ID,
    BIO_POS_LABELS, BIO_POS_LABEL2ID, BIO_POS_ID2LABEL, NUM_BIO_POS,
    DEP_RELS, DEP2ID, ID2DEP, NUM_DEP_RELS,
    LANG_MAP, LANG2ID, NUM_LANGS,
    BYT5_PAD, BYT5_EOS, BYT5_OFFSET,
    encode_lemma_to_bytes,
    Biaffine, MLP,
    load_all_sentences, ends_with_punct, group_into_paragraphs,
    collate_fn,
    compute_loss,
    evaluate,
)


# ============================================================
# Data (byte-level)
# ============================================================
def build_byte_labels(paragraph):
    """Build byte-level labels + dependency annotations for a paragraph."""
    all_bytes = []
    bio_pos_ids = []
    sent_start_ids = []
    dep_annotations = []

    for sent_idx, sent in enumerate(paragraph):
        tokens = sent["tokens"]
        sent_dep = {"token_spans": [], "heads": [], "dep_rels": [], "lemmas": [], "pos_ids": []}

        if sent_idx > 0:
            prev_tokens = paragraph[sent_idx - 1]["tokens"]
            if prev_tokens and not prev_tokens[-1].get("whitespace", ""):
                all_bytes.append(0x20)
                bio_pos_ids.append(BIO_POS_LABEL2ID["O"])
                sent_start_ids.append(0)

        for tok_idx, tok in enumerate(tokens):
            text = tok["text"]
            ws = tok.get("whitespace", "")
            pos = tok.get("pos", "X")

            token_start = len(all_bytes)
            text_bytes = text.encode("utf-8")
            for bi, b in enumerate(text_bytes):
                label = f"B-{pos}" if bi == 0 else f"I-{pos}"
                bio_pos_ids.append(BIO_POS_LABEL2ID.get(label, BIO_POS_LABEL2ID["O"]))
                sent_start_ids.append(1 if tok_idx == 0 and bi == 0 else 0)
                all_bytes.append(b)
            token_end = len(all_bytes)

            sent_dep["token_spans"].append((token_start, token_end))
            sent_dep["heads"].append(tok.get("head", 0))
            dep_rel = tok.get("dep", "dep")
            sent_dep["dep_rels"].append(DEP2ID.get(dep_rel, DEP2ID.get("dep", 0)))
            sent_dep["lemmas"].append(tok.get("lemma", text))
            sent_dep["pos_ids"].append(UPOS2ID.get(pos, UPOS2ID.get("X", 16)))

            for wc in ws.encode("utf-8"):
                all_bytes.append(wc)
                bio_pos_ids.append(BIO_POS_LABEL2ID["O"])
                sent_start_ids.append(0)

        dep_annotations.append(sent_dep)

    return all_bytes, bio_pos_ids, sent_start_ids, dep_annotations


class ByT5Dataset(Dataset):
    def __init__(self, paragraphs, max_length=2048):
        self.paragraphs = paragraphs
        self.max_length = max_length

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        paragraph = self.paragraphs[idx]
        raw_bytes, bio_pos_ids, sent_start_ids, dep_annotations = build_byte_labels(paragraph)

        max_bytes = self.max_length - 1  # Reserve 1 for EOS
        if len(raw_bytes) > max_bytes:
            raw_bytes = raw_bytes[:max_bytes]
            bio_pos_ids = bio_pos_ids[:max_bytes]
            sent_start_ids = sent_start_ids[:max_bytes]
            new_dep = []
            for sd in dep_annotations:
                spans, heads, rels, lemmas, pids = [], [], [], [], []
                for i, (s, e) in enumerate(sd["token_spans"]):
                    if e <= max_bytes:
                        spans.append((s, e))
                        heads.append(sd["heads"][i])
                        rels.append(sd["dep_rels"][i])
                        lemmas.append(sd["lemmas"][i])
                        pids.append(sd["pos_ids"][i])
                if spans:
                    new_dep.append({"token_spans": spans, "heads": heads, "dep_rels": rels,
                                    "lemmas": lemmas, "pos_ids": pids})
            dep_annotations = new_dep

        n_bytes = len(raw_bytes)

        # ByT5 input: byte_value + 3, then EOS, then pad
        input_ids = torch.full((self.max_length,), BYT5_PAD, dtype=torch.long)
        for i, b in enumerate(raw_bytes):
            input_ids[i] = b + BYT5_OFFSET
        input_ids[n_bytes] = BYT5_EOS

        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:n_bytes + 1] = 1  # Include EOS

        # Labels at byte positions (no CLS offset)
        bio_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        bio_labels[:n_bytes] = torch.tensor(bio_pos_ids, dtype=torch.long)
        sent_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        sent_labels[:n_bytes] = torch.tensor(sent_start_ids, dtype=torch.long)

        # Lemma byte IDs (spans already in byte coordinates, no offset needed)
        for sd in dep_annotations:
            sd["lemma_byte_ids"] = [encode_lemma_to_bytes(l) for l in sd["lemmas"]]

        lang_id = LANG2ID.get(paragraph[0].get("lang", "eng"), 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bio_pos_labels": bio_labels,
            "sent_labels": sent_labels,
            "dep_annotations": dep_annotations,
            "lang_id": lang_id,
        }


def decode_bio_to_tokens_bytes(text, bio_pred_ids, sent_pred_ids):
    """Convert byte-level BIO predictions to tokens and sentences."""
    text_bytes = text.encode("utf-8")
    sentences = []
    current_sent = []
    current_tok_bytes = []
    current_tok_pos = None

    for i, (byte_val, bio_id, sent_id) in enumerate(zip(text_bytes, bio_pred_ids, sent_pred_ids)):
        label = BIO_POS_ID2LABEL.get(bio_id, "O")

        if sent_id == 1 and i > 0:
            if current_tok_bytes:
                tok_text = bytes(current_tok_bytes).decode("utf-8", errors="replace")
                current_sent.append({"text": tok_text, "pos": current_tok_pos})
                current_tok_bytes = []
            if current_sent:
                sentences.append(current_sent)
                current_sent = []

        if label == "O":
            if current_tok_bytes:
                tok_text = bytes(current_tok_bytes).decode("utf-8", errors="replace")
                current_sent.append({"text": tok_text, "pos": current_tok_pos})
                current_tok_bytes = []
        elif label.startswith("B-"):
            if current_tok_bytes:
                tok_text = bytes(current_tok_bytes).decode("utf-8", errors="replace")
                current_sent.append({"text": tok_text, "pos": current_tok_pos})
            current_tok_bytes = [byte_val]
            current_tok_pos = label[2:]
        elif label.startswith("I-"):
            current_tok_bytes.append(byte_val)

    if current_tok_bytes:
        tok_text = bytes(current_tok_bytes).decode("utf-8", errors="replace")
        current_sent.append({"text": tok_text, "pos": current_tok_pos})
    if current_sent:
        sentences.append(current_sent)

    return sentences


# ============================================================
# Model
# ============================================================
class ByT5ForNLP(nn.Module):
    def __init__(self, num_bio_pos, num_dep_rels, num_langs=NUM_LANGS,
                 arc_dim=256, rel_dim=128, mlp_drop=0.33,
                 lora_r=64, lora_alpha=128, lora_dropout=0.05,
                 model_name="google/byt5-xxl"):
        super().__init__()

        # Load ByT5 and apply LoRA
        base_model = T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q", "k", "v", "o"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        self.byt5 = get_peft_model(base_model, lora_config)
        self.byt5.print_trainable_parameters()

        hidden = base_model.config.d_model
        self.hidden = hidden

        # Language embedding
        self.lang_emb = nn.Embedding(num_langs, hidden)

        # Byte-level heads
        self.bio_pos_head = nn.Linear(hidden, num_bio_pos)
        self.sent_head = nn.Linear(hidden, 2)

        # Dependency parsing
        self.root_emb = nn.Parameter(torch.randn(hidden) * 0.02)
        self.arc_dep_mlp = MLP(hidden, arc_dim, mlp_drop)
        self.arc_head_mlp = MLP(hidden, arc_dim, mlp_drop)
        self.rel_dep_mlp = MLP(hidden, rel_dim, mlp_drop)
        self.rel_head_mlp = MLP(hidden, rel_dim, mlp_drop)
        self.arc_biaffine = Biaffine(arc_dim, 1)
        self.rel_biaffine = Biaffine(rel_dim, num_dep_rels)

        # Lemma indicator/POS embeddings (no projection — shared d_model)
        self.lemma_indicator_emb = nn.Parameter(torch.randn(hidden) * 0.02)
        self.lemma_pos_emb = nn.Embedding(len(UPOS_TAGS), hidden)

    def encode(self, input_ids, attention_mask, lang_ids=None):
        """Run ByT5 encoder, return hidden states with language embedding."""
        encoder = self.byt5.get_encoder()
        h = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if lang_ids is not None:
            h = h + self.lang_emb(lang_ids).unsqueeze(1)
        return h

    def pool_tokens(self, byte_hidden, dep_info, device):
        if not dep_info:
            return None, None
        num_sents = len(dep_info)
        max_toks = max(len(d["token_spans"]) for d in dep_info)
        reps = byte_hidden.new_zeros(num_sents, max_toks + 1, self.hidden)
        mask = torch.zeros(num_sents, max_toks + 1, dtype=torch.bool, device=device)
        for si, d in enumerate(dep_info):
            reps[si, 0] = self.root_emb
            mask[si, 0] = True
            for ti, (s, e) in enumerate(d["token_spans"]):
                reps[si, ti + 1] = byte_hidden[d["batch_idx"], s:e].mean(0)
                mask[si, ti + 1] = True
        return reps, mask

    def forward_lemma(self, h, attention_mask, dep_info, device, max_tokens_per_sent=None):
        """Run ByT5 decoder for lemmatization with teacher forcing."""
        items = []
        for d in dep_info:
            if "lemma_byte_ids" not in d or not d["lemma_byte_ids"]:
                continue
            bi = d["batch_idx"]
            spans = d["token_spans"]
            lemma_bytes = d["lemma_byte_ids"]
            pos_ids = d["pos_ids"]
            n = len(spans)
            if n == 0:
                continue

            if max_tokens_per_sent and n > max_tokens_per_sent:
                indices = random.sample(range(n), max_tokens_per_sent)
            else:
                indices = range(n)

            base_h = h[bi]
            src_len = int(attention_mask[bi].sum().item())

            for ti in indices:
                s, e = spans[ti]
                h_mod = base_h[:src_len].clone()
                h_mod[s:e] = h_mod[s:e] + self.lemma_indicator_emb + self.lemma_pos_emb.weight[pos_ids[ti]]

                byte_ids = lemma_bytes[ti]
                dec_in = [BYT5_PAD] + byte_ids[:-1]
                items.append((h_mod, src_len, dec_in, byte_ids))

        if not items:
            return None, None

        N = len(items)
        max_src = max(it[1] for it in items)
        max_tgt = max(len(it[2]) for it in items)

        enc_states = h.new_zeros(N, max_src, self.hidden)
        enc_mask = h.new_zeros(N, max_src)
        dec_input = torch.full((N, max_tgt), BYT5_PAD, dtype=torch.long, device=device)
        labels = torch.full((N, max_tgt), -100, dtype=torch.long, device=device)

        for i, (h_mod, slen, dinp, labs) in enumerate(items):
            enc_states[i, :slen] = h_mod
            enc_mask[i, :slen] = 1.0
            dec_input[i, :len(dinp)] = torch.tensor(dinp, dtype=torch.long, device=device)
            labels[i, :len(labs)] = torch.tensor(labs, dtype=torch.long, device=device)

        encoder_outputs = BaseModelOutput(last_hidden_state=enc_states)
        outputs = self.byt5(
            decoder_input_ids=dec_input,
            encoder_outputs=encoder_outputs,
            attention_mask=enc_mask,
        )
        return outputs.logits, labels

    @torch.no_grad()
    def generate_lemmas(self, h, attention_mask, token_spans, pos_ids, max_len=32):
        """Greedy-decode lemmas from encoder hidden states (batch=1)."""
        device = h.device
        base_h = h[0]
        src_len = int(attention_mask[0].sum().item())

        N = len(token_spans)
        if N == 0:
            return []

        enc_states = base_h.new_zeros(N, src_len, self.hidden)
        enc_mask = base_h.new_ones(N, src_len)

        for i, ((s, e), pid) in enumerate(zip(token_spans, pos_ids)):
            h_mod = base_h[:src_len].clone()
            h_mod[s:e] = h_mod[s:e] + self.lemma_indicator_emb + self.lemma_pos_emb.weight[pid]
            enc_states[i] = h_mod

        dec_input = torch.full((N, 1), BYT5_PAD, dtype=torch.long, device=device)
        finished = torch.zeros(N, dtype=torch.bool, device=device)
        generated = [[] for _ in range(N)]

        for _ in range(max_len):
            encoder_outputs = BaseModelOutput(last_hidden_state=enc_states)
            outputs = self.byt5(
                decoder_input_ids=dec_input,
                encoder_outputs=encoder_outputs,
                attention_mask=enc_mask,
            )
            logits = outputs.logits[:, -1, :]
            next_tok = logits.argmax(-1)

            for i in range(N):
                if not finished[i]:
                    t = next_tok[i].item()
                    if t == BYT5_EOS:
                        finished[i] = True
                    else:
                        generated[i].append(t)

            if finished.all():
                break
            dec_input = torch.cat([dec_input, next_tok.unsqueeze(1)], dim=1)

        lemmas = []
        for byte_ids in generated:
            raw = bytes(max(0, b - BYT5_OFFSET) for b in byte_ids)
            try:
                lemmas.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                lemmas.append("")
        return lemmas

    def forward(self, input_ids, attention_mask, dep_info=None, lang_ids=None,
                max_lemma_tokens_per_sent=None):
        h = self.encode(input_ids, attention_mask, lang_ids)
        bio_logits = self.bio_pos_head(h)
        sent_logits = self.sent_head(h)

        arc_scores, rel_scores = None, None
        if dep_info:
            tok_reps, tok_mask = self.pool_tokens(h, dep_info, h.device)
            if tok_reps is not None:
                dep_r = tok_reps[:, 1:]
                head_r = tok_reps
                arc_scores = self.arc_biaffine(self.arc_dep_mlp(dep_r), self.arc_head_mlp(head_r))
                rel_scores = self.rel_biaffine(self.rel_dep_mlp(dep_r), self.rel_head_mlp(head_r))

        lemma_logits, lemma_labels = None, None
        if dep_info and any("lemma_byte_ids" in d and d["lemma_byte_ids"] for d in dep_info):
            lemma_logits, lemma_labels = self.forward_lemma(
                h, attention_mask, dep_info, h.device,
                max_tokens_per_sent=max_lemma_tokens_per_sent,
            )

        return bio_logits, sent_logits, arc_scores, rel_scores, lemma_logits, lemma_labels


# ============================================================
# Training
# ============================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if HAS_WANDB and args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    print("Loading data...")
    sentences = load_all_sentences(args.data_dir)
    paragraphs = group_into_paragraphs(sentences, seed=args.seed)
    print(f"{len(paragraphs)} paragraphs")

    rng = random.Random(args.seed)
    rng.shuffle(paragraphs)
    n_train = int(0.9 * len(paragraphs))
    n_val = int(0.05 * len(paragraphs))
    train_p, val_p = paragraphs[:n_train], paragraphs[n_train:n_train + n_val]
    print(f"Train: {len(train_p)}, Val: {len(val_p)}")

    train_ds = ByT5Dataset(train_p, args.max_length)
    val_ds = ByT5Dataset(val_p, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    print("Loading model...")
    model = ByT5ForNLP(
        num_bio_pos=NUM_BIO_POS, num_dep_rels=NUM_DEP_RELS,
        arc_dim=args.arc_dim, rel_dim=args.rel_dim,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        model_name=args.model_name,
    ).to(device)

    # LR groups: LoRA params (2e-5), heads + new embeddings (1e-3)
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]
    print(f"LoRA params: {sum(p.numel() for p in lora_params):,}")
    print(f"Head params: {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lora_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.num_epochs // args.grad_accum
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "label_config.json", "w") as f:
        json.dump({
            "bio_pos_labels": BIO_POS_LABELS,
            "dep_rels": DEP_RELS,
            "upos_tags": UPOS_TAGS,
        }, f, indent=2)

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
            amask = batch["attention_mask"].to(device)
            bio_lab = batch["bio_pos_labels"].to(device)
            sent_lab = batch["sent_labels"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            dep_info = batch["dep_info"]

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    bio_log, sent_log, arc_sc, rel_sc, lem_log, lem_lab = model(
                        ids, amask, dep_info, lang_ids,
                        max_lemma_tokens_per_sent=args.lemma_tokens_per_sent)
                    losses = compute_loss(bio_log, sent_log, arc_sc, rel_sc, bio_lab, sent_lab, dep_info, device,
                                          lemma_logits=lem_log, lemma_labels=lem_lab,
                                          lemma_loss_weight=args.lemma_loss_weight,
                                          )
                scaler.scale(losses["total"] / args.grad_accum).backward()
            else:
                bio_log, sent_log, arc_sc, rel_sc, lem_log, lem_lab = model(
                    ids, amask, dep_info, lang_ids,
                    max_lemma_tokens_per_sent=args.lemma_tokens_per_sent)
                losses = compute_loss(bio_log, sent_log, arc_sc, rel_sc, bio_lab, sent_lab, dep_info, device,
                                      lemma_logits=lem_log, lemma_labels=lem_lab,
                                      lemma_loss_weight=args.lemma_loss_weight,
                                      bio_crf=model.bio_crf)
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
                             bio=f"{losses['bio_pos'].item():.3f}",
                             arc=f"{losses['arc'].item():.3f}",
                             lem=f"{losses['lemma'].item():.3f}")

            if HAS_WANDB and args.wandb and step > 0 and step % 10 == 0:
                wandb.log({f"train/{k}": v.item() for k, v in losses.items()}, step=step)
                wandb.log({"train/lr": scheduler.get_last_lr()[0]}, step=step)

        avg = {k: v / ep_steps for k, v in ep_loss.items()}
        print(f"\nEpoch {epoch + 1} train: " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))

        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1} val: " + ", ".join(f"{k}={v:.4f}" for k, v in val.items()))

        if HAS_WANDB and args.wandb:
            wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)

        # Save only trainable params
        trainable_state = {n: p.data for n, p in model.named_parameters() if p.requires_grad}

        if val["loss_total"] < best_val_loss:
            best_val_loss = val["loss_total"]
            print(f"  -> New best val loss: {best_val_loss:.4f}")
            torch.save({
                "model_state_dict": trainable_state,
                "epoch": epoch, "val_metrics": val, "args": vars(args),
            }, out_dir / "best_model.pt")

        torch.save({
            "model_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "step": step,
        }, out_dir / "latest_checkpoint.pt")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {out_dir}")

    # Upload to HuggingFace Hub
    hf_username = os.getenv("HF_USERNAME")
    if hf_username:
        from huggingface_hub import HfApi
        hf_repo_name = f"{hf_username}/lexide-byt5-xxl-nlp"
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
            print("Model saved locally only.")

    if HAS_WANDB and args.wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="output/byt5_xxl")
    p.add_argument("--model_name", default="google/byt5-xxl")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lora_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--arc_dim", type=int, default=256)
    p.add_argument("--rel_dim", type=int, default=128)
    p.add_argument("--lemma_tokens_per_sent", type=int, default=4)
    p.add_argument("--lemma_loss_weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="lexide-pipeline")
    p.add_argument("--run_name", default="byt5-xxl-lora")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()

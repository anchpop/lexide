#!/usr/bin/env python3
"""Byte-level encoder-decoder lemmatizer (~20M params).

Per-token lemmatization using:
- Surface form bytes as encoder input
- POS tag embedding
- Frozen XLM-R contextual embedding for sentence context
- Autoregressive byte-level decoder for lemma generation
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

LANG_MAP = {
    "eng": "English", "deu": "German", "fra": "French", "spa": "Spanish",
    "kor": "Korean", "por": "Portuguese", "ita": "Italian", "rus": "Russian",
}
LANG2ID = {lang: i for i, lang in enumerate(LANG_MAP)}
NUM_LANGS = len(LANG2ID)

# Byte vocab
PAD = 0
BOS = 1
EOS = 2
BYTE_OFFSET = 3  # byte values 0-255 → token IDs 3-258
VOCAB_SIZE = 259


def bytes_to_ids(s):
    return [b + BYTE_OFFSET for b in s.encode("utf-8")]


def ids_to_str(ids):
    raw = bytes(max(0, b - BYTE_OFFSET) for b in ids)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""


# ============================================================
# Data
# ============================================================
def load_all_sentences(data_dir, include_generated=True):
    sentences = []
    data_dir = Path(data_dir)
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

    # Filter bad sentences
    clean = []
    filtered = 0
    for sent in sentences:
        tokens = sent["tokens"]
        n = len(tokens)
        if any(t.get("text", "") == "" for t in tokens):
            filtered += 1
            continue
        if any(t.get("head", 0) > n or t.get("head", 0) < 0 for t in tokens):
            filtered += 1
            continue
        clean.append(sent)
    if filtered:
        print(f"Filtered {filtered} sentences with bad data (empty tokens or OOB heads)")

    print(f"Total: {len(clean)} sentences")
    return clean


class LemmaDataset(Dataset):
    """Each item is a sentence. XLM-R runs on the sentence, then each token
    becomes a lemmatization example."""
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

        enc = self.tokenizer(
            words, is_split_into_words=True,
            max_length=self.max_length, truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        wids = enc.word_ids()

        word_starts = []
        prev_wid = None
        for i, wid in enumerate(wids):
            if wid is not None and wid != prev_wid:
                word_starts.append(i)
            prev_wid = wid

        n_words = len(word_starts)
        tokens = tokens[:n_words]

        surface_byte_ids = []
        pos_ids = []
        lemma_byte_ids = []
        for tok in tokens:
            surface_byte_ids.append(bytes_to_ids(tok["text"]))
            pos_ids.append(UPOS2ID.get(tok.get("pos", "X"), UPOS2ID["X"]))
            lemma = tok.get("lemma", tok["text"])
            lemma_byte_ids.append(bytes_to_ids(lemma) + [EOS])

        lang_id = LANG2ID.get(sent.get("lang", "eng"), 0)

        return {
            "xlmr_input_ids": input_ids,
            "xlmr_attention_mask": attention_mask,
            "word_starts": word_starts,
            "surface_byte_ids": surface_byte_ids,
            "pos_ids": pos_ids,
            "lemma_byte_ids": lemma_byte_ids,
            "n_words": n_words,
            "lang_id": lang_id,
        }


def collate_fn(batch):
    # XLM-R inputs: pad to max subword length
    max_sub = max(b["xlmr_input_ids"].shape[0] for b in batch)
    bs = len(batch)

    xlmr_ids = torch.zeros(bs, max_sub, dtype=torch.long)
    xlmr_mask = torch.zeros(bs, max_sub, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["xlmr_input_ids"].shape[0]
        xlmr_ids[i, :n] = b["xlmr_input_ids"]
        xlmr_mask[i, :n] = b["xlmr_attention_mask"]

    # Flatten all tokens across the batch for the lemmatizer
    all_surface = []
    all_lemma = []
    all_pos = []
    all_lang = []
    # Track which sentence/word each token came from (for gathering XLM-R embeddings)
    token_sent_idx = []
    token_word_start = []

    for i, b in enumerate(batch):
        for j in range(b["n_words"]):
            all_surface.append(b["surface_byte_ids"][j])
            all_lemma.append(b["lemma_byte_ids"][j])
            all_pos.append(b["pos_ids"][j])
            all_lang.append(b["lang_id"])
            token_sent_idx.append(i)
            token_word_start.append(b["word_starts"][j])

    # Pad surface bytes
    n_tokens = len(all_surface)
    max_surf = max(len(s) for s in all_surface) if all_surface else 1
    max_lem = max(len(l) for l in all_lemma) if all_lemma else 1

    surface_ids = torch.full((n_tokens, max_surf), PAD, dtype=torch.long)
    surface_mask = torch.zeros(n_tokens, max_surf, dtype=torch.bool)
    lemma_ids = torch.full((n_tokens, max_lem), PAD, dtype=torch.long)
    lemma_mask = torch.zeros(n_tokens, max_lem, dtype=torch.bool)

    for i, (s, l) in enumerate(zip(all_surface, all_lemma)):
        surface_ids[i, :len(s)] = torch.tensor(s)
        surface_mask[i, :len(s)] = True
        lemma_ids[i, :len(l)] = torch.tensor(l)
        lemma_mask[i, :len(l)] = True

    return {
        "xlmr_ids": xlmr_ids,
        "xlmr_mask": xlmr_mask,
        "surface_ids": surface_ids,
        "surface_mask": surface_mask,
        "lemma_ids": lemma_ids,
        "lemma_mask": lemma_mask,
        "pos_ids": torch.tensor(all_pos, dtype=torch.long),
        "lang_ids": torch.tensor(all_lang, dtype=torch.long),
        "token_sent_idx": torch.tensor(token_sent_idx, dtype=torch.long),
        "token_word_start": torch.tensor(token_word_start, dtype=torch.long),
    }


# ============================================================
# Model
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LemmaModel(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_encoder_layers=4,
                 num_decoder_layers=4, d_ff=1024, dropout=0.1,
                 num_pos=NUM_POS, num_langs=NUM_LANGS, xlmr_dim=1024):
        super().__init__()
        self.d_model = d_model

        # Encoder embeddings
        self.byte_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(num_pos, d_model)
        self.lang_emb = nn.Embedding(num_langs, d_model)
        self.xlmr_proj = nn.Linear(xlmr_dim, d_model)
        self.enc_pos_enc = PositionalEncoding(d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder embeddings
        self.dec_byte_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.dec_pos_enc = PositionalEncoding(d_model)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output
        self.output_proj = nn.Linear(d_model, VOCAB_SIZE)

    def encode(self, surface_ids, surface_mask, pos_ids, lang_ids, xlmr_emb):
        """Encode surface form bytes with POS, language, and XLM-R context."""
        x = self.byte_emb(surface_ids)  # (N, src_len, d_model)
        x = x + self.pos_emb(pos_ids).unsqueeze(1)  # broadcast POS over all bytes
        x = x + self.lang_emb(lang_ids).unsqueeze(1)
        x = x + self.xlmr_proj(xlmr_emb).unsqueeze(1)  # broadcast XLM-R emb
        x = self.enc_pos_enc(x)
        src_key_padding_mask = ~surface_mask
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode_train(self, memory, surface_mask, lemma_ids):
        """Teacher-forced decoding. lemma_ids includes EOS."""
        # Shift right: BOS + lemma[:-1]
        tgt_in = torch.full_like(lemma_ids, PAD)
        tgt_in[:, 0] = BOS
        tgt_in[:, 1:] = lemma_ids[:, :-1]

        tgt_emb = self.dec_byte_emb(tgt_in)
        tgt_emb = self.dec_pos_enc(tgt_emb)

        tgt_len = tgt_in.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt_in.device)
        memory_key_padding_mask = ~surface_mask

        out = self.decoder(
            tgt_emb, memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)  # (N, tgt_len, vocab)

    def forward(self, surface_ids, surface_mask, pos_ids, lang_ids, xlmr_emb, lemma_ids):
        """Training forward: returns logits and labels for cross-entropy."""
        memory = self.encode(surface_ids, surface_mask, pos_ids, lang_ids, xlmr_emb)
        logits = self.decode_train(memory, surface_mask, lemma_ids)
        return logits

    @torch.no_grad()
    def generate(self, surface_ids, surface_mask, pos_ids, lang_ids, xlmr_emb, max_len=64):
        """Greedy autoregressive decoding."""
        memory = self.encode(surface_ids, surface_mask, pos_ids, lang_ids, xlmr_emb)
        N = surface_ids.shape[0]
        device = surface_ids.device

        generated = torch.full((N, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(N, dtype=torch.bool, device=device)
        results = [[] for _ in range(N)]

        for _ in range(max_len):
            tgt_emb = self.dec_byte_emb(generated)
            tgt_emb = self.dec_pos_enc(tgt_emb)
            tgt_len = generated.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            memory_key_padding_mask = ~surface_mask

            out = self.decoder(
                tgt_emb, memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            logits = self.output_proj(out[:, -1, :])
            next_tok = logits.argmax(-1)

            for i in range(N):
                if not finished[i]:
                    t = next_tok[i].item()
                    if t == EOS:
                        finished[i] = True
                    else:
                        results[i].append(t)

            if finished.all():
                break

            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)

        return [ids_to_str(r) for r in results]


# ============================================================
# Training
# ============================================================
@torch.no_grad()
def evaluate(model, xlmr, loader, device):
    model.eval()
    stats = defaultdict(float)
    n_tokens = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        xlmr_ids = batch["xlmr_ids"].to(device)
        xlmr_mask = batch["xlmr_mask"].to(device)
        surface_ids = batch["surface_ids"].to(device)
        surface_mask = batch["surface_mask"].to(device)
        lemma_ids = batch["lemma_ids"].to(device)
        lemma_mask = batch["lemma_mask"].to(device)
        pos_ids = batch["pos_ids"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        sent_idx = batch["token_sent_idx"].to(device)
        word_start = batch["token_word_start"].to(device)

        # Get XLM-R embeddings
        xlmr_out = xlmr(input_ids=xlmr_ids, attention_mask=xlmr_mask).last_hidden_state
        xlmr_emb = xlmr_out[sent_idx, word_start]  # (n_tokens, xlmr_dim)

        logits = model(surface_ids, surface_mask, pos_ids, lang_ids, xlmr_emb, lemma_ids)

        # Loss on valid positions
        valid = lemma_mask
        loss = F.cross_entropy(logits[valid], lemma_ids[valid], ignore_index=PAD)
        stats["loss"] += loss.item() * valid.sum().item()

        # Token-level accuracy (exact match)
        preds = logits.argmax(-1)
        for i in range(surface_ids.shape[0]):
            n = lemma_mask[i].sum().item()
            pred_seq = preds[i, :n].cpu().tolist()
            gold_seq = lemma_ids[i, :n].cpu().tolist()
            stats["total"] += 1
            if pred_seq == gold_seq:
                stats["exact_match"] += 1

        n_tokens += valid.sum().item()

    return {
        "loss": stats["loss"] / max(n_tokens, 1),
        "exact_match": stats["exact_match"] / max(stats["total"], 1),
    }


@torch.no_grad()
def evaluate_sentence_accuracy(model, xlmr, sentences, tokenizer, device, max_length=512):
    """Full-sentence accuracy: % of sentences where every lemma is correct."""
    model.eval()
    per_lang = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total = 0

    for sent in tqdm(sentences, desc="SentAcc"):
        tokens = sent["tokens"]
        lang_code = sent.get("lang", "eng")
        lang_id = LANG2ID.get(lang_code, 0)
        words = [t["text"] for t in tokens]
        if not words:
            continue

        # XLM-R forward
        enc = tokenizer(words, is_split_into_words=True, max_length=max_length,
                        truncation=True, return_tensors="pt")
        xlmr_ids = enc["input_ids"].to(device)
        xlmr_mask = enc["attention_mask"].to(device)
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

        xlmr_out = xlmr(input_ids=xlmr_ids, attention_mask=xlmr_mask).last_hidden_state
        ws_tensor = torch.tensor(word_starts, device=device)
        xlmr_emb = xlmr_out[0, ws_tensor]  # (n_words, xlmr_dim)

        # Prepare surface bytes
        max_surf = max(len(t["text"].encode("utf-8")) for t in tokens)
        surface_ids = torch.full((n_words, max_surf), PAD, dtype=torch.long, device=device)
        surface_mask = torch.zeros(n_words, max_surf, dtype=torch.bool, device=device)
        pos_tensor = torch.tensor([UPOS2ID.get(t.get("pos", "X"), UPOS2ID["X"]) for t in tokens],
                                  dtype=torch.long, device=device)
        lang_tensor = torch.full((n_words,), lang_id, dtype=torch.long, device=device)

        for i, tok in enumerate(tokens):
            sb = bytes_to_ids(tok["text"])
            surface_ids[i, :len(sb)] = torch.tensor(sb, device=device)
            surface_mask[i, :len(sb)] = True

        pred_lemmas = model.generate(surface_ids, surface_mask, pos_tensor, lang_tensor, xlmr_emb)
        gold_lemmas = [t.get("lemma", t["text"]) for t in tokens]

        all_correct = all(p == g for p, g in zip(pred_lemmas, gold_lemmas))
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
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    # Load frozen XLM-R for contextual embeddings
    print("Loading XLM-R (frozen)...")
    xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    if args.xlmr_checkpoint:
        print(f"  Loading finetuned XLM-R from {args.xlmr_checkpoint}")
        from xlmr_train import XLMRForNLP
        ckpt = torch.load(args.xlmr_checkpoint, map_location=device, weights_only=False)
        xlmr_model = XLMRForNLP()
        xlmr_model.load_state_dict(ckpt["model_state_dict"])
        xlmr = xlmr_model.xlmr.to(device)
    else:
        xlmr = XLMRobertaModel.from_pretrained("xlm-roberta-large").to(device)
    xlmr.eval()
    for p in xlmr.parameters():
        p.requires_grad = False

    train_ds = LemmaDataset(train_s, xlmr_tokenizer, args.max_length)
    val_ds = LemmaDataset(val_s, xlmr_tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    model = LemmaModel(
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_enc_layers,
        num_decoder_layers=args.num_dec_layers,
        d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    print(f"Lemma model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    step = 0

    for epoch in range(args.num_epochs):
        model.train()
        ep_loss = 0.0
        ep_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in pbar:
            xlmr_ids = batch["xlmr_ids"].to(device)
            xlmr_mask = batch["xlmr_mask"].to(device)
            surface_ids = batch["surface_ids"].to(device)
            surface_mask = batch["surface_mask"].to(device)
            lemma_ids = batch["lemma_ids"].to(device)
            lemma_mask = batch["lemma_mask"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            sent_idx = batch["token_sent_idx"].to(device)
            word_start = batch["token_word_start"].to(device)

            # Get XLM-R embeddings (frozen)
            with torch.no_grad():
                xlmr_out = xlmr(input_ids=xlmr_ids, attention_mask=xlmr_mask).last_hidden_state
            xlmr_emb = xlmr_out[sent_idx, word_start]  # (n_tokens, xlmr_dim)

            logits = model(surface_ids, surface_mask, pos_ids, lang_ids, xlmr_emb, lemma_ids)

            valid = lemma_mask
            loss = F.cross_entropy(logits[valid], lemma_ids[valid], ignore_index=PAD)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            n = valid.sum().item()
            ep_loss += loss.item() * n
            ep_tokens += n

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if HAS_WANDB and args.wandb and step % 50 == 0:
                wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=step)

        train_loss = ep_loss / max(ep_tokens, 1)
        print(f"\nEpoch {epoch + 1} train: loss={train_loss:.4f}")

        val = evaluate(model, xlmr, val_loader, device)
        print(f"Epoch {epoch + 1} val: " + ", ".join(f"{k}={v:.4f}" for k, v in val.items()))

        if HAS_WANDB and args.wandb:
            wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            print(f"  -> New best val loss: {best_val_loss:.4f}")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_metrics": val, "args": vars(args),
            }, out_dir / "best_model.pt")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {out_dir}")

    # Full-sentence accuracy on test set
    test_s = sentences[n_train + n_val:]
    print(f"\n--- Full-sentence lemma accuracy on {len(test_s)} test sentences ---")
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    sent_eval = evaluate_sentence_accuracy(model, xlmr, test_s, xlmr_tokenizer, device, args.max_length)
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
        hf_repo_name = f"{hf_username}/lexide-lemmatizer"
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
    p.add_argument("--output_dir", default="output/lemma")
    p.add_argument("--xlmr_checkpoint", default=None, help="Path to finetuned XLM-R checkpoint (omit for pretrained)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_enc_layers", type=int, default=4)
    p.add_argument("--num_dec_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="lexide-pipeline")
    p.add_argument("--run_name", default="lemmatizer")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()

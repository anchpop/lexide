#!/usr/bin/env python3
"""Stage 1: CANINE-S fine-tuning for character-level NLP.

Joint model for:
- Token segmentation (BIO tagging)
- POS tagging (joint with BIO)
- Sentence boundary detection
- Dependency parsing (biaffine)
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
from transformers import CanineModel, CanineTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
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

BIO_POS_LABELS = (
    ["O"]
    + [f"B-{pos}" for pos in UPOS_TAGS]
    + [f"I-{pos}" for pos in UPOS_TAGS]
)
BIO_POS_LABEL2ID = {l: i for i, l in enumerate(BIO_POS_LABELS)}
BIO_POS_ID2LABEL = {i: l for l, i in BIO_POS_LABEL2ID.items()}
NUM_BIO_POS = len(BIO_POS_LABELS)

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
    "kor": "Korean", "por": "Portuguese", "ita": "Italian",
}
LANG2ID = {lang: i for i, lang in enumerate(LANG_MAP)}
NUM_LANGS = len(LANG2ID)
UPOS2ID = {t: i for i, t in enumerate(UPOS_TAGS)}

# ByT5 byte encoding constants
BYT5_PAD = 0
BYT5_EOS = 1
BYT5_OFFSET = 3  # raw bytes are offset by 3 in ByT5 vocab


def encode_lemma_to_bytes(lemma_str, max_len=32):
    """Convert lemma string to ByT5 byte token IDs."""
    byte_ids = [b + BYT5_OFFSET for b in lemma_str.encode('utf-8')][:max_len - 1]
    byte_ids.append(BYT5_EOS)
    return byte_ids


# ============================================================
# Data
# ============================================================
def load_all_sentences(data_dir):
    sentences = []
    data_dir = Path(data_dir)
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
    print(f"Total: {len(sentences)} sentences")
    return sentences


def ends_with_punct(text):
    text = text.strip()
    return bool(text) and text[-1] in ".!?:;。！？"


def group_into_paragraphs(sentences, seed=42):
    rng = random.Random(seed)
    rng.shuffle(sentences)
    paragraphs = []
    i = 0
    while i < len(sentences):
        group = [sentences[i]]
        i += 1
        if not ends_with_punct(group[0]["sentence"]):
            paragraphs.append(group)
            continue
        max_add = 0 if rng.random() < 0.1 else rng.randint(1, 4)
        added = 0
        while added < max_add and i < len(sentences):
            group.append(sentences[i])
            i += 1
            added += 1
            if not ends_with_punct(group[-1]["sentence"]):
                break
        paragraphs.append(group)
    return paragraphs


def build_char_labels(paragraph):
    """Build character-level labels + dependency annotations for a paragraph."""
    chars = []
    bio_pos_ids = []
    sent_start_ids = []
    dep_annotations = []

    for sent_idx, sent in enumerate(paragraph):
        tokens = sent["tokens"]
        sent_dep = {"token_spans": [], "heads": [], "dep_rels": [], "lemmas": [], "pos_ids": []}

        # Space between sentences if previous sentence didn't end with whitespace
        if sent_idx > 0:
            prev_tokens = paragraph[sent_idx - 1]["tokens"]
            if prev_tokens and not prev_tokens[-1].get("whitespace", ""):
                chars.append(" ")
                bio_pos_ids.append(BIO_POS_LABEL2ID["O"])
                sent_start_ids.append(0)

        for tok_idx, tok in enumerate(tokens):
            text = tok["text"]
            ws = tok.get("whitespace", "")
            pos = tok.get("pos", "X")

            token_start = len(chars)
            for ci, ch in enumerate(text):
                label = f"B-{pos}" if ci == 0 else f"I-{pos}"
                bio_pos_ids.append(BIO_POS_LABEL2ID.get(label, BIO_POS_LABEL2ID["O"]))
                sent_start_ids.append(1 if tok_idx == 0 and ci == 0 else 0)
                chars.append(ch)
            token_end = len(chars)

            sent_dep["token_spans"].append((token_start, token_end))
            sent_dep["heads"].append(tok.get("head", 0))
            dep_rel = tok.get("dep", "dep")
            sent_dep["dep_rels"].append(DEP2ID.get(dep_rel, DEP2ID.get("dep", 0)))
            sent_dep["lemmas"].append(tok.get("lemma", text))
            sent_dep["pos_ids"].append(UPOS2ID.get(pos, UPOS2ID.get("X", 16)))

            for wc in ws:
                chars.append(wc)
                bio_pos_ids.append(BIO_POS_LABEL2ID["O"])
                sent_start_ids.append(0)

        dep_annotations.append(sent_dep)

    return "".join(chars), bio_pos_ids, sent_start_ids, dep_annotations


class Stage1Dataset(Dataset):
    def __init__(self, paragraphs, tokenizer, max_length=2048):
        self.paragraphs = paragraphs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        paragraph = self.paragraphs[idx]
        text, bio_pos_ids, sent_start_ids, dep_annotations = build_char_labels(paragraph)

        max_chars = self.max_length - 2  # CLS + SEP
        if len(text) > max_chars:
            text = text[:max_chars]
            bio_pos_ids = bio_pos_ids[:max_chars]
            sent_start_ids = sent_start_ids[:max_chars]
            new_dep = []
            for sd in dep_annotations:
                spans, heads, rels, lemmas, pids = [], [], [], [], []
                for i, (s, e) in enumerate(sd["token_spans"]):
                    if e <= max_chars:
                        spans.append((s, e))
                        heads.append(sd["heads"][i])
                        rels.append(sd["dep_rels"][i])
                        lemmas.append(sd["lemmas"][i])
                        pids.append(sd["pos_ids"][i])
                if spans:
                    new_dep.append({"token_spans": spans, "heads": heads, "dep_rels": rels,
                                    "lemmas": lemmas, "pos_ids": pids})
            dep_annotations = new_dep

        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        n_chars = len(bio_pos_ids)
        bio_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        bio_labels[1:n_chars + 1] = torch.tensor(bio_pos_ids, dtype=torch.long)
        sent_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        sent_labels[1:n_chars + 1] = torch.tensor(sent_start_ids, dtype=torch.long)

        # Convert lemmas to byte IDs and offset spans by +1 for CLS token
        for sd in dep_annotations:
            sd["lemma_byte_ids"] = [encode_lemma_to_bytes(l) for l in sd["lemmas"]]
            sd["token_spans"] = [(s + 1, e + 1) for s, e in sd["token_spans"]]

        # Language ID from first sentence in paragraph
        lang_id = LANG2ID.get(paragraph[0].get("lang", "eng"), 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bio_pos_labels": bio_labels,
            "sent_labels": sent_labels,
            "dep_annotations": dep_annotations,
            "lang_id": lang_id,
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    bio_pos_labels = torch.stack([b["bio_pos_labels"] for b in batch])
    sent_labels = torch.stack([b["sent_labels"] for b in batch])
    lang_ids = torch.tensor([b["lang_id"] for b in batch], dtype=torch.long)

    dep_info = []
    for bi, b in enumerate(batch):
        for sd in b["dep_annotations"]:
            dep_info.append({
                "batch_idx": bi,
                "token_spans": sd["token_spans"],
                "heads": sd["heads"],
                "dep_rels": sd["dep_rels"],
                "lemma_byte_ids": sd.get("lemma_byte_ids", []),
                "pos_ids": sd.get("pos_ids", []),
            })

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bio_pos_labels": bio_pos_labels,
        "sent_labels": sent_labels,
        "lang_ids": lang_ids,
        "dep_info": dep_info,
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
            return s.squeeze(1)  # (B, n, m)
        return s.permute(0, 2, 3, 1)  # (B, n, m, out)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.33):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CanineForNLP(nn.Module):
    def __init__(self, num_bio_pos, num_dep_rels, num_langs=NUM_LANGS, hidden=768,
                 arc_dim=256, rel_dim=128, mlp_drop=0.33):
        super().__init__()
        self.canine = CanineModel.from_pretrained("google/canine-s")
        self.hidden = hidden

        self.lang_emb = nn.Embedding(num_langs, hidden)

        self.bio_pos_head = nn.Linear(hidden, num_bio_pos)
        self.sent_head = nn.Linear(hidden, 2)

        self.root_emb = nn.Parameter(torch.randn(hidden) * 0.02)
        self.arc_dep_mlp = MLP(hidden, arc_dim, mlp_drop)
        self.arc_head_mlp = MLP(hidden, arc_dim, mlp_drop)
        self.rel_dep_mlp = MLP(hidden, rel_dim, mlp_drop)
        self.rel_head_mlp = MLP(hidden, rel_dim, mlp_drop)
        self.arc_biaffine = Biaffine(arc_dim, 1)
        self.rel_biaffine = Biaffine(rel_dim, num_dep_rels)

        # Lemma decoder (ByT5-small pretrained decoder)
        byt5 = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
        self.lemma_decoder = byt5.decoder
        self.lemma_lm_head = byt5.lm_head
        self.byt5_d_model = byt5.config.d_model  # 1472
        del byt5.encoder

        self.enc_to_dec_proj = nn.Linear(hidden, self.byt5_d_model)
        self.lemma_indicator_emb = nn.Parameter(torch.randn(self.byt5_d_model) * 0.02)
        self.lemma_pos_emb = nn.Embedding(len(UPOS_TAGS), self.byt5_d_model)

    def pool_tokens(self, char_hidden, dep_info, device):
        if not dep_info:
            return None, None
        num_sents = len(dep_info)
        max_toks = max(len(d["token_spans"]) for d in dep_info)
        reps = char_hidden.new_zeros(num_sents, max_toks + 1, self.hidden)
        mask = torch.zeros(num_sents, max_toks + 1, dtype=torch.bool, device=device)
        for si, d in enumerate(dep_info):
            reps[si, 0] = self.root_emb
            mask[si, 0] = True
            for ti, (s, e) in enumerate(d["token_spans"]):
                reps[si, ti + 1] = char_hidden[d["batch_idx"], s:e].mean(0)
                mask[si, ti + 1] = True
        return reps, mask

    def encode(self, input_ids, attention_mask, lang_ids=None):
        """Run CANINE encoder and return hidden states with language embedding."""
        h = self.canine(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if lang_ids is not None:
            h = h + self.lang_emb(lang_ids).unsqueeze(1)
        return h

    def forward_lemma(self, h, attention_mask, dep_info, device, max_tokens_per_sent=None):
        """Run ByT5 decoder for lemmatization with teacher forcing.

        Returns (logits, labels) or (None, None) if no lemma data.
        """
        h_proj = self.enc_to_dec_proj(h)

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

            base_h = h_proj[bi]
            src_len = int(attention_mask[bi].sum().item())

            for ti in indices:
                s, e = spans[ti]
                h_mod = base_h[:src_len].clone()
                h_mod[s:e] = h_mod[s:e] + self.lemma_indicator_emb + self.lemma_pos_emb.weight[pos_ids[ti]]

                byte_ids = lemma_bytes[ti]
                dec_in = [BYT5_PAD] + byte_ids[:-1]  # shift right
                items.append((h_mod, src_len, dec_in, byte_ids))

        if not items:
            return None, None

        N = len(items)
        max_src = max(it[1] for it in items)
        max_tgt = max(len(it[2]) for it in items)

        enc_states = h.new_zeros(N, max_src, self.byt5_d_model)
        enc_mask = h.new_zeros(N, max_src)
        dec_input = torch.full((N, max_tgt), BYT5_PAD, dtype=torch.long, device=device)
        labels = torch.full((N, max_tgt), -100, dtype=torch.long, device=device)

        for i, (h_mod, slen, dinp, labs) in enumerate(items):
            enc_states[i, :slen] = h_mod
            enc_mask[i, :slen] = 1.0
            dec_input[i, :len(dinp)] = torch.tensor(dinp, dtype=torch.long, device=device)
            labels[i, :len(labs)] = torch.tensor(labs, dtype=torch.long, device=device)

        decoder_out = self.lemma_decoder(
            input_ids=dec_input,
            encoder_hidden_states=enc_states,
            encoder_attention_mask=enc_mask,
        )
        logits = self.lemma_lm_head(decoder_out.last_hidden_state)
        return logits, labels

    @torch.no_grad()
    def generate_lemmas(self, h, attention_mask, token_spans, pos_ids, max_len=32):
        """Greedy-decode lemmas given pre-computed CANINE hidden states (batch size 1)."""
        device = h.device
        h_proj = self.enc_to_dec_proj(h)
        base_h = h_proj[0]
        src_len = int(attention_mask[0].sum().item())

        N = len(token_spans)
        if N == 0:
            return []

        enc_states = base_h.new_zeros(N, src_len, self.byt5_d_model)
        enc_mask = base_h.new_ones(N, src_len)

        for i, ((s, e), pid) in enumerate(zip(token_spans, pos_ids)):
            h_mod = base_h[:src_len].clone()
            h_mod[s:e] = h_mod[s:e] + self.lemma_indicator_emb + self.lemma_pos_emb.weight[pid]
            enc_states[i] = h_mod

        dec_input = torch.full((N, 1), BYT5_PAD, dtype=torch.long, device=device)
        finished = torch.zeros(N, dtype=torch.bool, device=device)
        generated = [[] for _ in range(N)]

        for _ in range(max_len):
            out = self.lemma_decoder(
                input_ids=dec_input,
                encoder_hidden_states=enc_states,
                encoder_attention_mask=enc_mask,
            )
            logits = self.lemma_lm_head(out.last_hidden_state[:, -1, :])
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
                lemmas.append(raw.decode('utf-8'))
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
# Loss
# ============================================================
def compute_loss(bio_logits, sent_logits, arc_scores, rel_scores,
                 bio_labels, sent_labels, dep_info, device,
                 lemma_logits=None, lemma_labels=None, lemma_loss_weight=1.0):
    bio_loss = F.cross_entropy(
        bio_logits.view(-1, NUM_BIO_POS), bio_labels.view(-1), ignore_index=-100,
    )

    valid_sent = sent_labels[sent_labels >= 0]
    if len(valid_sent) > 0:
        n_pos = (valid_sent == 1).float().sum()
        n_neg = (valid_sent == 0).float().sum()
        w = torch.tensor([1.0, max(1.0, (n_neg / max(n_pos, 1)).item())], device=device)
    else:
        w = torch.ones(2, device=device)
    sent_loss = F.cross_entropy(
        sent_logits.view(-1, 2), sent_labels.view(-1), weight=w, ignore_index=-100,
    )

    arc_loss = torch.tensor(0.0, device=device)
    rel_loss = torch.tensor(0.0, device=device)

    if arc_scores is not None and dep_info:
        max_toks = arc_scores.shape[1]
        all_heads, all_rels, all_mask = [], [], []
        for d in dep_info:
            n = len(d["heads"])
            h = torch.zeros(max_toks, dtype=torch.long, device=device)
            r = torch.zeros(max_toks, dtype=torch.long, device=device)
            m = torch.zeros(max_toks, dtype=torch.bool, device=device)
            h[:n] = torch.tensor(d["heads"], dtype=torch.long, device=device)
            r[:n] = torch.tensor(d["dep_rels"], dtype=torch.long, device=device)
            m[:n] = True
            all_heads.append(h)
            all_rels.append(r)
            all_mask.append(m)

        heads_t = torch.stack(all_heads)
        rels_t = torch.stack(all_rels)
        mask_t = torch.stack(all_mask)

        arc_loss = F.cross_entropy(arc_scores[mask_t], heads_t[mask_t])

        n_rels = rel_scores.shape[-1]
        gold_head_idx = heads_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n_rels)
        rel_at_head = rel_scores.gather(2, gold_head_idx).squeeze(2)
        rel_loss = F.cross_entropy(rel_at_head[mask_t], rels_t[mask_t])

    lemma_loss = torch.tensor(0.0, device=device)
    if lemma_logits is not None and lemma_labels is not None:
        lemma_loss = F.cross_entropy(
            lemma_logits.view(-1, lemma_logits.size(-1)),
            lemma_labels.view(-1),
            ignore_index=-100,
        ) * lemma_loss_weight

    return {
        "total": bio_loss + sent_loss + arc_loss + rel_loss + lemma_loss,
        "bio_pos": bio_loss, "sent": sent_loss, "arc": arc_loss, "rel": rel_loss,
        "lemma": lemma_loss,
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
        bio_lab = batch["bio_pos_labels"].to(device)
        sent_lab = batch["sent_labels"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        dep_info = batch["dep_info"]

        bio_log, sent_log, arc_sc, rel_sc, lem_log, lem_lab = model(
            ids, mask, dep_info, lang_ids, max_lemma_tokens_per_sent=8)
        losses = compute_loss(bio_log, sent_log, arc_sc, rel_sc, bio_lab, sent_lab, dep_info, device,
                              lemma_logits=lem_log, lemma_labels=lem_lab)
        for k, v in losses.items():
            stats[f"loss_{k}"] += v.item()
        n_batches += 1

        # BIO-POS accuracy
        valid = bio_lab >= 0
        preds = bio_log.argmax(-1)
        stats["bio_correct"] += (preds[valid] == bio_lab[valid]).sum().item()
        stats["bio_total"] += valid.sum().item()

        # Sentence boundary
        sv = sent_lab >= 0
        sp = sent_log.argmax(-1)
        pp = sp[sv] == 1
        gp = sent_lab[sv] == 1
        stats["sent_tp"] += (pp & gp).sum().item()
        stats["sent_fp"] += (pp & ~gp).sum().item()
        stats["sent_fn"] += (~pp & gp).sum().item()

        # Dependency parsing
        if arc_sc is not None and dep_info:
            arc_pred = arc_sc.argmax(-1)
            for si, d in enumerate(dep_info):
                n = len(d["heads"])
                gh = torch.tensor(d["heads"], device=device)
                gr = torch.tensor(d["dep_rels"], device=device)
                ph = arc_pred[si, :n]
                correct_arc = ph == gh
                stats["arc_correct"] += correct_arc.sum().item()
                stats["arc_total"] += n
                pr = rel_sc[si, :n]  # (n, max_heads+1, n_rels)
                pred_rels = pr[torch.arange(n, device=device), ph].argmax(-1)
                stats["las_correct"] += (correct_arc & (pred_rels == gr)).sum().item()

    m = {}
    m["bio_pos_acc"] = stats["bio_correct"] / max(stats["bio_total"], 1)
    p = stats["sent_tp"] / max(stats["sent_tp"] + stats["sent_fp"], 1)
    r = stats["sent_tp"] / max(stats["sent_tp"] + stats["sent_fn"], 1)
    m["sent_f1"] = 2 * p * r / max(p + r, 1e-8)
    m["sent_prec"] = p
    m["sent_rec"] = r
    m["uas"] = stats["arc_correct"] / max(stats["arc_total"], 1)
    m["las"] = stats["las_correct"] / max(stats["arc_total"], 1)
    for k in ["loss_total", "loss_bio_pos", "loss_sent", "loss_arc", "loss_rel", "loss_lemma"]:
        m[k] = stats[k] / max(n_batches, 1)
    return m


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
    paragraphs = group_into_paragraphs(sentences, seed=args.seed)
    print(f"{len(paragraphs)} paragraphs")

    rng = random.Random(args.seed)
    rng.shuffle(paragraphs)
    n_train = int(0.9 * len(paragraphs))
    n_val = int(0.05 * len(paragraphs))
    train_p, val_p = paragraphs[:n_train], paragraphs[n_train:n_train + n_val]
    print(f"Train: {len(train_p)}, Val: {len(val_p)}")

    tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
    train_ds = Stage1Dataset(train_p, tokenizer, args.max_length)
    val_ds = Stage1Dataset(val_p, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    model = CanineForNLP(
        num_bio_pos=NUM_BIO_POS, num_dep_rels=NUM_DEP_RELS,
        arc_dim=args.arc_dim, rel_dim=args.rel_dim,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    backbone = list(model.canine.parameters())
    decoder_params = list(model.lemma_decoder.parameters()) + list(model.lemma_lm_head.parameters())
    decoder_ids = {id(p) for p in decoder_params}
    backbone_ids = {id(p) for p in backbone}
    heads = [p for p in model.parameters() if id(p) not in backbone_ids and id(p) not in decoder_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": heads, "lr": args.head_lr},
        {"params": decoder_params, "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.num_epochs // args.grad_accum
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save label config
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
                                          lemma_loss_weight=args.lemma_loss_weight)
                scaler.scale(losses["total"] / args.grad_accum).backward()
            else:
                bio_log, sent_log, arc_sc, rel_sc, lem_log, lem_lab = model(
                    ids, amask, dep_info, lang_ids,
                    max_lemma_tokens_per_sent=args.lemma_tokens_per_sent)
                losses = compute_loss(bio_log, sent_log, arc_sc, rel_sc, bio_lab, sent_lab, dep_info, device,
                                      lemma_logits=lem_log, lemma_labels=lem_lab,
                                      lemma_loss_weight=args.lemma_loss_weight)
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

    # Upload to HuggingFace Hub
    hf_username = os.getenv("HF_USERNAME")
    if hf_username:
        from huggingface_hub import HfApi
        hf_repo_name = f"{hf_username}/lexide-canine-nlp"
        print(f"Uploading Stage 1 model to HuggingFace Hub: {hf_repo_name}")
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
    p.add_argument("--output_dir", default="output/stage1")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
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
    p.add_argument("--decoder_lr", type=float, default=1e-4)
    p.add_argument("--lemma_tokens_per_sent", type=int, default=4)
    p.add_argument("--lemma_loss_weight", type=float, default=1.0)
    p.add_argument("--run_name", default="stage1-canine")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()

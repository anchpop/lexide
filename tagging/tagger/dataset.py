"""Torch datasets + collation for the multi-task tagger and the char boundary tagger.

The tagger dataset tokenizes each sentence with the encoder's own fast tokenizer and,
using the returned character offset mapping, records for every *gold* word the index of
the subword that its first character falls in. Downstream the model gathers exactly that
one vector per word, so gold token boundaries never need to agree with subword boundaries.
"""
import json

import torch
from torch.utils.data import Dataset


def load_vocab(path):
    v = json.loads(open(path, encoding="utf-8").read())
    return {
        "pos": {p: i for i, p in enumerate(v["pos"])},
        "dep": {d: i for i, d in enumerate(v["dep"])},
        "pos_list": v["pos"],
        "dep_list": v["dep"],
        "lemma_scripts": v["lemma_scripts"],
    }


def read_jsonl(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            out.append(json.loads(line))
    return out


class TaggerDataset(Dataset):
    def __init__(self, records, tokenizer, vocab, max_subwords=192, max_words=64):
        self.records = records
        self.tok = tokenizer
        self.vocab = vocab
        self.max_subwords = max_subwords
        self.max_words = max_words

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        text = r["text"]
        enc = self.tok(text, return_offsets_mapping=True, truncation=True,
                       max_length=self.max_subwords, return_tensors=None)
        offsets = enc["offset_mapping"]
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        # For each subword, its (start,end) char span; special tokens have (0,0).
        # Map char position -> first subword index whose span covers it.
        # Build an array: for each char, the subword index (or -1).
        char_to_sub = {}
        for si, (a, b) in enumerate(offsets):
            if b == 0 and a == 0:
                continue  # special / empty
            for c in range(a, b):
                if c not in char_to_sub:
                    char_to_sub[c] = si

        pos_map = self.vocab["pos"]
        dep_map = self.vocab["dep"]
        words_first = []
        pos_ids = []
        rel_ids = []
        heads = []
        lemma_ids = []
        kept_word_positions = []  # original 1-indexed token positions that survived truncation
        for wi, tk in enumerate(r["tokens"]):
            sub = char_to_sub.get(tk["start"])
            if sub is None:
                # first char of the word fell outside the (truncated) subword span
                sub = char_to_sub.get(tk["end"] - 1)
            if sub is None:
                continue
            if len(words_first) >= self.max_words:
                break
            words_first.append(sub)
            pos_ids.append(pos_map.get(tk["pos"], pos_map.get("X", 0)))
            rel_ids.append(dep_map.get(tk["dep"], 0))
            heads.append(tk["head"])
            lemma_ids.append(tk["lemma_script"])
            kept_word_positions.append(wi + 1)  # 1-indexed

        # remap heads to indices into the *kept* word list (+ROOT at 0). Heads pointing to a
        # dropped word (rare, only under truncation) fall back to ROOT (0).
        pos_to_new = {orig: new + 1 for new, orig in enumerate(kept_word_positions)}
        remapped_heads = []
        for h in heads:
            if h == 0:
                remapped_heads.append(0)
            else:
                remapped_heads.append(pos_to_new.get(h, 0))

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "word_first_sub": words_first,
            "pos": pos_ids,
            "rel": rel_ids,
            "head": remapped_heads,
            "lemma": lemma_ids,
        }


def pad_collate(batch, pad_id):
    B = len(batch)
    max_sub = max(len(b["input_ids"]) for b in batch)
    max_w = max(len(b["word_first_sub"]) for b in batch)
    max_w = max(max_w, 1)

    input_ids = torch.full((B, max_sub), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_sub), dtype=torch.long)
    word_first = torch.zeros((B, max_w), dtype=torch.long)
    word_mask = torch.zeros((B, max_w), dtype=torch.long)
    pos = torch.zeros((B, max_w), dtype=torch.long)
    rel = torch.zeros((B, max_w), dtype=torch.long)
    head = torch.zeros((B, max_w), dtype=torch.long)
    lemma = torch.zeros((B, max_w), dtype=torch.long)

    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"])
        attn[i, :L] = torch.tensor(b["attention_mask"])
        W = len(b["word_first_sub"])
        if W:
            word_first[i, :W] = torch.tensor(b["word_first_sub"])
            word_mask[i, :W] = 1
            pos[i, :W] = torch.tensor(b["pos"])
            rel[i, :W] = torch.tensor(b["rel"])
            head[i, :W] = torch.tensor(b["head"])
            lemma[i, :W] = torch.tensor(b["lemma"])

    return {
        "input_ids": input_ids, "attention_mask": attn,
        "word_first_sub": word_first, "word_mask": word_mask,
        "pos": pos, "rel": rel, "head": head, "lemma": lemma,
    }


# --------------------------------------------------------------------------------------
# Char boundary tagger data: raw text -> per-byte O/B/I labels
# --------------------------------------------------------------------------------------
PAD_BYTE, BOS_BYTE, EOS_BYTE = 256, 257, 258


def encode_bytes_and_labels(text, tokens, max_bytes=512):
    """Return (byte_ids, char_labels) where labels are over the UTF-8 byte stream.

    A token span [start,end) in characters is projected onto bytes; the first byte of a
    token is B, remaining bytes of that token are I, bytes outside any token are O.
    Labelling on bytes (not chars) keeps it aligned with the byte-level model input.
    """
    # char index -> label, then expand to bytes
    char_label = ["O"] * len(text)
    for tk in tokens:
        s, e = tk["start"], tk["end"]
        if s >= len(text):
            continue
        char_label[s] = "B"
        for c in range(s + 1, min(e, len(text))):
            char_label[c] = "I"
    byte_ids = [BOS_BYTE]
    labels = [0]  # O for BOS
    lab_map = {"O": 0, "B": 1, "I": 2}
    for ch, lab in zip(text, char_label):
        bs = ch.encode("utf-8")
        for j, byte in enumerate(bs):
            byte_ids.append(byte)
            # only the first byte of the first char of a token is B; continuation bytes are I
            labels.append(lab_map[lab] if j == 0 else lab_map["I" if lab != "O" else "O"])
    byte_ids.append(EOS_BYTE)
    labels.append(0)
    return byte_ids[:max_bytes], labels[:max_bytes]


class CharBoundaryDataset(Dataset):
    def __init__(self, records, max_bytes=512):
        self.records = records
        self.max_bytes = max_bytes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        byte_ids, labels = encode_bytes_and_labels(r["text"], r["tokens"], self.max_bytes)
        return {"byte_ids": byte_ids, "labels": labels}


def char_collate(batch):
    B = len(batch)
    L = max(len(b["byte_ids"]) for b in batch)
    byte_ids = torch.full((B, L), PAD_BYTE, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["byte_ids"])
        byte_ids[i, :n] = torch.tensor(b["byte_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
    return {"byte_ids": byte_ids, "labels": labels}

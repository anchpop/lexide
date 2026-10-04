"""Disk-indexed corpus and sort-ish, padded-token-budget joint batches."""
import json
import mmap
from pathlib import Path
import random

import numpy as np
import torch

from data_prep import LANGS


class IndexedCorpus:
    """Compact offsets, not millions of Python token dictionaries, live in RAM."""
    def __init__(self, path):
        self.path = Path(path)
        cache = self.path.with_suffix(".index.npz")
        stat = self.path.stat()
        identity = np.array([stat.st_size, stat.st_mtime_ns], dtype=np.int64)
        valid = False
        if cache.exists():
            with np.load(cache) as index:
                valid = np.array_equal(index["identity"], identity)
                if valid:
                    self.offsets, self.langs, self.gold = (index[k] for k in ("offsets", "langs", "gold"))
        if not valid:
            from array import array
            offsets, langs, gold = array("Q"), array("B"), array("B")
            with self.path.open("rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    r = json.loads(line)
                    offsets.append(offset)
                    langs.append(LANGS.index(r["lang"]))
                    gold.append(r.get("kind") == "gold")
            self.offsets = np.asarray(offsets, dtype=np.uint64)
            self.langs = np.asarray(langs, dtype=np.uint8)
            self.gold = np.asarray(gold, dtype=bool)
            np.savez(cache, identity=identity, offsets=self.offsets, langs=self.langs, gold=self.gold)
        self.file = self.path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def __getitem__(self, i):
        offset = int(self.offsets[i])
        end = self.data.find(b"\n", offset)
        return json.loads(self.data[offset:end if end >= 0 else len(self.data)])

    def sample(self, cap, seed, limit_per_lang=0):
        rng = np.random.default_rng(seed)
        selected = []
        counts = {}
        for i, lang in enumerate(LANGS):
            gold = np.flatnonzero((self.langs == i) & self.gold)
            silver = np.flatnonzero((self.langs == i) & ~self.gold)
            n = min(len(silver), max(0, cap - len(gold))) if cap else len(silver)
            ids = np.concatenate([gold, rng.choice(silver, n, replace=False)])
            # Explicit tiny/smoke limit is separate from normal epoch sampling.
            if limit_per_lang and len(ids) > limit_per_lang:
                ids = rng.choice(ids, limit_per_lang, replace=False)
            counts[lang] = len(ids)
            selected.extend(ids.tolist())
        rng.shuffle(selected)
        return selected, counts


def prepare(record, enc, vocab, char_buckets, truncate):
    text = record["text"]
    offsets = enc["offset_mapping"]
    # Truncation is signalled by an overflowing original tokenization, not merely
    # max(offsets): uncovered trailing whitespace belongs to the complete sentence.
    end = max((b for a, b in offsets), default=0) if truncate else len(text)
    text = text[:end]
    mapping = [-1] * len(text)
    first = set()
    for si, (a, b) in enumerate(offsets):
        if a < b:
            first.add(a)
            for c in range(a, min(b, len(text))):
                if mapping[c] < 0:
                    mapping[c] = si
    features = [int(ch.isspace()) + 2 * int(c in first) + 4 * int(mapping[c] < 0)
                for c, ch in enumerate(text)]
    boundary = [0] * len(text)
    tokens = record.get("tokens", [])
    kept = [(i + 1, t) for i, t in enumerate(tokens) if 0 <= t["start"] < t["end"] <= len(text)]
    remap = {i: j + 1 for j, (i, _) in enumerate(kept)}
    remap[0] = 0
    for t in tokens:
        if t["end"] > len(text):
            # A partial word must not teach the boundary head that its prefix is O.
            for c in range(t["start"], len(text)):
                boundary[c] = -100
    for _, t in kept:
        boundary[t["start"]] = 1
        boundary[t["start"] + 1:t["end"]] = [2] * (t["end"] - t["start"] - 1)
    pos = {v: i for i, v in enumerate(vocab["pos"])}
    dep = {v: i for i, v in enumerate(vocab["dep"])}
    langs = vocab.get("langs", LANGS)
    return {"record": record, "text": text, "input_ids": enc["input_ids"],
            "char_to_sub": mapping, "char_ids": [ord(c) % char_buckets + 1 for c in text],
            "char_features": features, "boundary": boundary,
            "lang_ids": langs.index(record["lang"]) + 1 if record.get("lang") in langs else 0,
            "starts": [t["start"] for _, t in kept], "ends": [t["end"] for _, t in kept],
            "pos": [pos.get(t["pos"], pos["X"]) for _, t in kept],
            "rel": [dep.get(t["dep"], 0) for _, t in kept],
            "lemma": [t.get("lemma_script", 0) for _, t in kept],
            "head": [remap.get(t["head"], -100) if t["head"] != i else -100 for i, t in kept]}


def encode_records(records, tokenizer, vocab, max_subwords=256, char_buckets=65536, training=True):
    enc = tokenizer([r["text"] for r in records], return_offsets_mapping=True,
                    truncation=False, verbose=False)
    result = []
    for i, r in enumerate(records):
        item = {k: enc[k][i] for k in ("input_ids", "offset_mapping")}
        truncated = training and len(item["input_ids"]) > max_subwords
        if truncated:
            # Preserve terminal special token while removing the tail.
            item = {k: v[:max_subwords - 1] + v[-1:] for k, v in item.items()}
        result.append(prepare(r, item, vocab, char_buckets, truncated))
    return result


def collate(items, pad_id, device="cpu"):
    b = len(items)
    s = max(len(x["input_ids"]) for x in items)
    c = max(1, max(len(x["char_ids"]) for x in items))
    w = max(1, max(len(x["starts"]) for x in items))
    batch = {"input_ids": torch.full((b, s), pad_id, dtype=torch.long),
             "attention_mask": torch.zeros(b, s, dtype=torch.long),
             "char_mask": torch.zeros(b, c, dtype=torch.bool),
             "word_mask": torch.zeros(b, w, dtype=torch.bool),
             "lang_ids": torch.tensor([x["lang_ids"] for x in items])}
    for key in ("char_to_sub", "char_ids", "char_features", "boundary"):
        batch[key] = torch.full((b, c), -100 if key == "boundary" else -1 if key == "char_to_sub" else 0, dtype=torch.long)
    for key in ("starts", "ends", "pos", "rel", "head", "lemma"):
        batch[key] = torch.zeros(b, w, dtype=torch.long)
    for i, item in enumerate(items):
        batch["attention_mask"][i, :len(item["input_ids"])] = 1
        batch["char_mask"][i, :len(item["char_ids"])] = True
        batch["word_mask"][i, :len(item["starts"])] = True
        for key in ("input_ids", "char_to_sub", "char_ids", "char_features", "boundary", "starts", "ends", "pos", "rel", "head", "lemma"):
            batch[key][i, :len(item[key])] = torch.tensor(item[key], dtype=torch.long)
    return {k: v.to(device) for k, v in batch.items()}


def budget_batches(corpus, indices, tokenizer, vocab, budget, max_subwords=256,
                   char_buckets=65536, seed=0, chunk_size=2048):
    """Sort actual subword lengths within shuffled chunks; budget charges padding."""
    rng = random.Random(seed)
    for start in range(0, len(indices), chunk_size):
        records = [corpus[i] for i in indices[start:start + chunk_size]]
        items = encode_records(records, tokenizer, vocab, max_subwords, char_buckets)
        items.sort(key=lambda x: len(x["input_ids"]))
        batches, batch = [], []
        for item in items:
            if batch and len(item["input_ids"]) * (len(batch) + 1) > budget:
                batches.append(batch)
                batch = []
            batch.append(item)
        if batch:
            batches.append(batch)
        rng.shuffle(batches)
        yield from batches

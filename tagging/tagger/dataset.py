"""Torch datasets + collation for the multi-task tagger and the char boundary tagger.

The tagger dataset tokenizes each sentence with the encoder's own fast tokenizer and,
using the returned character offset mapping, records for every *gold* word the index of
the subword that its first character falls in. Downstream the model gathers exactly that
one vector per word, so gold token boundaries never need to agree with subword boundaries.
"""
import json
import random

import torch
from torch.utils.data import Dataset

from prior import PRIOR_NONE, prior_ids_for


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
            r = json.loads(line)
            # keep the file position: the prior sidecar is indexed by it, and callers
            # filter/subset the records afterwards
            r["_idx"] = i
            out.append(r)
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

# Language-conditioned BOS: the sequence's first token is either the generic BOS
# (language unknown) or a per-language variant, telling the model the language without
# shifting any byte offsets. Order is fixed (alphabetical); the safetensors export and
# the Rust byte_bio reimplementation rely on these exact ids.
LANG_ORDER = ["deu", "eng", "fra", "hin", "ita", "jpn", "kor", "por", "rus", "spa", "tha",
              "zho-hans"]
LANG_BOS = {lang: 259 + i for i, lang in enumerate(LANG_ORDER)}
CHAR_VOCAB_SIZE = 259 + len(LANG_ORDER)  # 271


def encode_bytes_and_labels(text, tokens, max_bytes=512, lang=None):
    """Return (byte_ids, char_labels) where labels are over the UTF-8 byte stream.

    A token span [start,end) in characters is projected onto bytes; the first byte of a
    token is B, remaining bytes of that token are I, bytes outside any token are O.
    Labelling on bytes (not chars) keeps it aligned with the byte-level model input.
    `lang` (a code from LANG_ORDER, or None) selects the language-conditioned BOS.
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
    byte_ids = [LANG_BOS.get(lang, BOS_BYTE)]
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


def _jpn_absorbs(text, head, nxt):
    """Does `nxt` continue the same word as `head`? (see merge_jpn_inflection)"""
    if nxt["start"] != head["end"]:
        return False  # written apart — whatever separates them keeps them separate
    head_text, nxt_text = text[head["start"]:head["end"]], text[nxt["start"]:nxt["end"]]
    # pull the て/で of a て-form onto its verb so the stem is not left stranded
    if nxt.get("pos") == "SCONJ" and nxt_text in ("て", "で"):
        return head.get("pos") in ("VERB", "ADJ")
    if nxt.get("pos") != "AUX":
        return False
    # only a predicate has an inflectional tail; a noun keeps its copula (学生|です)
    if head.get("pos") not in ("VERB", "ADJ", "AUX"):
        return False
    if head_text.endswith(("て", "で")):
        return False  # 食べて|いる — both halves are words, so the split stays
    if nxt_text == "な":
        return False  # 綿密|な — attributive marker, treated like a particle
    return True


def merge_jpn_inflection(text, tokens):
    """Japanese verb/adjective + its auxiliary chain is one word: 食べ|まし|た -> 食べました.

    Japanese inflection is agglutinative, so splitting a predicate from its auxiliaries
    strands pieces (食べ, まし, だっ, られ) that are not words and that no learner can be
    shown. The morpheme layer, not the tokenizer, is where the internal structure lives.
    Mirrors JapaneseCorrector::post_corrections in yap's clean-nlp-data, which applies the
    same rule at labelling time; this applies it to silver already on disk, where the
    Gemma teacher still emits the old finer split.
    """
    out = []
    for tk in tokens:
        if out and _jpn_absorbs(text, out[-1], tk):
            out[-1] = {**out[-1], "end": tk["end"]}
            continue
        out.append(tk)
    return out


def maybe_merge_inflection(record, enabled):
    """Token spans for `record`, merged if this language needs it and `enabled` is set."""
    if enabled and record.get("lang") == "jpn":
        return merge_jpn_inflection(record["text"], record["tokens"])
    return record["tokens"]


class CharBoundaryDataset(Dataset):
    def __init__(self, records, max_bytes=512, lang_dropout=None, merge_inflection=False,
                 use_prior=False, wordbanks=None, prior_sidecar=None,
                 prior_dropout=0.0):
        """lang_dropout=None trains language-blind (generic BOS always, the pre-lang
        behavior); a float p trains language-conditioned, replacing the record's lang
        with the generic BOS with probability p so the model also works lang-free.

        merge_inflection is for the *tokenizer* only — the segmenter shares this class but
        its spans are sentences, which have no inflection to merge."""
        self.records = records
        self.max_bytes = max_bytes
        self.lang_dropout = lang_dropout
        self.merge_inflection = merge_inflection
        self.use_prior = use_prior
        self.wordbanks = wordbanks
        self.prior_sidecar = prior_sidecar
        # Fraction of examples whose proposal is blanked to PRIOR_NONE — "no proposal
        # available" — so the model keeps a route to the answer that does not go through
        # the prior. Measured on v11, which trained without this: Japanese scores 93.3 with
        # the dictionary and 21.2 without, even with the language token. That is not a model
        # using a hint, it is a model that has outsourced the task.
        #
        # Blanking to NONE rather than to the whitespace proposal is deliberate. Whitespace
        # on Japanese does not say "I don't know", it asserts the sentence is one word;
        # training against that would teach the model to distrust whitespace, and the prior
        # embedding is shared, so the lesson would leak to the languages where whitespace is
        # exactly right.
        self.prior_dropout = prior_dropout

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        lang = None
        if self.lang_dropout is not None and random.random() >= self.lang_dropout:
            lang = r.get("lang")
        tokens = maybe_merge_inflection(r, self.merge_inflection)
        byte_ids, labels = encode_bytes_and_labels(r["text"], tokens, self.max_bytes, lang)
        item = {"byte_ids": byte_ids, "labels": labels}
        if self.use_prior:
            if self.prior_dropout and random.random() < self.prior_dropout:
                n = min(len(r["text"].encode("utf-8")) + 2, self.max_bytes)
                item["prior_ids"] = [PRIOR_NONE] * n
            elif self.prior_sidecar is not None:
                # precomputed by the Rust binary that also runs at inference
                item["prior_ids"] = self.prior_sidecar[r["_idx"]]
            else:
                # the prior always knows the language — it is computed from the text we
                # hold, not from the lang token the model may have had dropped out
                item["prior_ids"] = prior_ids_for(r["text"], r.get("lang"), self.max_bytes,
                                                  wordbanks=self.wordbanks)
        return item


def char_collate(batch):
    B = len(batch)
    L = max(len(b["byte_ids"]) for b in batch)
    byte_ids = torch.full((B, L), PAD_BYTE, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    has_prior = "prior_ids" in batch[0]
    prior_ids = torch.full((B, L), PRIOR_NONE, dtype=torch.long) if has_prior else None
    for i, b in enumerate(batch):
        n = len(b["byte_ids"])
        byte_ids[i, :n] = torch.tensor(b["byte_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        if has_prior:
            p = b["prior_ids"]
            prior_ids[i, :len(p)] = torch.tensor(p)
    out = {"byte_ids": byte_ids, "labels": labels}
    if has_prior:
        out["prior_ids"] = prior_ids
    return out

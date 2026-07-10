"""End-to-end inference: raw text -> tokens with POS / lemma / dependency head+relation.

Chains the two trained models: the char boundary tagger segments the string into our
tokens, then the multi-task tagger labels them. This is the deployable replacement for
the autoregressive Gemma pipeline.

    from predict import Pipeline
    pipe = Pipeline("output/tagger/best", "output/tokenizer/tokenizer.pt")
    print(pipe("Eine Fundgrube."))
"""
import argparse
import json
import os

import torch

from data_prep import apply_script
from dataset import BOS_BYTE, EOS_BYTE
from model import CharBoundaryTagger, MultiTaskTagger


def spans_from_byte_labels(text, byte_labels):
    """Map per-byte O/B/I labels back to character spans in `text`.

    byte_labels is aligned to [BOS] + utf-8(text) + [EOS]; we walk the text's bytes and
    read the label at each character's first byte.
    """
    # Build char -> first-byte-index (in the BOS-prefixed byte stream) mapping.
    spans = []
    pos = 1  # skip BOS
    cur_start = None
    for ci, ch in enumerate(text):
        nbytes = len(ch.encode("utf-8"))
        lab = byte_labels[pos] if pos < len(byte_labels) else 0
        if lab == 1:  # B
            if cur_start is not None:
                spans.append((cur_start, ci))
            cur_start = ci
        elif lab == 0:  # O -> close any open token
            if cur_start is not None:
                spans.append((cur_start, ci))
                cur_start = None
        # lab == 2 (I) -> continue
        pos += nbytes
    if cur_start is not None:
        spans.append((cur_start, len(text)))
    return spans


class Pipeline:
    def __init__(self, tagger_dir, tokenizer_path=None, device=None, lemma_table_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lemma_table = None
        if lemma_table_path:
            from lemma_lookup import LemmaTable
            self.lemma_table = LemmaTable.load(lemma_table_path)
        meta = json.load(open(os.path.join(tagger_dir, "meta.json")))
        vocab = json.load(open(os.path.join(tagger_dir, "vocab.json"), encoding="utf-8"))
        self.pos_list = vocab["pos"]
        self.dep_list = vocab["dep"]
        self.scripts = vocab["lemma_scripts"]
        from transformers import AutoTokenizer
        self.hf_tok = AutoTokenizer.from_pretrained(tagger_dir, use_fast=True)
        self.tagger = MultiTaskTagger(meta["encoder"], len(self.pos_list),
                                      len(self.dep_list), len(self.scripts))
        self.tagger.load_state_dict(torch.load(os.path.join(tagger_dir, "model.pt"),
                                               map_location=self.device))
        self.tagger.to(self.device).eval()

        self.char_tok = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.char_tok = CharBoundaryTagger()
            self.char_tok.load_state_dict(torch.load(tokenizer_path, map_location=self.device))
            self.char_tok.to(self.device).eval()

    @torch.no_grad()
    def segment(self, text):
        """Raw text -> list of (start,end) char spans using the char boundary tagger."""
        if self.char_tok is None:
            raise RuntimeError("no char tokenizer loaded; pass gold spans instead")
        byte_ids = [BOS_BYTE]
        for ch in text:
            byte_ids.extend(ch.encode("utf-8"))
        byte_ids.append(EOS_BYTE)
        x = torch.tensor([byte_ids], device=self.device)
        labels = self.char_tok(x)[0].argmax(-1).tolist()
        return spans_from_byte_labels(text, labels)

    @torch.no_grad()
    def tag(self, text, spans):
        """Given token spans, produce per-token labels."""
        enc = self.hf_tok(text, return_offsets_mapping=True, truncation=True,
                          max_length=192, return_tensors=None)
        offsets = enc["offset_mapping"]
        char_to_sub = {}
        for si, (a, b) in enumerate(offsets):
            if a == 0 and b == 0:
                continue
            for c in range(a, b):
                char_to_sub.setdefault(c, si)
        word_first = []
        keep = []
        for (s, e) in spans:
            sub = char_to_sub.get(s, char_to_sub.get(e - 1))
            if sub is None:
                continue
            word_first.append(sub)
            keep.append((s, e))
        if not word_first:
            return []
        input_ids = torch.tensor([enc["input_ids"]], device=self.device)
        attn = torch.tensor([enc["attention_mask"]], device=self.device)
        wf = torch.tensor([word_first], device=self.device)
        wm = torch.ones_like(wf)
        out = self.tagger(input_ids=input_ids, attention_mask=attn,
                          word_first_sub=wf, word_mask=wm)
        pos = out.pos_logits[0].argmax(-1).tolist()
        arc = out.arc_scores[0].argmax(-1).tolist()  # head index into [ROOT, w1, ...]
        W = len(keep)
        rel_at = out.rel_scores[0][torch.arange(W), out.arc_scores[0].argmax(-1)]
        rel = rel_at.argmax(-1).tolist()
        lemma = out.lemma_logits[0].argmax(-1).tolist()
        result = []
        for i, (s, e) in enumerate(keep):
            form = text[s:e]
            script = self.scripts[lemma[i]] if lemma[i] < len(self.scripts) else None
            lem = apply_script(form, script) if script else form
            pos_tag = self.pos_list[pos[i]]
            if self.lemma_table is not None:
                # OOD floor: fill in a real lemma where the model punted to copy on a content form
                lem = self.lemma_table.resolve(form, pos_tag, lem)
            result.append({
                "text": form, "start": s, "end": e,
                "pos": pos_tag,
                "lemma": lem,
                "head": arc[i],                       # 0 = ROOT, else 1-indexed token
                "dep": self.dep_list[rel[i]],
            })
        return result

    def __call__(self, text):
        spans = self.segment(text)
        return self.tag(text, spans)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tagger-dir", default="output/tagger/best")
    ap.add_argument("--tokenizer", default="output/tokenizer/tokenizer.pt")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    pipe = Pipeline(args.tagger_dir, args.tokenizer)
    for tok in pipe(args.text):
        print(json.dumps(tok, ensure_ascii=False))

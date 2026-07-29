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
from dataset import BOS_BYTE, EOS_BYTE, LANG_BOS
from model import CharBoundaryTagger, MultiTaskTagger


def load_char_model(path, device):
    """Load a byte-minGRU checkpoint, recovering every dimension (vocab incl. language
    tokens, emb, hidden, layers) from tensor shapes — old 259-vocab and new lang-token
    checkpoints both load."""
    state = torch.load(path, map_location=device)
    vocab, emb_dim = state["emb.weight"].shape
    hidden = state["layers.0.fwd.to_z.weight"].shape[0]
    layers = sum(1 for k in state if k.endswith(".fwd.to_z.weight"))
    # The boundary prior is recovered the same way: presence from the tensor, and whether
    # it was added or concatenated from layer 0's input width against the byte embedding.
    prior_vocab, prior_dim, prior_mode = 0, 8, "add"
    if "prior_emb.weight" in state:
        prior_vocab, prior_dim = state["prior_emb.weight"].shape
        in0 = state["layers.0.fwd.to_z.weight"].shape[1]
        prior_mode = "concat" if in0 == emb_dim + prior_dim else "add"
    model = CharBoundaryTagger(vocab_size=vocab, emb_dim=emb_dim,
                               hidden_dim=hidden, layers=layers,
                               prior_vocab=prior_vocab, prior_mode=prior_mode,
                               prior_dim=prior_dim)
    model.load_state_dict(state)
    return model.to(device).eval()


def byte_encode(text, lang=None, model=None):
    """[LANG_xxx or BOS] + utf8(text) + [EOS] byte ids. Unknown lang codes — or any lang
    when `model` is a pre-lang-token checkpoint (259-row embedding) — fall back to the
    generic BOS, so callers can pass whatever they have."""
    first = LANG_BOS.get(lang, BOS_BYTE)
    if model is not None and first >= model.emb.num_embeddings:
        first = BOS_BYTE
    byte_ids = [first]
    for ch in text:
        byte_ids.extend(ch.encode("utf-8"))
    byte_ids.append(EOS_BYTE)
    return byte_ids


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
    def __init__(self, tagger_dir, tokenizer_path=None, device=None, lemma_table_path=None,
                 segmenter_path=None, prior_dir=None):
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
        self.unidic = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.char_tok = load_char_model(tokenizer_path, self.device)
            # A prior-trained tokenizer has learned to lean on the proposal, so running it
            # prior-free would silently degrade exactly the languages the prior was added
            # for — and in concat mode would not run at all. The artifact ships beside the
            # weights (see release.sh); it must be the *same* one the Rust library reads.
            if self.char_tok.prior_emb is not None:
                from unidic_artifact import UniDic
                # defaults to beside the weights; the serve points this at the shared
                # onnx/ copy so the 83MB artifact is published once, not twice
                art = os.path.join(prior_dir or os.path.dirname(tokenizer_path),
                                   "jpn-unidic.bin")
                if os.path.exists(art):
                    self.unidic = UniDic.load(art)
                else:
                    raise RuntimeError(
                        f"{tokenizer_path} was trained with a boundary prior but "
                        f"{art} is missing — Japanese would fall back to whitespace and "
                        f"diverge from the Rust library")

        # Optional sentence segmenter (same byte-minGRU architecture, sentence-scale O/B/I).
        self.segmenter = None
        if segmenter_path and os.path.exists(segmenter_path):
            self.segmenter = load_char_model(segmenter_path, self.device)

    @torch.no_grad()
    def segment(self, text, lang=None):
        """Raw text -> list of (start,end) char spans using the char boundary tagger."""
        if self.char_tok is None:
            raise RuntimeError("no char tokenizer loaded; pass gold spans instead")
        byte_ids = byte_encode(text, lang, self.char_tok)
        x = torch.tensor([byte_ids], device=self.device)
        p = None
        if self.char_tok.prior_emb is not None:
            from prior import prior_ids_for
            ids = prior_ids_for(text, lang, max_bytes=len(byte_ids), wordbanks={},
                                unidic=self.unidic)
            p = torch.tensor([ids], device=self.device)
        labels = self.char_tok(x, p)[0].argmax(-1).tolist()
        return spans_from_byte_labels(text, labels)

    @torch.no_grad()
    def segment_sentences(self, text, lang=None):
        """Split a passage into its sentence strings using the byte sentence segmenter.

        Same B/I/O span recovery as `segment`, but the spans are sentences, so the gaps
        (whitespace / headings / separators between sentences) are dropped.
        """
        if self.segmenter is None:
            raise RuntimeError("no sentence segmenter loaded")
        x = torch.tensor([byte_encode(text, lang, self.segmenter)], device=self.device)
        labels = self.segmenter(x)[0].argmax(-1).tolist()
        return [text[s:e] for s, e in spans_from_byte_labels(text, labels)]

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

    def __call__(self, text, lang=None):
        spans = self.segment(text, lang)
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

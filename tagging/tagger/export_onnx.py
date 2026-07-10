"""Export the trained MultiTaskTagger to ONNX for the Rust inference path.

The exported graph takes (input_ids, attention_mask, word_first_sub, word_mask) and returns
(pos_logits, lemma_logits, arc_scores, rel_scores) — i.e. the encoder + offset-based word
pooling + POS/lemma/biaffine heads, all in one forward. The arc-score padding mask (the -inf
fill) is left OUT of the graph on purpose (it's ONNX-unfriendly and trivial to apply in Rust at
argmax time, where batch=1 and the word count is known).

Run in a torch env with the tagger source importable:
    python export_onnx.py --model-dir <tagger/best> --out tagger.onnx
Verifies the ONNX outputs match PyTorch before writing is considered successful.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from model import MultiTaskTagger


class ExportTagger(nn.Module):
    """Wraps the trained model to return plain tensors (no loss, no dataclass) for ONNX."""

    def __init__(self, m: MultiTaskTagger):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, word_first_sub, word_mask):
        m = self.m
        enc = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        words = m.pool_words(enc, word_first_sub, word_mask)          # [B, W, H]
        pos_logits = m.pos_head(words)
        lemma_logits = m.lemma_head(words)

        B, W, H = words.shape
        root = m.root.expand(B, 1, H)
        head_cands = torch.cat([root, words], dim=1)                  # [B, W+1, H]

        arc_scores = m.arc_biaf(m.arc_dep(words), m.arc_head(head_cands))       # [B, W, W+1]
        rel_scores = m.rel_biaf(m.rel_dep(words), m.rel_head(head_cands))       # [B, W, W+1, n_dep]
        return pos_logits, lemma_logits, arc_scores, rel_scores


def load_model(model_dir):
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    vocab = json.load(open(os.path.join(model_dir, "vocab.json"), encoding="utf-8"))
    n_pos, n_dep, n_lemma = len(vocab["pos"]), len(vocab["dep"]), len(vocab["lemma_scripts"])
    model = MultiTaskTagger(meta["encoder"], n_pos, n_dep, n_lemma)
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta, vocab


def dummy_inputs(S=7, W=4):
    # one sentence: S subwords, W words; word_first_sub points to valid subword indices
    input_ids = torch.randint(5, 250000, (1, S), dtype=torch.long)
    input_ids[0, 0] = 0            # <s>
    input_ids[0, -1] = 2          # </s>
    attention_mask = torch.ones(1, S, dtype=torch.long)
    word_first_sub = torch.tensor([[1, 2, 3, 4]][:1], dtype=torch.long)[:, :W]
    if word_first_sub.shape[1] < W:
        word_first_sub = torch.arange(1, W + 1).unsqueeze(0)
    word_first_sub = word_first_sub.clamp(max=S - 1)
    word_mask = torch.ones(1, W, dtype=torch.long)
    return input_ids, attention_mask, word_first_sub, word_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="tagger.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model, meta, vocab = load_model(args.model_dir)
    wrapper = ExportTagger(model).eval()
    inputs = dummy_inputs()

    with torch.no_grad():
        ref = wrapper(*inputs)

    dyn = {
        "input_ids": {0: "batch", 1: "subwords"},
        "attention_mask": {0: "batch", 1: "subwords"},
        "word_first_sub": {0: "batch", 1: "words"},
        "word_mask": {0: "batch", 1: "words"},
        "pos_logits": {0: "batch", 1: "words"},
        "lemma_logits": {0: "batch", 1: "words"},
        "arc_scores": {0: "batch", 1: "words", 2: "words1"},
        "rel_scores": {0: "batch", 1: "words", 2: "words1"},
    }
    torch.onnx.export(
        wrapper, inputs, args.out,
        input_names=["input_ids", "attention_mask", "word_first_sub", "word_mask"],
        output_names=["pos_logits", "lemma_logits", "arc_scores", "rel_scores"],
        dynamic_axes=dyn, opset_version=args.opset, do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter — reliable for transformer + custom heads
    )
    print(f"[export] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

    # verify against onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    feed = {k: v.numpy() for k, v in zip(
        ["input_ids", "attention_mask", "word_first_sub", "word_mask"], inputs)}
    got = sess.run(None, feed)
    names = ["pos_logits", "lemma_logits", "arc_scores", "rel_scores"]
    ok = True
    for name, r, g in zip(names, ref, got):
        diff = np.abs(r.numpy() - g).max()
        flag = "OK" if diff < 1e-3 else "MISMATCH"
        if diff >= 1e-3:
            ok = False
        print(f"[verify] {name:12} max|torch-onnx|={diff:.2e} {flag}")
    # also verify at a DIFFERENT shape (dynamic axes really work)
    inputs2 = dummy_inputs(S=11, W=6)
    with torch.no_grad():
        ref2 = wrapper(*inputs2)
    feed2 = {k: v.numpy() for k, v in zip(
        ["input_ids", "attention_mask", "word_first_sub", "word_mask"], inputs2)}
    got2 = sess.run(None, feed2)
    d2 = max(np.abs(r.numpy() - g).max() for r, g in zip(ref2, got2))
    print(f"[verify] second shape (S=11,W=6) max diff={d2:.2e} {'OK' if d2 < 1e-3 else 'MISMATCH'}")
    assert ok and d2 < 1e-3, "ONNX outputs diverge from PyTorch"
    print("[export] ONNX matches PyTorch ✓")


if __name__ == "__main__":
    main()

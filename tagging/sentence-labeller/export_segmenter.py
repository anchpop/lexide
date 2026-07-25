"""Export the trained sentence segmenter to f32 safetensors + parity fixtures.

The byte-minGRU has a sequential scan that doesn't ONNX-export cleanly and, at ~0.31M
params, needs no runtime — the Rust side reimplements the forward pass and just loads the
weights. This mirrors `tagger/export_char_modal.py` (the token boundary tagger's export)
but runs locally, since the segmenter trains locally.

    .venv-seg/bin/python sentence-labeller/export_segmenter.py \
        --ckpt sentence-labeller/output/segmenter.pt --out-dir data/onnx

Writes `sentence_segmenter.safetensors` and `sentence_segmenter_fixtures.json`
(per-byte O/B/I labels + recovered sentence char spans + first-row logits) so the Rust
reimplementation can be verified bit-for-bit at the argmax level and close numerically.
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tagger"))
from dataset import BOS_BYTE, EOS_BYTE, LANG_BOS  # noqa: E402
from model import CharBoundaryTagger  # noqa: E402


def spans_from_byte_labels(text, byte_labels):
    """Per-byte O/B/I (aligned to [BOS]+utf8(text)+[EOS]) -> (start,end) char spans."""
    spans, pos, cur = [], 1, None
    for ci, ch in enumerate(text):
        lab = byte_labels[pos] if pos < len(byte_labels) else 0
        if lab == 1:
            if cur is not None:
                spans.append((cur, ci))
            cur = ci
        elif lab == 0:
            if cur is not None:
                spans.append((cur, ci))
                cur = None
        pos += len(ch.encode("utf-8"))
    if cur is not None:
        spans.append((cur, len(text)))
    return spans


# Multi-sentence passages exercising quotes, dashes, newlines, digits, multibyte scripts,
# and (v2) abbreviations + quote attributions. Each is recorded twice: with its language
# token and language-free (generic BOS), so the Rust parity test covers both paths.
FIXTURE_TEXTS = [
    ("eng", "He said, \"Hi!\"  Then he left. She waved back."),
    ("deu", "Guten Morgen. Wie geht es dir heute?\n\nMir geht es gut, danke!"),
    ("fra", "« Bonjour ! » dit-il. Puis il partit sans un mot."),
    ("spa", "¿Dónde está la biblioteca? Está cerca de la plaza. ¡Vamos!"),
    ("rus", "Я пришёл домой. Было тихо — слишком тихо. Что-то было не так."),
    ("jpn", "私は猫が好きです。犬も好きです。あなたは？"),
    ("kor", "고양이가 좋아요. 강아지도 좋아요. 당신은 어때요?"),
    ("hin", "मुझे किताबें पसंद हैं। यह किताब बहुत अच्छी है।"),
    ("ita", "Andiamo al mare domani. Fa caldo, vero? Sì, molto."),
    ("por", "Vamos à praia. O tempo está ótimo!\nA água está fria."),
    ("eng", "Mr. Dursley arrived at 3 p.m. sharp. \"Is this the place?\" she asked."),
    ("deu", "Dr. Weber wohnt in der Hauptstr. 5. Er kommt z. B. montags vorbei."),
]


def load_model(ckpt_path):
    """Rebuild the model from the checkpoint alone — every dimension (vocab incl. any
    language tokens, emb, hidden, layer count) is recoverable from tensor shapes, so old
    259-vocab and new lang-token checkpoints both load without config guesswork."""
    state = torch.load(ckpt_path, map_location="cpu")
    vocab, emb_dim = state["emb.weight"].shape
    hidden = state["layers.0.fwd.to_z.weight"].shape[0]
    layers = sum(1 for k in state if k.endswith(".fwd.to_z.weight"))
    model = CharBoundaryTagger(vocab_size=vocab, emb_dim=emb_dim,
                               hidden_dim=hidden, layers=layers)
    model.load_state_dict(state)
    model.eval()
    print(f"[load] vocab={vocab} emb={emb_dim} hidden={hidden} layers={layers}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sentence-labeller/output/segmenter.pt")
    ap.add_argument("--out-dir", default="data/onnx")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.ckpt)

    flat = {k: v.to(torch.float32).contiguous() for k, v in model.state_dict().items()}
    st_path = os.path.join(args.out_dir, "sentence_segmenter.safetensors")
    save_file(flat, st_path)
    n = sum(v.numel() for v in flat.values())
    print(f"[export] {st_path}: {len(flat)} tensors, {n/1e6:.3f}M params")

    has_lang_tokens = model.emb.num_embeddings > 259
    fixtures = []
    with torch.no_grad():
        for lang, text in FIXTURE_TEXTS:
            # Each text once language-free (generic BOS) and, when the checkpoint has
            # language tokens, once with its language token.
            for use_lang in ([None, lang] if has_lang_tokens else [None]):
                byte_ids = [LANG_BOS.get(use_lang, BOS_BYTE)]
                for ch in text:
                    byte_ids.extend(ch.encode("utf-8"))
                byte_ids.append(EOS_BYTE)
                logits = model(torch.tensor([byte_ids]))[0]
                labels = logits.argmax(-1).tolist()
                spans = spans_from_byte_labels(text, labels)
                fixtures.append({
                    "text": text,
                    "lang": use_lang,
                    "byte_labels": labels,
                    "spans": [[s, e] for s, e in spans],
                    "sentences": [text[s:e] for s, e in spans],
                    "first_logits": [round(x, 6) for x in logits[1].tolist()],
                })
                print(f"[fixture] lang={use_lang} {text[:40]!r}...: {len(spans)} sentences")
    fx_path = os.path.join(args.out_dir, "sentence_segmenter_fixtures.json")
    with open(fx_path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=1)
    print(f"[export] wrote {fx_path}")


if __name__ == "__main__":
    main()

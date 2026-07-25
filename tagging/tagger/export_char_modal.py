"""Export the CharBoundaryTagger weights to safetensors for the Rust inference path.

    modal run tagger/export_char_modal.py

The byte-minGRU tokenizer doesn't export cleanly to ONNX (its sequential scan would be
trace-unrolled at a fixed length), and at ~0.31M params it doesn't need a runtime — the Rust
side reimplements the forward pass and just needs the weights. This downloads
anchpop/lexide-parsley/tokenizer/tokenizer.pt, re-saves the state dict as f32 safetensors,
and also dumps reference outputs (per-byte labels + char spans) for a multilingual set of
strings so the Rust reimplementation can be verified bit-for-bit at the argmax level.
Artifacts land on the lexide-onnx volume next to tagger.onnx:

    modal volume get lexide-onnx char_tokenizer.safetensors data/onnx/
    modal volume get lexide-onnx char_tokenizer_fixtures.json data/onnx/
"""
from pathlib import Path

import modal

_tagger_src = Path(__file__).resolve().parent
app = modal.App("lexide-char-export")
hf_secret = modal.Secret.from_name("huggingface-secret")
onnx_vol = modal.Volume.from_name("lexide-onnx", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.13.0", "safetensors", "huggingface_hub", "numpy<2")
    .add_local_dir(str(_tagger_src), "/root/tagger", copy=True,
                   ignore=["output/**", ".venv/**", "__pycache__/**", "wandb/**", "*.pyc"])
)

# Multilingual reference strings — exercise multibyte scripts, apostrophes, digits,
# dashes, and (v2) abbreviations / multi-word names. Each is recorded language-free and,
# for lang-token checkpoints, with its language token.
FIXTURE_TEXTS = [
    ("deu", "Eine Fundgrube."),
    ("eng", "The cats were sleeping."),
    ("fra", "L'homme n'est pas venu, n'est-ce pas ?"),
    ("spa", "¿Dónde está la biblioteca?"),
    ("rus", "Я им доверяю — правда."),
    ("jpn", "私は猫が好きです。"),
    ("kor", "고양이가 좋아요."),
    ("hin", "मुझे बिल्लियाँ पसंद हैं।"),
    ("deu", "Ich weiß, dass es 3,5 km sind."),
    ("por", "Vamos à praia amanhã!"),
    ("eng", "Mr. Dursley visited the Eiffel Tower at 3 p.m."),
    ("fra", "M. Dupont a vu la tour Eiffel hier."),
]


@app.function(image=image, cpu=2.0, memory=4096, secrets=[hf_secret],
              volumes={"/vol": onnx_vol}, timeout=600)
def export():
    import json
    import os
    import sys

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    sys.path.insert(0, "/root/tagger")
    from dataset import BOS_BYTE, EOS_BYTE, LANG_BOS
    from model import CharBoundaryTagger
    from predict import spans_from_byte_labels

    pt = hf_hub_download("anchpop/lexide-parsley", "tokenizer/tokenizer.pt",
                         token=os.environ["HF_TOKEN"])
    state = torch.load(pt, map_location="cpu")
    # Every dimension is recoverable from tensor shapes, so old 259-vocab and new
    # lang-token checkpoints both load without config guesswork.
    vocab, emb_dim = state["emb.weight"].shape
    hidden = state["layers.0.fwd.to_z.weight"].shape[0]
    layers = sum(1 for k in state if k.endswith(".fwd.to_z.weight"))
    print(f"[load] vocab={vocab} emb={emb_dim} hidden={hidden} layers={layers}")
    model = CharBoundaryTagger(vocab_size=vocab, emb_dim=emb_dim,
                               hidden_dim=hidden, layers=layers)
    model.load_state_dict(state)
    model.eval()

    flat = {k: v.to(torch.float32).contiguous() for k, v in model.state_dict().items()}
    save_file(flat, "/vol/char_tokenizer.safetensors")
    n = sum(v.numel() for v in flat.values())
    print(f"[export] char_tokenizer.safetensors: {len(flat)} tensors, {n/1e6:.2f}M params")

    has_lang_tokens = model.emb.num_embeddings > 259
    fixtures = []
    with torch.no_grad():
        for lang, text in FIXTURE_TEXTS:
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
                    # first-row logits let the Rust test check numerics, not just argmax
                    "first_logits": [round(x, 6) for x in logits[1].tolist()],
                })
                print(f"[fixture] lang={use_lang} {text!r}: {len(spans)} tokens")
    with open("/vol/char_tokenizer_fixtures.json", "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=1)
    onnx_vol.commit()
    print("[export] wrote char_tokenizer.safetensors + char_tokenizer_fixtures.json to lexide-onnx")


@app.local_entrypoint()
def main():
    export.remote()

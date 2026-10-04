# Lexide Modal Deployment

**Parsley** (`modal_serve_tagger.py`) serves joint tagging on L4 and sentence segmentation
on CPU. The legacy Gemma vLLM serve (`modal_serve.py`) remains as a silver-data teacher
and for existing consumers; this release does not change yap's endpoint selection.

## Gemma vLLM serve (legacy teacher)

Gemma 4 31B merged with the lexide LoRA adapter, served by vLLM on an A100-80GB with
scale-to-zero (idle cost ≈ 0, which is why it stays deployed).

## Setup

1. Install Modal CLI:
```bash
pip install modal
```

2. Authenticate:
```bash
modal setup
```

3. Create HuggingFace secret:
```bash
modal secret create huggingface-secret HF_TOKEN=your_hf_token_here
```

## Usage

### Download and merge model (run once):
```bash
modal run modal/modal_serve.py --action merge
```

### Test inference:
```bash
modal run modal/modal_serve.py --action test
```

### Deploy to production:
```bash
modal deploy modal/modal_serve.py
```

## Model Configuration

- **Base model**: `google/gemma-4-31B-it`
- **LoRA adapter**: `anchpop/lexide-gemma-4-31B-it`
- **Merged model path**: `/models/merged-gemma4` (on the `lexide-models` Modal volume)
- **Endpoint**: `https://anchpop--lexide-gemma-4-31b-vllm-serve.modal.run` (also the lexide
  crate's `RemoteConfig` default; scale-to-zero). The older gemma-3-27b serve was retired
  and its HF repo deleted — only this gemma-4-31b serve remains.

---

# Parsley — joint tokenizer/tagger

`modal_serve_tagger.py` serves the 24-layer joint bge-m3 model from
`anchpop/lexide-parsley/training-runs/joint-v2-24L/best` at revision
`09a1f8b32248cc303132026ec488f4ad65085ecb`. The revision is an image env layer,
so changing it invalidates baked-weight caches.

Both public URLs are explicitly pinned:

- Tag: `https://anchpop--lexide-parsley-parsley-tag.modal.run`
- Segment: `https://anchpop--lexide-parsley-parsley-segment.modal.run`

Tag requests use an L4 with a shared batching queue (up to 256 sentences / 32k
padded characters per forward), bf16 encoder/heads and fp32 character LSTM.
Up to four containers, scale-to-zero after 300s. Segmentation uses an independent
2-CPU container with the unchanged byte-minGRU checkpoint; segment-only calls
never start a GPU.

```bash
modal deploy modal/modal_serve_tagger.py
modal run modal/verify_parsley.py
curl -X POST https://anchpop--lexide-parsley-parsley-tag.modal.run \
  -H 'content-type: application/json' \
  -d '{"sentences":["Eine Fundgrube."],"lang":"deu"}'
```

The response remains `{"results":[[{"text":...,"start":...,"end":...,"pos":...,
"lemma":...,"dep":...,"head":...}]]}`, consumed by `Lexide::from_parsley_server`.
`lang` conditions the model; it does not select a dictionary. Empty strings return
empty token lists. All 12 languages, including Japanese, use the same model.

Measured on this checkpoint (2026-10-04, `bench_parsley.py`, 16 clients × 100 sentences):
850 sentences/s on one L4 (~$0.26 per million at $0.80/h), versus 7.5/s for the
Gemma A100 serve (~$90/million). Cold start was ~45–50s. GPU bf16 can shift near-tied
labels relative to CPU fp32; Rust parity fixtures therefore come from CPU PyTorch,
not this endpoint. Release verification matched direct L4 inference on 48/48 sentences;
CPU fp32 differed on one Thai dependency head. The temporary `lexide-parsley-joint`
deployment is stopped. V1's CPU tagging chain and Wiktionary overrides are history.

On this NixOS box, if the Modal venv interpreter is broken, run its packages with
Python 3.13 using `PYTHONPATH=~/.modal-venv/lib/python3.13/site-packages`.

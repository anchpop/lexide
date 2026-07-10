# Lexide Modal Deployment

Two serves live here: the **legacy Gemma 4 31B vLLM serve** (`modal_serve.py` — replaced
online by parsley, but kept as the silver-data teacher and the Japanese fallback; see
`../train/README.md`) and **parsley 🌿**, the small CPU tagger (`modal_serve_tagger.py`,
second half of this file).

## Gemma vLLM serve (legacy — teacher + jpn fallback)

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
  crate's `RemoteConfig` default; the older gemma-3 app is still deployed too, both
  scale-to-zero)

---

# parsley 🌿 — the small CPU tagger (replaces the Gemma serve)

`modal_serve_tagger.py` serves the encoder tagger + minGRU tokenizer (`anchpop/lexide-parsley`)
on **CPU** with scale-to-zero. One forward pass per sentence, no GPU — idle cost ≈ nothing, a
warm container answers in ms. This is the online endpoint for user-submitted sentences.

Reuses the existing `huggingface-secret` and `lexide-models` volume from the Gemma serve.

**Deployed endpoint** (2026-07-10): `https://anchpop--lexide-parsley-parsley-tag.modal.run`
Point the Rust lib at it with `Lexide::from_parsley_server(url)`.

## Deploy

```bash
# from a checkout with data/lemma_tables/ populated (see tagger/LEMMA_LOOKUP.md), and `modal setup` done
modal run    modal/modal_serve_tagger.py     # smoke test (deu + eng)
modal deploy modal/modal_serve_tagger.py     # deploy the web endpoint (prints the URL)
```

## Call it

```bash
curl -X POST "$PARSLEY_URL" -H 'content-type: application/json' \
  -d '{"sentences": ["Eine Fundgrube.", "Die Katze schläft."], "lang": "deu"}'
# -> {"results": [[{"text":"Eine","pos":"DET","lemma":"ein","head":2,"dep":"det"}, ...], ...]}
```

`lang` is optional and only selects the Wiktionary lemma floor (the tagger is multilingual);
omit it for model-only lemmas. Built-in lemma tables: whatever is in `data/lemma_tables/`
at deploy time (currently all 9 served languages, with training-data candidate priors —
see `tagger/LEMMA_LOOKUP.md`). Deploys happen via `../release.sh`, which re-records the
Rust parity fixtures right after.

## Cold starts

Warm requests are ~0.4s. Cold start (after the 5-min scaledown) is **~26s**, essentially all
model-load-into-RAM. Things tried, with measured effect:

| change | cold start |
|--------|-----------|
| baseline (volume download, all tables loaded) | 26s |
| **memory snapshots** (`enable_memory_snapshot`) | **56s — reverted** |
| lazy per-language table loading | 34s |
| CPU-only torch wheel (smaller image) | **26s** (current) |

**Memory snapshots did not work here** — Modal rebuilt the snapshot on every cold start
(verified in logs: `Creating CPU memory snapshot` + the model load ran each time) rather than
restoring it, even with the model baked into the image so `snap=True` was local-only. Net
negative, so it's off. The tables load lazily (per language, on first use — one table is <0.5s)
so cold start doesn't parse all ~185MB up front, and torch is the CPU-only wheel to keep the
image small.

To actually eliminate cold starts: **`min_containers=1`** (one always-warm CPU container — the
reliable fix, at the cost of one ~$/hr container running 24/7). The deeper way to shrink the
~26s itself is the int8/ONNX model optimization below.

## Other notes

- **Cost:** `cpu=2, memory=4096`, `min_containers=0` (scale to zero = ~free when idle).
- **Not shipped:** Japanese (POS/LAS ~85/65) — gate it or keep on Gemma.
- **Model optimization (follow-up):** the encoder is fp32. ONNX-export + int8 dynamic quantization
  drops it to ~280MB and speeds CPU load/inference (also shrinks the cold-start floor).
- **Output format:** returns structured JSON. If the consumer expects the old Gemma tab-separated
  text (`idx⇥token⇥ws⇥POS⇥lemma⇥dep⇥head` with `-----`), add a formatter in `tag()`.

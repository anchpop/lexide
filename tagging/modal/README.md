# Lexide Modal Deployment

Serves the Gemma 3 27B model with LoRA adapter using vLLM on Modal.

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

- **Base model**: `google/gemma-3-27b-it`
- **LoRA adapter**: `anchpop/lexide-gemma-3-27b-it`
- **Merged model path**: `/models/merged` (on Modal volume)

---

# parsley 🌿 — the small CPU tagger (replaces the Gemma serve)

`modal_serve_tagger.py` serves the encoder tagger + minGRU tokenizer (`anchpop/lexide-tagger`)
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
(currently `deu`, `rus` — generate the rest with `tagger/parse_wiktextract.py`).

## Knobs / notes

- **Cost:** `cpu=2, memory=4096`, `min_containers=0` (scale to zero). First request after idle
  pays a ~15–30s cold start (torch + ~1GB model load); set `min_containers=1` to eliminate it
  at the cost of one always-warm CPU container.
- **Modal version:** uses `@modal.fastapi_endpoint` and `add_local_dir(..., ignore=[...])` — on
  older Modal these are `@modal.web_endpoint` and the `ignore` kwarg may differ; adjust if deploy
  complains.
- **Not shipped:** Japanese (POS/LAS ~85/65) — gate it or keep on Gemma.
- **Optimization (follow-up):** currently torch-CPU fp32. ONNX-export + int8 dynamic quantization
  drops the encoder to ~280MB and speeds CPU inference; do it once metrics are confirmed to hold.
- **Output format:** returns structured JSON. If the consumer expects the old Gemma tab-separated
  text (`idx⇥token⇥ws⇥POS⇥lemma⇥dep⇥head` with `-----`), add a formatter in `tag()`.

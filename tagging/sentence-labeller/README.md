# sentence-labeller — parsley's sentence segmenter

Builds the data for, and trains, the **byte-level sentence segmenter**: a bidirectional
minGRU over UTF-8 bytes predicting per-byte **O/B/I**, where `B` = a sentence begins here,
`I` = inside a sentence, `O` = a gap (whitespace, headings, separators *between* sentences).
Sentence char spans are recovered from each `B…I` run — the exact same shape as the token
boundary tagger (`tagger/`), just at sentence scale, so the model class, the Rust byte-minGRU
reimplementation, and the safetensors export path are all shared.

The result plugs into parsley as `Lexide::segment_sentences` (local, in-process) and the
serve `/segment` endpoint (remote): turn a document — or each entry of a list — into its
sentences, then feed each to `analyze`.

## Data

Each labelled record is `{id, lang, text, sections}` where `sections` is an ordered list of
`{"type": "sentence"|"gap", "content": "…"}` that concatenates back to `text` exactly. Sources:

| file | how | role |
|------|-----|------|
| `data/labelled-all-languages.jsonl`, `data/labelled-large.jsonl` | LLM-labelled real Harry Potter passages (`label-sentences` bin) | real signal → val/test |
| `data/synthetic-labelled.jsonl` | LLM-labelled synthetic prose (`generate` → `label-sentences`) | real signal → val/test |
| `data/mechanical-augmented.jsonl` | mechanical composition from the tokenization sentence pools with varied gaps / wrappers / leaders (`augment` bin) | synthetic → train only |

The Rust binaries (`cargo run --release --bin <name>`):
- `generate` — write original multi-paragraph passages in each language (LLM).
- `label-sentences` (default bin) — split passages into `sentence`/`gap` sections (LLM),
  validating character-exact reconstruction.
- `augment` — mechanically compose passages from `data/big/<lang>/…tokenization.jsonl`,
  wrapping sentences in quotes/brackets, prefixing list leaders, and inserting a wide variety
  of gap strings — cheap variety the LLM data doesn't cover.

`extract_samples.py` pulls the raw HP passages from the epub archive.

## Train (local GPU)

```bash
# 1. flatten sections -> per-byte sentence spans, split train/val/test
python3 sentence-labeller/sentence_data_prep.py         # -> sentence-labeller/processed/

# 2. train the byte-minGRU (0.31M params; trains on the box's GPU in minutes)
LD_LIBRARY_PATH=<gcc-lib>:/run/opengl-driver/lib \
  .venv-seg/bin/python sentence-labeller/train_segmenter.py \
    --data-dir sentence-labeller/processed --out-dir sentence-labeller/output

# 3. export weights + parity fixtures for the Rust reimplementation
.venv-seg/bin/python sentence-labeller/export_segmenter.py \
    --ckpt sentence-labeller/output/segmenter.pt --out-dir data/onnx
```

`<gcc-lib>` is a `gcc-*-lib/lib` nix-store dir with `libstdc++.so.6` (the pip torch wheel
needs it on `LD_LIBRARY_PATH` alongside the driver's `/run/opengl-driver/lib`). Training
reports sentence-span **F1** (a sentence is correct iff its exact `[start,end)` char span is
predicted) overall and per language, on the real LLM-labelled val/test.

## Release

`../release.sh` picks up `output/segmenter.pt` automatically: it exports the segmenter into
`data/onnx/`, runs the Rust bit-parity test, uploads `onnx/sentence_segmenter.safetensors`
(for the Rust backend) and `segmenter/segmenter.pt` (for the serve) to HF
`anchpop/lexide-parsley`, and redeploys the serve.

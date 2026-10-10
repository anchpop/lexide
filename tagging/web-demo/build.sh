#!/usr/bin/env bash
# Build the wasm demo: compile the crate to www/pkg and drop the byte-minGRU
# weights next to the page (the page falls back to fetching them from HF if
# they're absent, so the copy is an offline-friendliness nicety).
# Serve with: python3 -m http.server -d www
set -euo pipefail
cd "$(dirname "$0")"

direnv exec /data/coding/yap wasm-pack build --target web --out-dir www/pkg

ONNX=../data/onnx
for f in char_tokenizer.safetensors sentence_segmenter.safetensors; do
    if [[ -f "$ONNX/$f" ]]; then
        cp "$ONNX/$f" www/
    else
        echo "note: $ONNX/$f not found — the page will fetch it from HF instead" >&2
    fi
done

PRONUNCIATION=../../pronunciation/.work/onnx-run2
for pair in model.int8.onnx:pronunciation.int8.onnx frame_matrix.json:pronunciation-frame-matrix.json; do
    source=${pair%%:*}
    target=${pair#*:}
    if [[ -f "$PRONUNCIATION/$source" ]]; then
        cp "$PRONUNCIATION/$source" "www/$target"
    else
        echo "note: $PRONUNCIATION/$source not found — export run 2 before using pronunciation (no remote fallback)" >&2
    fi
done

python3 build-assets.py

echo "done — serve with: python3 -m http.server -d www"

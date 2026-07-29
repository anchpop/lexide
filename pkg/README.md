# parsley web demo — sentence segmentation + tokenization in the browser

A static page that runs the two byte-minGRU models (the sentence segmenter and
the char tokenizer, 0.99M params / 3.96 MB each) fully in-browser via WASM.
Paste a passage: it's split into sentences (gaps between them shown dropped),
and each sentence into token spans — the same `[BOS] + utf8 + [EOS]` O/B/I
pipeline as `lexide/src/local/`, reusing that crate's `byte_bio.rs` verbatim
(included by `#[path]`, so there's a single source of truth; the wasm build is
parity-tested against the Python reference fixtures).

The full tagger (POS/lemma/deps) is *not* in the demo — the XLM-R ONNX graph is
1.1 GB fp32 (~280 MB int8), which is not casual-demo territory. See
`../OVERVIEW.md`.

## Build & run

```sh
./build.sh                        # wasm-pack build + copy weights from ../data/onnx
python3 -m http.server -d www     # then open http://localhost:8000
```

The page loads the weights from its own directory, falling back to
`huggingface.co/anchpop/lexide-parsley/resolve/main/onnx/` — so the built
`www/` works even without the local artifacts (or hosted anywhere static).

`wasm-pack` comes from the yap flake (`direnv exec /data/coding/yap`); the
wasm binary is ~195 KB.

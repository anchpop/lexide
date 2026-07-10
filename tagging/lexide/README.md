# Lexide

A Rust library for multilingual NLP analysis: tokenization, POS tagging, lemmatization,
and dependency parsing for 10 languages (deu eng fra hin ita jpn kor por rus spa).

Two backends, selected by cargo feature:

- **`local`** — runs the parsley tagger in-process on CPU: a byte-level minGRU segmenter
  (pure Rust), the multi-task XLM-R encoder via ONNX Runtime (`ort`), and Wiktionary lemma
  tables in a compact `fst` format. Analyzes a sentence in tens of milliseconds, no network;
  loading is disk-bound on the 1.1 GB fp32 graph (~seconds; int8 quantization will shrink it).
- **`remote`** — an HTTP client for the Modal endpoints: the parsley CPU serve
  (`Lexide::from_parsley_server`, JSON tokens) or the legacy Gemma vLLM serve
  (`Lexide::from_server`, tab-separated completions).

Both produce identical `Tokenization`s — the local pipeline is verified token-for-token
against the parsley serve (`tests/parsley_parity.rs`).

## Usage

```toml
[dependencies]
lexide = { git = "https://github.com/anchpop/lexide.git", features = ["remote"] }
```

```rust
use lexide::{Language, Lexide};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Remote (parsley serve):
    let lexide = Lexide::from_parsley_server("https://anchpop--lexide-parsley-parsley-tag.modal.run")?;

    // Or local, with the `local` feature (reads LEXIDE_MODEL_DIR, see below):
    // let lexide = Lexide::from_pretrained(lexide::LocalConfig::default()).await?;

    let result = lexide.analyze("The cats were sleeping.", Language::English).await?;
    for token in &result.tokens {
        println!("{} [{}] lemma={} dep={} head={}",
                 token.text, token.pos, token.lemma, token.dep, token.head);
    }
    assert_eq!(result.reconstruct_text(), "The cats were sleeping.");
    Ok(())
}
```

## Local model artifacts

The `local` backend reads a directory (`LocalConfig::model_dir`, or the `LEXIDE_MODEL_DIR`
env var, default `data/onnx`) containing:

| file | what | from |
|------|------|------|
| `tagger.onnx` | XLM-R encoder + POS/lemma/biaffine heads, one graph | `tagger/export_onnx.py` |
| `tokenizer.json` | XLM-R fast tokenizer | HF `anchpop/lexide-tagger` |
| `vocab.json` | POS / dep / lemma edit-script vocabularies | HF `anchpop/lexide-tagger` |
| `char_tokenizer.safetensors` | byte-minGRU segmenter weights | `tagger/export_char_modal.py` |
| `lemma_fst/wikt_{lang}.fst` | optional per-language lemma tables | `build-lemma-fst` (below) |

The first four live on the `lexide-onnx` Modal volume
(`modal volume get lexide-onnx tagger.onnx …`). The lemma tables are built from the
Wiktionary JSON tables (`tagging/data/lemma_tables/`, see `tagger/LEMMA_LOOKUP.md`):

```bash
cargo run --release --features local --bin build-lemma-fst -- \
    --in ../data/lemma_tables --out ../data/onnx/lemma_fst
```

Multi-candidate entries are resolved at build time using training-data priors
(`wikt_priors_{lang}.json`, built by `tagger/build_lemma_priors.py`, picked up automatically
from the `--in` directory) — training's lemmatization wins over homographs like eng
`love→lofe`. Missing tables are fine — lemmas are then model-only, same as the server
without tables.

## Matching

`lexide::matching` provides `TextMatcher`, `LemmaMatcher`, `DiscontinuousLemmaMatcher`, and
`DependencyMatcher` for finding vocabulary/patterns in analyzed sentences
(see `examples/matching.rs`).

## Development

```bash
cargo test --lib --features remote                 # unit tests
cargo test --features local                        # + model-dependent tests (need artifacts;
                                                   #   they skip themselves otherwise)
cargo run --release --features local --example simple
```

The parity test (`tests/parsley_parity.rs`) replays recorded parsley responses across all
10 languages and asserts the local pipeline reproduces them exactly.

## License

MIT OR Apache-2.0

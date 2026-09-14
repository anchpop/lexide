# Lexide

**[Live demo](https://anchpop.github.io/lexide/)** — the sentence segmenter + tokenizer
running in your browser, generating the equivalent Rust as you type.

A Rust library for multilingual NLP analysis: sentence segmentation, tokenization,
POS tagging, lemmatization, and dependency parsing for 10 languages
(deu eng fra hin ita jpn kor por rus spa), plus offline pronunciation decoding,
CTC scoring, and forced alignment from model frame probabilities.

Capabilities selected by cargo feature:

- **`pronunciation`** — pure CPU frame-matrix decoding, nonblank-first phoneme
  decoding, CTC likelihood, and forced alignment. No model download, GPU,
  async runtime, or HTTP client; usable from native Rust or WebAssembly.

- **`pronunciation-remote`** — typed async HTTP client and model-identity discovery
  for the hosted phonemizer; includes `pronunciation`.

- **`segment`** — just the sentence segmenter: a 1M-param byte-level minGRU, pure Rust,
  one ~4 MB model download. The lightest entry point (see below).
- **`local`** — the full parsley tagger in-process on CPU: the byte-minGRU models, the
  multi-task XLM-R encoder via ONNX Runtime (`ort`), and Wiktionary lemma
  tables in a compact `fst` format. Analyzes a sentence in tens of milliseconds, no network;
  loading is disk-bound on the 1.1 GB fp32 graph (~seconds; int8 quantization will shrink it).
- **`remote`** — an HTTP client for the Modal endpoints: the parsley CPU serve
  (`Lexide::from_parsley_server`, JSON tokens) or the legacy Gemma vLLM serve
  (`Lexide::from_server`, tab-separated completions).

Local and remote produce identical `Tokenization`s — the local pipeline is verified
token-for-token against the parsley serve (`tests/parsley_parity.rs`).

## Pronunciation

```toml
[dependencies]
lexide = { version = "0.3", features = ["pronunciation"] }
```

Deserialize the endpoint's `frame_matrix` object into
`lexide::pronunciation::FrameMatrixPayload`, then decode and rescore locally:

```rust
use lexide::pronunciation::{FrameMatrix, FrameMatrixPayload};

fn rescore(payload: &FrameMatrixPayload, target: &[String]) -> anyhow::Result<()> {
    let matrix = FrameMatrix::decode(payload)?;
    let decoded = matrix.decode_path()?;
    for run in &decoded.runs {
        println!("{}: {}..{}", matrix.vocab[run.id], run.start_frame, run.end_frame);
    }
    println!("{:?}", matrix.score_target(target));
    Ok(())
}
```

The wire format is row-major `[T, V]`, little-endian float16, zlib + base64,
with tokenizer vocabulary labels and a blank ID. Decoding validates dimensions,
byte length, vocabulary, and log-probabilities; it preserves the original joint
probabilities for `log_likelihood` and `force_align`.

Free decoding first emits blank when its log-probability is at least `ln(0.5)`;
otherwise it picks the best eligible phone. For example, 60% nonblank probability
split 40/30/30 between phones is speech even though no individual joint phone
probability beats blank. Special labels and masked `-inf` entries are excluded.
Float16 quantization can move values near the threshold across it.

`decode_path(&[f32], frames, vocab_size, blank_id)` also accepts raw matrices,
without vocabulary-based special-label filtering. Its `DecodedPath` contains
per-frame IDs including blanks and `PhoneRun`s with **exclusive** end frames.
`FrameMatrix::decode_path` adds vocabulary filtering. `greedy_ids` returns the
collapsed phone IDs; `id` looks up exact wire labels; `log_probs` exposes unchanged
joint log-probabilities; `speech_fraction` measures nonblank frames in a range.
`force_align` retains the original **inclusive** `AlignedPhoneme::end_frame`.
`score_target` reports unknown/special target tokens in `oov` and scores the rest;
impossible or empty targets have no likelihood. All payload/score/path types
support serde serialization.

`half`, `flate2`, and `base64` are optional pronunciation dependencies. `serde`
remains a core dependency because the existing text types already require it;
the text API remains available without features. Tokio is enabled only by
`local`/`remote`, and reqwest by `remote`/`pronunciation-remote`.

### Hosted phonemizer

`pronunciation` also exports the serde wire schema: `ModelIdentity`,
`PredictRequest`, `EmittedPhoneme`, `PredictResponse`, and `BatchResponse`, with
typed alternatives, diagnostic frames, target scores (including all-OOV errors),
and per-item batch errors. Optional fields tolerate older endpoints; unknown
server fields are ignored. Single and batch envelopes expose optional `model_id`,
`model_revision`, `decoder_version`, and `deploy_marker`, so newer deployments
can describe themselves on each inference response.

Enable `pronunciation-remote` for `pronunciation::remote::PhonemizerClient`:

```rust,no_run
use lexide::pronunciation::{cache_version, PredictRequest};
use lexide::pronunciation::remote::PhonemizerClient;

async fn predict(audio: Vec<f32>) -> anyhow::Result<()> {
    let client = PhonemizerClient::new(
        "https://anchpop--wav2vec2-phoneme-wav2vec2phoneme-predict.modal.run",
    )?;
    let identity = client.identity().await?;
    let version = cache_version(&identity);
    let response = client.predict(&PredictRequest {
        audio, return_frame_matrix: true, ..Default::default()
    }).await?;
    println!("{version}: {:?}", response.phonemes);
    Ok(())
}
```

Audio is raw mono samples; defaults match the endpoint (16 kHz, top-k 3).
`identity()` uses the deployed protocol: **POST `{"marker_only": true}` to the
predict URL**, not a nonexistent GET health route. `check_identity(expected)`
returns that identity only if its deploy marker matches. This is a one-shot
probe, not a guarantee about later containers: validate each prediction's marker
(or the batch envelope's marker) before caching.

`predict_batch(&[PredictRequest])` sends 1–64 requests and returns ordered
`BatchResult::Prediction` / `BatchResult::Error` entries. The batch marker remains
on the envelope. Modal's batch URL is derived from the `-predict.modal.run`
suffix; use `with_endpoints(http_client, predict_url, batch_url)` for custom URLs,
authentication or timeouts. There is no caching or retry layer. Native callers
supply a Tokio runtime for reqwest; this feature does not enable lexide's optional
Tokio dependency or its text client.

`cache_version(&identity)` yields
`<model_id with '/' replaced by '_'>@<first 12 revision chars>__nonblank_v1`, e.g.
`anchpop_lexide-pronunciation@edcbbbf43a7f__nonblank_v1`. It uses the crate's
`DECODER_VERSION`, not the optional server-reported decoder version; callers
using server-decoded predictions should check that version when present.
**Any decoder change must bump `DECODER_VERSION`.** Caching stays with the caller.
The schema and offline decoder remain WebAssembly-friendly without the remote
feature.

## Sentence segmentation

```bash
cargo add lexide --features segment
```

```rust
let parsley = lexide::Segmenter::from_pretrained()?; // ~4 MB download, cached
assert_eq!(
    parsley.segment_in(
        "Dr. Smith arrived at 3 p.m. — he wasn't late. \"Is this the place?\" she asked.",
        lexide::Language::English,
    ),
    vec![
        "Dr. Smith arrived at 3 p.m. — he wasn't late.",
        "\"Is this the place?\" she asked.",
    ],
);
```

Gaps between sentences (whitespace, headings, separators) are dropped; punctuation that
*frames* a sentence (its quotes, a leading dialogue dash) stays attached.
`segment` skips the language hint, `segment_detailed` also returns each sentence's
`[start, end)` char span. See `examples/segment.rs`. (On the remote backend, use the async
`RemoteClient::segment_sentences` against a parsley `/segment` endpoint.)

## Tagging

```toml
[dependencies]
lexide = { version = "0.3", features = ["remote"] }
```

```rust
use lexide::{Language, Lexide};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Remote (parsley serve):
    let lexide = Lexide::from_parsley_server("https://anchpop--lexide-parsley-parsley-tag.modal.run")?;

    // Or local, with the `local` feature — downloads the model from HF on first use
    // (~1.2 GB into the standard HF cache), no setup needed:
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

`analyze` expects one sentence — use `Segmenter` (above) to turn documents into sentences
first.

## Local model artifacts

By default the `local` backend downloads everything it needs from HF
`anchpop/lexide-parsley/onnx/` into the standard HuggingFace cache on first load — no
setup. To use a local directory instead (offline, or artifacts you built yourself), set
`LocalConfig::model_dir` or the `LEXIDE_MODEL_DIR` env var to a directory containing:

| file | what | built by |
|------|------|----------|
| `tagger.onnx` | XLM-R encoder + POS/lemma/biaffine heads, one graph | `tagger/export_onnx.py` |
| `tokenizer.json` | XLM-R fast tokenizer | (from tagger training) |
| `vocab.json` | POS / dep / lemma edit-script vocabularies | (from tagger training) |
| `char_tokenizer.safetensors` | byte-minGRU token boundary tagger weights | `tagger/export_char_modal.py` |
| `sentence_segmenter.safetensors` | byte-minGRU sentence segmenter weights (optional) | `sentence-labeller/export_segmenter.py` |
| `lemma_fst/wikt_{lang}.fst` | optional per-language lemma tables | `build-lemma-fst` (below) |

To fetch that manually rather than through the crate:

```bash
hf download anchpop/lexide-parsley --include "onnx/*" --local-dir . && export LEXIDE_MODEL_DIR=./onnx
```

(also mirrored on the `lexide-onnx` Modal volume: `modal volume get lexide-onnx …`).
To rebuild the lemma tables from the Wiktionary JSON (`tagging/data/lemma_tables/`,
see `tagger/LEMMA_LOOKUP.md`):

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

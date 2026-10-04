# Lexide

**[Live demo](https://anchpop.github.io/lexide/)** — the sentence segmenter + tokenizer
running in your browser, generating the equivalent Rust as you type.

A Rust library for multilingual NLP analysis: sentence segmentation, tokenization,
POS tagging, lemmatization, and dependency parsing for 12 languages
(deu eng fra hin ita jpn kor por rus spa tha zho-hans), plus offline pronunciation decoding,
CTC scoring, and forced alignment from model frame probabilities.

Capabilities selected by cargo feature:

- **`pronunciation`** — pure CPU frame-matrix decoding, nonblank-first phoneme
  decoding, CTC likelihood, and forced alignment. No model download, GPU,
  async runtime, or HTTP client; usable from native Rust or WebAssembly.

- **`pronunciation-remote`** — typed async HTTP client and model-identity discovery
  for the hosted phonemizer; includes `pronunciation`.

- **`segment`** — just the sentence segmenter: a 1M-param byte-level minGRU, pure Rust,
  one ~4 MB model download. The lightest entry point (see below).
- **`local`** — joint parsley in-process on CPU: 24-layer bge-m3, character BiLSTM,
  boundary and word heads via fp32 ONNX Runtime. About 2.4 GB of model artifacts;
  no separate tokenizer model, boundary priors or lemma dictionaries.
- **`remote`** — an HTTP client for the Modal endpoints: the parsley L4 serve
  (`Lexide::from_parsley_server`, JSON tokens) or the legacy Gemma vLLM serve
  (`Lexide::from_server`, tab-separated completions).

Local inference is verified token-for-token against **CPU fp32 PyTorch** on 293
multilingual fixtures (`tests/parsley_parity.rs`). The GPU serve uses bf16, so near-tied
labels can differ. Version 0.5.0 changes the local backend; it is not yet on crates.io.
Until publication, use a path dependency on this checkout instead of the version below.

## Pronunciation

```toml
[dependencies]
lexide = { version = "0.5", features = ["pronunciation"] }
```

Deserialize the endpoint's `frame_matrix` object into
`lexide::pronunciation::FrameMatrixPayload`, then decode and rescore locally:

```rust
use lexide::pronunciation::{FrameMatrix, FrameMatrixPayload, Phoneme};

fn rescore(payload: &FrameMatrixPayload, target: &[Phoneme]) -> anyhow::Result<()> {
    let matrix = FrameMatrix::decode(payload)?;
    let decoded = matrix.decode_path()?;
    for run in &decoded.runs {
        println!("{}: {}..{}", matrix.vocab[run.id], run.start_frame, run.end_frame);
    }
    println!("{:?}", matrix.score_target(target));
    Ok(())
}
```

Scoring uses shared `g2p_types::Phoneme` values. `Phonemized` carries a
`Vec<Phoneme>` plus word boundaries and optional prosody; both are reexported here.
Pass g2p's output directly, or import existing tokenized dictionary IPA with
`Phonemized::from_ipa_tokens("b ɔ̃ ʒ u ʁ | m a d a m")?`.
This conversion rejects unknown tokens. `Phonemized::from_words` assembles
already-typed words without a string round trip.

`PredictResponse::score(&[Phonemized], Option<Language>)` compares accepted
readings by normalized edit distance and returns the closest reading, alignment,
error ratio and missing-word diagnostic. `None` selects generic normalization;
`Some(Language::French)` or `Some(Language::German)` also applies that language's
comparison rules. Ties preserve input order, an empty candidate list returns
`None`, and malformed word spans return an error. `score.failure_reason(threshold)`
applies the empty-output, missing-word and mismatch gates.

`FrameMatrix::score_target(&[Phoneme])` uses exact supplied tokens for CTC.
`FrameMatrix::align_segments(&[Phonemized])` returns half-open frame ranges.
These methods currently score segmental phones, not stress/tone/pitch. Lexide
never calls a g2p engine; targets do not trigger model/build identity checks.

`FrameMatrix::phonemes()` and `PredictResponse::phonemes()` return
`Result<Vec<Phoneme>>`. They reject unsupported emitted labels rather than
choosing a different sound. The response accessor separates the service's stress
prefix from the segmental token; the original wire response still retains it.
Raw payloads and response DTOs intentionally retain strings, control tokens and
future fields. The stored raw response remains the lossless source of truth.
Alignment operations and comparison results contain typed phones and serialize
as the same IPA strings as before. Remote `PredictRequest::target_phonemes` is
also typed. Model vocabulary indices remain checkpoint-specific.

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
`score_target` reports recognized phonemes absent from this checkpoint in `oov`
and scores the rest;
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
        return_frame_matrix: true, ..PredictRequest::from_samples(&audio)
    }).await?;
    println!("{version}: {:?}", response.phonemes);
    Ok(())
}
```

`PredictRequest::from_samples` encodes mono samples as standard base64 of
little-endian float32 bytes in `audio_f32_b64`; defaults match the endpoint
(16 kHz, top-k 3). Override `sample_rate` for other source rates.
`identity()` uses the deployed protocol: **POST `{"marker_only": true}` to the
predict URL**, not a nonexistent GET health route. It rejects any response
containing `load_error`, even alongside valid model fields, and includes the
error value in its diagnostic. `check_identity(expected)`
returns that identity only if its deploy marker matches. This is a one-shot
probe, not a guarantee about later containers. Configure
`with_expected_deploy_marker(expected)` to check every live response before caching.

`predict_many(Vec<AudioClip<Id>>)` accepts any number of clips and streams
`(Id, Result<RawPrediction>)` as work completes. Give each clip an ID, a duration
hint, and `AudioInput::File(path)` or `AudioInput::Bytes(encoded_audio)`.
The ID need not be unique; it is carried through unchanged. Files are opened
lazily. Keep them alive until their results arrive. ffmpeg decodes file/byte
inputs to mono 16 kHz, with symmetric padding to 0.6 seconds for short clips.
These inputs request top-k 10, frame matrices and all heads. For custom request
options or already-encoded float32 samples, use `AudioInput::Request`.

```rust,no_run
# async fn example(client: &lexide::pronunciation::remote::PhonemizerClient) -> anyhow::Result<()> {
use futures::StreamExt;
use lexide::pronunciation::remote::{AudioClip, AudioInput};
use std::time::Duration;

let results = client.predict_many(vec![AudioClip {
    id: "clip-1",
    cache_context: Some("expected phoneme sequence".into()),
    duration: Duration::from_secs(3),
    audio: AudioInput::File("clip-1.wav".into()),
}]);
futures::pin_mut!(results);
while let Some((id, response)) = results.next().await {
    println!("{id}: {:?}", response?.decode()?.phonemes);
}
# Ok(()) }
```

The client sorts by duration, prepares up to eight clips at a time, and sends
up to two requests concurrently, with at most 64 successfully prepared clips
per request. Invalid files fail individually. Failed HTTP batches retry up to
five attempts with 5/10/15/20-second delays for transport/decode failures and
408/425/429/500/502/503/504 statuses. Other statuses and per-item failures are
not retried. Results retain unknown item and envelope fields for caching.

For concurrent individual callers, `predict_audio(AudioInput, Option<&str>)` coalesces work
for 200 ms through a bounded queue shared by client clones. Both paths share
the HTTP concurrency limit. Dropping a batch stream stops scheduling further
clips; already-started blocking audio decodes may finish. Dropping all client
clones lets the individual-call worker drain its queue and stop.

The low-level `predict_batch` and `predict_batch_raw` methods expose one HTTP
request for protocol-level callers; the arbitrary-sized API owns retries and
scheduling. `with_activity(Arc<RequestActivity>)` reports actual HTTP attempts,
retries and time without an active request. It does not measure GPU utilization.

Modal's batch URL is derived from the `-predict.modal.run` suffix; use
`with_endpoints(http_client, predict_url, batch_url)` for custom URLs or
`with_http_client` for authentication/timeouts. Native callers supply a Tokio
runtime.

Configure caching on the client, following tysm's builder pattern:

```rust,ignore
let client = PhonemizerClient::new(endpoint)?
    .with_cache_directory(".cache"); // or .with_cache(shared_osmo_store)
let offline = client.clone().with_cached_only();
let refresh = client.clone().with_cache_policy(CachePolicy::Refresh);
```

Caching is opt-in. Lexide hashes encoded audio contents automatically: identical
file contents and `Bytes` share an entry, independent of filename. Files are
hashed with a bounded streaming buffer. `AudioInput::Request` hashes the entire
prepared request, including inference options, in a separate namespace.

The optional `cache_context` argument/field adds identity, such as expected
phonemes or an explicit cache-busting value. `None` caches by audio alone;
context never replaces the audio identity. Lexide adds no model, version, or
endpoint to the key. A valid hit skips ffmpeg, identity probes and inference,
but file inputs must still be readable to compute their identity. Missing or malformed
entries are inferred and replaced in normal read-through mode. `CachedOnly`
returns a miss error without preparation, identity probes, or inference;
`Refresh` deliberately ignores hits and writes successful new results.

The cached value is `RawPrediction`, preserving unknown item/envelope fields.
Live responses are validated and must contain a decodable frame matrix before
being stored. `cached(key)` supports inspection/export without audio or network
and distinguishes missing from malformed entries. `audio_cache_key(hash, context)`
reconstructs an encoded-audio key for exports using an already recorded XXH3 hash;
normal prediction callers never need to build keys. `with_identity_check()` lazily
probes once on the first miss; explicit evaluation identities/markers can be set
with `with_expected_identity` and `with_expected_deploy_marker`. These validate
live responses, never invalidate historical cache hits. Low-level protocol methods
(`predict`, `predict_raw`, `predict_batch`, `predict_batch_raw`, `identity`) bypass
the high-level cache policy. osmo sync remains an explicit caller operation.

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
lexide = { version = "0.5", features = ["remote"] }
```

```rust
use lexide::{Language, Lexide};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Remote (parsley serve):
    let lexide = Lexide::from_parsley_server("https://anchpop--lexide-parsley-parsley-tag.modal.run")?;

    // Or local, with the `local` feature — downloads the model from HF on first use
    // (~2.4 GB into the standard HF cache), no setup needed:
    // let lexide = Lexide::from_pretrained(lexide::LocalConfig::default()).await?;

    let result = lexide.analyze("The cats were sleeping.", Language::English).await?;
    for token in result.tokens() {
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

The `local` backend downloads `anchpop/lexide-parsley/joint/` at the immutable
`MODEL_REVISION` in [`src/local/mod.rs`](src/local/mod.rs) into the HF cache. Override with
`LocalConfig::model_dir` or `LEXIDE_MODEL_DIR`. Required files:

- `encoder.onnx` and `encoder.onnx.data` — bge-m3, character LSTM, boundary head.
- `heads.onnx` — POS, lemma edit-script, dependency arc/relation heads.
- `tokenizer.json`, `vocab.json`, `config.json` — matching tokenizer and metadata.

All are produced by `tagger/export_joint_onnx.py`. `LocalConfig::threads` controls
ONNX intra-op parallelism. Inputs must fit 8192 subwords; passages should be segmented.
The independent segmenter still downloads `onnx/sentence_segmenter.safetensors`.
The rest of `onnx/` is preserved for 0.4.x clients, not used by 0.5 local.

Measured on a Ryzen 9 3900X (293 multilingual length-spread sentences, warm filesystem
cache, fp32; no other evaluation jobs running):

| Intra-op threads | Load | Warm sentences/s | RSS |
|---|---:|---:|---:|
| 1 | 2.49 s | 5.97 | 2.20 GiB |
| 4 | 2.44 s | 10.64 | 2.20 GiB |
| 12 | 2.21 s | 12.15 | 2.20 GiB |

The full 12k test predictions exactly match CPU PyTorch fp32 and Python ORT:
macro token/POS/lemma/UAS/LAS F1 = 99.14 / 97.87 / 98.11 / 87.36 / 85.11.

```bash
# Uses the pinned HF artifacts unless LEXIDE_MODEL_DIR is set.
cargo run --release --features local --example bench_local -- 1
cargo run --release --features local --example bench_local -- 4
cargo run --release --features local --example tag_jsonl < ../data/processed-joint/test.jsonl
```

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

## Standalone tokenization types

Use `lexide-types = "0.1"` for `Text`, `Lemma`, `LemmaPos`, `Token`,
`Tokenization`, `PartOfSpeech`/`pos`, and `DependencyRelation`/`dep` without
inference, async, or networking dependencies. `lexide` reexports these types.
The optional `lexide-types/rkyv` feature archives only `Whitespace`.

`Tokenization::new(sentence, tokens)` validates nonempty, whitespace-free token
text and exact reconstruction of the original sentence. JSON carries both
`sentence` and `tokens`, and deserialization applies the same validation.
Read via `sentence()` and `tokens()`, or consume via `into_tokens()` / `into_parts()`.

`Token::whitespace` is a `Whitespace`: `None` (`""`), `Space` (`" "`), `Nbsp`
(U+00A0), or `NarrowNbsp` (U+202F). It serializes as the literal string, not the
variant name. Other gaps (including repeated spaces or skipped punctuation)
are errors; punctuation must be represented in token text, not hidden in a gap.
Gemma reconstruction repair runs before validation of the final tokens.

## License

MIT OR Apache-2.0

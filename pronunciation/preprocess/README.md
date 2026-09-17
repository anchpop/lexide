# Pronunciation dataset pipeline

One Rust crate drives the existing raw manifests through audits, training labels,
acoustic processing, packing and Hugging Face upload. It does not acquire audio.
Run commands from `pronunciation/` (or use an absolute manifest path).

```sh
# Inspect the full stage order without API calls or writes.
cargo run --release --manifest-path preprocess/Cargo.toml -- run --dry-run

# Prepare and upload. Until the aligner is retrained, skip narrowing.
cargo run --release --manifest-path preprocess/Cargo.toml -- run --skip-narrowing \
  --python /opt/homebrew/Caskroom/miniconda/base/bin/python3
```

`run` executes **audit → French stress → language filter → labels → VAD → speaker
clusters → measurements → narrowing → exclusions/packing → upload**. Each stage
must succeed before the next starts. `--skip-narrowing` skips both measurement and
narrowing and preserves existing narrowed files; training configured to use those
files continues to use their previous labels. `--skip-upload` prepares locally.
There are also `--skip-audit`, `--skip-stress`, `--skip-filter`, `--skip-vad`,
`--skip-speaker-cluster` and `--no-pack` flags for intentionally partial runs.

Individual stages use the same implementation:

```sh
cargo run --release --manifest-path preprocess/Cargo.toml -- audit --sources fleurs tatoeba tts
cargo run --release --manifest-path preprocess/Cargo.toml -- audit --sources fleurs --rescore
cargo run --release --manifest-path preprocess/Cargo.toml -- stress
cargo run --release --manifest-path preprocess/Cargo.toml -- filter --langs eng spa
cargo run --release --manifest-path preprocess/Cargo.toml -- labels --langs eng fra
cargo run --release --manifest-path preprocess/Cargo.toml -- vad --langs eng fra
cargo run --release --manifest-path preprocess/Cargo.toml -- speakers
cargo run --release --manifest-path preprocess/Cargo.toml -- measure
cargo run --release --manifest-path preprocess/Cargo.toml -- narrow
cargo run --release --manifest-path preprocess/Cargo.toml -- pack
cargo run --release --manifest-path preprocess/Cargo.toml -- upload
```

The full run audits FLEURS and Tatoeba by default, matching the old shell script;
`--sources` also supports TTS and Kathbath. ASR requests use Groq Whisper with the
intended language forced where supported. `--detect-language` opts out.
Transcripts are cached under `data/audio/.cache/asr`, keyed by audio content,
model and request settings. G2P labels and audit scores are recomputed locally.
Changing a sentence, dialect or g2p build therefore does not resend unchanged
audio. Completed requests survive a later request or scoring failure.

Older audit JSONL files can be rescored with `audit --rescore` without Groq calls;
this uses their saved transcripts with the current manifest text and dialect.
It requires a saved successful transcript for each selected recording. Old rows
lack an audio-content fingerprint, so a normal audit does not silently promote
them into the new transcript cache: its first pass on uncached audio calls Groq.
No production pass or cache migration is performed by building this crate.

Rust owns the HTTP audit, g2p calls, dialect selection, scoring, French stress,
language filtering and VAD. Python retains audio filtering, training-schema
adaptation, stress overrides, supervision masks, acoustic DSP, speaker embeddings
and clustering, and Hugging Face's resumable uploader. The old shell/Python
pipeline drivers and standalone stress/filter/VAD binaries are removed. The Modal
services stay Python and are unchanged. Deployment is explicit:

```sh
cargo run --release --manifest-path preprocess/Cargo.toml -- deploy-aligner
```

The current aligner's merged-label refusal guard applies before measurements and
narrowing. Neither `run` nor `measure` deploys automatically. Acoustic measurement
and speaker caches are tied to the canonical corpus directory; those stages
reject a different `--data-dir` rather than mix corpora.

Explicit g2p refusals exclude a label row, or omit phoneme error when an audit
reference cannot be labeled. An unlabelable ASR transcript against a valid
reference remains a mismatch. Engine/infrastructure errors fail the stage.
Vocabulary errors preserve prior label output, incomplete VAD preserves prior
VAD output, and failed LLM/audit stages preserve their prior sidecars. Language
subsets retain exclusions for other languages. Packing always includes the whole
corpus, even with `--langs`; caches are excluded from both tarballs and uploads.

Use `--jobs N` for parallel local language jobs and `--asr-workers N` for ASR
concurrency. Label-job logs go under `.work/preprocess_parallel`. LLM stages keep
the existing `tysm` caches under `train/{lang-filter,relabel-french}/.cache`.
Root `.env` is loaded first, then `pronunciation/.env` overrides it.
`--env-file PATH` selects a single explicit env file instead. Only stages
that need them require `GROQ_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN` or Modal auth.
The Rust build requires cmake and a C compiler for g2p; some engines need `uv`.

Python defaults to `python3`; on NixOS pass `--python scripts/py-linux.sh`. Use the
existing environment with numpy/soundfile/tqdm and the acoustic, Modal, sklearn
and Hugging Face dependencies for the stages you run. Keep the crate in the
checkout so it can locate its helper scripts.

For a local scratch-data run without any services:

```sh
cargo run --manifest-path preprocess/Cargo.toml -- run \
  --data-dir /tmp/audio-sample --train-dir /tmp/sample-sidecars \
  --output-tar /tmp/sample.tar --jobs 2 --python /path/to/python3 \
  --skip-audit --skip-stress --skip-filter --skip-speaker-cluster \
  --skip-narrowing --skip-upload
```

Historical engine-comparison tools still use the Python g2p client, tracked in
YAP-26; neither production labels nor ASR audits use that transport.

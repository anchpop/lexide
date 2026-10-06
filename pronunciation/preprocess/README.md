# Pronunciation dataset pipeline

One Rust crate drives the existing raw manifests through audits, training labels,
acoustic processing, packing and Hugging Face upload. It does not acquire audio.
Run commands from `pronunciation/` (or use an absolute manifest path).

```sh
# Inspect the full stage order without API calls or writes.
cargo run --release --manifest-path preprocess/Cargo.toml -- run --dry-run

# Prepare and upload. Narrowing is off until it is regenerated against the current labels.
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
`--sources` also supports TTS, Kathbath, MLS, CV, AISHELL-1 and AISHELL-3
(`tts kathbath mls cv aishell1 aishell3`). ASR defaults to Groq Whisper with the
intended language forced where supported. `--asr-backend cloudflare` explicitly
selects Workers AI `@cf/openai/whisper-large-v3-turbo`; `--asr-model` overrides the
chosen provider's model ID. `--detect-language` omits the language hint.
Cloudflare sends base64 audio, `task: transcribe` and optional `language`, without
undocumented temperature parameters or transcript hints. Credentials are
`CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN` in `pronunciation/.env` (the API
token needs Workers AI Read/Edit). No credential values are printed.
Cloudflare results use separate provider/request cache keys; the original Groq
cache keys stay unchanged. Audit rows record `asr_backend`; offline rescoring
retains the original provider rather than relabeling an old Groq transcript.
Cloudflare requests are paced at 600/minute per process, below the documented
720/minute ASR task limit; shared account traffic may still cause 429 retries.
`Retry-After` remains honored. Missing segment statistics remain null, not made up.

`--limit N` caps **total selected clips globally** across sources and languages,
including cached clips—not N per language. Selection follows the given source
order, sorted languages and manifest row order. Limited runs replace only the
selected audit rows and retain all other sidecar rows; they do not certify full
strict-source coverage.

```sh
# PAID, only after authorization: ten selected German MLS clips.
cargo run --release --manifest-path preprocess/Cargo.toml -- audit \
  --asr-backend cloudflare --sources mls --langs deu --limit 10
# PAID: all three newly quarantined source collections. No labels/VAD here.
cargo run --release --manifest-path preprocess/Cargo.toml -- audit \
  --asr-backend cloudflare --sources mls aishell1 aishell3
```

Strict sources always compare g2p phone sequences, even if `--text-only` is
supplied. That flag skips phoneme metrics only for legacy sources. The phonetic
comparison is local and does not add paid inference requests.

Official Cloudflare references:
[model and schema](https://developers.cloudflare.com/workers-ai/models/whisper-large-v3-turbo/),
[pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/),
[limits](https://developers.cloudflare.com/workers-ai/platform/limits/).
Checked 2026-10-05: the model page quotes $0.000513/audio minute; the pricing
table rounds this to $0.0005 and specifies 46.63 neurons/minute. At
$0.011/1,000 neurons, estimate $0.0307758/audio hour before the shared daily
10,000-neuron free allocation. Minimum-duration/rounding rules are not documented
there, so duration-based estimates are not billing guarantees.

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
and clustering, and Hugging Face's resumable uploader. Before g2p, prepare skips
clips in the training exclusion sidecars when the audited sentence hash still
matches, as well as silent and (unless allowed) noncommercial recordings. The old
shell/Python pipeline drivers and standalone stress/filter/VAD binaries are
removed. The Modal services stay Python and are unchanged. Acoustic measurement
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

All phonemization callers use the g2p Rust library; no g2p executable is needed.

### Strict read-speech quarantine

`mls`, `cv`, `aishell1` and `aishell3` require successful current audit coverage
before preparation or training. Admission uses `phone_match` with
`phone_match_version: 1`: phonemize reference and saved Whisper text with the
same g2p language/variety resolver as labels, then compare **only the phone
sequences**, ignoring stress/tone/pitch factors. Homophones, digits, spelling and
accent differences are allowed if and only if those phone sequences match.
A g2p refusal on either side (including an unrepresentable phone) rejects; engine
or infrastructure errors fail the audit. `phone_match_reason` distinguishes
`identical_phones`, `phone_disagreement` and `g2p_refusal`, with side-specific
refusal details. Existing sources retain their prior PER/CER/WER thresholds.

`word_match` / `word_match_version: 1` remain diagnostics only: exact words after
NFC/lowercase and punctuation/whitespace boundaries, with no spelling repairs.
They do not decide strict admission.

Missing, failed, word-only or sentence-hash-stale audits stop the selected
preparation scope before audio processing. Raw manifest rows must also match
`g2p_selection` (original `g2p_language`, `variety`, `espeak_voice` inputs); finalized
labels must match the audit's resolved `g2p_language` and `g2p_identity`. This
prevents a same-text dialect or label-build change from reusing stale coverage.
Training independently checks all strict-source label rows against explicitly
supplied `--audit-path` files; `train.sh` includes all four source sidecars when
present. Downloading a manifest does not admit it to training. Old labels/VAD are
never regenerated by acquisition.

```sh
# PAID: obtain missing ASR transcripts, then locally score the strict gate.
cargo run --release --manifest-path preprocess/Cargo.toml -- audit \
  --sources mls cv aishell1 aishell3
# FREE only when every selected clip already has a saved successful transcript:
cargo run --release --manifest-path preprocess/Cargo.toml -- audit \
  --sources mls --rescore
```

### Incremental labels and VAD

`--new-only` selects files absent from the canonical output. `--label-sources`
restricts sources and implies new-only for **both** labels and VAD. These flags do
not restrict the audit or LLM stages; `--sources` remains the separate audit flag.
Use standalone stages for an expansion, not a whole `run` that packs old labels.

```sh
cargo run --release --manifest-path preprocess/Cargo.toml -- labels \
  --langs zho-hans --label-sources aishell1 aishell3 \
  --python scripts/py-linux.sh
cargo run --release --manifest-path preprocess/Cargo.toml -- vad \
  --langs zho-hans --label-sources aishell1 aishell3
# Or all sources, only missing files:
cargo run --release --manifest-path preprocess/Cargo.toml -- labels --new-only \
  --python scripts/py-linux.sh
```

Incremental writes atomically copy the original bytes and append new JSONL rows.
Existing rows are not reserialized or replaced, even if their text has changed;
use an intentional full rebuild to change labels. A file with no final newline is
rejected rather than silently altering its last row. Strict coverage is required
for selected new rows; training still validates coverage for all strict rows.
Old and new label builds can coexist in append mode; each row retains its own
`g2p_identity`. Incremental VAD only examines canonical labeled clips.

### Bulk LLM stages: OpenAI Batch

French stress and language filtering use **gpt-6-luna**, via the shared Batch
transport. No uncached synchronous Chat Completions requests remain. Existing
`tysm` caches are read in cached-only mode, with unchanged model/prompts/schema;
new results and job state live in a `batch/` child of the same cache directory.
Using `--train-dir` also relocates these caches, which is useful for scratch tests.

```sh
# Offline: builds deduplicated request JSONL without a key or network requests.
cargo run --release --manifest-path preprocess/Cargo.toml -- filter \
  --langs eng --batch-dry-run
cargo run --release --manifest-path preprocess/Cargo.toml -- stress --batch-dry-run
# PAID: same commands without --batch-dry-run upload, submit, poll and ingest.
```

JSONL chunks stay below 50,000 requests / 200 MB. Responses are joined by
`custom_id`, validated, and cached before sidecars are replaced. Batch IDs are
persisted in `*.state.json`; rerun the same command to resume. Prior un-ingested
jobs are recovered before constructing new batches, avoiding duplicate paid
requests after an interrupted ingestion. `*.result.json` and downloaded output /
error JSONL retain terminal status and per-request errors.

A process/network failure during submission can leave `submitting: true` without
a Batch ID. This intentionally stops instead of paying twice: inspect the OpenAI
Batch dashboard for metadata `lexide_input_hash`, then insert its `batch_id` into
that state file. If no job was created, remove the state file and retry. Failed,
expired or cancelled jobs also stop loudly; inspect their error/output files and
resolve the cause before explicitly retrying. Do not delete a running job's state.
`--batch-dry-run` is accepted only for standalone `filter` / `stress`, preventing
an ostensibly offline full run from reaching another paid stage.

Official references: [model](https://developers.openai.com/api/docs/models/gpt-6-luna),
[Batch guide](https://developers.openai.com/api/docs/guides/batch),
[pricing](https://developers.openai.com/api/docs/pricing). The published Batch
rates checked 2026-10-05 are $0.05 input / $0.25 output per 1M tokens for requests
up to 272K input tokens; cached input is $0.005. These short-text stages are below
that tier. Estimate input/output token counts separately; ASR is billed separately.

### Acquire public / authorized read-speech archives

From `pronunciation/` on NixOS (these commands do **not** audit or label audio):

```sh
scripts/py-linux.sh data/download_aishell.py --source aishell1 \
  --cache-dir /path/with/150GiB-free/aishell-cache
scripts/py-linux.sh data/download_aishell.py --source aishell3 \
  --cache-dir /path/with/150GiB-free/aishell-cache
# Reuse a complete official download with --archive /path/data_aishell3.tgz.

scripts/py-linux.sh data/download_cv.py --lang eng \
  --archive /path/authorized-cv-en.tar.gz --cache-dir /path/cv-cache \
  --dataset-url https://mozilladatacollective.com/datasets/SELECTED-DATASET \
  --license CC0-1.0
```

AISHELL uses [SLR33](https://www.openslr.org/33/) and
[SLR93](https://www.openslr.org/93/), listed as Apache-2.0. SLR33 additionally says
“free for academic use”; retain and review the upstream terms before commercial
redistribution. All audio with supplied transcripts is retained, including
upstream train/dev/test splits (not an AISHELL benchmark evaluation). Missing
transcripts are reported, never guessed. AISHELL-1 transcript segmentation spaces
are removed to form Hanzi sentences and retained as `original_transcript`.
AISHELL-3 supplies `annotated_pinyin` and `original_annotation` provenance.
Original speaker IDs are namespaced in `voice`. Audio becomes 16-kHz mono PCM16;
AISHELL-3's supplied audio is 44.1 kHz. Manifest prefix SHA256, resumable appends,
atomic audio writes and header spotchecks guard acquisition. Cached extraction
and archives remain local; budget disk accordingly.

Common Voice requires **your own Mozilla Data Collective authorization**:

1. [Sign up](https://mozilladatacollective.com/auth/signup) or sign in.
2. Select Common Voice **scripted speech**, release and language under
   [datasets](https://mozilladatacollective.com/datasets).
3. Read and accept that dataset's terms in the website; complete any required
   approval. Do not assume all MDC datasets share the same license.
4. Download the authorized archive through the site and pass it to the importer.
   Optional authorized API downloads require a key from profile credentials;
   API-only terms acceptance is not supported. See the
   [official API docs](https://mozilladatacollective.com/api-reference/docs).

The downloader never bypasses those gates or obtains somebody else's signed URL.
Targets are 15h per `ara ces dan deu eng fas fra hin ita jpn kor por rus spa tha
zho-hans` (the last maps to `zh-CN`), at most 15min per `client_id`, and at most
16s per clip. Only `validated.tsv` rows with up_votes≥2 / down_votes=0 survive.
Blank or explicitly recognized native-region accents are retained; explicit
nonnative and unknown nonempty accents are dropped. The conservative exact
region allowlist can underfill a target; inspect the per-language metadata
coverage report rather than relaxing votes or contributor limits. Voices are
`cv:<client_id>`. Repeated imports count existing WAV durations against budgets.

### Shared phoneme inventory

Rust consumes `g2p::Phonemized` with `Vec<g2p::Phoneme>` throughout label generation
and ASR comparisons. JSON still contains the exact IPA spellings expected by
Python. After Python finalization and acoustic narrowing, Rust validates each
output row against that same enum before later stages proceed. Unsupported
labels fail with the file and line number; they are never coerced to `<unk>`.
Raw historical files are not rewritten by this API change.

Fresh-model vocabulary and relabeling requirements are documented in [../train/VOCABULARY.md](../train/VOCABULARY.md). Validation rejects controls, legacy artifacts, and phones outside the model subset; it does not silently rewrite or drop them.

<!-- Moved from yap/modal-envs/PRONUNCIATION_BATCHING.md at a4b4e8f8c0c784a38be0fbdcf9adf966663db69a. -->

# Explicit pronunciation batch API

`Wav2Vec2Phoneme.predict_batch` accepts one HTTP POST containing **1–64 clips**.
`Wav2Vec2Phoneme.transcribe_batch` provides the same operation over Modal RPC.
Production URL: https://anchpop--wav2vec2-phoneme-wav2vec2phoneme-predict-batch.modal.run

The service resamples and normalizes each clip separately, sorts by length,
runs similar-length microbatches, trims padding from the outputs, and restores
the caller's order. Each item supports the existing single-request fields:
`audio_f32_b64`, `sample_rate`, `language`, `top_k`, `return_frames`,
`return_frame_matrix`, and `target_phonemes`.

```python
import base64
import numpy as np
import requests

# Pack mono samples as little-endian IEEE-754 float32 (no WAV header).
def encode_audio(samples):
    return base64.b64encode(np.asarray(samples, dtype="<f4").tobytes()).decode()
response = requests.post(batch_url, json={
    "requests": [
        {"audio_f32_b64": encode_audio(audio_a), "sample_rate": 16000, "language": "eng"},
        {"audio_f32_b64": encode_audio(audio_b), "sample_rate": 16000, "language": "tha",
         "return_frame_matrix": True},
    ],
}, timeout=180)
response.raise_for_status()
results = response.json()["results"]
```

Both single and batch HTTP endpoints accept this compact format. Legacy
`audio` float arrays are also accepted during caller rollout. Compact input
takes precedence if both fields are supplied; invalid compact input is rejected.

The HTTP batch response is `{"results": [...], "deploy_marker": "...",
"model_id": "...", "model_revision": "...", "decoder_version": "nonblank_v1"}`.
The same four identity fields accompany every single prediction (including an
empty phoneme result) and `marker_only` identity probe. Each successful
batch item has the existing
pronunciation response fields. An invalid item receives
`{"error": {"type": "ValueError", "message": "..."}}` at its original index;
healthy items still run. Batch items and RPC transcription results retain
their existing shape; the HTTP envelope carries the identity.

An invalid batch envelope/count or single-clip input receives HTTP 422 with
the original `detail` string and the four identity fields beside it. Model
load failures receive HTTP 503 with identity inside the existing `detail`
object alongside `load_error`; failed-load `marker_only` probes remain HTTP
200 with `load_error` and identity at the top level. Framework-generated
validation errors (for example malformed JSON before a handler runs) are
unchanged and do not carry model identity. Check each batch result for
`error` even when HTTP returns 200.

This batches only the clips explicitly submitted together. It adds no
cross-request collection window. The existing single-clip endpoint remains
available. A request completes after its microbatches finish; outputs do not
stream. The 64-item limit is a request size, not 64 concurrent GPU forwards.

## Limits and deployment configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `WAV2VEC2_BATCH_SIZE` | `8` | Maximum clips per GPU forward |
| `WAV2VEC2_MAX_LENGTH_RATIO` | `1.25` | Longest/shortest duration within a microbatch |
| `WAV2VEC2_MAX_PADDED_SECONDS` | `120` | Longest duration × clip count per microbatch |
| `WAV2VEC2_GPU` | `L40S` | GPU choice |
| `WAV2VEC2_MAX_CONTAINERS` | Modal default | Optional container cap |
| `WAV2VEC2_BATCH_DIAGNOSTICS` | `0` | Add batching/sample-count diagnostics to results |

An individual clip exceeding the padded budget runs alone. GPU OOM splits a
microbatch recursively; failure of a singleton is reported for that clip.
Group-normalized checkpoints use singleton forwards because time padding
changes their normalization. No audio is truncated to satisfy a budget.
Decoded label strings are cached, and GPU outputs move to CPU in bulk before
frame decoding to remove repeated GPU synchronization.

Batch shape can change FP16 probabilities and predictions even with correct
masking. The 64-clip production-path test preserved phonemes in its sample;
this is not a guarantee of singleton-equivalent output on every clip.

## Deployment ownership

This app and its self-contained image now live in lexide, not yap. Yap consumes
its unchanged URLs; its deployment workflow no longer deploys this app. The
`compare-audio-models` eval tool expects a sibling lexide checkout at
`../lexide` when run from the yap repo root.

When deployment is authorized, run from the lexide repo root (stop first so
warm containers cannot keep serving old code):

```bash
~/.modal-venv/bin/modal app stop wav2vec2-phoneme
~/.modal-venv/bin/modal deploy pronunciation/modal/wav2vec2_phoneme.py
```

No deployment is required to run the CPU contract tests below.

## Local tests (NixOS)

Use the pronunciation wrapper for its native-library loader paths. Invoke it
with `bash` (the file is not executable). This machine already has torch and
pytest in the test venv, Modal in the Modal CLI venv, and FastAPI in the data
venv. Keep the test venv first on `PYTHONPATH` so those fallback paths do not
replace its native dependencies. No package installation is needed.

Exact working command (Python 3.13 venvs on this machine):

```bash
PYTHONPATH="$HOME/.venv-lexide-tests/lib/python3.13/site-packages:$HOME/.modal-venv/lib/python3.13/site-packages:$HOME/.venv-lexide-data/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}" \
LEXIDE_DATA_VENV="$HOME/.venv-lexide-tests" \
bash /data/coding/lexide/pronunciation/scripts/py-linux.sh \
  -m pytest /data/coding/lexide/pronunciation/modal -q
```

Verified: **54 passed**, with one existing Starlette TestClient/httpx deprecation
warning. These CPU tests import the relocated app/image definition and exercise
the real response handlers with fake model forwards; they do not deploy, load
production weights, or validate GPU/cloud execution.

## Self-describing frame matrices (schema 1)

`return_frame_matrix: true` on either endpoint returns the same trimmed artifact:

```json
{
  "schema_version": 1,
  "producer": {"model_id": "...", "model_revision": "...", "deploy_marker": "...", "decoder_version": "nonblank_v1"},
  "trained_against_g2p": "g2p/0.4.0 espeak-ng/aa907af78d5665d8 thai/ad66331eca29d4ea korean/9e4bc6b854f6a903",
  "sample_rate": 16000,
  "frame_rate_ms": 20.0,
  "heads": {
    "phone": {"shape": [123, 461], "labels": ["<pad>", "..."], "blank_id": 0, "dtype": "float16", "encoding": "zlib+base64", "value_semantics": "joint_log_probability", "data": "..."},
    "nonblank": {"shape": [123], "labels": ["nonblank"], "dtype": "float16", "encoding": "zlib+base64", "value_semantics": "sigmoid_probability", "data": "..."},
    "stress": {"shape": [123, 3], "labels": ["none", "primary", "secondary"], "dtype": "float16", "encoding": "zlib+base64", "value_semantics": "probability", "data": "..."}
  }
}
```

Dimensions above are illustrative. Values are **little-endian, row-major fp16**,
then zlib-compressed and standard-base64 encoded. Phone values are natural-log
joint probabilities (blank from the nonblank gate, phones marginalized over
stress/prosody), with masked special-token columns retained as `-inf`. Labels
come from `tokenizer.get_vocab()`, not the display decoder. Nonblank is the raw
sigmoid; stress and auxiliary heads contain full softmax distributions, not IDs.
All heads share the phone head's `T`, after trimming padded frames.

No language means no auxiliary heads. `language: "tha"`, `"zho-hans"`, or `"jpn"`
adds the applicable checkpoint head, with explicit `language` and `target`
metadata. `return_all_heads: true` requests the whole available inventory,
regardless of language. It does **not** change the auxiliary labels in emitted
`phonemes`/diagnostic `frames`, which remain language-specific. Distinct keys
`tha_tone`, `zho_hans_tone`, `jpn_pitch_accent` prevent colliding tone heads.

| Head | Ordered class labels |
| --- | --- |
| `tha_tone` | not_bearer, mid, low, falling, high, rising |
| `zho_hans_tone` | not_bearer, high_level, rising, dipping, falling, neutral |
| `jpn_pitch_accent` | not_bearer, low, high |

The current three-class Japanese head represents per-mora H/L, not downstep.
These distributions are predictions, not claims of acoustic correctness; a
checkpoint can emit a degenerate distribution. New checkpoint label conventions
must update declarations (or supply explicit `labels` in their head specs).

`sample_rate` is the processor feature extractor's rate, **after resampling**;
`frame_rate_ms = 1000 * product(backbone.config.conv_stride) / sample_rate`.
The trained sidechannel's fixed geometry is validated rather than silently
reinterpreted for another rate. `T * frame_rate_ms` is slightly shorter than the
waveform because convolution needs a full receptive field; this interval is not
an exact end timestamp or a declared frame-center offset.

`trained_against_g2p` is the exact `g2p::identity()` string of the source label
build. It is an explicit checkpoint-bound assertion alongside `MODEL_REVISION`,
not inferred from weights or the serving machine's current G2P. Undeclared
checkpoints (including the old production default) return **null**, not a made-up
identity. An operator may explicitly declare `WAV2VEC2_TRAINED_AGAINST_G2P` for a
new checkpoint. **Update this assertion whenever the checkpoint changes.**

This field is source-label provenance evidence, **not a full compatibility
guarantee**: it does not cover `TOKEN_REMAP`, per-language `LANG_PHONEME_REMAP`,
narrowing selection, French stress overrides, or accent supervision masks.
No compatibility/refusal policy is implemented here.

Rust `FrameMatrixPayload` dispatches explicitly on version: an absent version
means legacy phone-only `{shape, vocab, blank_id, dtype, encoding, data}`;
unknown/malformed versions are errors, never a fallback. `FrameMatrix::decode`
keeps typed producer, comparable training identity, timebase, and every head's
wire descriptor and decoded values. Legacy artifacts have unknown provenance
and timebase, and an empty head inventory. Existing phone scoring APIs retain
their behavior. Python's pre-existing `zero_infinity=True` CTC scorer may report
zero log probability for an impossible repeated target, while Rust correctly
returns `None`; the matrix format does not change or hide that discrepancy.

### Reproducing an eval-only round trip

After an authorized eval deployment, use `verify_frame_matrix.py --marker MARKER
--output /path/to/artifacts --clip jpn=/path/to/clip.wav --clip tha=/path/to/clip.wav`.
It checks freshness, singleton HTTP/batch byte identity, inventory selection,
full distributions, timing, and records actual compressed/HTTP bytes. Its
"without suprasegmentals" size is explicitly a projection (phone + nonblank),
not a separate server option. Then run
`cargo run --manifest-path tagging/lexide/Cargo.toml --features pronunciation
--example verify_frame_matrix -- /path/to/artifacts/*-score.json` to verify every
wire field and decoded fp16 value, legacy decoding, and same-target scoring.
The tools do not manage cloud jobs; stop the eval app after verification.

### Demo rollout ordering

The older deployed web-demo WASM expects the flat legacy payload and cannot
read schema 1. **Before promoting this endpoint format to production**, rebuild
with `bash tagging/web-demo/build.sh` and deploy the backwards-compatible demo
(page, immutable JS assets, and the entire `www/pkg` together). The build hashes
JS/CSS assets, but WASM-bindgen `pkg` URLs are stable: refresh/cache-bust those
assets and already-open older pages during rollout. No production or web
promotion is part of the eval verification. New demo timing comes from the
matrix's declared interval throughout the phone track, selection and playback;
only a legacy payload gets the historical 20 ms fallback.

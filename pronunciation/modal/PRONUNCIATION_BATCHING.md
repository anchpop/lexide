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

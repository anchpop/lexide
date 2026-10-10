# Lexide web demos

Two static pages with shared navigation and theme settings:

- `/` — Parsley sentence segmentation and tokenization, entirely in-browser.
- `/pronunciation.html` — Lexide audio-to-IPA transcription, entirely in-browser.

Both URLs work directly on GitHub Pages, including refreshes and project subpaths.

## Segmentation

The segmentation page runs the two byte-minGRU models (the sentence segmenter and
the char tokenizer, 0.99M params / 3.96 MB each) fully in-browser via WASM.
Paste a passage: it's split into sentences (gaps between them shown dropped),
and each sentence into token spans — the same `[BOS] + utf8 + [EOS]` O/B/I
pipeline retained from v1. Its byte models and boundary-prior code now live in
`web-demo/src/`, independently of the joint Rust tagger. The demo's concat-prior
fixture still verifies the historical implementation against PyTorch.

The full joint tagger (POS/lemma/deps) is *not* in the demo: its fp32 bge-m3 artifacts
are about 2.4 GB. See `../OVERVIEW.md`. This page is not a joint-parsley quality demo.

## Build & run

```sh
./build.sh                        # wasm-pack build + copy segmentation weights
python3 -m http.server -d www     # then open http://localhost:8000
```

The segmentation page loads its weights from its own directory, falling back to
`huggingface.co/anchpop/lexide-parsley/resolve/main/onnx/`. The pronunciation
model always loads from Hugging Face.

`wasm-pack` comes from the yap flake (`direnv exec /data/coding/yap`); the
shared demo WASM binary is ~516 KB (separate from ONNX Runtime's WASM).

## Pronunciation

Upload or drag and drop a browser-decodable audio file (up to 20 seconds / 25 MB),
or record with the microphone. Transcription starts automatically when the file
is ready or recording stops. Audio stays in the browser. Cancel terminates the
inference worker, including model loading or an active run; Retry starts a fresh
worker. Successful runs reuse the loaded session.

The model is `anchpop/lexide-pronunciation-small` run 2 (`a5472e4a`, 24.93M
parameters). Its int8 export lives in that repo under `onnx/`, and the worker pins
the commit that added it (`58bd3171`). The module worker lazily loads that model and `onnxruntime-web@1.30.0` from jsDelivr
on the first transcription. It uses the single-threaded WASM backend, so static
Pages needs no cross-origin isolation headers. The browser supplies raw mono
16 kHz float32 audio, matching the saved processor's `do_normalize=False`.
The ONNX graph itself removes the waveform mean and divides by
`sqrt(population_variance + 1e-7)`, then normalizes log-mel features with one scalar
mean/variance per clip (`epsilon=1e-5`). JavaScript must not normalize again.

### Export the pinned model

To re-export (e.g. for a new run), download the checkpoint into
`pronunciation/.work/onnx-run2/checkpoint` with `huggingface_hub.snapshot_download`,
then from the repository root:

```sh
LEXIDE_DATA_VENV=~/.venv-lexide-tests pronunciation/scripts/py-linux.sh \
  pronunciation/inference/export_onnx.py pronunciation/.work/onnx-run2/checkpoint \
  --output pronunciation/.work/onnx-run2
```

The venv needs onnx, onnxscript and onnxruntime besides the training stack.
`benchmark.json` records parity, sizes and CPU timings: int8 is 30 MB and took
0.16 s for 5 s of audio on one native core. Upload `model.int8.onnx` and
`frame_matrix.json` to the model repo's `onnx/`, then pin the new commit in
`www/pronunciation-worker.js`. `tests/run.sh` validates the exported vocabulary
through the g2p types when that export directory exists.

`www/pronunciation-decoder.mjs` loads the WASM bindings. Local phone log
probabilities go directly into a float32 constructor with vocabulary/blank
metadata; g2p types validate the vocabulary. The demo pins the same g2p revision
as pronunciation preprocessing, supporting run 2's g2p 0.7.0 inventory.
Matrix validation and decoding reuse `lexide/src/pronunciation/mod.rs` via
`#[path]`, with no JavaScript decoder. Legacy compressed float16 artifacts
remain supported by the separate wire constructor for regression tests.

Decoding is **nonblank-first**: blank wins at probability >= 0.5; otherwise the
highest-probability eligible phone wins, even when its individual joint
probability is below blank. Special tokens and masked negative-infinity scores
are excluded. Adjacent repeated phone IDs collapse; blanks separate repeats.
This is not a beam search for the sequence with highest summed CTC probability.
This matches Modal's nonblank-first semantics, deriving the gate from the blank
log probability rather than joint argmax or the separately exported nonblank head.
Legacy float16 rounding can change near-ties and the half-probability boundary;
local float32 outputs do not pass through float16. Stress stays attached to
frames and never splits a phone run. Confidence and alternatives average
conditionally normalized phone probabilities over the emitted run, excluding
blank and specials; alternatives with zero mass throughout the run are omitted.

Select a sound directly in the IPA to inspect its confidence and alternatives,
and seek to its approximate audio position. Arrow keys move between sounds;
**Copy IPA** copies the complete plain transcription. Recording has an elapsed
time indicator, and the file picker and drop target share the same validation.

The IPA view displays bare phonemes in square brackets, without stress, tone,
pitch-accent annotations, or inferred word boundaries. Stress remains in the
original frame records, computed by argmax of the raw stress probabilities.
All auxiliary tone/pitch heads are retained in downloads, without a language hint.
Below the IPA, the result includes a shared phoneme/spectrogram time axis with
one playback control and one playhead. The initial audio preview is hidden once
results are shown. Click or drag the spectrogram to seek; arrow keys move 20 ms
(Shift: 100 ms), and Home/End jump to the clip edges. Phoneme blocks
use their actual CTC emission spans; blank gaps remain visible. Clicking a sound
highlights its span and seeks the audio. Playback updates the playhead, active
phoneme, IPA highlight and inspector together. The spectrogram supports click to
seek, keyboard seeking directly on the spectrogram, horizontal scrolling and 1×/2×/4×
zoom. Long clips scroll to follow playback. No stress annotations are added.

The spectrogram is computed locally in a module worker (`spectrogram-worker.js`),
using the same mono 16 kHz samples sent to the model. `spectrogram.mjs` computes a
centered 512-sample Hann-window STFT at a 160-sample hop (32 ms window, 10 ms hop),
with zero padding at edges. Color shows -80 to 0 dBFS amplitude, with linear
frequency from 0 to 8 kHz. It requires no additional service or audio upload.

The collapsed **Model details** section contains the diagnostic frame timeline
and raw-output download. The timeline shows the phoneme path, including CTC blanks, and its
slider seeks audio. Times use the scratch encoder's 320-sample hop
at 16 kHz (20 ms); CTC emission runs are not forced-aligned phoneme boundaries.

**Download model output** saves every raw float32 head as a row-major JSON array
with shape/labels/semantics, frame stress, `repo@revision` model identity, runtime,
and the locally decoded path and phonemes. Masked negative infinity is encoded as
`"-Infinity"` (restore with `Number(value)`), never silently changed to JSON null.
The local artifact is not the compressed Modal wire format; re-decode its phone
head with `rawMatrix(Float32Array.from(head.data, Number), head.shape[0], head)`.
Audio samples are not included.

Run the decoder checks with:

```sh
./tests/run.sh  # build actual nodejs-target WASM, run JS and asset tests
```

No API key, audio upload or inference server is used. The first run downloads
~30 MB of model weights plus the runtime; later runs reuse the worker session.
Microphone access requires HTTPS or localhost and browser permission.

## GitHub Pages

`.github/workflows/web-demo.yml` publishes on every push to `main` that touches
the demo or the shared decoder: it runs `build.sh` and copies `www/` over the
`gh-pages` branch. Segmentation weights aren't in git, so the copies already on
`gh-pages` stay. Keep the branch's `.nojekyll` file. There is no SPA rewrite or
server to configure.

### Browser asset versions

After any browser source edit, run `python3 build-assets.py` (also included in
`build.sh`). It emits content-hashed scripts and styles into `www/assets/`,
rewrites module/worker imports to matching hashed dependencies, and updates
both HTML pages. This prevents new markup from using stale cached JavaScript.
Run this command before serving locally as well.

Publishing copies over `gh-pages` instead of replacing it, so previously
published hashed assets are kept: cached older HTML still needs its own
matching versions. Source JS/CSS files in `www/` remain the editable originals.

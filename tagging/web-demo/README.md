# Lexide web demos

Two static pages with shared navigation and theme settings:

- `/` — Parsley sentence segmentation and tokenization, entirely in-browser.
- `/pronunciation.html` — Lexide audio-to-IPA transcription, using the hosted model.

Both URLs work directly on GitHub Pages, including refreshes and project subpaths.

## Segmentation

The segmentation page runs the two byte-minGRU models (the sentence segmenter and
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

## Pronunciation

Upload or drag and drop a browser-decodable audio file (up to 20 seconds / 25 MB), or record with
the microphone. Transcription starts automatically when the file is ready or
recording stops. Cancel stops the browser request; Retry appears after a failed
or canceled request. The browser converts audio to 16 kHz
mono and sends it to the existing production Modal endpoint:

```
https://anchpop--wav2vec2-phoneme-wav2vec2phoneme-predict.modal.run
```

This is the `anchpop/lexide-pronunciation` model endpoint also used by Yap.
The page requests `{audio: number[], sample_rate: 16000, top_k: 3,
return_frames: true, return_frame_matrix: true}`. The response includes the
full phoneme log-probability matrix as zlib/base64-compressed, row-major float16,
its vocabulary and blank ID, per-frame stress labels, and the deployment marker.

`www/pronunciation-decoder.mjs` is a thin asynchronous loader for the WASM
bindings. Wire validation, decompression and decoding reuse
`lexide/src/pronunciation/mod.rs` via `#[path]`, with no JavaScript decoder.
Decoding is **nonblank-first**: blank wins at probability >= 0.5; otherwise the
highest-probability eligible phone wins, even when its individual joint
probability is below blank. Special tokens and masked negative-infinity scores
are excluded. Adjacent repeated phone IDs collapse; blanks separate repeats.
This is not a beam search for the sequence with highest summed CTC probability.
Float16 rounding can change near-ties and the half-probability boundary:
encoded `ln(0.5)` rounds slightly below the threshold and therefore emits a
phone; the decoder does not adjust quantized inputs. Stress stays attached to
frames and never splits a phone run. Confidence and alternatives average
conditionally normalized phone probabilities over the emitted run, excluding
blank and specials; alternatives with zero mass throughout the run are omitted.

Select a sound directly in the IPA to inspect its confidence and alternatives,
and seek to its approximate audio position. Arrow keys move between sounds;
**Copy IPA** copies the complete plain transcription. Recording has an elapsed
time indicator, and the file picker and drop target share the same validation.

The IPA view displays bare phonemes in square brackets, without stress, tone,
pitch-accent annotations, or inferred word boundaries. Stress remains in the
original frame records for future views; its probabilities are not currently
exposed by this endpoint. No language hint or auxiliary tone/pitch head is
requested. Below the IPA, the result includes a shared phoneme/spectrogram time axis with
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
slider seeks audio. Times use the production wav2vec2 encoder's 320-sample hop
at 16 kHz (20 ms); CTC emission runs are not forced-aligned phoneme boundaries.

**Download model output** saves the original response (including compressed
matrix and frame labels), the deployment marker, and the locally decoded
phonemes as JSON. This allows inspecting or re-decoding the result later
without running the model again. Audio samples are not included in this file.

Run the decoder checks with:

```sh
./tests/run.sh  # build actual nodejs-target WASM, run JS and asset tests
```

No API key or model weights are shipped to the browser for pronunciation.
The endpoint permits cross-origin requests from GitHub Pages; it must remain
available for transcription to work. Audio is sent automatically after a successful upload/drop or finished recording.
Cold starts can take a minute; requests time out after three minutes and can
be canceled (canceling stops the browser request, not necessarily server work).
Microphone access requires HTTPS or localhost and browser permission.

## GitHub Pages

The published site lives on the `gh-pages` branch. Build with `./build.sh`, then
copy the **entire contents** of `www/` into the Pages branch before publishing,
including `demo.css`, `theme.js`, `pronunciation.html`, `pronunciation.js`,
`pronunciation.css`, `pronunciation-decoder.mjs`, `audio-explorer.js`,
`spectrogram.mjs`, `spectrogram-worker.js`, `pkg/`, and the segmentation weights. Keep the branch's `.nojekyll` file.
There is no SPA rewrite or server to configure. Both pages use the shared WASM package; run `build.sh` before serving `www/`.

### Browser asset versions

After any browser source edit, run `python3 build-assets.py` (also included in
`build.sh`). It emits content-hashed scripts and styles into `www/assets/`,
rewrites module/worker imports to matching hashed dependencies, and updates
both HTML pages. This prevents new markup from using stale cached JavaScript.
Run this command before serving locally as well.

When publishing, copy `www/assets/` alongside the HTML and **retain previously
published hashed assets** on `gh-pages`: cached older HTML still needs its own
matching versions. Source JS/CSS files in `www/` remain the editable originals.

# On-device TTS (spike)

A small text-to-speech model for yap, built on the
[tiny-voca](https://github.com/n4ze3m/tiny-voca) recipe (Apache-2.0; vendored
in `voca/` from commit `62e2adef`, see `voca/LICENSE-tiny-voca`): a
FastSpeech 2-style transformer that predicts durations, pitch, voicing and
energy, and drives a source-filter synthesizer instead of a neural vocoder.
About 4.5M parameters.

What we changed:

- **Input tokens come from yap's pack phonemes** (`voca/text.py`): the per-gram
  g2p sequences yap stores for every sentence (`language_utils::sentence_phonemes`),
  so the model is trained on exactly what the app will feed it. The phone
  inventory is built from a whole pack dump (`voca/inventory.json`).
- **Durations come from the lexide phonetizer**: CTC forced alignment of its
  frame matrix from the production batch endpoint, instead of tiny-voca's
  espeak wav2vec2 recogniser.
- **Training audio is Gemini TTS** (Achernar, with yap's "warm and welcoming" direction).
  The first spike (`runs/eng-spike`) used Cloud TTS Chirp 3 HD Achernar.
- **Less buzz**: harmonics above a predicted per-frame cutoff get random phase, and a
  ~1M-parameter post-filter corrects the synthesized spectrum (`voca/model.py`).
- **Pauses**: yap's `#` word breaks become word-boundary tokens; the aligner keeps
  short silences at punctuation; `speak` floors comma/stop pauses at 150/300 ms.

## Pipeline

```sh
# 1. Every phonemized sentence of a pack, with its grams (from the yap repo)
cargo run --release -p language-utils --example dump_sentence_phonemes -- out/eng_for_fra > ../lexide/tts/data/eng_sentences.jsonl

# Python below runs through the NixOS wrapper:
#   LEXIDE_DATA_VENV=~/.venv-lexide-tts bash ../pronunciation/scripts/py-linux.sh ...
# 2. Inventory, then a script greedy on phone-bigram coverage
scripts/build_inventory.py data/eng_sentences.jsonl voca/inventory.json --language eng
scripts/select_sentences.py data/eng_sentences.jsonl data/eng_script.jsonl --hours 1 --chars-per-second 19
# 3. Synthesize (GOOGLE_CLOUD_API_KEY), then fetch phonetizer frame matrices
scripts/generate.py data/eng_script.jsonl data/eng_gemini_achernar --provider gemini --language-code en-US --voice Achernar --style "warm and welcoming" --workers 6
scripts/fetch_frames.py data/eng_gemini_achernar --language eng
# 4. Features + durations (PYTHONPATH=.)
-m voca.build --set data/eng_gemini_achernar --output features/eng_gemini_achernar --workers 24
# 5. Train on Lambda (uploads features/), then score samples (GROQ_API_KEY)
set -a; . ../.env; set +a   # HF_TOKEN, WANDB_API_KEY
sky launch -c tts-x sky_train.yaml --env RUN_NAME=eng-x [--env CONFIG=configs/istft.json] --secret HF_TOKEN --secret WANDB_API_KEY
# runs push to the private HF repo anchpop/lexide-tts-runs every 5 min and log to W&B project lexide-tts
scripts/score.py runs/eng-gemini/samples-validation --language eng
```

`scripts/calibrate_onset.py` measures the phonetizer's emission timing against
pitch-tracker voicing flips; rerun it for a new phonetizer checkpoint.

## StyleTTS 2 / Kokoro architecture

`configs/styletts.json` (`"architecture": "styletts"`) selects `voca/styletts.py`:
StyleTTS 2's prosody predictor, AdaIN decoder and Kokoro's iSTFTNet generator
with the harmonic NSF source, trained from scratch on our phonemes, durations,
pitch, voicing and energy (see the module docstring for what differs from
upstream). Training adds StyleTTS 2's style encoders, multi-period,
multi-resolution and WavLM critics and its loss weights (`voca/losses.py`);
none of that ships in `inference.pt`, which stores the mean training style
instead. Training needs `transformers` (WavLM-base-plus). Sizes are config
fields (`generator_channels` 144/224/320 is a ~2/4/8M generator).

```sh
sky launch -c tts-x sky_train.yaml --env RUN_NAME=eng-styletts --env CONFIG=configs/styletts.json --secret HF_TOKEN --secret WANDB_API_KEY
# CPU real-time factors against the current model and Kokoro-82M (random weights, forced durations)
PYTHONPATH=. python scripts/time_cpu.py --output runs/cpu-timing.json
```

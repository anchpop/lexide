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
  so the model is trained on exactly what the app will feed it. **One model per
  language**: each has its own phone inventory, built from that language's whole
  pack dump (`voca/inventories/<language>.json`); `VOCA_LANGUAGE=<language>`
  (yap's code, default `eng`) picks it for every tool and training run. Stress,
  tone (Mandarin, Thai) and Japanese accent (phrase starts `%`, nuclei `ꜜ`)
  are parallel per-phone features, and so is what yap knows that a generic TTS
  has to guess: each word's part of speech (yap's own labels), each gram's
  corpus rarity, and whether a break is inside a gram or between grams (its
  chunking). Never the lesson's target word: readings must not depend on the
  exercise. The inventory also says which phones are voiced by definition
  (English: every vowel). `PYTHONPATH=. python -m pytest tests`.
- **A word-level prosody head** (StyleTTS architecture): per word occurrence, a
  prominence score ranked within its phrase, a contour shape (level, rise,
  fall, rise-fall, fall-rise), pitch excursion and phrase-final movement,
  supervised by targets measured from the recording (`voca.data.word_prosody`).
  Its predictions condition durations and the F0/energy branch, teacher-forced
  with the recording's at first.
- **Durations come from the lexide phonetizer**: CTC forced alignment of its
  frame matrix from the production batch endpoint, instead of tiny-voca's
  espeak wav2vec2 recogniser.
- **Training audio is Gemini TTS**, with yap's "warm and welcoming" direction written in the
  language itself (`Language::tts_direction`): Achernar for 10 h per language, and Kore,
  Leda, Puck, Iapetus and Orus for 30 min each. Every voice also reads 50 shared training
  sentences and the same 60 validation sentences. Training draws each voice with
  probability proportional to its hours^0.5 (`--sampling-alpha`), so the small voices get
  about a tenth of the batches each instead of 4%; `voca.evaluate` reports per voice.
  The first spike (`runs/eng-spike`) used Cloud TTS Chirp 3 HD Achernar.
- **Less buzz**: harmonics above a predicted per-frame cutoff get random phase, and a
  ~1M-parameter post-filter corrects the synthesized spectrum (`voca/model.py`).
- **Pauses**: yap's `#` word breaks become word-boundary tokens; the aligner keeps
  short silences at punctuation; `speak` floors comma/stop pauses at 150/300 ms.

## Pipeline

```sh
# 1. Every phonemized sentence of a pack, with its grams (from the yap repo; English is eng_for_fra, the others <language>_for_eng)
cp out/jpn_for_eng/tts_sentences.jsonl ../lexide/tts/data/jpn_sentences.jsonl

# Python below: PYTHONPATH=. and the GPU venv with the nix LD_LIBRARY_PATH (see the agent reports).
# 2. The language's inventory
scripts/build_inventory.py data/jpn_sentences.jsonl          # -> voca/inventories/jpn.json
# 3. Scripts, Gemini audio (GEMINI_API_KEY), phonetizer frames, features, upload (HF_TOKEN): resumable
scripts/language_data.py jpn
# 4. phonebert, then the TTS model, on the language's frontend
VOCA_LANGUAGE=jpn python -m voca.phonebert prepare data/jpn_sentences.jsonl features/jpn_phonebert_v5 --exclude data/scripts/jpn_*.jsonl
set -a; . ../.env; set +a   # HF_TOKEN, WANDB_API_KEY
sky launch -c pb-jpn sky_phonebert.yaml --env LANGUAGE=jpn --secret HF_TOKEN --secret WANDB_API_KEY
sky launch -c tts-x sky_train.yaml --env LANGUAGE=jpn --env RUN_NAME=jpn-x --env CONFIG=configs/styletts.json --env PHONEME_ENCODER=phonebert-jpn-v5/encoder.pt --secret HF_TOKEN --secret WANDB_API_KEY
# runs push to the private HF repo anchpop/lexide-tts-runs every 5 min and log to W&B project lexide-tts
scripts/score.py runs/eng-gemini/samples-validation --language eng
```

Run the Python commands from `tts/` with `PYTHONPATH=.` and set `VOCA_LANGUAGE` to the pack language.
For batch recording, run the `select` stage first, then `python scripts/generate_batch.py jpn`.
The batch ledger in `data/` lets interrupted downloads resume. Run the `generate` stage to retry
missing clips and write manifests, then run `frames build push`. For single words and short
phrases, use `scripts/select_words.py` on the same dump and `scripts/words_data.sh jpn`.

English edge scripts are built from the curated text in `data/edge/`. After selecting the regular
scripts, use Cargo, CMake and a C compiler to build the fallback phonemizer with `cargo build --release --locked --manifest-path g2p-jsonl/Cargo.toml`.
Run `python scripts/edge_grams.py data/eng_sentences.jsonl`, then `python scripts/generate_batch.py eng --part edge`.
The scripts use yap's readings when the labelled sentence matches the spoken text. On the other
lines, g2p supplies per-word phones for the text sent to Gemini. To update regular scripts with
new pack readings for the same text, use `scripts/regram.py`.

`voca.build` writes the reason for each rejected clip in `rejected.jsonl`. It checks the
letters, digits and punctuation of the grams against the recording script, alignment confidence,
excess speech and a minimum of 80 frames for the waveform critics. Pass `--reuse features/<old-set>`
to realign recordings using their existing acoustic tensors. For an existing build,
`scripts/drop_short.py` creates filtered manifests over the original tensors.

`scripts/calibrate_onset.py` measures the phonetizer's emission timing against
pitch-tracker voicing flips; rerun it for a new phonetizer checkpoint.

## StyleTTS 2 / Kokoro architecture

`configs/styletts.json` (`"architecture": "styletts"`) selects `voca/styletts.py`:
StyleTTS 2's prosody predictor, AdaIN decoder and Kokoro's iSTFTNet generator
with the harmonic NSF source, trained from scratch on our phonemes, durations,
pitch, voicing and energy (see the module docstring for what differs from
upstream). The decoder receives each frame's position in its phone. During training a posterior
encoder infers delivery from recorded pitch, energy and timing. At synthesis the text and voice
predict a distribution over deliveries; `voca.speak --take` seeds a repeatable draw. The decoder
uses the stored acoustic style for the voice. Training uses multi-period, multi-resolution and
speech-model critics, mHuBERT-147 and the losses in `voca/losses.py`.
The adversarial terms keep StyleTTS 2's
fixed weights (`--weights` `gan`, `slm_gan`; `--feature-matching` 2) and fade in over
`--gan-ramp` steps (`gan_scale` logs ramp x weight); `--gan-ratio` and
`--gan-scale-max` balance TinyVoca's only. The generator low-passes its source
coefficients before the strided (coarse) injection, which otherwise folds a held
vowel's harmonics into an inharmonic line. `voca.speak` trims leading silence to
4 frames before the first phone or sustained sound (about 220 ms sooner). The
generator also images the voice around its synthesis frame rate (3 kHz at hop 16):
`deployed_clock_sensitivity_db` measures how much of voiced 1.2-4.4 kHz follows its
output grid, and `--weights '{"shift": 1}'` (off by default) trains against it, by
rendering two utterances per batch again at a sub-hop delay and penalising the
difference (`shift_consistency`). Sizes are config fields
(`generator_channels` 144/224/320 is a ~2/4/8M generator).

At every save, the EMA weights that ship synthesize the held-out sentences as
`voca.speak` does and are scored per word against the recordings
(`voca/evaluate.py`: main-word agreement, accent and other-word pitch
excursion, phrase-final movement, pauses, vowel voicing), logged as
`deployed_*`. The same scores for any checkpoint:
`python -m voca.evaluate --checkpoint runs/x/inference.pt --validation features/eng_gemini_10h_v3/validation.jsonl`.

Feature sets and checkpoints record the front end's fingerprint (`voca.text.FRONTEND_VERSION`,
the symbols and the features); one from another front end is refused as stale. Version 2
(`eng_gemini_10h_v2`, `eng_phonebert_v2`) keeps punctuation inside grams, names
word occurrences with clitics attached, adds tone and accent features, and
checks pYIN against a Praat pitch track. Version 3 (`eng_gemini_10h_v3`,
`eng_phonebert_v3`; the same tokens and durations) adds part of speech, rarity
and boundary type, so the pack dump and scripts carry each gram's `pos` and
`count`. An older run can still be branched onto newer features:
`--init-from <last.pt> --new-data` (new feature tables start at zero for "none";
the predictor LSTMs grow inputs for the word features).

```sh
# English v10 used the 80-frame-filtered v5 manifests over v4 audio, and a pretrained encoder.
sky launch -c tts-x sky_train.yaml --env LANGUAGE=eng --env RUN_NAME=eng-style-voices-v10 --env CONFIG=configs/styletts.json \
  --env FEATURES=eng_gemini_multi_v5 --env PHONEME_ENCODER=phonebert-eng-v3-f4/encoder.pt --env STEPS=20000 \
  --env GAN_START=8000 --env EXTRA_ARGS='--cuda-graphs --bf16 --weights {"shift":1}' --secret HF_TOKEN --secret WANDB_API_KEY
# CPU real-time factors against the current model and Kokoro-82M (random weights, forced durations)
PYTHONPATH=. python scripts/time_cpu.py --output runs/cpu-timing.json
```

## Public phoneme-only models

Distillation trains the two text encoders on paired text from yap and stock espeak-ng.
It keeps the teacher's decoder, voice and prosody weights. Use zero added encoder layers
for every language (the default). Set `LANGUAGE_CODE` for the espeak script and
`VOCA_LANGUAGE` for model tools. Each input row needs `text` and the teacher's `grams`.
Keep the same sentence order in both files. Pairs with unknown phones or different token
counts are skipped. The student sees stress, tone and accent with POS, rarity and boundary hints zeroed.

```sh
export VOCA_LANGUAGE=eng LANGUAGE_CODE=eng PYTHONPATH=.
mkdir -p data/distill
python plain/stock_espeak.py data/eng_sentences.jsonl data/distill/plain.jsonl
python -m voca.distill --teacher runs/eng-style-voices-v10/inference.pt \
  --pairs data/eng_sentences.jsonl data/distill/plain.jsonl --output runs/eng-distilled
python -m voca.speak --checkpoint runs/eng-distilled/inference.pt --script data/distill/plain.jsonl --limit 1 --output samples
```

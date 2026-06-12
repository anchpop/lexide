#!/usr/bin/env bash
#
# Process whatever raw data is currently in data/audio/<lang>/manifest.jsonl
# and push the result to the HuggingFace dataset. Idempotent and safe to
# re-run; each step skips itself when its inputs haven't changed.
#
# Steps:
#   1. Audit FLEURS audio (Groq Whisper transcribe + aggressive content-overlap
#      decision) → train/fleurs_asr_exclusions.jsonl
#   2. Audit Tatoeba audio (same) → train/tatoeba_asr_exclusions.jsonl
#   3. relabel-french: LLM-labeled rhythmic-group stress → data/audio/fra/stress_overrides.jsonl
#   4. lang-filter: flag clips whose transcript isn't entirely the target
#      language (Pimsleur mixes in foreign example/instruction text that espeak
#      then mislabels silently) → train/lang_exclusions.jsonl (training excludes)
#   5. preprocess.py: per-lang phonemes.jsonl + vad.jsonl from the manifest
#      (phonemize via espeak-ng; framewise VAD via vad_compute Rust binary)
#   6. narrow: measure_corpus.py (Modal align, cache-aware → no-op when phonemes
#      unchanged) + narrow.py (acoustic nasal + English flap → phonemes_narrowed.jsonl,
#      the file training reads). Harmonic A1-P0 is recomputed locally, not cached.
#   7. upload_audio_to_hf.py: push data/audio/ to anchpop/lexide-pronunciation-audio
#
# This script does NOT acquire new data. If you have new Pimsleur lessons
# or Tatoeba records to ingest, run the appropriate downloader first:
#   python3 data/download_pimsleur.py [--recover-partials] ...
#   python3 data/download_tatoeba.py ...
# Then re-run this script.
#
# Required env vars (loaded from .env at repo root or pronunciation/.env):
#   GROQ_API_KEY    — Whisper audits (Groq Cloud)
#   OPENAI_API_KEY  — gpt-5.4-nano calls in relabel-french + lang-filter
#   HF_TOKEN        — HuggingFace dataset upload
# Also needs Modal auth (~/.modal.toml) for step 6's alignment — cached, so it's
# a no-op (no GPU spend) when no phonemes changed since the last run.
#
# Run from any directory; the script cd's into pronunciation/.

set -euo pipefail

# Resolve pronunciation/ root from the script's own path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Load .env files (repo root first, then local override).
if [ -f ../.env ]; then set -a; . ../.env; set +a; fi
if [ -f .env ]; then set -a; . .env; set +a; fi

require_env() {
  local name=$1
  if [ -z "${!name:-}" ]; then
    echo "ERROR: $name is not set (put it in pronunciation/.env or ../.env)" >&2
    exit 1
  fi
}
require_env GROQ_API_KEY
require_env OPENAI_API_KEY
require_env HF_TOKEN

echo "=== Step 1/7: FLEURS audit (Groq Whisper, phoneme-PER) ==="
# One source-parameterized auditor for FLEURS + Tatoeba (steps 1 & 2). Transcribes
# each clip via Groq Whisper (language forced), phonemizes expected + transcript
# through the same espeak pipeline as phonemes.jsonl, and writes per-clip phoneme
# error to train/<source>_asr_exclusions.jsonl. Resumable (skips clips already in
# the output), so rerun after a re-download to fill only new clips.
python3 scripts/audit_asr_groq.py --source fleurs

echo
echo "=== Step 2/7: Tatoeba audit (Groq Whisper, phoneme-PER) ==="
python3 scripts/audit_asr_groq.py --source tatoeba

echo
echo "=== Step 3/7: French rhythmic-group stress relabel ==="
# espeak emits per-word stress for French, which is systematically wrong:
# French stress falls on the final syllable of each rhythmic group, not on
# every word. tysm's prompt-aware caching makes re-runs free if no rows
# changed; otherwise only new rows hit the LLM.
# Run from the crate dir because the binary uses paths relative to its own
# location (../../data/audio/fra/manifest.jsonl).
(cd train/relabel-french && cargo run --release --quiet)

echo
echo "=== Step 4/7: Language-contamination filter (gpt-5.4-nano) ==="
# Pimsleur courses teach other languages, so some clips' transcripts are
# foreign / mixed-language; espeak then phonemizes the foreign text as the
# target language → silently wrong labels. Flag clips whose transcript isn't
# entirely the target language → train/lang_exclusions.jsonl (asr_exclusions
# schema, hash-gated on the sentence; training auto-excludes them). tysm's
# prompt-aware caching makes re-runs free except for new/changed sentences.
# Run from the crate dir (default paths are relative to it); ensure the tysm
# cache dir exists (it panics otherwise).
(cd train/lang-filter && mkdir -p .cache && cargo run --release --quiet)

echo
echo "=== Step 5/7: Phonemize + recompute VAD ==="
# preprocess.py rebuilds vad.jsonl via vad_compute as it goes, keeping VAD
# coverage in lockstep with phonemes. --skip-vad if you regenerated phonemes
# for a label-only fix that didn't change which audio files are referenced.
python3 train/scripts/preprocess.py

echo
echo "=== Step 6/7: Narrow (acoustic nasal + English flap) ==="
# Alignment boundaries come from Modal; measure_corpus is cache-aware (keyed by
# the exact phonemes), so this is a no-op with no GPU spend unless step 5 changed
# some clip's phonemes — those clips re-align, the rest are served from cache.
# Redeploy first so the container code matches the pinned revision measure_corpus
# asserts. Then narrow.py rewrites tokens the acoustics justify → the canonical
# phonemes_narrowed.jsonl that train.sh (--use-narrowed) reads. narrow recomputes
# harmonic A1-P0 locally (no cache) — a few min of DSP, no Modal.
(cd espeak_audit && python3 -m modal deploy modal_aligner.py >/dev/null)
python3 espeak_audit/measure_corpus.py
python3 espeak_audit/narrow.py

echo
echo "=== Step 7/7: Upload to HF dataset ==="
# --large = upload_large_folder: splits into ≤25k-file commits (HF's per-commit
# cap) and resumes from .cache/.huggingface if interrupted. The plain single-commit
# path 413s on this dataset (hundreds of thousands of files > 25k/commit).
python3 scripts/upload_audio_to_hf.py --large

echo
echo "=== DONE ==="
echo "Optional sanity checks (not run automatically):"
echo "  # mixed-language clips are now handled by Step 4 (lang-filter →"
echo "  #   train/lang_exclusions.jsonl); audit_pimsleur_mixing.py was the"
echo "  #   heuristic precursor."
echo "  python3 scripts/audit_nasals.py           # nasal-vowel labeling spot-check"

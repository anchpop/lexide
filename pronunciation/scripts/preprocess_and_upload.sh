#!/usr/bin/env bash
#
# Process whatever raw data is currently in data/audio/<lang>/manifest.jsonl
# and push the result to the HuggingFace dataset. Idempotent and safe to
# re-run; each step skips itself when its inputs haven't changed.
#
# Steps:
#   1. Audit FLEURS audio (Whisper + phoneme-level CER) → train/fleurs_asr_exclusions.jsonl
#   2. Audit Tatoeba audio (same) → train/tatoeba_asr_exclusions.jsonl
#   3. relabel-french: LLM-labeled rhythmic-group stress → data/audio/fra/stress_overrides.jsonl
#   4. preprocess.py: per-lang phonemes.jsonl + vad.jsonl from the manifest
#      (phonemize via espeak-ng; framewise VAD via vad_compute Rust binary)
#   5. upload_audio_to_hf.py: push data/audio/ to anchpop/lexide-pronunciation-audio
#
# This script does NOT acquire new data. If you have new Pimsleur lessons
# or Tatoeba records to ingest, run the appropriate downloader first:
#   python3 data/download_pimsleur.py [--recover-partials] ...
#   python3 data/download_tatoeba.py ...
# Then re-run this script.
#
# Required env vars (loaded from .env at repo root or pronunciation/.env):
#   GROQ_API_KEY    — Whisper audits (Groq Cloud)
#   OPENAI_API_KEY  — gpt-5.4-nano calls in relabel-french
#   HF_TOKEN        — HuggingFace dataset upload
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

echo "=== Step 1/5: FLEURS audit (Groq Whisper) ==="
python3 scripts/audit_fleurs_groq.py

echo
echo "=== Step 2/5: Tatoeba audit (Groq Whisper) ==="
python3 scripts/audit_tatoeba_groq.py

echo
echo "=== Step 3/5: French rhythmic-group stress relabel ==="
# espeak emits per-word stress for French, which is systematically wrong:
# French stress falls on the final syllable of each rhythmic group, not on
# every word. tysm's prompt-aware caching makes re-runs free if no rows
# changed; otherwise only new rows hit the LLM.
cargo run --release --manifest-path train/relabel-french/Cargo.toml --quiet

echo
echo "=== Step 4/5: Phonemize + recompute VAD ==="
# preprocess.py rebuilds vad.jsonl via vad_compute as it goes, keeping VAD
# coverage in lockstep with phonemes. --skip-vad if you regenerated phonemes
# for a label-only fix that didn't change which audio files are referenced.
python3 train/scripts/preprocess.py

echo
echo "=== Step 5/5: Upload to HF dataset ==="
python3 scripts/upload_audio_to_hf.py

echo
echo "=== DONE ==="
echo "Optional sanity checks (not run automatically):"
echo "  python3 scripts/audit_pimsleur_mixing.py  # flag clips with English mixed in"
echo "  python3 scripts/audit_nasals.py           # nasal-vowel labeling spot-check"

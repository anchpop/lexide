#!/usr/bin/env bash
#
# Orchestrator: run the full data-refresh pipeline end-to-end after
# downloads finish. Idempotent and resumeable — each script tolerates
# being re-run.
#
# Steps:
#   1. Tatoeba audit (Groq Whisper, phoneme-level CER) → train/tatoeba_asr_exclusions.jsonl
#   2. Pimsleur extraction (VAD + Whisper) — for whatever languages the
#      `ia download p_rty` task has finished by now.
#   3. preprocess.py to regenerate per-lang phonemes.jsonl from the
#      now-bigger manifest.jsonl files.
#   4. upload_audio_to_hf.py to push to the HF dataset repo.
#
# Required env vars:
#   GROQ_API_KEY — for both audit and Pimsleur transcription
#   HF_TOKEN     — for the final upload
# Both can sit in pronunciation/.env or the repo root .env.
#
# Run from pronunciation/ directory.

set -euo pipefail

# Pull keys from a .env if present.
if [ -f ../.env ]; then set -a; . ../.env; set +a; fi
if [ -f .env ]; then set -a; . .env; set +a; fi

if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "ERROR: GROQ_API_KEY is not set" >&2
  exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set" >&2
  exit 1
fi

echo "=== Step 1/4: Tatoeba audit ==="
python3 scripts/audit_tatoeba_groq.py

echo
echo "=== Step 2/4: Pimsleur extraction (whatever's available on T7) ==="
if [ -d "/Volumes/T7/p_rty/Pimsleur Complete Collection" ]; then
  python3 data/download_pimsleur.py
else
  echo "  T7 not mounted; skipping Pimsleur step"
fi

echo
echo "=== Step 3/4: Regenerate per-lang phonemes.jsonl ==="
python3 train/scripts/preprocess.py

echo
echo "=== Step 4/4: Upload to HF dataset ==="
python3 scripts/upload_audio_to_hf.py

echo
echo "=== DONE ==="
echo "Next training run (sky_*.yaml → train_unified.py) will auto-load:"
echo "  - the enriched anchpop/lexide-pronunciation-audio dataset"
echo "  - fleurs_asr_exclusions.jsonl + tatoeba_asr_exclusions.jsonl"

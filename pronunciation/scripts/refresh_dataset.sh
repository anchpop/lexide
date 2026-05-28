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

echo "=== Step 1/5: Tatoeba audit ==="
python3 scripts/audit_tatoeba_groq.py

echo
echo "=== Step 2/5: Backfill Pimsleur resume markers ==="
python3 scripts/backfill_pimsleur_markers.py

echo
echo "=== Step 3/5: Pimsleur extraction (whatever's available on T7) ==="
# Use full whisper-large-v3 for Pimsleur because clips are very short
# (1-5 sec) — Whisper's short-audio regime is where turbo's accuracy
# gap widens. Pimsleur transcripts BECOME the training labels here,
# so transcript quality matters more than throughput.
if [ -d "/Volumes/T7/p_rty/Pimsleur Complete Collection" ]; then
  python3 data/download_pimsleur.py --model whisper-large-v3
else
  echo "  T7 not mounted; skipping Pimsleur step"
fi

echo
echo "=== Step 4/5: Regenerate per-lang phonemes.jsonl ==="
python3 train/scripts/preprocess.py

echo
echo "=== Step 5/5: Upload to HF dataset ==="
python3 scripts/upload_audio_to_hf.py

echo
echo "=== Optional: Pimsleur mixing audit ==="
echo "  Not run automatically (it's a sanity check, not blocking)."
echo "  To check for VAD-mixed English/target clips:"
echo "    python3 scripts/audit_pimsleur_mixing.py"

echo
echo "=== DONE ==="
echo "Next training run (sky_*.yaml → train_unified.py) will auto-load:"
echo "  - the enriched anchpop/lexide-pronunciation-audio dataset"
echo "  - fleurs_asr_exclusions.jsonl + tatoeba_asr_exclusions.jsonl"

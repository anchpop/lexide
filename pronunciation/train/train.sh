#!/usr/bin/env bash
#
# Shared launcher for every sky_*.yaml in this directory.
#
# Each variant's `run:` block calls `bash train/train.sh <variant flags>`.
# The variant flags are forwarded to train_unified.py via "$@" and override
# (or extend) the defaults below.
#
# What this script does, in order:
#   - export the env vars every run needs
#   - install panphon (idempotent on warm clusters; required for any run
#     touching feature heads)
#   - pull the latest anchpop/lexide-pronunciation-audio HF dataset
#   - launch train_unified with the common flags + variant-specific "$@"
#
# Run-specific flags expected to come in via "$@":
#   --vad-weight {0 | 0.05}
#   --use-features / --use-aux-features / --feature-emission-weight / ...
#   --regularized-heads
#   --save-dir checkpoints-...
#   --hf-repo anchpop/lexide-pronunciation-unified-...
#   --resume-from / --resume-epoch    (for resume variants)

set -euo pipefail

cd ~/sky_workdir

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="lexide-pronunciation"
# Set by SkyPilot from the `secrets:` block in the calling sky.yaml.
export WANDB_API_KEY=${WANDB_API_KEY:-}
export HF_TOKEN=${HF_TOKEN:-}
# Combats CUDA allocator fragmentation. Harmless when memory isn't tight;
# rescued an OOM during sky_articulatory_aux_regularized job 6 where 34 GB
# was "reserved but unallocated".
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Idempotent: no-op if already installed (sky exec on warm cluster).
pip install --quiet panphon

# Pull (or refresh) audio + phoneme labels from the private HF dataset —
# UNLESS data was pre-mounted via sky's file_mounts (rsync from local). The
# HF↔Lambda link is slow (an hour-plus for our ~19 GB); rsync from local
# usually drains the same payload in 15-30 min. If any lang's phonemes.jsonl
# is already present we trust the mount and skip the download.
if compgen -G "$HOME/data/*/phonemes.jsonl" > /dev/null; then
  echo "Using pre-mounted dataset at ~/data (skipping HF download)"
else
  hf download anchpop/lexide-pronunciation-audio --repo-type dataset --local-dir ~/data
fi

# Common train_unified knobs. Variant-specific flags come in via "$@" and
# argparse's last-wins behavior lets variants override any of these if needed.
python -m src.train_unified \
  --data-dir ~/data \
  --model-name facebook/wav2vec2-xls-r-2b \
  --processor-source facebook/wav2vec2-xlsr-53-espeak-cv-ft \
  --gradient-checkpointing \
  --epochs 7 \
  --batch-size 16 \
  --backbone-lr 1e-5 \
  --head-lr 1e-3 \
  --stress-weight 0.3 \
  --stress-warmup-steps 400 \
  --num-workers 16 \
  --bf16 \
  "$@"

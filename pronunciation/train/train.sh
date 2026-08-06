#!/usr/bin/env bash
#
# Launcher for sky_train.yaml.
#
# What this script does, in order:
#   - export the env vars every run needs
#   - install panphon (idempotent on warm clusters)
#   - assert the staged dataset is present at ~/data
#   - launch train_unified, forwarding "$@"
#
# This used to pass seventeen flags pinning the champion recipe. Every one of
# them turned out to already be train_unified.py's argparse default — model,
# lrs, epochs, batch, bf16, worker count, and the mel-sidechannel / mlp-heads /
# audio-degrade / use-narrowed booleans alike — so they only obscured which
# settings a run actually chose. They're gone: the defaults ARE the champion,
# and anything passed via "$@" now means "deliberately deviate from it".

set -euo pipefail

# `python -m src.train_unified` resolves src/ relative to CWD, and src/ lives
# at ~/sky_workdir/train/src/ (workdir is the pronunciation dir).
cd ~/sky_workdir/train

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

# The dataset arrives as one tarball staged by sky's file_mounts and untarred
# into ~/data by the calling yaml's run: block. There is no download fallback
# on purpose: ~450k loose wavs defeat every remote transport we tried — rsync
# crawls on them, and the Hub's 256-commits/hour ceiling turned a loose-file
# dataset repo into a multi-day upload that stalled outright. One rsync'd file
# beats both. Build it with `python train/scripts/preprocess.py`.
#
# Fail closed: a missing mount used to silently fall through to a slow
# re-download, but the worse failure is training on a partial corpus.
if ! compgen -G "$HOME/data/*/phonemes.jsonl" > /dev/null; then
  echo "ERROR: no dataset at ~/data (expected ~/data/<lang>/phonemes.jsonl)." >&2
  echo "  The calling sky_*.yaml should untar ~/data.tar into ~/data before this runs." >&2
  echo "  Rebuild the tar locally with: python train/scripts/preprocess.py" >&2
  exit 1
fi
echo "Using dataset at ~/data: $(ls ~/data | tr '\n' ' ')"

# --data-dir is the only setting this script pins: it's where the yaml untarred
# the dataset, not a modelling choice. Everything else is left at its default.
python -m src.train_unified \
  --data-dir ~/data \
  "$@"

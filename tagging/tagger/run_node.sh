#!/usr/bin/env bash
# Node-side orchestration for the tagger training job.
# Runs on the Lambda GPU node. Trains the multi-task tagger (with an on-GPU smoke phase
# first), pushes it to HF, then trains the char boundary tokenizer and pushes that too.
# Artifacts MUST go to HF because sky autodown destroys the node disk.
set -euo pipefail

cd ~/sky_workdir
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-lexide-parsley}"
export HF_HUB_DISABLE_XET=1   # classic LFS; the xet backend stalled on prior runs

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Ensure a driver-compatible CUDA torch. A prior setup may have installed the default pip
# wheel (now cu13), which silently falls back to CPU on Lambda's ~570 driver. Force cu128.
if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "torch cannot see CUDA -> installing cu128 build"
  pip install --quiet "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
fi
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); assert torch.cuda.is_available(), 'no CUDA after torch-ensure'"

export HF_USER=$(python3 -c "from huggingface_hub import whoami; import os; print(whoami(token=os.environ['HF_TOKEN'])['name'])")
REPO="${HF_USER}/lexide-parsley"
echo "=== HF target repo: ${REPO} ==="

cd tagger

# ---------------- multi-task tagger (smoke phase runs first, inside train.py) ----------
echo "=== TRAIN TAGGER ==="
python3 train.py \
  --data-dir ../data/processed \
  --out-dir output/tagger \
  --encoder "${ENCODER:-xlm-roberta-base}" \
  --amp --smoke --wandb \
  --batch-size "${BATCH:-64}" \
  --eval-batch-size 128 \
  --epochs "${EPOCHS:-2}" \
  --eval-every "${EVAL_EVERY:-3000}" \
  --workers 8

echo "=== PUSH TAGGER -> ${REPO} ==="
python3 - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = f"{os.environ['HF_USER']}/lexide-parsley"
tok = os.environ["HF_TOKEN"]
create_repo(repo, exist_ok=True, token=tok)
HfApi().upload_folder(folder_path="output/tagger", path_in_repo="tagger",
                      repo_id=repo, token=tok)
print("pushed tagger to", repo)
PY

# ---------------- char boundary tokenizer ----------------------------------------------
# The prior flags are not optional extras: a tokenizer trained without them scores jpn 74.6
# against 92.1 (OVERVIEW.md), and tha/zho-hans have no whitespace to fall back on either.
# --prior-sidecar reads the proposals precomputed on the box by the Rust `emit-priors`, so
# training and inference share one implementation. Its absence is a hard error rather than
# a silent downgrade to a prior-free model that would then be pushed to the shipping path.
if [ "${SKIP_TOK:-0}" != "1" ]; then
echo "=== TRAIN CHAR TOKENIZER ==="
SIDECAR="${TOK_PRIOR_SIDECAR:-../data/processed/priors}"
for split in train val; do
  [ -f "${SIDECAR}.${split}" ] || {
    echo "ERROR: no prior sidecar at ${SIDECAR}.${split}. Build it on the box before" >&2
    echo "       launching: cargo run --bin emit-priors -- --unidic data/priors/jpn-unidic.bin" >&2
    echo "       --wordbank-dir data/priors/wordbanks --out data/processed/priors.${split}" >&2
    echo "       < data/processed/${split}.jsonl   (see tagger/README.md)" >&2
    exit 1
  }
done
# TOK_LIMIT=0 (the default) trains on the whole corpus. A limit takes a *prefix* of
# train.jsonl, which is written language-by-language — so it does not subsample, it drops
# whichever languages sort last. That is how the low-resource languages this run exists to
# improve would be the ones excluded.
TOK_LIMIT_ARG=()
[ "${TOK_LIMIT:-0}" != "0" ] && TOK_LIMIT_ARG=(--train-limit "${TOK_LIMIT}")
# TOK_BATCH: 128 fits a 24GB A10, which is what sky_tagger.yaml's accelerator list can land
# on. 256 (what byte-v13 used on a 48GB A6000) OOMs here at ~step 300 in the minGRU scan —
# and the batches are heavier than they were, because Thai is 3 UTF-8 bytes per character so
# far more sequences reach the 512-byte cap. expandable_segments costs nothing and buys back
# the ~1.2GB that fragments out of the doubling scan's allocation pattern.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python3 train_tokenizer.py \
  --data-dir ../data/processed \
  --out-dir output/tokenizer \
  --epochs "${TOK_EPOCHS:-1}" \
  "${TOK_LIMIT_ARG[@]}" \
  --batch-size "${TOK_BATCH:-128}" \
  --eval-every 4000 \
  --use-prior --prior-mode concat --prior-dim 8 \
  --prior-sidecar "${SIDECAR}" \
  --wandb

echo "=== PUSH TOKENIZER -> ${REPO} ==="
python3 - <<'PY'
import os
from huggingface_hub import HfApi
repo = f"{os.environ['HF_USER']}/lexide-parsley"
HfApi().upload_folder(folder_path="output/tokenizer", path_in_repo="tokenizer",
                      repo_id=repo, token=os.environ["HF_TOKEN"])
print("pushed tokenizer to", repo)
PY
fi

echo "=== ALL DONE ==="

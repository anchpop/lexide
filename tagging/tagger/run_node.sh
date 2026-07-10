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
echo "=== TRAIN CHAR TOKENIZER ==="
python3 train_tokenizer.py \
  --data-dir ../data/processed \
  --out-dir output/tokenizer \
  --epochs "${TOK_EPOCHS:-1}" \
  --train-limit "${TOK_LIMIT:-800000}" \
  --batch-size 256 \
  --eval-every 4000 \
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

echo "=== ALL DONE ==="

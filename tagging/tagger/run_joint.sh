#!/usr/bin/env bash
# GH200/H100/A100 launch entrypoint. No cloud provisioning happens here.
# Unmeasured planning estimate: bge-m3/18 layers, 4096 padded pieces/batch,
# FP32 AdamW + bf16 activations should fit in ~25–40 GB, ~0.4–1.5 s/step.
# Character lengths affect memory too. Measure the smoke, then adjust the token
# budget; these are assumptions, not measured GPU results or a $70 guarantee.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir/.."
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
: "${HF_TOKEN:?HF_TOKEN must be exported for checkpoints to survive autodown}"
python3 -c 'import torch; assert torch.cuda.is_available(), "Refusing CPU training"'
run="${RUN_NAME:-joint-v1}"
output="$script_dir/output/$run"
data="${JOINT_DATA:-data/processed-joint}"
args=(
  --data "$data" --output "$output"
  --encoder "${ENCODER:-BAAI/bge-m3}" --encoder-layers "${ENCODER_LAYERS:-18}"
  --encoder-revision "${ENCODER_REVISION:-5617a9f61b028005a4858fdac845db406aefb181}"  # yap's bge-m3 pin
  --epochs "${EPOCHS:-2}" --lang-cap "${LANG_CAP:-300000}"
  --token-budget "${TOKEN_BUDGET:-4096}" --eval-every "${EVAL_EVERY:-1000}"
  --hf-repo anchpop/lexide-parsley --hf-path "training-runs/$run"
  --smoke
)
if [[ -n "${WANDB_PROJECT:-}" ]]; then args+=(--wandb-project "$WANDB_PROJECT"); fi
python3 -u "$script_dir/train_joint.py" "${args[@]}" "$@"
python3 "$script_dir/predict_joint.py" --checkpoint "$output/best" \
  --input "$data/test.jsonl" --output "$output/test_predictions.jsonl" --device cuda
python3 "$script_dir/eval_e2e.py" --gold "$data/test.jsonl" \
  --predictions "$output/test_predictions.jsonl" --output "$output/test_metrics"
# Training already uploads improvements (at most every 30 minutes) and forces
# a final best upload. This last upload adds held-out scores and predictions.
python3 - "$output" "$run" <<'PY'
import sys
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="anchpop/lexide-parsley", folder_path=sys.argv[1],
                      path_in_repo=f"training-runs/{sys.argv[2]}", ignore_patterns=["smoke/*", "*.tmp"],
                      commit_message="parsley-joint final test metrics")
PY

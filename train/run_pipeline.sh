#!/bin/bash
set -e

# Run the full 2-stage NLP pipeline training
# Usage: ./run_pipeline.sh [local|sky]

MODE=${1:-sky}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load env vars
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ "$MODE" = "local" ]; then
    echo "Running locally..."

    python src/stage1_train.py \
        --data_dir data \
        --output_dir output/stage1 \
        --max_length 2048 \
        --batch_size 4 \
        --grad_accum 4 \
        --num_epochs 10 \
        --backbone_lr 2e-5 \
        --head_lr 1e-3 \
        --decoder_lr 1e-4 \
        --lemma_tokens_per_sent 4 \
        --num_workers 2 \
        --wandb \
        --run_name stage1-unified-local

    python src/eval_pipeline.py \
        --data_dir data \
        --model_path output/stage1/best_model.pt \
        --output_dir output/eval \
        --eval_sent_boundary

elif [ "$MODE" = "sky" ]; then
    echo "Launching on Lambda via SkyPilot..."
    echo "Using autostop=10min, autodown=true"

    sky launch sky_pipeline.yaml \
        --env WANDB_API_KEY="$WANDB_API_KEY" \
        --env HF_TOKEN="$HF_TOKEN" \
        --down \
        -y

else
    echo "Usage: ./run_pipeline.sh [local|sky]"
    exit 1
fi

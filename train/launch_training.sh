#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="gemma-multilingual-nlp"
export TOKENIZERS_PARALLELISM=false

echo "Starting Gemma fine-tuning for multilingual NLP tasks..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Activate virtual environment and run with uv
uv run accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    src/train.py

echo "Training completed!"
# Lexide - Multilingual Linguistic Analysis with Gemma

Fine-tuning Gemma 3 for multilingual part-of-speech tagging, lemmatization, and dependency parsing.

This repository contains the training code, the training data, and a REX library that can be used to actually use the model with a convenient API.

[More info on Huggingface!](https://huggingface.co/collections/anchpop/lexide-nlp-models)

## Overview

This project fine-tunes Google's Gemma 3 1B model to perform linguistic analysis across 7 languages:
- English (~11K samples)
- German (~11K samples)
- French (~11K samples)
- Italian (~11K samples)
- Korean (~10K samples)
- Portuguese (~11K samples)
- Spanish (~11K samples)

The model learns to analyze sentences and output structured linguistic information including POS tags, lemmas, and syntactic dependencies.

## Quick Start

### Skypilot

1. [Install skypilot](https://docs.skypilot.co/en/latest/getting-started/installation.html)

2. Start job

```bash
sky start lexide
sky launch -c lexide  --secret HF_TOKEN --secret WANDB_API_KEY sky.yaml
```

### Installation

```bash
# Clone repository
git clone <repo-url>
cd lexide

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Pull data files (requires Git LFS)
git lfs pull
```

### Training

```bash
# Login to Hugging Face (for Gemma model access)
huggingface-cli login

# Start training
./launch_training.sh
```
### Inference

```bash
# Test the latest checkpoint
uv run python test_checkpoint.py

# Interactive inference
uv run python inference_example.py
```

## Model Output Format

Given an input sentence, the model outputs linguistic analysis in this format:

**Input:**
```
Language: English
Sentence: I don't have them.
Task: Analyze tokens (idx,token,ws,POS,lemma,dep,head)
```

**Output:**
```
Here's the token analysis:

1	I	none	PRON	I	nsubj	4
2	 do	none	AUX	do	aux	4
3	n't	none	PART	not	advmod	4
4	 have	none	VERB	have	ROOT	0
5	 them	none	PRON	they	obj	4
6	.	none	PUNCT	.	punct	4

</analysis>
```

The output format is tab-separated with columns: index, token, whitespace (none/_ for space), POS tag, lemma, dependency label, and head index.

## Architecture

- **Base Model**: Google Gemma 3 1B-IT (instruction-tuned)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: Q, K, V, O projections + MLP layers
- **Training**: Mixed precision (bfloat16) with gradient accumulation

## Project Structure

```
lexide/
├── train/
│   ├── data/                      # Training data (JSONL format)
│   │   ├── cleaned_eng.jsonl     # English samples
│   │   ├── cleaned_deu.jsonl     # German samples
│   │   ├── cleaned_fra.jsonl     # French samples
│   │   ├── cleaned_ita.jsonl     # Italian samples
│   │   ├── cleaned_kor.jsonl     # Korean samples
│   │   ├── cleaned_por.jsonl     # Portuguese samples
│   │   └── cleaned_spa.jsonl     # Spanish samples
│   ├── src/
│   │   ├── data_loader.py        # Data loading and preprocessing
│   │   ├── train.py              # Main training script
│   │   ├── inference.py          # Inference utilities
│   │   └── evaluate.py           # Model evaluation
│   ├── launch_training.sh        # Training launch script
│   ├── main.py                   # Training entry point
│   ├── inference_example.py      # Interactive inference
│   ├── test_checkpoint.py        # Quick testing script
│   └── sky.yaml                  # Skypilot configuration
├── lexide/                        # Rust library for linguistic analysis
└── modal/                         # Modal deployment scripts
```

## Configuration

Edit `config.yaml` to adjust training parameters:

```yaml
model:
  name: "google/gemma-3-1b-it"  # Model variant
  use_4bit: false               # Quantization settings

training:
  batch_size: 8                 # Adjust based on GPU memory
  learning_rate: 2e-4
  num_epochs: 3
  
lora:
  r: 16                        # LoRA rank
  alpha: 32                    # LoRA scaling
```

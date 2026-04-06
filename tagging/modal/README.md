# Lexide Modal Deployment

Serves the Gemma 3 27B model with LoRA adapter using vLLM on Modal.

## Setup

1. Install Modal CLI:
```bash
pip install modal
```

2. Authenticate:
```bash
modal setup
```

3. Create HuggingFace secret:
```bash
modal secret create huggingface-secret HF_TOKEN=your_hf_token_here
```

## Usage

### Download and merge model (run once):
```bash
modal run modal/modal_serve.py --action merge
```

### Test inference:
```bash
modal run modal/modal_serve.py --action test
```

### Deploy to production:
```bash
modal deploy modal/modal_serve.py
```

## Model Configuration

- **Base model**: `google/gemma-3-27b-it`
- **LoRA adapter**: `anchpop/lexide-gemma-3-27b-it`
- **Merged model path**: `/models/merged` (on Modal volume)

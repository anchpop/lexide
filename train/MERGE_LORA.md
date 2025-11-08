# Merging LoRA Adapter for Inference

After training a LoRA adapter, you need to merge it with the base model to create a standalone model for inference in Rust.

## Why Merge?

- **LoRA adapters** are lightweight fine-tuning weights that need to be combined with the base model
- **Candle** (the Rust inference framework) doesn't have full LoRA support yet
- **Merged models** are self-contained and easier to deploy

## How to Merge

### 1. Run the merge script

```bash
cd train
python merge_lora.py
```

This will:
1. Download the base Gemma 3 1B model
2. Download your LoRA adapter from HuggingFace
3. Merge them together
4. Upload the merged model to `anchpop/lexide-gemma-3-1b-it-merged`

### 2. Custom options

```bash
# Just save locally, don't upload
python merge_lora.py --no-push

# Use a different output name
python merge_lora.py --output yourusername/my-model-name

# Use a different base model
python merge_lora.py --base-model google/gemma-3-1b-it
```

### 3. Update your Rust code

After merging and uploading, update your Rust code to use the merged model:

```rust
let config = LexideConfig {
    model_repo: "anchpop/lexide-gemma-3-1b-it-merged".to_string(),
    ..Default::default()
};
```

## File Comparison

### LoRA Adapter Repository
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - LoRA weights (~52 MB)
- `tokenizer.json` - Tokenizer
- ❌ No `config.json` (missing!)
- ❌ No full model weights

### Merged Model Repository
- `config.json` - ✅ Full model config
- `model.safetensors` - ✅ Full merged weights (~5-6 GB)
- `tokenizer.json` - Tokenizer
- Ready for Candle inference! ✅

## Troubleshooting

### Out of Memory
If you get OOM errors during merging:
```bash
# Use CPU offloading
export CUDA_VISIBLE_DEVICES=""
python merge_lora.py
```

### HuggingFace Token
If you need to upload a private model:
```bash
huggingface-cli login
python merge_lora.py
```

## Alternative: Use the Base Model Config

If you don't want to merge (e.g., to save space), you could modify the Rust code to:
1. Load config from base model (`google/gemma-3-1b-it`)
2. Load weights from your adapter

However, this requires implementing LoRA application in Candle, which is more complex.

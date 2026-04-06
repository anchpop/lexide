#!/usr/bin/env python3
"""
Modal deployment script for serving Gemma 4 31B with LoRA adapter using vLLM.

This script:
1. Downloads the base Gemma 4 31B model
2. Downloads the LoRA adapter from anchpop/lexide-gemma-4-31B-it
3. Merges the weights
4. Serves the merged model with vLLM

Usage:
    modal deploy modal/modal_serve.py
"""

import modal

# Create Modal app
app = modal.App("lexide-gemma-4-31b-vllm")

# Define the container image with all dependencies
image = (
    modal.Image.from_registry("vllm/vllm-openai:gemma4", add_python="3.11")
    .entrypoint([])  # Clear vLLM entrypoint so Modal can inject its own
    .pip_install("peft", "huggingface_hub", "pillow", "torchvision", "librosa", "soundfile")
)

# Model configuration
BASE_MODEL = "google/gemma-4-31B-it"
LORA_ADAPTER = "anchpop/lexide-gemma-4-31B-it"
MODEL_DIR = "/models/merged-gemma4"

# Shared volume for model weights
volume = modal.Volume.from_name("lexide-models", create_if_missing=True)

# Hugging Face secret for authentication
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    gpu="A100-80GB",  # 27B model needs substantial GPU memory
    volumes={"/models": volume},
    secrets=[hf_secret],
    timeout=3600,  # 1 hour timeout for downloading and merging
)
def download_and_merge_model():
    """
    Download base model and LoRA adapter, then merge them.
    This runs once to prepare the model for serving.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import os

    # The HF_TOKEN environment variable is automatically set from the Modal secret
    print(f"Downloading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Temporarily replace Gemma4ClippableLinear with nn.Linear so PEFT can load the adapter
    # (LoRA targets q_proj/v_proj etc. which exist in both text and vision, but
    #  vision uses Gemma4ClippableLinear which PEFT doesn't support)
    replaced = {}
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        for name, module in base_model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, Gemma4ClippableLinear):
                    replaced[(name, child_name)] = child
                    setattr(module, child_name, child.linear)
        print(f"Temporarily replaced {len(replaced)} Gemma4ClippableLinear layers for PEFT")
    except ImportError:
        pass

    print(f"Downloading LoRA adapter: {LORA_ADAPTER}")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER,
        token=os.environ.get("HF_TOKEN"),
    )

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    # Restore Gemma4ClippableLinear wrappers so vLLM can load the model
    if replaced:
        for (parent_name, child_name), original in replaced.items():
            # Find the parent module in the merged model
            parent = merged_model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            # The inner linear may have been updated by LoRA merge, grab current weights
            current_linear = getattr(parent, child_name)
            original.linear.weight = current_linear.weight
            if current_linear.bias is not None:
                original.linear.bias = current_linear.bias
            setattr(parent, child_name, original)
        print(f"Restored {len(replaced)} Gemma4ClippableLinear wrappers")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    print("Loading processor (for vLLM multimodal compat)...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Save merged model to persistent volume
    print(f"Saving merged model to {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    merged_model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)

    # Fix config.json for vLLM compatibility (add rope_type if missing)
    import json
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    if "rope_scaling" in config and not config.get("rope_scaling", {}).get("rope_type"):
        config["rope_scaling"]["rope_type"] = config["rope_scaling"].get("type", "linear")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Fixed rope_scaling in config.json for vLLM compatibility")

    # Commit volume changes
    volume.commit()
    print("Model merge complete and saved to volume!")


VLLM_PORT = 8000


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": volume},
    secrets=[hf_secret],
    scaledown_window=5 * 60,  # Keep container warm for 5 minutes
    timeout=10 * 60,
)
@modal.concurrent(max_inputs=360, target_inputs=160)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    """
    Serve the vLLM model with OpenAI-compatible API.
    This runs vLLM's built-in server which handles batching automatically.
    """
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_DIR,
        "--uvicorn-log-level=info",
        "--served-model-name",
        "lexide-gemma-4-31b",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.9",
        "--trust-remote-code",
    ]

    print(" ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[hf_secret],
    timeout=300,
)
def add_processor():
    """Download and save the AutoProcessor to the existing merged model dir."""
    import os
    from transformers import AutoProcessor
    volume.reload()
    print("Downloading processor from base model...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Saving processor to {MODEL_DIR}")
    processor.save_pretrained(MODEL_DIR)
    volume.commit()
    print("Done!")


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=60,
)
def fix_config():
    """Fix the saved config.json for vLLM compatibility."""
    import json, os
    volume.reload()
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    # Remove any bogus top-level rope_scaling we may have added previously
    if "rope_scaling" in config:
        del config["rope_scaling"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Removed bogus top-level rope_scaling from config.json")
    else:
        print("Config looks clean")
    volume.commit()


@app.local_entrypoint()
def main(action: str = "merge"):
    """
    Local entrypoint for testing and setup.

    Args:
        action: Either "merge" to download and merge the model, or "test" to test inference
    """
    if action == "merge":
        print("Downloading and merging model...")
        download_and_merge_model.remote()
        print("\n✓ Model merge complete!")
        print("\nNext steps:")
        print("  1. Test the model: modal run modal_serve.py --action test")
        print("  2. Deploy to production: modal deploy modal/modal_serve.py")
    elif action == "test":
        import requests

        print("Testing model inference...")

        # Get the server URL
        url = serve.web_url
        print(f"Server URL: {url}")

        # Use the same prompt format as the Rust code (matches parsing.rs::create_prompt)
        test_sentence = "The quick brown fox jumps over the lazy dog."
        test_prompt = f"""Language: English
Sentence: {test_sentence}
Task: Analyze tokens (idx,token,ws,POS,lemma,dep,head)
"""

        # Use OpenAI-compatible chat completions endpoint
        response = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "lexide-gemma-4-31b",
                "messages": [{"role": "user", "content": test_prompt}],
                "temperature": 0.1,
                "max_tokens": 512,
            },
            timeout=300,
        )
        response.raise_for_status()

        result = response.json()["choices"][0]["message"]["content"]

        print(f"\n✓ Test successful!")
        print(f"\nPrompt:\n{test_prompt}")
        print(f"Result:\n{result}")
    elif action == "fix":
        print("Fixing config.json on volume...")
        fix_config.remote()
        print("Done!")
    elif action == "add-processor":
        print("Adding processor to merged model dir...")
        add_processor.remote()
        print("Done!")
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: merge, test, fix, add-processor")

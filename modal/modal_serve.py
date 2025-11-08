#!/usr/bin/env python3
"""
Modal deployment script for serving Gemma 3 27B with LoRA adapter using vLLM.

This script:
1. Downloads the base Gemma 3 27B model
2. Downloads the LoRA adapter from anchpop/lexide-gemma-3-27b-it
3. Merges the weights
4. Serves the merged model with vLLM

Usage:
    modal deploy modal/modal_serve.py
"""

import modal

# Create Modal app
app = modal.App("lexide-gemma-3-27b-vllm")

# Define the container image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm>=0.6.11",  # Gemma 3 support added in recent versions
    "torch>=2.4.0",
    "transformers",
    "peft",
    "huggingface_hub",
    "fastapi[standard]",
)

# Model configuration
BASE_MODEL = "google/gemma-3-27b-it"
LORA_ADAPTER = "anchpop/lexide-gemma-3-27b-it"
MODEL_DIR = "/models/merged"

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

    print(f"Downloading LoRA adapter: {LORA_ADAPTER}")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER,
        token=os.environ.get("HF_TOKEN"),
    )

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Load image processor (Gemma 3 is multimodal)
    print("Loading image processor...")
    from transformers import AutoImageProcessor

    try:
        image_processor = AutoImageProcessor.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as e:
        print(f"Warning: Could not load image processor: {e}")
        image_processor = None

    # Save merged model to persistent volume
    print(f"Saving merged model to {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    merged_model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Save image processor if it exists
    if image_processor is not None:
        image_processor.save_pretrained(MODEL_DIR)

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
@modal.concurrent(max_inputs=180)  # Handle up to 180 concurrent requests
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
        "--uvicorn-log-level=info",
        MODEL_DIR,
        "--served-model-name",
        "lexide-gemma-3-27b",
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
                "model": "lexide-gemma-3-27b",
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
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: merge, test")

#!/usr/bin/env python3
"""
Merge LoRA adapter weights with base model and upload to HuggingFace Hub.
This creates a standalone model that can be used directly for inference.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def merge_and_upload(
    base_model_name: str = "google/gemma-3-1b-it",
    adapter_model_name: str = "anchpop/lexide-gemma-3-1b-it",
    output_name: str = "anchpop/lexide-gemma-3-1b-it-merged",
    push_to_hub: bool = True,
):
    """
    Merge LoRA adapter with base model and optionally upload to HuggingFace Hub.

    Args:
        base_model_name: Name of the base model on HuggingFace
        adapter_model_name: Name of the LoRA adapter on HuggingFace
        output_name: Name for the merged model (for HuggingFace Hub)
        push_to_hub: Whether to push to HuggingFace Hub
    """

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter: {adapter_model_name}")
    model = PeftModel.from_pretrained(base_model, adapter_model_name)

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_model_name,
        trust_remote_code=True
    )

    # Save locally first
    local_output_dir = "./merged_model"
    print(f"Saving merged model to {local_output_dir}")
    merged_model.save_pretrained(local_output_dir)
    tokenizer.save_pretrained(local_output_dir)

    print(f"✓ Merged model saved locally to {local_output_dir}")

    if push_to_hub:
        print(f"\nPushing merged model to HuggingFace Hub: {output_name}")
        try:
            merged_model.push_to_hub(output_name, private=False)
            tokenizer.push_to_hub(output_name, private=False)
            print(f"✓ Model uploaded to: https://huggingface.co/{output_name}")
        except Exception as e:
            print(f"Error uploading to HuggingFace: {e}")
            print(f"Model is saved locally in {local_output_dir}")
    else:
        print(f"\nSkipping upload to HuggingFace Hub")
        print(f"Model is saved locally in {local_output_dir}")

    print("\n✓ Done!")
    print(f"\nTo use in Rust, update your config to use: {output_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base-model",
        default="google/gemma-3-1b-it",
        help="Base model name (default: google/gemma-3-1b-it)"
    )
    parser.add_argument(
        "--adapter",
        default="anchpop/lexide-gemma-3-1b-it",
        help="LoRA adapter name (default: anchpop/lexide-gemma-3-1b-it)"
    )
    parser.add_argument(
        "--output",
        default="anchpop/lexide-gemma-3-1b-it-merged",
        help="Output model name for HuggingFace Hub"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push to HuggingFace Hub, only save locally"
    )

    args = parser.parse_args()

    merge_and_upload(
        base_model_name=args.base_model,
        adapter_model_name=args.adapter,
        output_name=args.output,
        push_to_hub=not args.no_push
    )

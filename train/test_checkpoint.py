#!/usr/bin/env python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def test_checkpoint(checkpoint_path=None):
    import os
    import glob
    
    # If no checkpoint specified, find the latest one
    if checkpoint_path is None:
        checkpoints = glob.glob("./checkpoints/checkpoint-*")
        if checkpoints:
            # Sort by step number and get the latest
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            checkpoint_path = checkpoints[-1]
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = None
            print("No checkpoints found, using base model")
    
    print("Loading model...")
    
    # Load base model and tokenizer
    base_model = "google/gemma-3-270m-it"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Load LoRA adapter from checkpoint if it exists
    if checkpoint_path and os.path.exists(checkpoint_path) and os.path.exists(f"{checkpoint_path}/adapter_config.json"):
        print(f"Loading LoRA adapter from {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        if checkpoint_path:
            print(f"No valid checkpoint at {checkpoint_path}, using base model only")
    model.eval()
    
    # Test sentences
    test_cases = [
        ("I don't have them.", "eng", "English"),
        ("Je ne les ai pas.", "fra", "French"),
        ("Ich werde bald zurück sein.", "deu", "German"),
        ("Estoy pensando.", "spa", "Spanish"),
        ("저거로군!", "kor", "Korean"),
        # Additional test cases
        ("The cat sits on the mat.", "eng", "English"),
        ("Das ist sehr gut.", "deu", "German"),
    ]
    
    print("\n" + "="*70)
    if checkpoint_path:
        step = checkpoint_path.split("-")[-1] if "checkpoint-" in checkpoint_path else "unknown"
        print(f"TESTING CHECKPOINT AT STEP {step}")
    else:
        print("TESTING BASE MODEL (NO FINE-TUNING)")
    print("="*70 + "\n")
    
    for sentence, lang_code, lang_name in test_cases:
        prompt = f"""Language: {lang_name}
Sentence: {sentence}
Task: Analyze tokens (idx,token,POS,lemma,dep,head)

Analysis:"""
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        print(f"Language: {lang_name}")
        print(f"Input: {sentence}")
        print("-" * 50)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.95,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id  # Should stop at EOS
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the analysis part
        if "Analysis:" in response:
            analysis = response.split("Analysis:")[-1].strip()
            print("Output:")
            print(analysis)
        else:
            print("Output:", response)
        
        print("\n" + "="*70 + "\n")
    
    print("Format: idx<tab>token<tab>POS<tab>lemma<tab>dep<tab>head")
    print("\nQuality check:")
    print("- Tab-separated values for efficiency")
    print("- POS tags (NOUN, VERB, ADJ, etc.)")
    print("- Lemmas (base forms)")
    print("- Dependencies (ROOT, nsubj, obj, etc.)")
    print("- Head indices (pointing to governor tokens)")

if __name__ == "__main__":
    test_checkpoint()
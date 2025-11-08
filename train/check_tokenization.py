#!/usr/bin/env python
"""Check how the training data is actually tokenized"""

from transformers import AutoTokenizer
from src.data_loader import MultilingualNLPDataset
import json

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

print(f"Tokenizer EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
print(f"Tokenizer PAD token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
print("-" * 60)

# Load a sample
dataset_loader = MultilingualNLPDataset(data_dir="data")

# Get one sample
import jsonlines
with jsonlines.open("data/eng.jsonl") as reader:
    sample = next(iter(reader))

processed = dataset_loader.process_sample(sample, "eng")
prompt_response = dataset_loader.create_prompt_response(processed)

print("Full text that will be tokenized:")
print(prompt_response["full_text"])
print("-" * 60)

# Tokenize it
tokens = tokenizer(
    prompt_response["full_text"],
    max_length=512,
    padding=False,  # Don't pad to see actual tokens
    truncation=True,
    return_tensors=None
)

print(f"\nNumber of tokens: {len(tokens['input_ids'])}")
print(f"Last 10 token IDs: {tokens['input_ids'][-10:]}")
print(f"Last 10 tokens decoded individually:")
for token_id in tokens['input_ids'][-10:]:
    print(f"  {token_id}: '{tokenizer.decode([token_id])}'")

print("\n" + "=" * 60)
print("CHECKING: Does the last token match EOS?")
if tokens['input_ids'][-1] == tokenizer.eos_token_id:
    print("✅ YES - EOS token is present at the end!")
else:
    print(f"❌ NO - Last token {tokens['input_ids'][-1]} is not EOS token {tokenizer.eos_token_id}")
    print("The tokenizer is NOT adding EOS automatically!")

print("\nTo fix: We need to explicitly add the EOS token to training data")
print("=" * 60)

# Now check with explicit EOS
print("\nChecking with explicit <eos> token added:")
text_with_eos = prompt_response["full_text"] + "<eos>"
tokens_with_eos = tokenizer(text_with_eos, max_length=512, padding=False, truncation=True, return_tensors=None)
print(f"Last 5 tokens: {tokens_with_eos['input_ids'][-5:]}")
for token_id in tokens_with_eos['input_ids'][-5:]:
    print(f"  {token_id}: '{tokenizer.decode([token_id])}'")
    
if tokens_with_eos['input_ids'][-1] == tokenizer.eos_token_id:
    print("✅ With explicit <eos>: EOS token is now present!")
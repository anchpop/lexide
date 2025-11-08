#!/usr/bin/env python
"""Test token efficiency of old vs new format"""

from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# Sample data
tokens = ["I", "do", "n't", "have", "them", "."]
pos_tags = ["PRON", "AUX", "PART", "VERB", "PRON", "PUNCT"]
lemmas = ["I", "do", "not", "have", "they", "."]
deps = ["nsubj", "aux", "neg", "ROOT", "dobj", "punct"]
heads = ["3", "3", "3", "3", "3", "3"]

# Old verbose format
old_format_parts = []
for i, token in enumerate(tokens):
    old_format_parts.append(
        f"{i}: {token} | POS: {pos_tags[i]} | "
        f"Lemma: {lemmas[i]} | Dep: {deps[i]} | "
        f"Head: {heads[i]}"
    )
old_format = "\n".join(old_format_parts)

# New compact format
new_format_parts = []
for i, token in enumerate(tokens):
    new_format_parts.append(
        f"{i}\t{token}\t{pos_tags[i]}\t"
        f"{lemmas[i]}\t{deps[i]}\t{heads[i]}"
    )
new_format = "\n".join(new_format_parts)

print("="*60)
print("TOKEN EFFICIENCY COMPARISON")
print("="*60)

print("\nOLD FORMAT:")
print(old_format)
old_tokens = tokenizer.encode(old_format, add_special_tokens=False)
print(f"\nCharacters: {len(old_format)}")
print(f"Tokens: {len(old_tokens)}")

print("\n" + "-"*60)

print("\nNEW FORMAT:")
print(new_format)
new_tokens = tokenizer.encode(new_format, add_special_tokens=False)
print(f"\nCharacters: {len(new_format)}")
print(f"Tokens: {len(new_tokens)}")

print("\n" + "="*60)
print("RESULTS:")
print(f"Character reduction: {len(old_format)} → {len(new_format)} ({100*(len(old_format)-len(new_format))/len(old_format):.1f}% saved)")
print(f"Token reduction: {len(old_tokens)} → {len(new_tokens)} ({100*(len(old_tokens)-len(new_tokens))/len(old_tokens):.1f}% saved)")
print(f"Tokens per linguistic item: {len(old_tokens)/6:.1f} → {len(new_tokens)/6:.1f}")

print("\nFor a typical 10-token sentence:")
print(f"  Old: ~{len(old_tokens)*10/6:.0f} tokens")
print(f"  New: ~{len(new_tokens)*10/6:.0f} tokens")
print(f"  Savings: ~{(len(old_tokens)-len(new_tokens))*10/6:.0f} tokens per sentence")
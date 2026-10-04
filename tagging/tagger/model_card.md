---
language:
- de
- en
- fr
- hi
- it
- ja
- ko
- pt
- ru
- es
- th
- zh
base_model: BAAI/bge-m3
tags:
- token-classification
- dependency-parsing
- lemmatization
- onnx
---
# Parsley

Parsley jointly tokenizes raw sentences and predicts POS, lemmas and dependencies
in 12 languages. A 24-layer bge-m3 encoder feeds a character BiLSTM, O/B/I boundaries,
and word heads over start/end character and start-subword states. Dependency trees
use single-root Chu–Liu–Edmonds. There are no dictionary overrides.

## Current model

- PyTorch checkpoint: `training-runs/joint-v2-24L/best/` at
  `09a1f8b32248cc303132026ec488f4ad65085ecb`.
- CPU fp32 ONNX: `joint/encoder.onnx`, `joint/encoder.onnx.data`, `joint/heads.onnx`,
  `joint/tokenizer.json`, `joint/vocab.json`, `joint/config.json`.
- Rust: the `lexide` crate's 0.5 local backend pins an immutable artifact revision;
  source and release instructions are in [lexide/tagging](https://github.com/anchpop/lexide/tree/main/tagging).
- Hosted tagging: `https://anchpop--lexide-parsley-parsley-tag.modal.run`.

Macro end-to-end LAS on the 12k held-out teacher-labelled test set is 85.1%, versus
71.3% for v1. This is teacher agreement, not an independent gold accuracy claim.
The fp32 ONNX export matches CPU PyTorch's final predictions on 288 multilingual
length-spread cases; Rust has 293 CPU-reference parity fixtures. The GPU service
uses bf16 and can differ on near-tied labels.

## Compatibility

`onnx/` is the **legacy v1** artifact set, preserved for lexide 0.4.x downloads.
Do not use it for the joint tagger. The independent byte sentence segmenter remains
at `onnx/sentence_segmenter.safetensors`. Training checkpoints and historical v1
files remain available; they are not the current joint inference format.

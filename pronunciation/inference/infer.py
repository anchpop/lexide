"""Inference: transcribe audio with phonemes + predicted stress.

Usage:
    python -m pronunciation.inference.infer path/to/audio.wav
"""

import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

STRESS_MARKS = {0: "", 1: "ˈ", 2: "ˌ"}


def build_stress_head(hidden_size: int, num_labels: int = 3):
    return nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_labels),
    )


def load_model_and_head(model_name="anchpop/lexide-pronunciation-phoneme-xls-r-2b",
                       hf_repo="anchpop/lexide-pronunciation-stress"):
    backbone = Wav2Vec2ForCTC.from_pretrained(model_name)
    backbone.eval()

    # Load checkpoint
    ckpt_path = hf_hub_download(hf_repo, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    stress_head = build_stress_head(backbone.config.hidden_size)
    stress_head.load_state_dict(ckpt["stress_head"])
    stress_head.eval()

    layer_weights = ckpt["layer_weights"]
    print(f"Loaded checkpoint. Layer weights softmax: top layer = {layer_weights.argmax().item()}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    return backbone, stress_head, layer_weights, processor


@torch.no_grad()
def transcribe(wav_path, backbone, stress_head, layer_weights, processor):
    audio, sr = sf.read(wav_path)
    assert sr == 16000, f"expected 16kHz audio, got {sr}"

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    outputs = backbone(inputs.input_values, output_hidden_states=True)

    # Weighted sum over layers (matches training)
    stacked = torch.stack(outputs.hidden_states, dim=0)  # (layers, 1, frames, hidden)
    weights = torch.softmax(layer_weights, dim=0)
    weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)  # (1, frames, hidden)

    # Stress predictions per frame
    stress_logits = stress_head(weighted)  # (1, frames, 3)
    stress_preds = stress_logits.argmax(dim=-1).squeeze(0)  # (frames,)

    # Phoneme predictions from frozen base model
    phoneme_ids = outputs.logits.argmax(dim=-1).squeeze(0)  # (frames,)
    blank_id = processor.tokenizer.pad_token_id

    # Decode: collapse repeats, drop blanks, attach stress to each phoneme
    result = []
    prev_id = -1
    for t in range(len(phoneme_ids)):
        tid = phoneme_ids[t].item()
        if tid == blank_id:
            prev_id = -1
            continue
        if tid == prev_id:
            continue  # repeat
        token = processor.tokenizer.convert_ids_to_tokens(tid)
        stress = stress_preds[t].item()
        result.append((token, stress))
        prev_id = tid

    return result


def format_output(tokens_with_stress):
    """Render with stress marks before each phoneme."""
    out = []
    for token, stress in tokens_with_stress:
        out.append(STRESS_MARKS[stress] + token)
    return "".join(out)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m pronunciation.inference.infer <audio.wav> [...]")
        sys.exit(1)

    backbone, stress_head, layer_weights, processor = load_model_and_head()

    for path in sys.argv[1:]:
        result = transcribe(path, backbone, stress_head, layer_weights, processor)
        print(f"\n{path}:")
        print(f"  phonemes+stress: {format_output(result)}")
        print(f"  raw: {result}")


if __name__ == "__main__":
    main()

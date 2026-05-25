"""Inference: transcribe audio with phonemes + stress using the unified model.

Usage:
    python -m pronunciation.inference.infer path/to/audio.wav [more.wav ...]

The unified model is `anchpop/lexide-pronunciation-unified-xls-r-2b`: a
Wav2Vec2 backbone with three factorized heads (nonblank/phoneme/stress) —
see train/src/factorized_ctc.py for training-time details. The repo ships
the backbone via standard `save_pretrained` plus a `factorized_heads.pt`
side-file holding the three head state dicts.
"""

import sys

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import (
    Wav2Vec2Model, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

REPO = "anchpop/lexide-pronunciation-unified-xls-r-2b"
STRESS_MARKS = {0: "", 1: "ˈ", 2: "ˌ"}


def build_heads(hidden_size: int, vocab_size: int, num_stress_labels: int = 3):
    """Match factorized_ctc.FactorizedCTCModel head shapes exactly."""
    nonblank_head = nn.Linear(hidden_size, 1)
    phoneme_head = nn.Linear(hidden_size, vocab_size)
    stress_head = nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_stress_labels),
    )
    return nonblank_head, phoneme_head, stress_head


def load_model_and_heads(repo: str = REPO):
    backbone = Wav2Vec2Model.from_pretrained(repo)
    backbone.eval()

    ckpt_path = hf_hub_download(repo, "factorized_heads.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    nonblank_head, phoneme_head, stress_head = build_heads(
        hidden_size=backbone.config.hidden_size,
        vocab_size=ckpt["vocab_size"],
        num_stress_labels=ckpt.get("num_stress_labels", 3),
    )
    nonblank_head.load_state_dict(ckpt["nonblank_head"])
    phoneme_head.load_state_dict(ckpt["phoneme_head"])
    stress_head.load_state_dict(ckpt["stress_head"])
    nonblank_head.eval(); phoneme_head.eval(); stress_head.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(repo)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(repo)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    return {
        "backbone": backbone,
        "nonblank_head": nonblank_head,
        "phoneme_head": phoneme_head,
        "stress_head": stress_head,
        "processor": processor,
        "blank_id": ckpt["blank_id"],
    }


@torch.no_grad()
def transcribe(wav_path, model):
    audio, sr = sf.read(wav_path)
    assert sr == 16000, f"expected 16kHz audio, got {sr}"

    inputs = model["processor"](audio, sampling_rate=16000, return_tensors="pt")
    out = model["backbone"](inputs.input_values)
    hidden_states = out.last_hidden_state  # (1, T, H)

    blank_id = model["blank_id"]
    l_nb = model["nonblank_head"](hidden_states).squeeze(-1)   # (1, T)
    l_ph = model["phoneme_head"](hidden_states)                # (1, T, V)
    l_ph = l_ph.clone()
    l_ph[..., blank_id] = float("-inf")

    log_p_blank    = F.logsigmoid(-l_nb).unsqueeze(-1)              # (1, T, 1)
    log_p_nonblank = F.logsigmoid(l_nb).unsqueeze(-1)               # (1, T, 1)
    log_p_phonemes = log_p_nonblank + F.log_softmax(l_ph, dim=-1)   # (1, T, V), blank=-inf
    log_probs = log_p_phonemes.clone()
    log_probs[..., blank_id] = log_p_blank.squeeze(-1)              # (1, T, V), valid distribution

    stress_logits = model["stress_head"](hidden_states)             # (1, T, 3)

    phoneme_ids  = log_probs.argmax(dim=-1).squeeze(0)
    stress_preds = stress_logits.argmax(dim=-1).squeeze(0)
    tokenizer = model["processor"].tokenizer

    # Greedy CTC collapse: drop blanks, drop repeats. Attach stress at the
    # frame where each phoneme is first emitted (the stress-head prediction
    # there reflects the model's stress call for that vowel/segment).
    result = []
    prev_id = -1
    for t in range(len(phoneme_ids)):
        tid = phoneme_ids[t].item()
        if tid == blank_id:
            prev_id = -1
            continue
        if tid == prev_id:
            continue
        token = tokenizer.convert_ids_to_tokens(tid)
        result.append((token, stress_preds[t].item()))
        prev_id = tid
    return result


def format_output(tokens_with_stress):
    return "".join(STRESS_MARKS[s] + t for t, s in tokens_with_stress)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m pronunciation.inference.infer <audio.wav> [...]")
        sys.exit(1)

    model = load_model_and_heads()

    for path in sys.argv[1:]:
        result = transcribe(path, model)
        print(f"\n{path}:")
        print(f"  phonemes+stress: {format_output(result)}")
        print(f"  raw: {result}")


if __name__ == "__main__":
    main()

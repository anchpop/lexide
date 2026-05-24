"""Inference: transcribe audio with phonemes + inline stress markers.

The phoneme model (e.g. anchpop/lexide-pronunciation-phoneme-xls-r-2b) emits
stress markers (ˈ for primary, ˌ for secondary) directly in the CTC stream,
so we just decode the model's argmax and walk the token sequence — no
separate stress head, no forced alignment.

Usage:
    python -m pronunciation.inference.infer path/to/audio.wav
"""

import sys

import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


PRIMARY_STRESS = "ˈ"
SECONDARY_STRESS = "ˌ"


def load_model(model_name="anchpop/lexide-pronunciation-phoneme-xls-r-2b"):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    return model, processor


@torch.no_grad()
def transcribe(wav_path, model, processor):
    """Return a list of (phoneme, stress) pairs.

    stress: 0 = none, 1 = primary, 2 = secondary. Stress applies to the
    phoneme it precedes in the IPA stream (espeak convention).
    """
    audio, sr = sf.read(wav_path)
    assert sr == 16000, f"expected 16kHz audio, got {sr}"

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    logits = model(inputs.input_values).logits.squeeze(0)  # (frames, vocab)
    pred_ids = logits.argmax(dim=-1).tolist()
    blank_id = processor.tokenizer.pad_token_id

    # CTC collapse: drop blanks, collapse consecutive duplicates
    tokens = []
    prev = -1
    for tid in pred_ids:
        if tid == blank_id:
            prev = -1
            continue
        if tid == prev:
            continue
        tokens.append(processor.tokenizer.convert_ids_to_tokens(tid))
        prev = tid

    # Attach stress markers to the following token
    result = []
    pending = 0
    for tok in tokens:
        if tok == PRIMARY_STRESS:
            pending = 1
        elif tok == SECONDARY_STRESS:
            pending = 2
        else:
            result.append((tok, pending))
            pending = 0
    return result


def format_output(tokens_with_stress):
    marks = {0: "", 1: PRIMARY_STRESS, 2: SECONDARY_STRESS}
    return "".join(marks[stress] + tok for tok, stress in tokens_with_stress)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m pronunciation.inference.infer <audio.wav> [...]")
        sys.exit(1)

    model, processor = load_model()
    for path in sys.argv[1:]:
        result = transcribe(path, model, processor)
        print(f"\n{path}:")
        print(f"  phonemes+stress: {format_output(result)}")
        print(f"  raw: {result}")


if __name__ == "__main__":
    main()

"""Inference: transcribe audio with phonemes + stress using a unified model.

Usage:
    python -m pronunciation.inference.infer path/to/audio.wav [more.wav ...]

Loads a unified factorized-CTC model from HuggingFace (or local path) via
`FactorizedCTCModel.load_from_dir`, which transparently handles both the
direct-phoneme-head and articulatory-feature-decomposed checkpoints.
Default repo is the latest unified model; pass `--repo` to override.

The HF repo also carries a saved processor (tokenizer + feature extractor),
so the same call site loads everything needed for inference.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import (
    Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,
)

# Add `pronunciation/train` to sys.path so we can import the model module
# whether infer.py is invoked from the project root, the inference/ dir,
# or as `python -m pronunciation.inference.infer`.
_HERE = Path(__file__).resolve().parent
_TRAIN_SRC = _HERE.parent / "train"
if str(_TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(_TRAIN_SRC))

from src.factorized_ctc import FactorizedCTCModel  # noqa: E402

DEFAULT_REPO = "anchpop/lexide-pronunciation-unified-vad"
STRESS_MARKS = {0: "", 1: "ˈ", 2: "ˌ"}


def load_repo_snapshot(repo: str) -> Path:
    """Return a local directory holding the HF repo (snapshot-downloaded)."""
    p = Path(repo)
    if p.is_dir():
        return p
    return Path(snapshot_download(repo_id=repo))


def load_model(repo: str):
    local_dir = load_repo_snapshot(repo)

    model = FactorizedCTCModel.load_from_dir(local_dir)
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(local_dir)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return model, processor


@torch.no_grad()
def transcribe(wav_path, model: FactorizedCTCModel, processor: Wav2Vec2Processor):
    audio, sr = sf.read(wav_path)
    assert sr == 16000, f"expected 16kHz audio, got {sr}"

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    out = model(inputs.input_values)
    log_probs = out["log_probs"]           # (1, T, V), proper distribution per frame
    stress_logits = out["stress_logits"]   # (1, T, 3)

    phoneme_ids = log_probs.argmax(-1).squeeze(0)
    stress_preds = stress_logits.argmax(-1).squeeze(0)
    tokenizer = processor.tokenizer
    blank_id = model.blank_id

    # Greedy CTC collapse: drop blanks, drop repeats. Attach stress at the
    # frame where each phoneme is first emitted.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="One or more 16kHz mono wav files.")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"HF repo or local path (default: {DEFAULT_REPO})")
    args = parser.parse_args()

    model, processor = load_model(args.repo)

    for path in args.paths:
        result = transcribe(path, model, processor)
        print(f"\n{path}:")
        print(f"  phonemes+stress: {format_output(result)}")
        print(f"  raw: {result}")


if __name__ == "__main__":
    main()

"""Inference: transcribe audio with phonemes + stress using a unified model.

Usage:
    python -m pronunciation.inference.infer --repo <HF-repo-or-local-dir> \\
        path/to/audio.wav [more.wav ...]

    # JSON output (one record per audio file) for downstream evaluation:
    python -m pronunciation.inference.infer --repo <repo> --format jsonl \\
        *.wav > predictions.jsonl

Loads a unified factorized-CTC model from HuggingFace (or local path) via
`FactorizedCTCModel.load_from_dir`, which transparently handles the
direct-phoneme-head, articulatory-feature-decomposed, and regularized-
heads checkpoint variants.

The HF repo also carries a saved processor (tokenizer + feature extractor),
so the same call site loads everything needed for inference.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
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

STRESS_MARKS = {0: "", 1: "ˈ", 2: "ˌ"}


def load_repo_snapshot(repo: str) -> Path:
    """Return a local directory holding the HF repo (snapshot-downloaded)."""
    p = Path(repo)
    if p.is_dir():
        return p
    return Path(snapshot_download(repo_id=repo))


def load_model(repo: str, device: torch.device):
    local_dir = load_repo_snapshot(repo)

    model = FactorizedCTCModel.load_from_dir(local_dir)
    model.eval()
    model.to(device)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(local_dir)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return model, processor


def _is_special_token(token: str) -> bool:
    """Special tokens look like `<pad>`, `<s>`, `</s>`, `<unk>` — bracket-wrapped."""
    return token.startswith("<") and token.endswith(">")


@torch.no_grad()
def transcribe(
    wav_path,
    model: FactorizedCTCModel,
    processor: Wav2Vec2Processor,
    device: torch.device,
    use_bf16: bool = False,
):
    audio, sr = sf.read(wav_path)
    assert sr == 16000, f"expected 16kHz audio, got {sr}"

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        out = model(input_values)

    log_probs = out["log_probs"]           # (1, T, V), proper distribution per frame
    stress_logits = out["stress_logits"]   # (1, T, 3)
    nonblank_logit = out["nonblank_logit"] # (1, T)

    phoneme_ids = log_probs.argmax(-1).squeeze(0)
    stress_preds = stress_logits.argmax(-1).squeeze(0)
    nonblank_probs = torch.sigmoid(nonblank_logit).squeeze(0)
    tokenizer = processor.tokenizer
    blank_id = model.blank_id

    # Greedy CTC collapse: drop blanks, drop repeats, drop special tokens.
    # Attach stress at the frame where each phoneme is first emitted.
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
        prev_id = tid
        if _is_special_token(token):
            continue
        result.append({
            "token": token,
            "stress": int(stress_preds[t].item()),
            "frame": t,
            "nonblank_prob": round(nonblank_probs[t].item(), 4),
        })
    return result


def format_output(tokens):
    """Render tokens as IPA with stress markers."""
    return "".join(STRESS_MARKS[r["stress"]] + r["token"] for r in tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="One or more 16kHz mono wav files.")
    parser.add_argument("--repo", required=True,
                        help="HF repo (e.g. anchpop/lexide-pronunciation-unified-articulatory-aux) "
                             "or local checkpoint directory.")
    parser.add_argument("--format", choices=["text", "jsonl"], default="text",
                        help="text: human-readable IPA per file. "
                             "jsonl: one JSON record per file with phonemes, stress, "
                             "frame indices, and nonblank probability — for downstream eval.")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu / mps. Default: cuda if available, else cpu.")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 autocast (CUDA only). Faster, slight precision drop.")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.format == "text":
        print(f"# Loading {args.repo} on {device}...", file=sys.stderr)
    model, processor = load_model(args.repo, device)

    for path in args.paths:
        tokens = transcribe(path, model, processor, device, use_bf16=args.bf16)
        if args.format == "jsonl":
            print(json.dumps({
                "path": str(path),
                "ipa": format_output(tokens),
                "tokens": tokens,
            }, ensure_ascii=False))
        else:
            print(f"\n{path}:")
            print(f"  ipa: {format_output(tokens)}")
            print(f"  tokens: {[t['token'] for t in tokens]}")


if __name__ == "__main__":
    main()

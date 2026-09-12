"""Inference: transcribe audio with phonemes + stress using a unified model.

Usage:
    python -m pronunciation.inference.infer --repo <HF-repo-or-local-dir> \\
        path/to/audio.wav [more.wav ...]

    # Length-bucketed offline batches (tune for your GPU and clip durations):
    python -m pronunciation.inference.infer --repo <repo> --batch-size 4 \\
        --max-batch-seconds 60 *.wav

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
import math
import sys
from pathlib import Path

import numpy as np
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


def plan_batches(lengths, batch_size=8, max_batch_samples=960000):
    """Length-sort within a caller-bounded window; budget includes padding.

    An oversized clip runs alone. Indices let callers restore input order.
    """
    if batch_size < 1 or max_batch_samples < 1:
        raise ValueError("batch_size and max_batch_samples must be positive")
    batch = []
    for index in sorted(range(len(lengths)), key=lengths.__getitem__):
        if batch and (len(batch) >= batch_size or
                      lengths[index] * (len(batch) + 1) > max_batch_samples):
            yield batch
            batch = []
        batch.append(index)
    if batch:
        yield batch


def read_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000 or audio.ndim != 1 or len(audio) == 0:
        raise ValueError(f"{path}: expected nonempty 16kHz mono audio")
    if not np.isfinite(audio).all():
        raise ValueError(f"{path}: audio contains nonfinite samples")
    return audio


def transcribe(wav_path, model, processor, device, use_bf16=False):
    return transcribe_batch([wav_path], model, processor, device, use_bf16)[0]


def transcribe_batch(paths, model, processor, device, use_bf16=False,
                     batch_size=1, max_batch_seconds=60.0):
    """Transcribe a bounded window of files, returning results in input order."""
    if not math.isfinite(max_batch_seconds) or max_batch_seconds <= 0:
        raise ValueError("max_batch_seconds must be finite and positive")
    audios = [read_audio(path) for path in paths]
    results = [None] * len(audios)
    for indices in plan_batches([len(a) for a in audios], batch_size,
                                max(1, int(max_batch_seconds * 16000))):
        decoded = transcribe_audio_batch(
            [audios[i] for i in indices], model, processor, device, use_bf16,
        )
        for i, tokens in zip(indices, decoded):
            results[i] = tokens
    return results


@torch.inference_mode()
def transcribe_audio_batch(audios, model, processor, device, use_bf16=False):
    """Run a batch, splitting on CUDA OOM; a failing singleton still raises."""
    try:
        return _transcribe_audio_batch(audios, model, processor, device, use_bf16)
    except torch.cuda.OutOfMemoryError:
        if len(audios) <= 1:
            raise
    # Outside the except block: release the failed forward's traceback/tensors
    # before retrying. Never drop a clip or substitute an empty prediction.
    torch.cuda.empty_cache()
    middle = len(audios) // 2
    return (transcribe_audio_batch(audios[:middle], model, processor, device, use_bf16)
            + transcribe_audio_batch(audios[middle:], model, processor, device, use_bf16))


@torch.inference_mode()
def _transcribe_audio_batch(audios, model, processor, device, use_bf16=False):
    """One forward pass over mono float32 arrays; never decode padded frames."""
    if not audios:
        return []
    lengths = torch.tensor([len(a) for a in audios], dtype=torch.long)
    frame_lengths = model.backbone._get_feat_extract_output_lengths(lengths)
    if (frame_lengths <= 0).any():
        raise ValueError("audio is too short for the model's feature extractor")
    # GroupNorm pools over time before the attention mask is applied. Padding
    # changes valid frames for these checkpoints; retain singleton semantics.
    # Cohere batching also remains singleton until its frontend is validated.
    if (len(audios) > 1 and
            (getattr(model.backbone.config, "feat_extract_norm", None) == "group"
             or getattr(model.backbone.config, "model_type", "") == "cohere_conformer_ctc")):
        return [transcribe_audio_batch([a], model, processor, device, use_bf16)[0]
                for a in audios]
    # Match training: acoustic sidechannels and Cohere take raw waveforms;
    # plain wav2vec2 checkpoints retain feature-extractor normalization.
    # Sidechannels align their own analysis windows inside the model. Only
    # right-pad for batching here; never add analysis-window pre-padding.
    needs_raw = (
        getattr(model.backbone.config, "model_type", "") == "cohere_conformer_ctc"
        or getattr(model, "mel_sidechannel", False)
        or getattr(model, "regularized_heads", False)
    )
    if needs_raw:
        input_values = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(a, dtype=torch.float32) for a in audios], batch_first=True,
        )
    else:
        # Normalize each utterance independently, before padding.
        input_values = processor(
            audios, sampling_rate=16000, return_tensors="pt", padding=True,
            return_attention_mask=True,
        ).input_values
    attention_mask = None
    if len(audios) > 1:
        attention_mask = (torch.arange(input_values.shape[1])[None, :] < lengths[:, None])
        attention_mask = attention_mask.long().to(device)
    input_values = input_values.to(device)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        out = model(input_values, attention_mask=attention_mask)

    # Three compact transfers per batch, instead of .item() synchronizing the
    # GPU for every frame/token. Keep the Python CTC run reduction on the CPU.
    ids = out["log_probs"].argmax(-1).cpu().numpy()
    stress = torch.log_softmax(out["stress_logits"].float(), -1).cpu().numpy()
    nonblank = torch.sigmoid(out["nonblank_logit"]).float().cpu().numpy()
    return [decode_frames(ids[i, :n], stress[i, :n], nonblank[i, :n],
                          processor.tokenizer, model.blank_id)
            for i, n in enumerate(frame_lengths.tolist())]


def decode_frames(phoneme_ids, stress_log_probs, nonblank_probs, tokenizer, blank_id):
    # Greedy CTC collapse: drop blanks, drop repeats, drop special tokens.
    # Attach stress by summing stress log-probs over each token's emitted run
    # (argmax equals averaging) — the same per-frame product the joint CTC
    # scores an alignment by. The VAD loss widens emissions across the phone's
    # span, so the run's first frame is a phone-onset boundary frame and the
    # least reliable single frame to read stress from.
    result = []
    stress_acc = None
    prev_id = -1
    for t in range(len(phoneme_ids)):
        tid = int(phoneme_ids[t])
        if tid == blank_id:
            prev_id = -1
            continue
        if tid == prev_id:
            if stress_acc is not None:
                stress_acc += stress_log_probs[t]
            continue
        token = tokenizer.convert_ids_to_tokens(tid)
        prev_id = tid
        if _is_special_token(token):
            stress_acc = None
            continue
        stress_acc = stress_log_probs[t].copy()
        result.append({
            "token": token,
            "stress": stress_acc,  # accumulated over the run; finalized below
            "frame": t,
            "nonblank_prob": round(float(nonblank_probs[t]), 4),
        })
    for r in result:
        r["stress"] = int(r["stress"].argmax())
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
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Maximum clips per model forward (default: 1). "
                             "Try 4 for throughput; batch shape can change predictions and emission timing.")
    parser.add_argument("--max-batch-seconds", type=float, default=60,
                        help="Padded audio budget per forward (default: 60 seconds).")
    parser.add_argument("--sort-window", type=int, default=128,
                        help="Files loaded and length-sorted at a time (default: 128).")
    args = parser.parse_args()
    if args.batch_size < 1 or args.sort_window < 1:
        parser.error("--batch-size and --sort-window must be positive")
    if not math.isfinite(args.max_batch_seconds) or args.max_batch_seconds <= 0:
        parser.error("--max-batch-seconds must be finite and positive")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.format == "text":
        print(f"# Loading {args.repo} on {device}...", file=sys.stderr)
    model, processor = load_model(args.repo, device)

    for start in range(0, len(args.paths), args.sort_window):
        paths = args.paths[start:start + args.sort_window]
        results = transcribe_batch(
            paths, model, processor, device, use_bf16=args.bf16,
            batch_size=args.batch_size, max_batch_seconds=args.max_batch_seconds,
        )
        for path, tokens in zip(paths, results):
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

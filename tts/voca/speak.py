"""Synthesize sentences from a pack dump or script (rows with ``grams``).

    python -m voca.speak --checkpoint runs/x/inference.pt --script data/eng_script.jsonl --split validation --output out/x
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch

from .config import Config, check_format
from .model import build_model
from .text import HINTS, frontend_fingerprint, frontend_identity, tokenize


# Minimum pauses (seconds) after commas and sentence stops inside an utterance:
# the teacher barely pauses at commas, and learners need the phrasing.
COMMA_PAUSE, STOP_PAUSE = 0.15, 0.3


def load_model(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    check_format(state, path)
    if state["frontend"]["fingerprint"] != frontend_fingerprint():
        raise ValueError(f"{path} was trained with another front end (language {state['frontend'].get('language')}, version "
                         f"{state['frontend'].get('version', 1)}; this process reads {frontend_identity()['language']}, version "
                         f"{frontend_identity()['version']}): set VOCA_LANGUAGE to its language, or retrain it")
    model = build_model(Config.from_dict(state["config"]))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def speak(model, grams, device="cpu", speaker=None, speed=1.0, pitch_range=1.0, seed=0, comma_pause=COMMA_PAUSE, stop_pause=STOP_PAUSE,
          refiner=None, pitch_jitter=0.0, take=0):
    """``speaker``: a name from the model's configuration (default: its first)."""
    phonemized = tokenize(grams)
    tokens = torch.tensor([phonemized.ids], device=device)
    lexical = torch.tensor([phonemized.lexical()], device=device)
    if not model.config.hints:
        lexical[..., HINTS] = 0
    speaker = model.config.speakers.index(speaker or model.config.speakers[0])
    audio, lengths, details = model.synthesize(tokens, lexical, torch.tensor([len(phonemized)], device=device),
                                               torch.tensor([speaker], device=device), speed, 0.0, seed, pitch_range,
                                               pitch_jitter=pitch_jitter, comma_pause=comma_pause, stop_pause=stop_pause, take=take)
    if refiner is not None:
        audio = refiner.refine(audio, details["acoustic"].float(), details["frame_lengths"], seed=seed)
    audio = audio[0, :int(lengths[0])].cpu().numpy()
    audio = audio[leading_silence(audio, details["durations"][0].cpu().numpy(), phonemized.ids, model) * model.config.hop:]
    return audio * min(1.0, 0.99 / max(float(np.abs(audio).max()), 1e-9)), details


def leading_silence(audio, durations, ids, model, margin=4, floor_db=-50.0):
    """Frames of silence to skip before playback: up to ``margin`` frames before
    the earlier of the first phone (as predicted) and the first two frames in a
    row above ``floor_db`` dBFS, so a quiet initial fricative or a breath before
    it is kept. The model predicts about 280 ms of leading silence (Gemini's
    recordings have it); this saves about 220 ms of latency."""
    hop = model.config.hop
    first_phone = int(durations[:next(i for i, t in enumerate(ids) if bool(model.phone_tokens[t]))].sum())
    frames = len(audio) // hop
    level = 10 * np.log10(np.maximum(np.mean(audio[:frames * hop].reshape(frames, hop) ** 2, axis=1), 1e-12))
    loud = np.flatnonzero((level[:-1] > floor_db) & (level[1:] > floor_db))
    return max(0, min(first_phone, int(loud[0]) if len(loud) else first_phone) - margin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speaker", help="one of the checkpoint's speakers (default: its first)")
    parser.add_argument("--pitch-range", type=float, default=1.0)
    parser.add_argument("--pitch-jitter", type=float, default=0.0, help="random pitch wobble for the oscillator, in semitones")
    parser.add_argument("--take", type=int, default=0, help="another take, another delivery of every sentence")
    parser.add_argument("--refiner", type=Path, help="voca.refine checkpoint trained on this --checkpoint")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    refiner = None
    if args.refiner:
        from .refine import load_refiner
        refiner = load_refiner(args.refiner, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.script, encoding="utf-8")]
    rows = [r for r in rows if args.split is None or r.get("split") == args.split][:args.limit]
    seconds = compute = 0.0
    with open(args.output / "manifest.jsonl", "w", encoding="utf-8") as manifest:
        for n, row in enumerate(rows):
            start = time.perf_counter()
            audio, _ = speak(model, row["grams"], args.device, args.speaker, pitch_range=args.pitch_range, refiner=refiner, take=len(rows) * args.take + n,
                             pitch_jitter=args.pitch_jitter)
            compute += time.perf_counter() - start
            seconds += len(audio) / model.config.sample_rate
            filename = f"{row.get('id', n)}.wav"
            sf.write(args.output / filename, audio, model.config.sample_rate)
            manifest.write(json.dumps({**row, "wav": filename}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} clips, {seconds:.1f} s of audio in {compute:.1f} s ({seconds / max(compute, 1e-9):.0f}x real time on {args.device})")


if __name__ == "__main__":
    main()

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

from .config import Config
from .model import build_model
from .text import frontend_fingerprint, tokenize


def load_model(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["frontend"]["fingerprint"] != frontend_fingerprint():
        raise ValueError("checkpoint was trained with a different token inventory")
    model = build_model(Config.from_dict(state["config"]))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def speak(model, grams, device="cpu", speaker=0, speed=1.0, pitch_range=1.0, seed=0, comma_pause=0.15, stop_pause=0.3,
          refiner=None, pitch_jitter=0.0):
    phonemized = tokenize(grams)
    tokens = torch.tensor([phonemized.ids], device=device)
    stress = torch.tensor([phonemized.stress], device=device)
    audio, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phonemized)], device=device),
                                               torch.tensor([speaker], device=device), speed, 0.0, seed, pitch_range,
                                               pitch_jitter=pitch_jitter, comma_pause=comma_pause, stop_pause=stop_pause)
    if refiner is not None:
        audio = refiner.refine(audio, details["acoustic"].float(), details["frame_lengths"], seed=seed)
    audio = audio[0, :int(lengths[0])].cpu().numpy()
    return audio * min(1.0, 0.99 / max(float(np.abs(audio).max()), 1e-9)), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pitch-range", type=float, default=1.0)
    parser.add_argument("--pitch-jitter", type=float, default=0.0, help="random pitch wobble for the oscillator, in semitones")
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
        for row in rows:
            start = time.perf_counter()
            audio, _ = speak(model, row["grams"], args.device, pitch_range=args.pitch_range, refiner=refiner,
                             pitch_jitter=args.pitch_jitter)
            compute += time.perf_counter() - start
            seconds += len(audio) / model.config.sample_rate
            sf.write(args.output / f"{row['id']}.wav", audio, model.config.sample_rate)
            manifest.write(json.dumps({**row, "wav": f"{row['id']}.wav"}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} clips, {seconds:.1f} s of audio in {compute:.1f} s ({seconds / max(compute, 1e-9):.0f}x real time on {args.device})")


if __name__ == "__main__":
    main()

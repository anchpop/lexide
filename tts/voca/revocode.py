"""Give a trained model a new waveform head: the text-to-acoustic part stays
frozen and only the new head (``--waveform``) is trained, on short crops of its
decoder states, against the recordings with the same spectral losses and
critics as voca.train. Writes an ordinary ``inference.pt`` (frozen base plus new
head) that voca.speak reads.

This is how an autoregressive head (``fargan``) is trained: stepping through
whole utterances one 64-sample step at a time would be far too slow, crops of
half a second are not.

    python -m voca.revocode --base runs/eng-nsf/inference.pt --waveform fargan --train features/x/train.jsonl \
        --validation features/x/validation.jsonl --output runs/eng-fargan
"""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import torch
from torch import nn

from .config import Config
from .data import AnnotatedDataset, BucketSampler, collate, to_device
from .losses import Discriminators, SpectralLoss
from .model import TinyVoca
from .text import frontend_fingerprint
from .train import TrainingSystem, adversarial_backward, export_model, start_tracking


@torch.no_grad()
def decoder_states(model, batch, own_prosody):
    """Decoder states [B, T, C] and recorded F0 [B, T] (Hz) with recorded durations,
    so they line up with the recording. Rows where ``own_prosody`` is set condition
    the decoder on the model's predicted pitch, voicing and energy, as at inference."""
    m = model
    hidden, token_mask = m.encode(batch["tokens"], batch["lexical"], batch["token_lengths"], batch["speakers"])
    frames, frame_mask, lengths = m.expand(hidden, batch["durations"], token_mask, batch["speakers"])
    prosody_hidden, prediction = m.predict_prosody(frames, frame_mask)
    target = batch["prosody"]
    condition = torch.where(own_prosody[:, None, None], m.prosody_condition(prediction.float()), target)
    acoustic = m.decode(prosody_hidden, condition.to(prosody_hidden.dtype), frame_mask)
    return acoustic.float(), m.frequency(target[..., 0]), lengths


def crop_batch(model, batch, frames, count, own_fraction, generator=None, centre=False):
    """``count`` aligned crops of ``frames`` acoustic frames per utterance: decoder
    states, F0, the recording, and the recording just before each crop."""
    hop, history = model.config.hop, getattr(model.vocoder, "HISTORY", 0)
    own = torch.rand(len(batch["frame_lengths"]), generator=generator).to(batch["wave"].device) < own_fraction
    acoustic, f0, lengths = decoder_states(model, batch, own)
    frames = min(frames, int(lengths.min()))
    if centre:
        start = ((lengths - frames) // 2)[:, None]
    else:
        start = (torch.rand(len(lengths), count, generator=generator).to(lengths.device) * (lengths - frames + 1)[:, None]).long()
    start = start.flatten()
    row = torch.arange(len(lengths), device=lengths.device).repeat_interleave(start.numel() // len(lengths))
    frame = start[:, None] + torch.arange(frames, device=lengths.device)
    sample = (start * hop)[:, None] + torch.arange(-history, frames * hop, device=lengths.device)
    wave = nn.functional.pad(batch["wave"], (history, 0))[row[:, None], sample + history]
    return {"acoustic": acoustic[row[:, None], frame], "f0": f0[row[:, None], frame], "history": wave[:, :history],
            "wave": wave[:, history:], "lengths": torch.full_like(start, frames)}


def synthesize(model, crop):
    extra = {"history": crop["history"]} if crop["history"].shape[-1] else {}
    return model.vocoder(crop["acoustic"], crop["f0"], crop["lengths"], **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="inference.pt of the trained model")
    parser.add_argument("--waveform", required=True, help="the new head, a Config.waveform value")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-frames", type=int, default=9000)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--crop-frames", type=int, default=48, help="acoustic frames per crop (48 = 0.5 s)")
    parser.add_argument("--crops", type=int, default=2, help="crops per utterance")
    parser.add_argument("--own-prosody", type=float, default=0.5, help="share of utterances decoded from predicted prosody")
    parser.add_argument("--gan-start", type=int, default=4000)
    parser.add_argument("--gan-ratio", type=float, default=0.7)
    parser.add_argument("--gan-ramp", type=int, default=3000)
    parser.add_argument("--feature-matching", type=float, default=30.0)
    parser.add_argument("--gan-scale-max", type=float, default=2000.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=14400)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(0)
    state = torch.load(args.base, map_location="cpu", weights_only=True)
    if state["frontend"]["fingerprint"] != frontend_fingerprint():
        raise ValueError("base was trained with a different token inventory")
    c = replace(Config.from_dict(state["config"]), waveform=args.waveform)
    model = TinyVoca(c)
    frozen = {k: v for k, v in state["model"].items() if not k.startswith("vocoder.")}
    missing, unexpected = model.load_state_dict(frozen, strict=False)
    if unexpected or any(not k.startswith("vocoder.") for k in missing):
        raise ValueError(f"base does not match: missing {missing}, unexpected {unexpected}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    vocoder = model.vocoder.train().requires_grad_(True)
    count = sum(p.numel() for p in model.parameters())
    print(json.dumps({"vocoder_parameters": sum(p.numel() for p in vocoder.parameters()), "inference_parameters": count}), flush=True)
    train = AnnotatedDataset(args.train, c, voiced_vowels=True)
    validation = AnnotatedDataset(args.validation, c, voiced_vowels=True)
    train.preload()
    validation.preload()
    spectral, critics = SpectralLoss(c.sample_rate).to(args.device), Discriminators().to(args.device)
    optimizer = torch.optim.AdamW(vocoder.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
    critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
    rate = lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
    schedules = [torch.optim.lr_scheduler.LambdaLR(o, rate) for o in (optimizer, critic_optimizer)]
    sampler = BucketSampler(train.lengths, train.token_counts, train.speakers, args.batch_frames, args.max_batch, 0, alpha=1.0)
    recipe = {k: getattr(args, k) for k in ("gan_start", "gan_ratio", "gan_ramp", "feature_matching", "gan_scale_max")}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(c.to_dict(), indent=2) + "\n")
    tracker = start_tracking(args.output.name, {"model": c.to_dict(), **{k: str(v) for k, v in vars(args).items()}})
    start, window, steps = time.monotonic(), defaultdict(float), 0
    for step in range(1, args.steps + 1):
        batch = to_device(collate([train[i] for i in sampler.next()[0]]), args.device)
        crop = crop_batch(model, batch, args.crop_frames, args.crops, args.own_prosody)
        wave = synthesize(model, crop)
        stft, mel = spectral(wave, crop["wave"], crop["lengths"] * c.hop)
        loss = TrainingSystem.WEIGHTS["stft"] * stft + TrainingSystem.WEIGHTS["mel"] * mel
        metrics = {"stft": stft, "mel": mel}
        optimizer.zero_grad(set_to_none=True)
        if step >= args.gan_start:
            index = torch.arange(wave.shape[-1], device=wave.device).expand_as(wave)
            ramp = min(1.0, (step - args.gan_start + 1) / args.gan_ramp)
            loss, gan = adversarial_backward(loss, loss, wave, wave, crop["wave"], index, critics, critic_optimizer, ramp, recipe)
            metrics.update(gan)
        else:
            loss.backward()
        metrics["grad_norm"] = nn.utils.clip_grad_norm_(vocoder.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        for schedule in schedules:
            schedule.step()
        for key, value in {**metrics, "loss": loss}.items():
            window[key] += float(value.detach())
        steps += 1
        done = step == args.steps or time.monotonic() - start > args.max_seconds
        if step % args.log_every == 0 or done:
            record = {k: v / steps for k, v in window.items()}
            record.update(step=step, elapsed_seconds=time.monotonic() - start, learning_rate=schedules[0].get_last_lr()[0])
            if step % args.save_every == 0 or done:
                vocoder.eval()
                generator, totals = torch.Generator().manual_seed(1), defaultdict(float)
                with torch.no_grad():
                    for i in range(len(validation)):
                        crop = crop_batch(model, to_device(collate([validation[i]]), args.device), 2 * args.crop_frames, 1, 0.0, generator, centre=True)
                        for key, value in zip(("validation_stft", "validation_mel"), spectral(synthesize(model, crop), crop["wave"], crop["lengths"] * c.hop)):
                            totals[key] += float(value) / len(validation)
                vocoder.train()
                record.update(totals)
                export_model(model, args.output / "inference.pt", step, state["annotation"])
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if tracker:
                tracker.log(record, step=step)
            window, steps = defaultdict(float), 0
        if done:
            break


if __name__ == "__main__":
    main()

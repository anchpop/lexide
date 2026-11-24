"""Shallow flow-matching refiner: a few-step generative correction of a trained
model's audio, in the spirit of DiffSinger's shallow diffusion.

The base TTS model is frozen. Its waveform (the draft) is moved toward the
recording in the log-magnitude STFT domain; the waveform is rebuilt with the
draft's own phase. Generation starts at the draft plus a little Gaussian noise
rather than at pure noise, and a rectified-flow velocity network walks it to a
realistic spectrum in a few Euler steps. Unlike the post-filter (one
deterministic multiplicative mask) it can add detail the draft lacks.

Magnitude only, because the draft's phase is not aligned with the recording's
(the oscillator starts at a random phase): a first version refined the complex
spectrum, as in SGMSE speech enhancement where both sides share one recording,
and learned to shrink unpredictable-phase harmonics toward zero, leaving speech
quiet and the start noise audible in pauses (runs/eng-nsf-refiner, PER 48%).

Training pairs come from teacher-forced durations and pitch, so the draft is
time-aligned with the recording; half the time the decoder sees its own
predicted prosody, as at inference.

    python -m voca.refine --base runs/eng-gemini/inference.pt --train features/x/train.jsonl \
        --validation features/x/validation.jsonl --output runs/refiner
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F

from .data import AnnotatedDataset, BucketSampler, collate, to_device
from .model import PostFilterBlock, length_mask, stft_frames
from .revocode import decoder_states
from .speak import load_model
from .train import start_tracking


@dataclass(frozen=True)
class RefinerConfig:
    fft: int = 1024              # 512/128 STFT processing sounded phasey (eng-nsf-pf, eng-istft)
    hop: int = 256
    channels: int = 192
    layers: int = 6
    kernel: int = 7
    start_noise: float = 0.1      # std of the noise added to the draft's normalized log magnitude
    steps: int = 4                # Euler steps at inference


class Refiner(nn.Module):
    def __init__(self, c: RefinerConfig, condition_channels, acoustic_hop):
        super().__init__()
        self.config, self.acoustic_hop = c, acoustic_hop
        bins = c.fft // 2 + 1
        self.input = nn.Linear(2 * bins, c.channels)            # state and draft log magnitudes
        self.condition = nn.Linear(condition_channels, c.channels)
        self.time = nn.Sequential(nn.Linear(64, c.channels), nn.GELU(), nn.Linear(c.channels, c.channels))
        self.blocks = nn.ModuleList(PostFilterBlock(c.channels, c.kernel) for _ in range(c.layers))
        self.output = nn.Linear(c.channels, bins)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("window", torch.hann_window(c.fft), persistent=False)
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 32)), persistent=False)

    # Normalized log magnitude [B, F, bins] (and the phase, [B, F, bins]) and back.
    def analyse(self, wave):
        c = self.config
        spectrum = torch.stft(F.pad(wave, (0, c.fft // 2)), c.fft, c.hop, window=self.window, center=True,
                              pad_mode="constant", return_complex=True).transpose(1, 2)
        return (torch.log(spectrum.abs() + 1e-5) + 5) / 5, spectrum.angle()

    def synthesize(self, level, phase, samples):
        c = self.config
        magnitude = (torch.exp(5 * level.clamp(max=3.0) - 5) - 1e-5).clamp_min(0)
        return torch.istft(torch.polar(magnitude, phase).transpose(1, 2), c.fft, c.hop, window=self.window,
                           center=True, length=samples + c.fft // 2)[:, :samples]

    def frame_mask(self, lengths, count):
        return stft_frames(lengths, self.acoustic_hop // self.config.hop, count)

    def velocity(self, state, draft, tau, acoustic, mask, owner):
        angle = tau[:, None] * self.frequencies[None]
        x = self.input(torch.cat((state, draft), dim=-1))
        x = x + self.condition(acoustic.gather(1, owner[..., None].expand(-1, -1, acoustic.shape[-1])))
        x = (x + self.time(torch.cat((angle.sin(), angle.cos()), dim=-1))[:, None]) * mask[..., None]
        for block in self.blocks:
            x = block(x, mask)
        return self.output(x) * mask[..., None]

    @torch.inference_mode()
    def refine(self, wave, acoustic, lengths, steps=None, seed=0):
        """wave [B, S] from the base model, acoustic [B, T, C] its decoder states."""
        c = self.config
        draft, phase = self.analyse(wave)
        mask, owner = self.frame_mask(lengths, draft.shape[1])
        generator = torch.Generator().manual_seed(seed + 104729)
        state = draft + c.start_noise * torch.randn(draft.shape, generator=generator).to(draft.device)
        steps = steps or c.steps
        for k in range(steps):
            tau = torch.full((len(wave),), k / steps, device=wave.device)
            state = state + self.velocity(state, draft, tau, acoustic, mask, owner) / steps
        audible = length_mask(lengths * self.acoustic_hop, wave.shape[-1])
        return self.synthesize(torch.where(mask[..., None], state, draft), phase, wave.shape[-1]) * audible


def save_refiner(refiner, base, step, path):
    torch.save({"config": asdict(refiner.config), "base": str(base), "step": step,
                "condition_channels": refiner.condition.in_features, "acoustic_hop": refiner.acoustic_hop,
                "model": refiner.state_dict()}, path)


def load_refiner(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    refiner = Refiner(RefinerConfig(**state["config"]), state["condition_channels"], state["acoustic_hop"])
    refiner.load_state_dict(state["model"], strict=True)
    return refiner.to(device).eval()


@torch.no_grad()
def drafts(base, batch, own_prosody):
    """The frozen base model's waveform and decoder states with recorded durations
    and pitch (so they line up with the recording)."""
    acoustic, f0, lengths = decoder_states(base, batch, own_prosody)
    phase = 2 * math.pi * torch.rand(len(lengths), device=lengths.device)
    return base.vocoder(acoustic, f0, lengths, phase=phase).float(), acoustic, lengths


def loss(refiner, base, batch, own_fraction=0.5):
    own = torch.rand(len(batch["frame_lengths"]), device=batch["wave"].device) < own_fraction
    wave, acoustic, lengths = drafts(base, batch, own)
    draft, _ = refiner.analyse(wave)
    target, _ = refiner.analyse(batch["wave"][:, :wave.shape[-1]])
    mask, owner = refiner.frame_mask(lengths, draft.shape[1])
    start = draft + refiner.config.start_noise * torch.randn_like(draft)
    tau = torch.rand(len(lengths), device=draft.device)
    state = start + tau[:, None, None] * (target - start)
    predicted = refiner.velocity(state, draft, tau, acoustic, mask, owner)
    error = (predicted - (target - start)).square().mean(-1)
    return (error * mask).sum() / mask.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-frames", type=int, default=9000)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subset", type=int)
    args = parser.parse_args()
    torch.manual_seed(0)
    base = load_model(args.base, args.device)
    for p in base.parameters():
        p.requires_grad_(False)
    c = base.config
    train = AnnotatedDataset(args.train, c, voiced_vowels=True)
    validation = AnnotatedDataset(args.validation, c, voiced_vowels=True)
    pool = list(range(len(train)))[:args.subset]
    refiner = Refiner(RefinerConfig(), c.channels, c.hop).to(args.device)
    print(f"refiner {sum(p.numel() for p in refiner.parameters()) / 1e6:.2f}M parameters", flush=True)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    sampler = BucketSampler([train.lengths[i] for i in pool], [train.token_counts[i] for i in pool], args.batch_frames, args.max_batch, 0)
    args.output.mkdir(parents=True, exist_ok=True)
    tracker = start_tracking(args.output.name, {"refiner": asdict(refiner.config), **{k: str(v) for k, v in vars(args).items()}})
    start, window, count = time.monotonic(), defaultdict(float), 0
    for step in range(1, args.steps + 1):
        batch = to_device(collate([train[pool[i]] for i in sampler.next()[0]]), args.device)
        value = loss(refiner, base, batch)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        window["loss"] += float(value.detach())
        window["grad_norm"] += float(norm)
        count += 1
        done = step == args.steps or time.monotonic() - start > args.max_seconds
        if step % args.log_every == 0 or done:
            record = {k: v / count for k, v in window.items()}
            record.update(step=step, elapsed_seconds=time.monotonic() - start, learning_rate=schedule.get_last_lr()[0])
            if step % args.save_every == 0 or done:
                refiner.eval()
                with torch.no_grad():
                    torch.manual_seed(1)
                    values = [float(loss(refiner, base, to_device(collate([validation[i]]), args.device))) for i in range(len(validation))]
                refiner.train()
                record["validation_loss"] = sum(values) / len(values)
                save_refiner(refiner, args.base, step, args.output / "refiner.pt")
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if tracker:
                tracker.log(record, step=step)
            window, count = defaultdict(float), 0
        if done:
            break


if __name__ == "__main__":
    main()

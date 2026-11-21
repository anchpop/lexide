"""From-scratch training on validated durations. Running this starts paid compute.

There is no learned aligner and no alignment loss: token durations come from
the annotation layer and must cover the acoustic frames exactly. The waveform
discriminators and the auxiliary mel head are training-only. No model or data
downloads and no infrastructure management happen here.
"""

import argparse
import dataclasses
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from .config import CHECKPOINT_FORMAT, Config
from .data import AnnotatedDataset, BucketSampler, collate, prefetch
from .losses import (Discriminators, MelConvergenceLoss, SpectralLoss, StyleCritics, discriminator_loss, generator_gan_loss,
                     masked_mean, slm_feature_loss)
from .model import TinyVoca
from .styletts import StyleEncoder, StyleVoca, frames_of
from .text import frontend_fingerprint, frontend_identity

SEMITONES = 12 / math.log(2)


def masked_std(value, mask):
    """Standard deviation per row over the masked positions, [B]."""
    count = mask.sum(1).clamp_min(1)
    mean = (value * mask).sum(1) / count
    return (((value - mean[:, None]).square() * mask).sum(1) / count + 1e-8).sqrt()


def prosody_terms(prediction, target, trusted, frame_mask):
    """Pitch, pitch-spread, voicing and energy losses for predicted [B, F, 3]
    (log pitch, voicing logit, energy), plus their metrics. Returns (terms, metrics)."""
    # Recorded pitch counts fully; interpolated stretches only keep the curve continuous.
    pitch = masked_mean((prediction[..., 0] - target[..., 0]).abs(), frame_mask * (trusted + 0.1 * (1 - trusted)))
    # An absolute-error fit alone predicts a flatter contour than people speak
    # (about three quarters of the recorded spread on held-out text). Match each
    # utterance's pitch spread as well, the classic global-variance correction.
    sung = frame_mask * target[..., 1] * trusted
    enough = (sung.sum(1) >= 10).float()
    predicted_spread, recorded_spread = masked_std(prediction[..., 0], sung), masked_std(target[..., 0], sung)
    pitch_spread = masked_mean((predicted_spread - recorded_spread).abs(), enough)
    voicing = masked_mean(F.binary_cross_entropy_with_logits(prediction[..., 1], target[..., 1], reduction="none"), frame_mask)
    energy = masked_mean((prediction[..., 2] - target[..., 2]).abs(), frame_mask)
    with torch.no_grad():
        metrics = dict(pitch_error_semitones=SEMITONES * masked_mean((prediction[..., 0] - target[..., 0]).abs(), sung),
                       pitch_spread_ratio=masked_mean(predicted_spread / recorded_spread.clamp_min(1e-3), enough),
                       voicing_accuracy=masked_mean(((prediction[..., 1] > 0) == target[..., 1].bool()).float(), frame_mask))
    return dict(pitch=pitch, pitch_spread=pitch_spread, voicing=voicing, energy=energy), metrics


@torch.no_grad()
def duration_metrics(predicted, presence, durations, pausable, lengths):
    """Free-running duration quality: predicted [B, N] frames, presence logits [B, N]."""
    predicted_frames = predicted.sum(1)
    return dict(pause_accuracy=masked_mean(((presence > 0) == (durations > 0)).float(), pausable),
                pause_length_ratio=(predicted * pausable).sum() / (durations * pausable).sum().clamp_min(1),
                duration_length_ratio=(predicted_frames / lengths).mean(),
                duration_length_error=((predicted_frames - lengths).abs() / lengths).mean())


class TrainingSystem(nn.Module):
    WEIGHTS = {"duration": 1.0, "pause": 1.0, "pitch": 2.0, "pitch_spread": 2.0, "voicing": 1.0, "energy": 1.0, "mel_head": 2.0,
               "stft": 2.0, "mel": 10.0}

    def __init__(self, config):
        super().__init__()
        self.model = TinyVoca(config)
        self.mel_head = nn.Linear(config.channels, config.n_mels)   # auxiliary, not exported
        self.spectral = SpectralLoss(config.sample_rate)

    def objective(self, batch, teacher_probability, weights=WEIGHTS, random_phase=True):
        """Loss on full utterances. Returns ``(loss, terms, auxiliary)``.

        ``teacher_probability`` is the per-utterance chance that the acoustic
        decoder is conditioned on recorded pitch/voicing/energy; otherwise it
        gets the model's own (detached) predictions, exactly as at inference.
        The oscillator always follows the recorded pitch here, because the
        waveform target does.
        """
        m, c = self.model, self.model.config
        hidden, token_mask = m.encode(batch["tokens"], batch["stress"], batch["token_lengths"], batch["speakers"])
        output = m.duration(hidden, token_mask).float()
        # Length is learned only where a token has frames; whether a pause-capable
        # token has any is a separate yes/no target.
        present = batch["durations"] > 0
        duration = masked_mean((output[..., 0] - torch.log1p(batch["durations"].float())).square(), token_mask & present)
        pausable = token_mask & ~m.phone_tokens[batch["tokens"]]
        pause = masked_mean(F.binary_cross_entropy_with_logits(output[..., 1], present.float(), reduction="none"), pausable)
        frames, frame_mask, lengths = m.expand(hidden, batch["durations"], token_mask, batch["speakers"])
        if not torch.equal(lengths, batch["frame_lengths"]):
            raise ValueError("durations do not sum to the acoustic frame count")
        prosody_hidden, prediction = m.predict_prosody(frames, frame_mask)
        prediction, target = prediction.float(), batch["prosody"]
        prosody, prosody_metrics = prosody_terms(prediction, target, batch["pitch_trusted"], frame_mask)
        teacher = torch.rand(len(lengths), device=lengths.device) < teacher_probability
        condition = torch.where(teacher[:, None, None], target, m.hard_prosody(prediction.detach()))
        acoustic = m.decode(prosody_hidden, condition.to(prosody_hidden.dtype), frame_mask)
        mel_target = (batch["mel"] - c.mel_mean) / c.mel_scale
        mel_head = masked_mean((self.mel_head(acoustic).float() - mel_target).abs().mean(-1), frame_mask)
        phase = 2 * math.pi * torch.rand(len(lengths), device=lengths.device) if random_phase else None
        wave = m.vocoder(acoustic, m.frequency(target[..., 0]), lengths, phase=phase)
        stft, mel = self.spectral(wave, batch["wave"], lengths * c.hop)
        terms = dict(duration=duration, pause=pause, **prosody, mel_head=mel_head, stft=stft, mel=mel)
        loss = sum(weights[name] * value for name, value in terms.items())
        with torch.no_grad():
            predicted = m.predict_durations(hidden, token_mask, batch["tokens"])
        terms.update(base_loss=loss.detach(), teacher_fraction=teacher.float().mean(), **prosody_metrics,
                     **duration_metrics(predicted, output[..., 1], batch["durations"], pausable, lengths))
        return loss, terms, {"wave": wave, "reconstruction": weights["stft"] * stft + weights["mel"] * mel}

    def critics(self):
        return Discriminators()

    def adversarial(self, loss, auxiliary, batch, critics, critic_optimizer, step, recipe):
        """One critic update, then backward of ``loss`` plus the balanced GAN term."""
        wave = auxiliary["wave"]
        fake, real, index = crops(wave, batch["wave"], batch["frame_lengths"] * self.model.config.hop, recipe["crop_samples"], recipe["crops"])
        return adversarial_backward(loss, auxiliary["reconstruction"], wave, fake, real, index, critics, critic_optimizer, step, recipe)


class StyleTrainingSystem(nn.Module):
    """StyleTTS 2's second-stage objective, minus its aligner, pitch network,
    diffusion and PL-BERT: style encoders read each clip's mel; the prosody
    predictor is supervised by our durations, pitch, voicing and energy; the
    decoder and the F0/energy branches run on equal-length random windows
    (``config.crop_frames``) as in StyleTTS 2. Weights are StyleTTS 2's
    (lambda_mel 5, lambda_ce 20, lambda_dur 1, lambda_gen 1, lambda_slm 1) and ours
    for the prosody targets. The style encoders, critics and WavLM never ship."""

    WEIGHTS = {"duration": 1.0, "duration_bins": 20.0, "pause": 1.0, "pitch": 2.0, "pitch_spread": 2.0, "voicing": 1.0, "energy": 1.0,
               "mel_convergence": 5.0, "slm": 1.0, "gan": 1.0, "slm_gan": 1.0}
    def __init__(self, config):
        super().__init__()
        self.model = StyleVoca(config)
        self.style_encoder, self.predictor_encoder = StyleEncoder(config), StyleEncoder(config)   # acoustic, prosodic
        self.mel_convergence = MelConvergenceLoss(config.sample_rate)
        self.spectral = SpectralLoss(config.sample_rate)   # metrics comparable with TinyVoca runs

    def critics(self):
        return StyleCritics(self.model.config.sample_rate)


    def objective(self, batch, teacher_probability, weights=WEIGHTS, random_phase=True):
        m, c = self.model, self.model.config
        tokens, token_lengths, speakers, durations = batch["tokens"], batch["token_lengths"], batch["speakers"], batch["durations"]
        hidden, token_mask = m.encode(tokens, batch["stress"], token_lengths, speakers)
        # The dataset checked that durations sum to the frame counts. Shapes come
        # from the host copies of the lengths, so nothing here waits for the GPU.
        host = batch["host"]
        index, frame_mask, lengths = m.align(durations, token_mask, frames=batch["mel"].shape[1])
        # One window per utterance, the same length across the batch; centred when validating.
        size = min(c.crop_frames, int(host["frame_lengths"].min()))
        offset = torch.rand(len(lengths), device=lengths.device) if self.training else torch.full_like(lengths, 0.5, dtype=torch.float)
        window = (offset * (lengths - size + 1)).long().clamp(max=lengths - size)[:, None] + torch.arange(size, device=lengths.device)
        mel = batch["mel"].gather(1, window[..., None].expand(-1, -1, c.n_mels))
        if self.training:
            mel = ((mel - c.mel_mean) / c.mel_scale).transpose(1, 2)[:, None]
            acoustic_style, prosodic_style = self.style_encoder(mel), self.predictor_encoder(mel)
            m.observe_style(torch.cat((acoustic_style, prosodic_style), dim=-1).detach(), speakers)
        else:
            acoustic_style, prosodic_style = m.style[speakers].chunk(2, dim=-1)
        states, logits = m.predict_tokens(hidden, tokens, batch["stress"], token_mask, host["token_lengths"], prosodic_style)
        logits = logits.float()
        # Durations as StyleTTS 2 learns them (bins "longer than k" + L1 on their sum),
        # on tokens that have frames; whether a pause-capable token has any is the presence logit.
        present = token_mask & (durations > 0)
        bins = (torch.arange(c.style_max_duration, device=durations.device) < durations[..., None]).float()
        duration_bins = masked_mean(F.binary_cross_entropy_with_logits(logits[..., :-1], bins, reduction="none").mean(-1), present)
        duration = masked_mean((torch.sigmoid(logits[..., :-1]).sum(-1) - durations.clamp(max=c.style_max_duration)).abs(), present)
        pausable = token_mask & ~m.phone_tokens[tokens]
        pause = masked_mean(F.binary_cross_entropy_with_logits(logits[..., -1], (durations > 0).float(), reduction="none"), pausable)
        crop = index.gather(1, window)
        prediction = m.predictor.F0Ntrain(frames_of(states, crop), prosodic_style, torch.full((len(lengths),), size)).float()
        target, trusted = batch["prosody"].gather(1, window[..., None].expand(-1, -1, 3)), batch["pitch_trusted"].gather(1, window)
        prosody, prosody_metrics = prosody_terms(prediction, target, trusted, torch.ones_like(trusted))
        teacher = torch.rand(len(lengths), device=lengths.device) < teacher_probability
        condition = torch.where(teacher[:, None, None], target, m.hard_prosody(prediction.detach()))
        # The source always follows the recorded pitch and voicing, because the waveform target does.
        wave = m.decoder(frames_of(hidden, crop), condition, m.frequency(target[..., 0]) * target[..., 1], acoustic_style)
        samples = (window[:, :1] * c.hop + torch.arange(size * c.hop, device=lengths.device))
        real = batch["wave"].gather(1, samples)
        terms = dict(duration=duration, duration_bins=duration_bins, pause=pause, **prosody, mel_convergence=self.mel_convergence(wave, real))
        if self.training:
            terms["slm"], slm = slm_feature_loss(wave, real, c.sample_rate)
        loss = sum(weights[name] * value for name, value in terms.items())
        with torch.no_grad():
            stft, mel = self.spectral(wave, real, torch.full_like(lengths, size * c.hop))
            predicted = m.predict_durations(logits, tokens, token_mask)
        terms.update(base_loss=loss.detach(), teacher_fraction=teacher.float().mean(), stft=stft, mel=mel, **prosody_metrics,
                     **duration_metrics(predicted, logits[..., -1], durations, pausable, lengths))
        return loss, terms, {"wave": wave, "real": real, "slm": slm if self.training else None}

    def adversarial(self, loss, auxiliary, batch, critics, critic_optimizer, step, recipe):
        """StyleTTS 2's GAN step with fixed weights: critics first (the WavLM head
        every 10th step), then backward of ``loss`` plus the generator terms."""
        fake, real = auxiliary["wave"], auxiliary["real"]
        weights = recipe["weights"]
        critic_optimizer.zero_grad(set_to_none=True)
        d_loss = critics.discriminator_loss(real, fake.detach(), slm=auxiliary["slm"] if step % 10 == 0 else None)
        d_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(critics.parameters(), 10)
        critic_optimizer.step()
        critics.requires_grad_(False)
        try:
            gan, slm_gan = critics.generator_loss(real, fake, auxiliary["slm"]["fake"])
            loss = loss + weights["gan"] * gan + weights["slm_gan"] * slm_gan
            loss.backward()
        finally:
            critics.requires_grad_(True)
        return loss, dict(discriminator=d_loss, adversarial=gan, slm_adversarial=slm_gan, critic_grad_norm=critic_grad)


TRAINING_SYSTEMS = {"tinyvoca": TrainingSystem, "styletts": StyleTrainingSystem}


def crops(fake, real, lengths, size, count):
    """Aligned random windows [B*count, size] from generated and recorded audio,
    plus the sample indices they cover ([B, count*size])."""
    size = min(size, int(lengths.min()))
    start = (torch.rand(len(lengths), count, device=fake.device) * (lengths[:, None] - size + 1)).long()
    index = (start[..., None] + torch.arange(size, device=fake.device)).flatten(1)
    return fake.gather(1, index).view(-1, size), real.gather(1, index).view(-1, size), index


def learning_rate(step, base, warmup, half_life, start=0):
    """Linear warm-up from ``start`` (0, or the step a run was branched at),
    then halving every ``half_life`` steps."""
    return base * min(1.0, (step - start + 1) / max(warmup, 1)) * 0.5 ** (step / half_life)


def teacher_probability(step, hold, ramp, final):
    """1 while the prosody predictor is still poor, then down to ``final``."""
    return 1.0 if step < hold else max(final, 1.0 - (1.0 - final) * (step - hold) / max(ramp, 1))


def adversarial_backward(loss, reconstruction, wave, fake, real, index, critics, critic_optimizer, step, recipe):
    """One critic update on the windows ``fake``/``real`` (taken from ``wave`` at
    ``index``), then backward of ``loss`` plus the balanced generator GAN term.
    Returns the total loss and the GAN metrics."""
    critic_optimizer.zero_grad(set_to_none=True)
    d_loss = discriminator_loss(critics(real), critics(fake.detach()))
    d_loss.backward()
    critic_grad = nn.utils.clip_grad_norm_(critics.parameters(), 10)
    critic_optimizer.step()
    critics.requires_grad_(False)
    try:
        with torch.no_grad():
            real_features = critics(real)
        adversarial, matching = generator_gan_loss(real_features, critics(fake))
        gan = adversarial + recipe["feature_matching"] * matching
        # Fixed weights cannot balance these terms: measured at the waveform, the
        # adversarial gradient was under 1% of the spectral one, i.e. switched off.
        # Scale the GAN term so that, inside the windows the critics saw, its
        # gradient is ``gan_ratio`` times the reconstruction gradient.
        reconstruction_gradient = torch.autograd.grad(reconstruction, wave, retain_graph=True)[0].gather(1, index)
        gan_gradient = torch.autograd.grad(gan, wave, retain_graph=True)[0].gather(1, index)
        ramp = min(1.0, (step - recipe["gan_start"] + 1) / max(recipe["gan_ramp"], 1))
        scale = (recipe["gan_ratio"] * ramp * reconstruction_gradient.norm() / gan_gradient.norm().clamp_min(1e-12))
        scale = scale.clamp(max=recipe["gan_scale_max"]).detach()
        loss = loss + scale * gan
        loss.backward()
    finally:
        critics.requires_grad_(True)
    return loss, dict(discriminator=d_loss, adversarial=adversarial, feature_matching=matching, gan_scale=scale, critic_grad_norm=critic_grad)


def optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe):
    weights = recipe["weights"]
    start = (recipe.get("branched_from") or {}).get("step", 0)
    for group in optimizer.param_groups:
        group["lr"] = group.get("scale", 1.0) * learning_rate(step, recipe["learning_rate"], recipe["warmup"], recipe["lr_half_life"], start)
    for group in critic_optimizer.param_groups:
        group["lr"] = learning_rate(step, recipe["learning_rate"], 1, recipe["lr_half_life"], start)
    optimizer.zero_grad(set_to_none=True)
    probability = teacher_probability(step, recipe["teacher_hold"], recipe["teacher_ramp"], recipe["teacher_final"])
    loss, metrics, auxiliary = system.objective(batch, probability, weights)
    if step >= recipe["gan_start"]:
        loss, gan_metrics = system.adversarial(loss, auxiliary, batch, critics, critic_optimizer, step, recipe)
        metrics.update(gan_metrics)
    else:
        loss.backward()
    # Gradient norms are returned, not checked: checking would make every step wait
    # for the GPU. The training loop stops at the next log step on a non-finite one.
    grad = nn.utils.clip_grad_norm_(system.parameters(), recipe["grad_clip"])
    optimizer.step()
    return {**{k: torch.as_tensor(v).detach() for k, v in metrics.items()}, "loss": loss.detach(), "grad_norm": grad}


def start_tracking(name, config):
    """A Weights & Biases run named after the output directory, when WANDB_API_KEY is set."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb
    return wandb.init(project="lexide-tts", name=name, id=name, resume="allow", config=config)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def export_model(model, path, step, annotation=None, weights=None):
    """``weights``: a state dict to ship instead of the model's own (the EMA)."""
    atomic_save({"format_version": CHECKPOINT_FORMAT, "config": model.config.to_dict(), "step": step,
                 "frontend": {**frontend_identity(), "fingerprint": frontend_fingerprint()}, "annotation": annotation,
                 "model": {k: v.detach().cpu() for k, v in (weights or model.state_dict()).items()}}, path)


@torch.no_grad()
def update_ema(ema, model, decay):
    """Exponential moving average of the weights, which is what ships: it smooths
    out the step-to-step jitter of adversarial training at no inference cost."""
    names = [name for name, value in model.state_dict().items() if value.dtype.is_floating_point]
    state = model.state_dict()
    torch._foreach_lerp_([ema[name] for name in names], [state[name] for name in names], 1 - decay)
    for name, value in state.items():
        if not value.dtype.is_floating_point:
            ema[name].copy_(value)


def load_compatible(module, state):
    """Load weights for a branched run. A tensor that grew (same rank, every
    dimension at least as large) keeps the old values in its leading block and
    its fresh initialisation elsewhere. Returns what was not an exact copy."""
    current = module.state_dict()
    report = {"grown": [], "new": [], "dropped": sorted(set(state) - set(current))}
    for name, tensor in current.items():
        if name not in state:
            report["new"].append(name)
        elif state[name].shape == tensor.shape:
            current[name] = state[name]
        elif state[name].ndim == tensor.ndim and all(a <= b for a, b in zip(state[name].shape, tensor.shape)):
            merged = tensor.clone()
            merged[tuple(slice(0, size) for size in state[name].shape)] = state[name]
            current[name] = merged
            report["grown"].append(name)
        else:
            raise ValueError(f"cannot branch: {name} changed shape from {tuple(state[name].shape)} to {tuple(tensor.shape)}")
    module.load_state_dict(current)
    return report


def balanced_indices(dataset, limit):
    """Stable round-robin over voices, independent of manifest order."""
    groups = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        groups[row["speaker"]].append(i)
    limit = min(max(limit, len(groups)), len(dataset.rows))
    selected = []
    for offset in range(limit):
        for voice in sorted(groups):
            if offset < len(groups[voice]):
                selected.append(groups[voice][offset])
                if len(selected) == limit:
                    return selected
    return selected


def check_leakage(train, validation):
    ids = {r["id"] for r in train.rows}
    texts = defaultdict(list)
    for row in train.rows:
        texts[row["text_hash"]].append(row["id"])
    collisions = []
    for row in validation.rows:
        if row["id"] in ids or row["text_hash"] in texts:
            collisions.append({"validation_id": row["id"], "train_ids": texts.get(row["text_hash"], [])[:3], "text": row.get("text", "")[:160]})
            if len(collisions) == 5:
                break
    if collisions:
        raise ValueError("train/validation leakage; split by sentence/text_id across BOTH voices. Examples: " + json.dumps(collisions, ensure_ascii=False))


@torch.no_grad()
def validate(system, dataset, indices, device, weights, batch_size=8):
    """Held-out metrics on a fixed balanced subset.

    ``teacher_forced`` conditions the decoder on recorded prosody, ``predicted``
    on the model's own; both use recorded durations and pitch for the oscillator
    so the spectra are comparable with the recording. Durations, pauses, pitch
    and voicing errors are those of the free-running predictors. This is a
    training monitor; pronunciation is judged separately with ASR and by listening.
    """
    system.eval()
    order = sorted(indices, key=lambda i: dataset.lengths[i])
    totals, weight = defaultdict(float), 0
    for start in range(0, len(order), batch_size):
        batch = collate([dataset[i] for i in order[start:start + batch_size]], device)
        size = len(batch["frame_lengths"])
        for name, probability in (("teacher_forced", 1.0), ("predicted", 0.0)):
            _, metrics, _ = system.objective(batch, probability, weights, random_phase=False)
            totals[f"validation_{name}_mel"] += float(metrics["mel"]) * size
            totals[f"validation_{name}_stft"] += float(metrics["stft"]) * size
        for key in ("duration", "pause_accuracy", "pause_length_ratio", "pitch_error_semitones", "pitch_spread_ratio", "voicing_accuracy",
                    "duration_length_ratio", "duration_length_error"):
            totals["validation_" + key] += float(metrics[key]) * size
        weight += size
    system.train()
    return {key: value / weight for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Annotation manifest (train.jsonl)")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phoneme-encoder", type=Path, help="voca.phonebert encoder.pt to start the prosody predictor's text encoder from")
    parser.add_argument("--phoneme-encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 ships the raw weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-frames", type=int, default=9000, help="Padded acoustic frames per batch")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--lr-half-life", type=float, default=100000)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--weights", type=json.loads, default={}, help="JSON overrides for the loss weights, e.g. '{\"mel\": 5}'")
    parser.add_argument("--gan-start", type=int, default=8000)
    parser.add_argument("--gan-ratio", type=float, default=0.7, help="Adversarial gradient as a fraction of the reconstruction gradient at the waveform")
    parser.add_argument("--gan-ramp", type=int, default=3000, help="Steps over which the adversarial term fades in while the critics warm up")
    parser.add_argument("--feature-matching", type=float, default=30.0, help="Feature-matching weight inside the adversarial term")
    parser.add_argument("--gan-scale-max", type=float, default=2000.0)
    parser.add_argument("--crop-samples", type=int, default=8192)
    parser.add_argument("--crops", type=int, default=2, help="Discriminator windows per utterance")
    parser.add_argument("--teacher-hold", type=int, default=3000, help="Steps with recorded prosody conditioning only")
    parser.add_argument("--teacher-ramp", type=int, default=7000)
    parser.add_argument("--teacher-final", type=float, default=0.0, help="Share of utterances still conditioned on recorded prosody "
                        "after the ramp (0 = the decoder sees only what it gets at inference)")
    parser.add_argument("--subset", type=int, help="Train on the first N utterances (balanced by voice): memorisation check")
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-every", type=int, default=10000, help="Keep a numbered checkpoint at this interval and at the GAN boundary")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--max-seconds", type=float, default=3600, help="Wall-time cap per invocation; save and exit at a step boundary")
    parser.add_argument("--deadline-file", type=Path, help="Budget JSON with an absolute 'deadline'; training saves and exits before it")
    parser.add_argument("--deadline-margin", type=float, default=300)
    parser.add_argument("--resume", type=Path, help="Continue the same run exactly (same recipe, data and configuration)")
    parser.add_argument("--init-from", type=Path, help="Start a NEW run from another run's checkpoint with a changed recipe: "
                        "weights, step counter and already-trained critics are taken over; the data order starts fresh")
    parser.add_argument("--new-data", action="store_true", help="With --init-from: the checkpoint comes from a different annotation "
                        "run or front end (same symbol inventory)")
    parser.add_argument("--stop-file", type=Path, help="Save and exit after a step when this file appears")
    parser.add_argument("--recorded-voicing", action="store_true", help="Use the pitch tracker's voicing as it is, "
                        "without marking vowel frames voiced")
    parser.add_argument("--no-preload", action="store_true", help="Load feature files lazily instead of before the first step")
    args = parser.parse_args()
    if min(args.steps, args.batch_frames, args.max_batch, args.save_every, args.keep_every, args.log_every, args.crops, args.gan_ramp) < 1:
        parser.error("steps, batch limits and intervals must be positive")
    if args.max_seconds <= 0 or args.learning_rate <= 0 or args.lr_half_life <= 0 or not 0 <= args.teacher_final <= 1:
        parser.error("time cap, learning rate and half life must be positive; teacher_final in [0, 1]")
    if min(args.gan_ratio, args.feature_matching, args.gan_scale_max) < 0:
        parser.error("GAN settings must be non-negative")
    if args.resume and args.init_from:
        parser.error("--init-from starts a new run; --resume continues one")

    # The budget file is re-read while training so a deliberate extension (or cut) takes effect.
    budget = {"deadline": math.inf, "checked": time.time()}
    if args.deadline_file:
        budget["deadline"] = float(json.loads(args.deadline_file.read_text())["deadline"]) - args.deadline_margin

    def deadline():
        if args.deadline_file and time.time() - budget["checked"] > 30:
            budget["checked"] = time.time()
            try:
                budget["deadline"] = float(json.loads(args.deadline_file.read_text())["deadline"]) - args.deadline_margin
            except (OSError, ValueError, KeyError):
                pass    # keep the last valid deadline while the file is being rewritten
        return budget["deadline"]
    if time.time() >= deadline():
        raise SystemExit("the cloud budget deadline has passed (or is too close); not starting")
    c = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    if args.phoneme_encoder:
        pretrained = torch.load(args.phoneme_encoder, map_location="cpu", weights_only=True)
        e = pretrained["config"]
        c = dataclasses.replace(c, phoneme_encoder_layers=pretrained["layers"], phoneme_encoder_width=e["channels"],
                                phoneme_encoder_heads=e["heads"], phoneme_encoder_feedforward=e["feedforward"])
    weights = TRAINING_SYSTEMS[c.architecture].WEIGHTS
    if set(args.weights) - set(weights):
        parser.error(f"--weights may only override {sorted(weights)}")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train = AnnotatedDataset(args.train, c, voiced_vowels=not args.recorded_voicing)
    validation = AnnotatedDataset(args.validation, c, voiced_vowels=not args.recorded_voicing)
    check_leakage(train, validation)
    if train.annotation != validation.annotation:
        raise ValueError("train and validation manifests come from different annotation runs")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("output directory is not empty; use --resume or a new directory")
    pool = balanced_indices(train, args.subset) if args.subset else list(range(len(train)))
    validation_indices = balanced_indices(validation, args.validation_size)
    if not args.no_preload:
        train.preload(pool)
        validation.preload(validation_indices)
    system = TRAINING_SYSTEMS[c.architecture](c).to(args.device)
    critics = system.critics().to(args.device)
    count = sum(p.numel() for p in system.model.parameters())
    if count > system.model.PARAMETER_BUDGET:
        raise ValueError(f"inference parameter budget exceeded: {count:,}")
    if args.phoneme_encoder:
        system.model.bert.load_state_dict(pretrained["encoder"])
    # StyleTTS 2 fine-tunes its pretrained PL-BERT at a tenth of the learning rate.
    encoder = getattr(system.model, "bert", None)
    bert = {id(p) for p in encoder.parameters()} if encoder else set()
    groups = [{"params": [p for p in system.parameters() if id(p) not in bert], "scale": 1.0}]
    if bert:
        groups.append({"params": list(encoder.parameters()), "scale": args.phoneme_encoder_lr_scale})
    fused = str(args.device).startswith("cuda")   # one kernel per step instead of several per tensor; same update
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.01, fused=fused)
    critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.0, fused=fused)
    sampler = BucketSampler([train.lengths[i] for i in pool], args.batch_frames, args.max_batch, args.seed)
    fingerprints = [train.fingerprint, validation.fingerprint]
    recipe = {key: getattr(args, key) for key in ("batch_frames", "max_batch", "learning_rate", "warmup", "lr_half_life", "grad_clip",
                                                   "gan_start", "gan_ratio", "gan_ramp", "feature_matching", "gan_scale_max",
                                                   "crop_samples", "crops", "teacher_hold", "teacher_ramp", "teacher_final",
                                                   "subset", "validation_size", "seed", "ema_decay", "phoneme_encoder_lr_scale")}
    recipe["voiced_vowel_targets"] = not args.recorded_voicing
    recipe["weights"] = {**weights, **{key: float(value) for key, value in args.weights.items()}}
    recipe["branched_from"] = None
    tracker = start_tracking(args.output.name, {"model": c.to_dict(), "recipe": recipe})
    step = 0

    def compatible(state, what, same_data=True):
        if state.get("format_version") != CHECKPOINT_FORMAT:
            raise ValueError(f"checkpoint predates phoneme tokens and supervised durations (format < 4); it cannot be {what}")
        if same_data and (state["frontend"] != frontend_fingerprint() or state["annotation"] != train.annotation):
            raise ValueError("checkpoint was trained with a different front end or annotation run "
                             "(branch with --init-from and --new-data to fine-tune on new annotations)")
        if state["config"] != c.to_dict():
            raise ValueError("checkpoint was trained with a different model configuration")
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        compatible(state, "resumed")
        recipe["branched_from"] = state["recipe"].get("branched_from")
        if state["datasets"] != fingerprints or state["recipe"] != recipe:
            raise ValueError("resume recipe or dataset mismatch (use --init-from to branch a new run with a changed recipe)")
        system.load_state_dict(state["system"])
        critics.load_state_dict(state["critics"])
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        sampler.load_state(state["sampler"])
        ema = state["ema"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if str(args.device).startswith("cuda") and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        step = state["step"]
    elif args.init_from:
        state = torch.load(args.init_from, map_location=args.device, weights_only=True)
        # The symbol inventory is fixed (vocabulary size is part of the configuration),
        # so a checkpoint can be fine-tuned on another annotation run or front end.
        compatible(state, "branched", same_data=not args.new_data)
        step = state["step"]
        loaded = load_compatible(system, state["system"])
        # Critics that have already been trained are taken over, otherwise a branch
        # in the adversarial phase would hand full-strength gradients from untrained
        # critics to the generator. Optimizer moments are kept when no tensor changed.
        loaded["critics"] = step > state["recipe"]["gan_start"]
        if loaded["critics"]:
            critics.load_state_dict(state["critics"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
        loaded["optimizer"] = not loaded["grown"] and not loaded["new"] and not loaded["dropped"]
        if loaded["optimizer"]:
            scales = [group["scale"] for group in optimizer.param_groups]
            optimizer.load_state_dict(state["optimizer"])   # restores the checkpoint's scales too: keep this run's
            for group, scale in zip(optimizer.param_groups, scales):
                group["scale"] = scale
        recipe["branched_from"] = {"step": step, "recipe": state["recipe"], "datasets": state["datasets"], "loaded": loaded}
        if args.new_data:
            recipe["branched_from"].update(frontend=state["frontend"], annotation=state["annotation"])
    if step >= args.steps:
        raise ValueError("checkpoint already reached requested total steps")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(c.to_dict(), indent=2) + "\n")
    (args.output / "recipe.json").write_text(json.dumps({**recipe, "annotation": train.annotation, "frontend": frontend_fingerprint(),
                                                         "train_utterances": len(pool), "datasets": fingerprints}, indent=2) + "\n")
    print(json.dumps({"inference_parameters": count, "training_only_parameters": sum(p.numel() for p in system.parameters()) - count
                      + sum(p.numel() for p in critics.parameters()), "train_utterances": len(pool), "start_step": step,
                      "branched": bool(args.init_from), "resumed": bool(args.resume)}), flush=True)
    if not args.resume:
        ema = {k: v.detach().clone() for k, v in system.model.state_dict().items()}
    start = time.monotonic()
    system.train()

    def snapshot():
        return {"format_version": CHECKPOINT_FORMAT, "config": c.to_dict(), "frontend": frontend_fingerprint(), "annotation": train.annotation,
                "step": step, "datasets": fingerprints, "recipe": recipe, "system": system.state_dict(), "critics": critics.state_dict(),
                "optimizer": optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(), "sampler": sampler_state, "ema": ema,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if str(args.device).startswith("cuda") else None}
    sampler_state = sampler.state()   # as of the last batch trained on, not the prefetched ones
    if step == args.gan_start and not args.resume:
        atomic_save(snapshot(), args.output / f"step-{step:08d}.pt")
    # Metrics stay on the GPU between log steps: reading one makes the CPU wait for
    # the GPU, which then idles while the next step's kernels are being queued.
    window, window_steps, window_start = None, 0, time.monotonic()
    totals = {"frames": 0, "utterances": 0}
    def draw():
        indices = sampler.next()
        return sampler.state(), collate([train[pool[i]] for i in indices], args.device)
    batches = prefetch(draw)
    while step < args.steps:
        sampler_state, batch = next(batches)
        metrics = optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe)
        update_ema(ema, system.model, args.ema_decay)
        step += 1
        names = sorted(metrics)
        values = torch.stack([metrics[name].float().reshape(()) for name in names])
        window = values if window is None else window + values
        totals["frames"] += int(batch["host"]["frame_lengths"].sum())
        totals["utterances"] += len(batch["host"]["frame_lengths"])
        window_steps += 1
        elapsed = time.monotonic() - start
        should_stop = (elapsed >= args.max_seconds or step == args.steps or time.time() >= deadline()
                       or (args.stop_file is not None and args.stop_file.exists()))
        retain = step == args.gan_start or step % args.keep_every == 0
        save = step % args.save_every == 0 or should_stop or retain
        if step % args.log_every == 0 or save:
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.monotonic() - window_start
            record = dict(zip(names, (window / window_steps).tolist()), **totals)
            if not all(math.isfinite(record[key]) for key in ("grad_norm", "critic_grad_norm") if key in record):
                raise FloatingPointError(f"non-finite gradient norm before step {step}; last saved checkpoint is intact")
            record.update(step=step, elapsed_seconds=elapsed, steps_per_second=window_steps / max(seconds, 1e-9),
                          audio_seconds_per_second=totals["frames"] * c.hop / c.sample_rate / max(seconds, 1e-9),
                          learning_rate=optimizer.param_groups[0]["lr"])
            if save:
                record.update(validate(system, validation, validation_indices, args.device, recipe["weights"]))
                state = snapshot()
                atomic_save(state, args.output / "last.pt")
                # inference.pt ships the EMA weights; validation metrics are of the raw ones.
                export_model(system.model, args.output / "inference.pt", step, train.annotation, ema)
                export_model(system.model, args.output / "inference-raw.pt", step, train.annotation)
                if retain:
                    atomic_save(state, args.output / f"step-{step:08d}.pt")
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if tracker:
                tracker.log(record, step=step)
            window, window_steps, window_start = None, 0, time.monotonic()
            totals = {"frames": 0, "utterances": 0}
        if should_stop:
            break


if __name__ == "__main__":
    main()

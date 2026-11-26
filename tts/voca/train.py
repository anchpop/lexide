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
from .config import CHECKPOINT_FORMAT, Config, check_format
from .data import AnnotatedDataset, BucketSampler, collate, pin, prefetch, to_device
from .losses import (Discriminators, MelConvergenceLoss, SpectralLoss, StyleCritics, discriminator_loss, generator_gan_loss,
                     masked_mean, slm_feature_loss)
from .evaluate import DeployedEvaluation
from .model import TinyVoca, build_model
from .styletts import WORD_SEMITONES, Reading, StyleEncoder, StyleVoca, frames_of, reading_divergence, recorded_delivery
from .text import frontend_fingerprint, frontend_identity, sentence_type

SEMITONES = 12 / math.log(2)


def masked_std(value, mask):
    """Standard deviation per row over the masked positions, [B]."""
    count = mask.sum(1).clamp_min(1)
    mean = (value * mask).sum(1) / count
    return (((value - mean[:, None]).square() * mask).sum(1) / count + 1e-8).sqrt()


def smoothed_spread(pitch, voiced):
    # Seven frames span 75 ms. Renormalize at voicing gaps and utterance edges.
    def smooth(x):
        return F.avg_pool1d(x[:, None], 7, stride=1, padding=3)[:, 0]
    contour = smooth(pitch * voiced) / smooth(voiced).clamp_min(1 / 7)
    return masked_std(contour, voiced)


def delta_percentiles(pitch, pairs):
    # Fixed-size sort and gather also work inside CUDA graphs. Masked deltas
    # sort after every real one; an empty set gives five zeros.
    delta = pitch.diff().abs().flatten()
    ordered = F.pad(delta.masked_fill(~pairs.flatten(), float("inf")), (0, 1), value=float("inf")).sort().values
    count = pairs.sum()
    last = (count - 1).clamp_min(0)
    rank = torch.stack([last * q for q in (.25, .50, .90, .99, .999)])
    low = torch.where(count > 0, ordered[rank.floor().long()], 0)
    high = torch.where(count > 0, ordered[rank.ceil().long()], 0)
    return torch.lerp(low, high, rank.frac())


def prosody_terms(prediction, target, trusted, frame_mask, phrase):
    """Pitch, pitch-spread, voicing and energy losses for predicted [B, F, 3]
    (log pitch, voicing logit, energy), plus their metrics. Returns (terms, metrics)."""
    # Recorded pitch counts fully; interpolated stretches only keep the curve continuous.
    pitch = masked_mean((prediction[..., 0] - target[..., 0]).abs(), frame_mask * (trusted + 0.1 * (1 - trusted)))
    # An absolute-error fit alone predicts a flatter contour than people speak
    # (about three quarters of the recorded spread on held-out text). Match each
    # utterance's pitch spread as well, the classic global-variance correction.
    sung = frame_mask * target[..., 1] * trusted
    enough = (sung.sum(1) >= 10).float()
    predicted_spread, recorded_spread = smoothed_spread(prediction[..., 0], sung), smoothed_spread(target[..., 0], sung)
    pitch_spread = masked_mean((predicted_spread - recorded_spread).abs(), enough)
    pairs = sung[:, 1:].bool() & sung[:, :-1].bool() & (phrase[:, 1:] == phrase[:, :-1])
    pitch_delta = (delta_percentiles(prediction[..., 0], pairs) - delta_percentiles(target[..., 0], pairs)).abs().mean()
    voicing = masked_mean(F.binary_cross_entropy_with_logits(prediction[..., 1], target[..., 1], reduction="none"), frame_mask)
    energy = masked_mean((prediction[..., 2] - target[..., 2]).abs(), frame_mask)
    with torch.no_grad():
        metrics = dict(pitch_error_semitones=SEMITONES * masked_mean((prediction[..., 0] - target[..., 0]).abs(), sung),
                       pitch_spread_ratio=masked_mean(predicted_spread / recorded_spread.clamp_min(1e-3), enough),
                       voicing_accuracy=masked_mean(((prediction[..., 1] > 0) == target[..., 1].bool()).float(), frame_mask))
    return dict(pitch=pitch, pitch_spread=pitch_spread, pitch_delta=pitch_delta, voicing=voicing, energy=energy), metrics


def word_terms(outputs, targets, words):
    """Losses of the word head (``voca.styletts.WordProsody``) outputs [B, W, 8] against
    ``voca.data.word_prosody`` targets [B, W, 4] on the batch's ``Words``, plus their metrics.

    Prominence is learned from comparisons inside each phrase: every pair of
    words is ordered as the recording orders them, weighted by how far apart they
    are, so near ties barely count and a phrase may have several prominent words
    or none. Shape is a classification; excursion and final movement are robust
    (Huber, quadratic within a semitone) regressions."""
    outputs = outputs.float()
    valid = torch.isfinite(targets) & words.mask[..., None]
    t = targets.nan_to_num()
    score, target = outputs[..., 0], t[..., 0]
    pairs = words.same & valid[:, :, None, 0] & valid[:, None, :, 0]
    gap = target[:, :, None] - target[:, None, :]
    weight = pairs * gap.abs().clamp(max=1.0)
    rank = (weight * F.softplus(-torch.sign(gap) * (score[:, :, None] - score[:, None, :]))).sum() / weight.sum().clamp_min(1e-6)
    shape_logits = outputs[..., 1:-2]
    shape = masked_mean(F.cross_entropy(shape_logits.transpose(1, 2), t[..., 1].long(), reduction="none"), valid[..., 1])

    def robust(column, target_column):
        loss = F.smooth_l1_loss(outputs[..., column], t[..., target_column] / WORD_SEMITONES, beta=1 / WORD_SEMITONES, reduction="none")
        return masked_mean(loss, valid[..., target_column])
    with torch.no_grad():
        def main(x):    # [B, W]: no other word of the phrase scores higher
            return ~((x[:, None, :] > x[:, :, None]) & pairs).any(-1)
        phrase_main = pairs.any(-1) & main(target)     # the recording's main words of phrases with 2+ words
        metrics = dict(word_main_agreement=masked_mean(main(score).float(), phrase_main),
                       word_shape_accuracy=masked_mean((shape_logits.argmax(-1) == t[..., 1].long()).float(), valid[..., 1]),
                       word_excursion_error=masked_mean((outputs[..., -2] * WORD_SEMITONES - t[..., 2]).abs(), valid[..., 2]))
    return dict(word_rank=rank, word_shape=shape, word_excursion=robust(-2, 2), word_final=robust(-1, 3)), metrics


@torch.no_grad()
def duration_metrics(predicted, presence, durations, pausable, lengths):
    """Free-running duration quality: predicted [B, N] frames, presence logits [B, N]."""
    predicted_frames = predicted.sum(1)
    return dict(pause_accuracy=masked_mean(((presence > 0) == (durations > 0)).float(), pausable),
                pause_length_ratio=(predicted * pausable).sum() / (durations * pausable).sum().clamp_min(1),
                duration_length_ratio=(predicted_frames / lengths).mean(),
                duration_length_error=((predicted_frames - lengths).abs() / lengths).mean())


class TrainingSystem(nn.Module):
    WEIGHTS = {"duration": 1.0, "pause": 1.0, "pitch": 2.0, "pitch_spread": 1.0, "pitch_delta": 1.0, "voicing": 1.0, "energy": 1.0, "mel_head": 2.0,
               "stft": 2.0, "mel": 10.0}

    def __init__(self, config):
        super().__init__()
        self.model = TinyVoca(config)
        self.mel_head = nn.Linear(config.channels, config.n_mels)   # auxiliary, not exported
        self.spectral = SpectralLoss(config.sample_rate)

    def objective(self, batch, teacher, weights=WEIGHTS, random_phase=True):
        """Loss on full utterances. Returns ``(loss, terms, auxiliary)``.

        ``teacher[0]`` is the per-utterance chance that the acoustic
        decoder is conditioned on recorded pitch/voicing/energy; otherwise it
        gets the model's own (detached) predictions, exactly as at inference.
        The oscillator always follows the recorded pitch here, because the
        waveform target does.
        """
        m, c = self.model, self.model.config
        hidden, token_mask = m.encode(batch["tokens"], batch["lexical"], batch["token_lengths"], batch["speakers"])
        output = m.duration(hidden, token_mask).float()
        # Length is learned only where a token has frames; whether a pause-capable
        # token has any is a separate yes/no target.
        present = batch["durations"] > 0
        duration = masked_mean((output[..., 0] - torch.log1p(batch["durations"].float())).square(), token_mask & present)
        pausable = token_mask & ~m.phone_tokens[batch["tokens"]]
        pause = masked_mean(F.binary_cross_entropy_with_logits(output[..., 1], present.float(), reduction="none"), pausable)
        frames, frame_mask, lengths = m.expand(hidden, batch["durations"], token_mask, batch["speakers"], batch["mel"].shape[1])
        if not torch.equal(lengths, batch["frame_lengths"]):
            raise ValueError("durations do not sum to the acoustic frame count")
        prosody_hidden, prediction = m.predict_prosody(frames, frame_mask)
        prediction, target = prediction.float(), batch["prosody"]
        index, _, _ = m.align(batch["durations"], token_mask, frame_mask.shape[1])
        phrase = m.phrase_breaks[batch["tokens"]].long().cumsum(1).gather(1, index)
        prosody, prosody_metrics = prosody_terms(prediction, target, batch["pitch_trusted"], frame_mask, phrase)
        teacher = torch.rand(len(lengths), device=lengths.device) < teacher[0]
        condition = torch.where(teacher[:, None, None], target, m.prosody_condition(prediction.detach()))
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

    FEATURE_MATCHING = 30.0    # the default of --feature-matching

    def critics(self):
        return Discriminators()

    def adversarial(self, loss, auxiliary, batch, critics, critic_optimizer, step, recipe, ramp):
        """One critic update, then backward of ``loss`` plus the balanced GAN term."""
        wave = auxiliary["wave"]
        fake, real, index = crops(wave, batch["wave"], batch["frame_lengths"] * self.model.config.hop, recipe["crop_samples"], recipe["crops"])
        return adversarial_backward(loss, auxiliary["reconstruction"], wave, fake, real, index, critics, critic_optimizer, ramp, recipe)


class StyleTrainingSystem(nn.Module):
    """StyleTTS 2's second-stage objective, minus its aligner, pitch network,
    diffusion and PL-BERT: style encoders read each clip's mel; the prosody
    predictor is supervised by our durations, pitch, voicing and energy, and its
    word head by word-level targets (``word_terms``) on whole sentences; the
    decoder and the F0/energy branches run on equal-length random windows
    (``config.crop_frames``) as in StyleTTS 2. Weights are StyleTTS 2's
    (lambda_mel 5, lambda_ce 20, lambda_dur 1, lambda_gen 1, lambda_slm 1) and ours
    for the prosody targets. The style encoders, critics and speech model never ship.

    The prosody predictor's style is a delivery (``voca.styletts.Delivery``) drawn from the posterior ``heard``,
    which reads the recording's timing, pitch and energy; ``delivery`` is the weight of its divergence, in nats
    per sentence, from the prior that the text alone gives.

    Fixed styles (``schedule``'s fourth value, ramped from 0 to 1): the
    per-utterance chance that the decoder gets the stored voice style (what inference uses: one running mean per
    voice, detached) instead of the style read from this recording.

    The adversarial terms are at StyleTTS 2's fixed weights (``gan``, with
    ``feature_matching`` on the waveform critics' feature matching, and
    ``slm_gan``), faded in over ``gan_ramp`` steps from the GAN start."""

    WEIGHTS = {"duration": 1.0, "duration_bins": 20.0, "pause": 1.0, "pitch": 2.0, "pitch_spread": 1.0, "pitch_delta": 1.0, "voicing": 1.0, "energy": 1.0,
               "word_rank": 1.0, "word_shape": 1.0, "word_excursion": 1.0, "word_final": 1.0,
               "mel_convergence": 5.0, "slm": 1.0, "gan": 1.0, "slm_gan": 1.0, "delivery": 0.05,
               "shift": 0.0}    # shift_consistency; opt in with --weights '{"shift": 1}'
    CROP_ENDS, CROP_STARTS = 0.25, 0.15    # shares of training windows anchored to the utterance's end and start
    SHIFT_ITEMS = 2                  # utterances of each batch the shift-consistency loss renders a second time
    SHIFT_BAND = (1000.0, 4500.0)    # Hz: where the generator's clock images were measured
    FEATURE_MATCHING = 2.0     # StyleTTS 2's
    SLM_EVERY = 10   # the speech-model critic head trains every this many steps

    def __init__(self, config):
        super().__init__()
        self.model = StyleVoca(config)
        self.style_encoder = channels_last(StyleEncoder(config))   # acoustic
        self.heard = Reading(config.style_prosody_channels + 5, len(config.speakers))
        self.mel_convergence = MelConvergenceLoss(config.sample_rate)
        self.spectral = SpectralLoss(config.sample_rate)   # metrics comparable with TinyVoca runs
        self.register_buffer("shift_window", torch.hann_window(512), persistent=False)

    def shift_consistency(self, generator, wave, features, style, source, voiced):
        """The generator's sensitivity to its own output grid, as a loss (Codex, round 5). The first
        ``SHIFT_ITEMS`` utterances are rendered again from the same excitation and input, delayed by
        a random fraction of the generator's hop, and the delay undone (``Generator.render``). A
        voice renders the same either way; images of it around the synthesis frame rate (3 kHz at
        hop 16, filling the gaps between harmonics in 1.8-4.4 kHz) move with the grid. Their
        difference's energy in ``SHIFT_BAND`` over voiced frames, relative to the band's energy."""
        k = min(self.SHIFT_ITEMS, len(wave))
        delay = torch.randint(1, generator.hop, (1,), device=wave.device)
        shifted = generator.render(features[:k], style[:k], source[:k], delay)
        rate = self.model.config.sample_rate
        low, high = (math.ceil(f * 512 / rate) for f in self.SHIFT_BAND)
        with torch.autocast(wave.device.type, enabled=False):
            a, b = (torch.stft(w.float(), 512, 128, window=self.shift_window, return_complex=True)[:, low:high] for w in (wave[:k], shifted))
        # STFT frames (every 128 samples) over voiced acoustic frames (256), the crop's edge frames left out.
        frame = (torch.arange(a.shape[-1], device=wave.device) // 2).clamp(max=voiced.shape[1] - 1)
        weight = voiced[:k, frame] * F.pad(torch.ones(a.shape[-1] - 4, device=wave.device), (2, 2))
        return ((a - b).abs().square() * weight[:, None]).sum() / ((a.abs().square() * weight[:, None]).sum() + 1e-6)

    def critics(self):
        return channels_last(StyleCritics(self.model.config.sample_rate))


    def objective(self, batch, scheduled, weights=WEIGHTS, random_phase=True):
        """``scheduled`` (``schedule``): the per-utterance chances that the decoder gets recorded
        prosody (``[0]``), that the predictor gets recorded word prosody (``[1]``) and that the
        decoder gets the stored voice style (``[3]``); and how much of the delivery's weight applies (``[4]``)."""
        m, c = self.model, self.model.config
        weights = {**weights, "delivery": weights["delivery"] * scheduled[4]}
        tokens, token_lengths, speakers, durations = batch["tokens"], batch["token_lengths"], batch["speakers"], batch["durations"]
        hidden, token_mask = m.encode(tokens, batch["lexical"], token_lengths, speakers)
        # The dataset checked that durations sum to the frame counts. Shapes come
        # from the batch's padded shape and the host-side ``shortest``, so nothing
        # here waits for the GPU and fixed-shape batches give fixed shapes throughout.
        index, frame_mask, lengths = m.align(durations, token_mask, frames=batch["mel"].shape[1])
        # One window per utterance, the same length across the batch; centred when validating. While training,
        # CROP_ENDS of them end with the utterance and CROP_STARTS start with it: uniform windows covered the
        # utterance's end in 2.8% of crops, and the decoder, generator and critics hardly saw final falls and releases.
        size = min(c.crop_frames, batch["host"]["shortest"])
        if self.training:
            anchor = torch.rand(len(lengths), device=lengths.device)
            offset = torch.where(anchor < self.CROP_ENDS, 1.0, torch.where(anchor < self.CROP_ENDS + self.CROP_STARTS, 0.0,
                                                                          torch.rand(len(lengths), device=lengths.device)))
        else:
            offset = torch.full_like(lengths, 0.5, dtype=torch.float)
        window = (offset * (lengths - size + 1)).long().clamp(max=lengths - size)[:, None] + torch.arange(size, device=lengths.device)
        mel = batch["mel"].gather(1, window[..., None].expand(-1, -1, c.n_mels))
        if self.training:
            mel = ((mel - c.mel_mean) / c.mel_scale).transpose(1, 2)[:, None]
            acoustic_style = self.style_encoder(mel)
            shipped = m.style[speakers]    # a copy, from before this batch updates the means
            m.observe_style(acoustic_style.detach(), speakers)
            fixed = torch.rand(len(lengths), device=lengths.device) < scheduled[3]
            acoustic_style = torch.where(fixed[:, None], shipped.to(acoustic_style.dtype), acoustic_style)
        else:
            acoustic_style = m.style[speakers]
        # The delivery: while training, one the recording may have; when validating, the text's most typical one.
        text = m.prosody_text(hidden, tokens, batch["lexical"], token_mask)
        prior = m.delivery.prior(text, speakers, token_mask)
        heard = torch.cat((text.detach(), recorded_delivery(durations, batch["prosody"], index, frame_mask)), dim=-1)
        posterior = self.heard(heard, speakers, token_mask)
        delivery = reading_divergence(posterior, prior).mean()
        prosodic_style = m.delivery.style(m.delivery.draw(posterior) if self.training else prior[0], speakers)
        # The word head and its losses see whole sentences; its features condition what follows on the
        # recording's word prosody only briefly (its own schedule), so that durations and F0/N learn
        # from the word features they will get at inference for most of training.
        word_teacher = torch.rand(len(lengths), device=lengths.device) < scheduled[1]
        words = m.words(tokens, token_mask)
        states, logits, word_outputs = m.predictor(text, prosodic_style, token_lengths, token_mask, words, (batch["word_prosody"], word_teacher))
        word, word_metrics = word_terms(word_outputs, batch["word_prosody"], words)
        logits = logits.float()
        # Durations as StyleTTS 2 learns them (bins "longer than k" + L1 on their sum),
        # on tokens that have frames; whether a pause-capable token has any is the presence logit.
        present = token_mask & (durations > 0)
        bins = (torch.arange(c.style_max_duration, device=durations.device) < durations[..., None]).float()
        duration_bins = masked_mean(F.binary_cross_entropy_with_logits(logits[..., :-1], bins, reduction="none").mean(-1), present)
        duration = masked_mean((torch.sigmoid(logits[..., :-1]).sum(-1) - durations.clamp(max=c.style_max_duration)).abs(), present)
        pausable = token_mask & ~m.phone_tokens[tokens]
        pause = masked_mean(F.binary_cross_entropy_with_logits(logits[..., -1], (durations > 0).float(), reduction="none"), pausable)
        prediction = m.melody(frames_of(states, index), durations, index, prosodic_style, lengths)
        phrase = m.phrase_breaks[tokens].long().cumsum(1).gather(1, index)
        prosody, prosody_metrics = prosody_terms(prediction, batch["prosody"], batch["pitch_trusted"], frame_mask.float(), phrase)
        # What is spoken with: vowels kept voiced, as the app's synthesis does (``StyleVoca.synthesize``).
        spoken = m.voice_vowels(prediction, tokens, index, frame_mask)
        target, prediction, spoken = frames_of(batch["prosody"], window), frames_of(prediction, window), frames_of(spoken, window)
        teacher = torch.rand(len(lengths), device=lengths.device) < scheduled[0]
        condition = torch.where(teacher[:, None, None], target, m.prosody_condition(spoken.detach()))
        # The source follows what it will speak with (the predicted pitch and voicing, without a gradient: the gradient
        # of a phase grows with every frame it was integrated over), as StyleTTS 2's and Kokoro's do.
        spoken_text = m.decoder_text(hidden, tokens, durations, index.gather(1, window), window)
        generator, features = m.decoder.generator, m.decoder.features(spoken_text, condition, acoustic_style)
        source = generator.excitation(m.frequency(condition[..., 0].detach()), condition[..., 1].detach(), features)
        wave = generator.render(features, acoustic_style, source)
        samples = (window[:, :1] * c.hop + torch.arange(size * c.hop, device=lengths.device))
        real = batch["wave"].gather(1, samples)
        terms = dict(duration=duration, duration_bins=duration_bins, pause=pause, delivery=delivery, **prosody, **word,
                     mel_convergence=self.mel_convergence(wave, real))
        if self.training:
            terms["slm"], slm = slm_feature_loss(wave, real, c.sample_rate)
            if weights["shift"]:
                terms["shift"] = self.shift_consistency(generator, wave, features, acoustic_style, source, target[..., 1])
        loss = sum(weights[name] * value for name, value in terms.items())
        with torch.no_grad():
            stft, mel = self.spectral(wave, real, torch.full_like(lengths, size * c.hop))
            predicted = m.predict_durations(logits, tokens, token_mask)
        terms.update(base_loss=loss.detach(), teacher_fraction=teacher.float().mean(), word_teacher_fraction=word_teacher.float().mean(),
                     stft=stft, mel=mel, **prosody_metrics, **word_metrics,
                     **duration_metrics(predicted, logits[..., -1], durations, pausable, lengths))
        if self.training:
            terms["fixed_style_fraction"] = fixed.float().mean()
        return loss, terms, {"wave": wave, "real": real, "slm": slm if self.training else None,
                             "reconstruction": weights["mel_convergence"] * terms["mel_convergence"]}

    def adversarial(self, loss, auxiliary, batch, critics, critic_optimizer, step, recipe, ramp):
        """StyleTTS 2's GAN step at fixed weights, ramped. The waveform critics judge a random ``critic_share`` of the
        utterances, at full length, once for both sides (``StyleCritics.joint_backward``); the speech-model head
        judges all of them, and trains every 10th step. The critics update after both gradients are taken."""
        fake, real, slm = auxiliary["wave"], auxiliary["real"], auxiliary["slm"]
        scale = torch.as_tensor(ramp * recipe["weights"]["gan"], device=fake.device)
        count = math.ceil(recipe["critic_share"] * len(fake))
        rows = torch.rand(len(fake), device=fake.device).topk(count).indices
        critic_optimizer.zero_grad(set_to_none=True)
        judged = fake[rows].detach().requires_grad_()
        d_loss, adversarial, matching, gradient = critics.joint_backward(real[rows], judged, scale, recipe["feature_matching"])
        if step % self.SLM_EVERY == 0:
            d_loss = d_loss + critics.head_backward(slm)
        critics.requires_grad_(False)
        try:
            slm_gan = (1 - critics.wd(slm["fake"])).square().mean()
            loss = loss + ramp * recipe["weights"]["slm_gan"] * slm_gan
            torch.autograd.backward([loss, fake], [None, torch.zeros_like(fake).index_copy(0, rows, gradient)])
        finally:
            critics.requires_grad_(True)
        critic_grad = nn.utils.clip_grad_norm_(critics.parameters(), 10)
        critic_optimizer.step()
        loss = loss.detach() + scale * (adversarial + recipe["feature_matching"] * matching)
        return loss, dict(discriminator=d_loss, adversarial=adversarial, feature_matching=matching, slm_adversarial=slm_gan,
                          critic_grad_norm=critic_grad, gan_scale=scale)


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


def adversarial_backward(loss, reconstruction, wave, fake, real, index, critics, critic_optimizer, ramp, recipe):
    """TinyVoca's critics (``Discriminators``): one update on the windows
    ``fake``/``real`` (taken from ``wave`` at ``index``), then backward of
    ``loss`` plus the balanced generator GAN term. Returns the total loss and the GAN metrics."""
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
        loss, balance = balanced_backward(loss, reconstruction, adversarial + recipe["feature_matching"] * matching, wave, index, ramp, recipe)
    finally:
        critics.requires_grad_(True)
    return loss, dict(discriminator=d_loss, adversarial=adversarial, feature_matching=matching, critic_grad_norm=critic_grad, **balance)


def balanced_backward(loss, reconstruction, gan, wave, index, ramp, recipe):
    """Backward of ``loss`` plus the generator's GAN term ``gan``, scaled so that
    its gradient at the waveform ``wave`` (inside the critics' windows ``index``,
    or all of it) is ``ramp * gan_ratio`` times the ``reconstruction`` term's,
    at most ``gan_scale_max``. Fixed weights cannot hold that balance: measured
    at the waveform, a fixed adversarial gradient was under 1% of the spectral
    one for TinyVoca, i.e. switched off. Everything stays on the GPU, so the
    step can be captured; ``ramp`` is a tensor in graphs.

    The GAN gradient at the waveform is computed once and sent down the
    generator with the rest of the loss's in one backward pass. Returns the total
    loss and the metrics: the scale, and the unscaled gradient ratio."""
    def at(term):
        gradient = torch.autograd.grad(term, wave, retain_graph=True)[0]
        return gradient, (gradient if index is None else gradient.gather(1, index)).norm()
    _, reconstruction_norm = at(reconstruction)
    gan_gradient, gan_norm = at(gan)
    ratio = gan_norm / reconstruction_norm.clamp_min(1e-12)
    scale = (recipe["gan_ratio"] * ramp / ratio.clamp_min(1e-12)).clamp(max=recipe["gan_scale_max"]).detach()
    torch.autograd.backward([loss, wave], [None, scale * gan_gradient])
    return loss.detach() + scale * gan.detach(), dict(gan_scale=scale, gan_gradient_ratio=ratio.detach())


def schedule(optimizer, critic_optimizer, step, recipe):
    """Set this step's learning rates; returns its teacher-forcing probabilities
    (recorded prosody for the decoder, recorded word prosody for the predictor),
    how far the adversarial terms have faded in (0 before the GAN starts) and the
    chance of the stored voice styles (styletts: 0 for ``fixed_style_hold`` steps, then up to 1),
    and how far the delivery's divergence has faded in (styletts: 0 for ``delivery_hold`` steps, so that
    the predictor first learns what a delivery is good for, then up to 1)."""
    start = (recipe.get("branched_from") or {}).get("step", 0)

    def set_rate(group, value):
        if torch.is_tensor(group["lr"]):
            group["lr"].fill_(value)   # captured optimizer steps read the rate from this tensor
        else:
            group["lr"] = value
    for group in optimizer.param_groups:
        set_rate(group, group.get("scale", 1.0) * learning_rate(step, recipe["learning_rate"], recipe["warmup"], recipe["lr_half_life"], start))
    for group in critic_optimizer.param_groups:
        set_rate(group, learning_rate(step, recipe["learning_rate"], 1, recipe["lr_half_life"], start))
    return (teacher_probability(step, recipe["teacher_hold"], recipe["teacher_ramp"], recipe["teacher_final"]),
            teacher_probability(step, recipe["word_teacher_hold"], recipe["word_teacher_ramp"], 0.0),
            min(1.0, max(0.0, (step - recipe["gan_start"] + 1) / recipe["gan_ramp"])),
            1.0 - teacher_probability(step, recipe["fixed_style_hold"], recipe["fixed_style_ramp"], 0.0),
            1.0 - teacher_probability(step, recipe["delivery_hold"], recipe["delivery_ramp"], 0.0))


def optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe):
    """One eager training step on a (pinned) CPU batch."""
    scheduled = schedule(optimizer, critic_optimizer, step, recipe)
    return train_step(system, critics, optimizer, critic_optimizer, to_device(batch, next(system.parameters()).device), step, recipe, scheduled)


def train_step(system, critics, optimizer, critic_optimizer, batch, step, recipe, scheduled):
    """Losses, backward and both optimizer updates, with the learning rates
    already set (``schedule``, whose other values are ``scheduled``): everything
    here can be captured in a CUDA graph."""
    weights = recipe["weights"]
    optimizer.zero_grad(set_to_none=True)
    # bf16: matrix products and convolutions in bfloat16 (the weights, gradients
    # and optimizer stay float32); spectra, norms and losses are computed in float32.
    with torch.autocast("cuda", torch.bfloat16, enabled=recipe.get("bf16", False)):
        loss, metrics, auxiliary = system.objective(batch, scheduled, weights)
        if step >= recipe["gan_start"]:
            loss, gan_metrics = system.adversarial(loss, auxiliary, batch, critics, critic_optimizer, step, recipe, scheduled[2])
            metrics.update(gan_metrics)
        else:
            loss.backward()
    # Gradient norms are returned, not checked: checking would make every step wait
    # for the GPU. The training loop stops at the next log step on a non-finite one.
    grad = nn.utils.clip_grad_norm_(system.parameters(), recipe["grad_clip"])
    optimizer.step()
    return {**{k: torch.as_tensor(v).detach() for k, v in metrics.items()}, "loss": loss.detach(), "grad_norm": grad}


class ChannelsLast(nn.Module):
    def forward(self, weight):
        return weight.contiguous(memory_format=torch.channels_last)


def channels_last(module):
    """Run the 2-D convolutions of ``module`` in NHWC, the layout cuDNN's fast
    kernels on recent GPUs use: in NCHW every call converts its input, weight
    and output (about 14% of a GAN step's GPU time). Same arithmetic; the
    parameters and state-dict keys are unchanged (the conversion follows the
    weight's normalization as one more parametrization)."""
    from torch.nn.utils import parametrize
    for conv in module.modules():
        if isinstance(conv, nn.Conv2d):
            parametrize.register_parametrization(conv, "weight", ChannelsLast(), unsafe=True)
            conv.register_forward_pre_hook(lambda _, inputs: (inputs[0].contiguous(memory_format=torch.channels_last), *inputs[1:]))
    return module


class GraphedSteps:
    """``optimize_step`` replayed from CUDA graphs, one per batch shape and phase
    (before the GAN, GAN, GAN with the speech-model head's update). Replaying skips the
    ~14k kernel launches per step that otherwise keep the GPU waiting on Python.

    The first step of each shape and phase runs eagerly, which creates what a
    capture must not (optimizer state, the speech model, cuFFT plans), and its graph is
    captured right after. Eager steps allocate from the graphs' memory pool, so
    they reuse the graphs' scratch memory instead of needing their own.

    A graph launches the cuFFT plans it was captured with, which live in
    PyTorch's plan cache, an LRU of ``max_size`` plans (4096). Replays never look
    a plan up, so the graphs' plans are the least recently used, and everything
    else that runs FFTs on this GPU (the deployed evaluation synthesizes hundreds
    of sentences of new lengths at every save) pushes them out: their workspaces
    are freed while the graphs still use them, and the next replay crashes
    ("misaligned address", "illegal memory access"). Runs also segfaulted on the
    first replay after an evaluation with the cache never full. So no graph
    outlives an evaluation: the training loop calls ``release`` after every
    one, which drops the graphs and the plans (recaptured on the next steps,
    about a minute). Between evaluations only captures add plans; should anything
    else fill the cache, the graphs are released before they could replay an
    evicted plan (the cache only shrinks when cleared, so a cache that is not
    full has never evicted one).
    """

    def __init__(self, system, critics, optimizer, critic_optimizer, recipe):
        if not isinstance(system, StyleTrainingSystem):
            raise ValueError("CUDA graphs need the styletts training step (no host waits, fixed shapes)")
        self.system, self.critics, self.optimizer, self.critic_optimizer, self.recipe = system, critics, optimizer, critic_optimizer, recipe
        self.device = next(system.parameters()).device
        self.stream = torch.cuda.Stream(self.device)
        self.scheduled = torch.zeros(5, device=self.device)   # ``schedule``'s values, rewritten before every replay
        self.pool, self.graphs = torch.cuda.MemPool(), {}
        self.plans = torch.backends.cuda.cufft_plan_cache[self.device.index]
        self.recaptures = 0

    def release(self):
        """Drop every captured graph and the cuFFT plans: the next steps capture them again, into
        the same pool. The gradients the captured backward left in ``.grad`` live in graph memory
        and are dropped first: recapturing while they were still referenced crashed."""
        torch.cuda.synchronize(self.device)
        for parameter in (*self.system.parameters(), *self.critics.parameters()):
            parameter.grad = None
        self.graphs.clear()
        self.plans.clear()
        self.recaptures += 1

    def __call__(self, batch, step):
        if self.plans.size >= self.plans.max_size:    # a captured graph's plan may be gone: see the class docstring
            self.release()
        self.scheduled.copy_(torch.tensor(schedule(self.optimizer, self.critic_optimizer, step, self.recipe)))
        gan = step >= self.recipe["gan_start"]
        tensors = {key: value for key, value in batch.items() if key != "host"}
        key = (gan, gan and step % self.system.SLM_EVERY == 0, batch["host"]["shortest"],
               *((name, tuple(value.shape)) for name, value in sorted(tensors.items())))
        if key in self.graphs:
            graph, static, outputs = self.graphs[key]
            for name, value in tensors.items():
                static[name].copy_(value, non_blocking=True)
            graph.replay()
            return outputs   # overwritten by the next replay of this graph
        static = to_device(batch, self.device)
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        # Backward on this thread: autograd's worker threads would allocate outside the pool.
        with torch.cuda.stream(self.stream), torch.cuda.use_mem_pool(self.pool), torch.autograd.set_multithreading_enabled(False):
            metrics = self.step(static, step)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool.id, stream=self.stream, capture_error_mode="thread_local"):
            outputs = self.step(static, step)   # recorded, not run
        current.wait_stream(self.stream)
        self.graphs[key] = graph, static, outputs
        return metrics

    def step(self, batch, step):
        return train_step(self.system, self.critics, self.optimizer, self.critic_optimizer, batch, step, self.recipe, self.scheduled)


def adamw(groups, device, **settings):
    """AdamW whose steps can be captured in CUDA graphs on a GPU: fused, with the
    step count and each group's learning rate in device tensors."""
    if not str(device).startswith("cuda"):
        return torch.optim.AdamW(groups, **settings)
    groups = [{**group, "lr": torch.tensor(float(settings["lr"]), device=device)} for group in groups]
    return torch.optim.AdamW(groups, **settings, fused=True, capturable=True)


def load_optimizer(optimizer, state):
    """A checkpoint's moments; the hyperparameters (learning-rate tensors, scales,
    fused/capturable) stay this run's, whatever the checkpoint was trained with."""
    settings = [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups]
    optimizer.load_state_dict(state)
    for group, setting in zip(optimizer.param_groups, settings):
        group.update(setting)
        for parameter in group["params"]:
            moments = optimizer.state.get(parameter, {})
            if "step" in moments and (group.get("fused") or group.get("capturable")):
                moments["step"] = moments["step"].to(parameter.device, torch.float32)


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


def balanced_indices(dataset, limit, per_voice=False):
    """Stable round-robin over voices, independent of manifest order: ``limit``
    in all, or with ``per_voice`` up to ``limit`` of each, taken round-robin over
    the kinds of sentence (script role, e.g. quoted, and ``sentence_type``) so
    every kind is in it when the voice has more sentences than ``limit``."""
    groups = defaultdict(list)
    for i, speaker in enumerate(dataset.speakers):
        groups[speaker].append(i)
    if per_voice:
        selected = []
        for voice in sorted(groups):
            kinds = defaultdict(list)
            for i in groups[voice]:
                row = dataset.rows[i]
                kinds[(row.get("role"), sentence_type(row["text"]))].append(i)
            queues = [kinds[k] for k in sorted(kinds, key=str)]
            mine = [q[n] for n in range(max(map(len, queues))) for q in queues if n < len(q)]
            selected += mine[:limit]
        return selected
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
        raise ValueError("train/validation leakage; split by sentence/text_id across all voices. Examples: " + json.dumps(collisions, ensure_ascii=False))


@torch.no_grad()
def validate(system, dataset, indices, device, weights, batch_size=8):
    """Held-out metrics on a fixed balanced subset, with the raw (not the shipped
    EMA) weights, the stored voice style and each text's most typical delivery.

    ``teacher_forced`` conditions the decoder on recorded prosody, ``predicted``
    on the model's own; both use recorded durations and pitch for the oscillator
    so the spectra are comparable with the recording, which also means neither
    is what a learner hears (``voca.evaluate`` scores that). Durations and
    pauses are the free-running predictors'; pitch and voicing are predicted at
    the recorded durations, over whole sentences. This is a
    training monitor; pronunciation is judged separately with ASR and by listening.
    """
    system.eval()
    order = sorted(indices, key=lambda i: dataset.lengths[i])
    totals, weight = defaultdict(float), 0
    for start in range(0, len(order), batch_size):
        batch = to_device(collate([dataset[i] for i in order[start:start + batch_size]]), device)
        size = len(batch["frame_lengths"])
        for name, probability in (("teacher_forced", 1.0), ("predicted", 0.0)):
            _, metrics, _ = system.objective(batch, (probability, probability, 0.0, 1.0, 1.0), weights, random_phase=False)
            totals[f"validation_{name}_mel"] += float(metrics["mel"]) * size
            totals[f"validation_{name}_stft"] += float(metrics["stft"]) * size
        for key in ("duration", "pause_accuracy", "pause_length_ratio", "pitch_error_semitones", "pitch_spread_ratio", "voicing_accuracy",
                    "duration_length_ratio", "duration_length_error", "word_rank", "word_main_agreement", "word_shape_accuracy",
                    "word_excursion_error", "word_final"):
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
    parser.add_argument("--gan-ratio", type=float, default=0.7, help="tinyvoca: adversarial gradient as a fraction of the "
                        "reconstruction gradient at the waveform (styletts: fixed weights, --weights gan / slm_gan)")
    parser.add_argument("--gan-ramp", type=int, default=3000, help="Steps over which the adversarial terms fade in while the critics warm up")
    parser.add_argument("--feature-matching", type=float, help="Feature-matching weight inside the adversarial term "
                        "(default: the architecture's, TrainingSystem.FEATURE_MATCHING / StyleTrainingSystem.FEATURE_MATCHING)")
    parser.add_argument("--gan-scale-max", type=float, default=2000.0, help="tinyvoca: the balanced GAN term's largest scale")
    parser.add_argument("--crop-samples", type=int, default=8192)
    parser.add_argument("--crops", type=int, default=2, help="Discriminator windows per utterance")
    parser.add_argument("--teacher-hold", type=int, default=3000, help="Steps with recorded prosody conditioning only")
    parser.add_argument("--teacher-ramp", type=int, default=7000)
    parser.add_argument("--teacher-final", type=float, default=0.0, help="Share of utterances still conditioned on recorded prosody "
                        "after the ramp (0 = the decoder sees only what it gets at inference)")
    parser.add_argument("--word-teacher-hold", type=int, default=500, help="Steps with recorded word prosody feeding durations and F0/N")
    parser.add_argument("--word-teacher-ramp", type=int, default=2000, help="Steps over which that falls to the word head's own predictions")
    parser.add_argument("--fixed-style-hold", type=int, default=2000, help="styletts: steps where the prosody predictor and the decoder "
                        "get each recording's own styles; then the share given the stored voice styles (as at inference) rises to 1")
    parser.add_argument("--delivery-hold", type=int, default=2000, help="styletts: steps before the delivery's divergence from the text's prior counts")
    parser.add_argument("--delivery-ramp", type=int, default=6000, help="Steps of its rise to full weight")
    parser.add_argument("--critic-share", type=float, default=0.5, help="styletts: the share of each batch's utterances the waveform critics judge")
    parser.add_argument("--fixed-style-ramp", type=int, help="Steps of that rise (default: to 70%% of --steps)")
    parser.add_argument("--subset", type=int, help="Train on the first N utterances (balanced by voice): memorisation check")
    parser.add_argument("--sampling-alpha", type=float, default=0.5, help="Each voice is drawn with probability proportional "
                        "to its hours to this power: 1 uniform by audio, 0 every voice equally (voca.data.TemperedSampler)")
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--deployed-size", type=int, default=64, help="Held-out sentences of each voice the shipped weights synthesize "
                        "at every save for the word-level prosody scores (voca.evaluate; 0: none)")
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
    parser.add_argument("--cuda-graphs", action="store_true", help="Replay the training step from CUDA graphs (styletts on a GPU)")
    parser.add_argument("--bf16", action="store_true", help="Mixed precision: bfloat16 matrix products and convolutions")
    args = parser.parse_args()
    if min(args.steps, args.batch_frames, args.max_batch, args.save_every, args.keep_every, args.log_every, args.crops, args.gan_ramp) < 1:
        parser.error("steps, batch limits and intervals must be positive")
    if args.max_seconds <= 0 or args.learning_rate <= 0 or args.lr_half_life <= 0 or not 0 <= args.teacher_final <= 1:
        parser.error("time cap, learning rate and half life must be positive; teacher_final in [0, 1]")
    if args.resume and args.init_from:
        parser.error("--init-from starts a new run; --resume continues one")
    if args.fixed_style_ramp is None:
        args.fixed_style_ramp = max(1, int(0.7 * args.steps) - args.fixed_style_hold)

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
    # The voices are the training data's (sorted: Achernar first, the default speaker).
    c = dataclasses.replace(c, speakers=tuple(sorted({json.loads(line)["speaker"] for line in args.train.open(encoding="utf-8")})))
    if args.phoneme_encoder:
        pretrained = torch.load(args.phoneme_encoder, map_location="cpu", weights_only=True)
        e = pretrained["config"]
        c = dataclasses.replace(c, phoneme_encoder_layers=pretrained["layers"], phoneme_encoder_distinct=pretrained.get("distinct", 0),
                                phoneme_encoder_width=e["channels"],
                                phoneme_encoder_heads=e["heads"], phoneme_encoder_feedforward=e["feedforward"])
    weights = TRAINING_SYSTEMS[c.architecture].WEIGHTS
    if set(args.weights) - set(weights):
        parser.error(f"--weights may only override {sorted(weights)}")
    if args.feature_matching is None:
        args.feature_matching = TRAINING_SYSTEMS[c.architecture].FEATURE_MATCHING
    if min(args.gan_ratio, args.feature_matching, args.gan_scale_max, args.sampling_alpha) < 0:
        parser.error("GAN settings and the sampling alpha must be non-negative")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train = AnnotatedDataset(args.train, c, voiced_vowels=not args.recorded_voicing)
    validation = AnnotatedDataset(args.validation, c, voiced_vowels=not args.recorded_voicing, reference=train.reference)
    check_leakage(train, validation)
    if train.annotation != validation.annotation:
        raise ValueError("train and validation manifests come from different annotation runs")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("output directory is not empty; use --resume or a new directory")
    pool = balanced_indices(train, args.subset) if args.subset else list(range(len(train)))
    validation_indices = balanced_indices(validation, args.validation_size)
    deployed_indices = balanced_indices(validation, args.deployed_size, per_voice=True) if args.deployed_size else []
    if not args.no_preload:
        train.preload(pool)
        validation.preload(sorted(set(validation_indices) | set(deployed_indices)))
    system = TRAINING_SYSTEMS[c.architecture](c).to(args.device)
    critics = system.critics().to(args.device)
    count = sum(p.numel() for p in system.model.parameters())
    if count > system.model.PARAMETER_BUDGET:
        raise ValueError(f"inference parameter budget exceeded: {count:,}")
    if args.phoneme_encoder:
        if pretrained.get("frontend") != frontend_fingerprint():
            raise ValueError(f"{args.phoneme_encoder} was pretrained on another front end's tokens; retrain it (voca.phonebert)")
        system.model.bert.load_state_dict(pretrained["encoder"])
    # StyleTTS 2 fine-tunes its pretrained PL-BERT at a tenth of the learning rate.
    encoder = getattr(system.model, "bert", None)
    bert = {id(p) for p in encoder.parameters()} if encoder else set()
    groups = [{"params": [p for p in system.parameters() if id(p) not in bert], "scale": 1.0}]
    if bert:
        groups.append({"params": list(encoder.parameters()), "scale": args.phoneme_encoder_lr_scale})
    optimizer = adamw(groups, args.device, lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
    critic_optimizer = adamw([{"params": list(critics.parameters())}], args.device, lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.0)
    sampler = BucketSampler([train.lengths[i] for i in pool], [train.token_counts[i] for i in pool], [train.speakers[i] for i in pool],
                            args.batch_frames, args.max_batch, args.seed, args.sampling_alpha)
    fingerprints = [train.fingerprint, validation.fingerprint]
    recipe = {key: getattr(args, key) for key in ("batch_frames", "max_batch", "learning_rate", "warmup", "lr_half_life", "grad_clip",
                                                   "gan_start", "gan_ratio", "gan_ramp", "feature_matching", "gan_scale_max",
                                                   "crop_samples", "crops", "teacher_hold", "teacher_ramp", "teacher_final",
                                                   "word_teacher_hold", "word_teacher_ramp", "sampling_alpha",
                                                   "subset", "validation_size", "seed", "ema_decay", "phoneme_encoder_lr_scale", "bf16",
                                                   "fixed_style_hold", "fixed_style_ramp", "delivery_hold", "delivery_ramp", "critic_share", "deployed_size")}
    recipe["voiced_vowel_targets"] = not args.recorded_voicing
    recipe["weights"] = {**weights, **{key: float(value) for key, value in args.weights.items()}}
    recipe["branched_from"] = None
    tracker = start_tracking(args.output.name, {"model": c.to_dict(), "recipe": recipe})
    step = 0

    def compatible(state, what, same_data=True):
        check_format(state, f"the checkpoint to be {what}")
        if same_data and (state["frontend"] != frontend_fingerprint() or state["annotation"] != train.annotation):
            raise ValueError("checkpoint was trained with a different front end or annotation run "
                             "(branch with --init-from and --new-data to fine-tune on new annotations)")
        if Config.from_dict(state["config"]).to_dict() != c.to_dict():   # fields added since take their defaults
            raise ValueError("checkpoint was trained with a different model configuration")
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        compatible(state, "resumed")
        recipe["branched_from"] = state["recipe"].get("branched_from")
        if state["datasets"] != fingerprints or state["recipe"] != recipe:
            raise ValueError("resume recipe or dataset mismatch (use --init-from to branch a new run with a changed recipe)")
        system.load_state_dict(state["system"])
        critics.load_state_dict(state["critics"])
        load_optimizer(optimizer, state["optimizer"])
        load_optimizer(critic_optimizer, state["critic_optimizer"])
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
            load_optimizer(critic_optimizer, state["critic_optimizer"])
        loaded["optimizer"] = not loaded["grown"] and not loaded["new"] and not loaded["dropped"]
        if loaded["optimizer"]:
            load_optimizer(optimizer, state["optimizer"])
        recipe["branched_from"] = {"step": step, "recipe": state["recipe"], "datasets": state["datasets"], "loaded": loaded}
        if args.new_data:
            recipe["branched_from"].update(frontend=state["frontend"], annotation=state["annotation"])
    if step >= args.steps:
        raise ValueError("checkpoint already reached requested total steps")
    # The weights that ship, synthesizing as the app does, scored per word (voca.evaluate).
    evaluation = DeployedEvaluation(validation, deployed_indices, c) if deployed_indices else None
    shipped = build_model(c).to(args.device).eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(c.to_dict(), indent=2) + "\n")
    (args.output / "recipe.json").write_text(json.dumps({**recipe, "annotation": train.annotation, "frontend": frontend_fingerprint(),
                                                         "train_utterances": len(pool), "datasets": fingerprints,
                                                         "deployed_reference": evaluation and evaluation.reference}, indent=2) + "\n")
    print(json.dumps({"inference_parameters": count, "training_only_parameters": sum(p.numel() for p in system.parameters()) - count
                      + sum(p.numel() for p in critics.parameters()), "train_utterances": len(pool), "start_step": step,
                      "branched": bool(args.init_from), "resumed": bool(args.resume),
                      "voice_shares": {voice: round(share, 4) for voice, share in sampler.items.shares.items()}}), flush=True)
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
        indices, shape = sampler.next()
        return sampler.state(), pin(collate([train[pool[i]] for i in indices], shape))
    batches = prefetch(draw)
    if args.cuda_graphs:
        graphed = GraphedSteps(system, critics, optimizer, critic_optimizer, recipe)
        run_step = lambda batch, step: graphed(batch, step)
    else:
        run_step = lambda batch, step: optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe)
    while step < args.steps:
        sampler_state, batch = next(batches)
        metrics = run_step(batch, step)
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
                          learning_rate=float(optimizer.param_groups[0]["lr"]))
            if args.cuda_graphs:
                record["graph_recaptures"] = graphed.recaptures
            if save:
                record.update(validate(system, validation, validation_indices, args.device, recipe["weights"]))
                if evaluation:
                    shipped.load_state_dict(ema)
                    record.update({f"deployed_{key}": value for key, value in evaluation(shipped, args.device).items()})
                state = snapshot()
                atomic_save(state, args.output / "last.pt")
                # inference.pt ships the EMA weights: the deployed_ metrics; the other validation metrics are of the raw ones.
                export_model(system.model, args.output / "inference.pt", step, train.annotation, ema)
                export_model(system.model, args.output / "inference-raw.pt", step, train.annotation)
                if retain:
                    atomic_save(state, args.output / f"step-{step:08d}.pt")
                if args.cuda_graphs:
                    graphed.release()     # no graph outlives an evaluation (GraphedSteps)
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

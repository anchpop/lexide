import functools
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from .model import add_lexical_embeddings, embed_tokens, position_in_token
from .styletts import frames_of


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def magnitude(wave, fft, hop, window):
    spectrum = torch.stft(wave.float(), fft, hop, window=window, center=True, pad_mode="constant", return_complex=True)
    # sqrt(re^2 + im^2 + eps): finite gradient at exact zeros (padding, silence).
    return torch.view_as_real(spectrum).square().sum(-1).add(1e-9).sqrt()


def mel_filterbank(sample_rate, fft, n_mels):
    """Slaney-scale triangular filters with area normalisation, [n_mels, bins]."""
    step, knee, slope = 200.0 / 3, 1000.0, math.log(6.4) / 27

    def to_hz(mel):
        return torch.where(mel < knee / step, mel * step, knee * torch.exp((mel - knee / step) * slope))
    top = knee / step + math.log(sample_rate / 2 / knee) / slope if sample_rate / 2 > knee else sample_rate / 2 / step
    points = to_hz(torch.linspace(0, top, n_mels + 2, dtype=torch.float64))
    bins = torch.linspace(0, sample_rate / 2, fft // 2 + 1, dtype=torch.float64)
    rising = (bins[None] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
    falling = (points[2:, None] - bins[None]) / (points[2:] - points[1:-1])[:, None]
    weights = torch.minimum(rising, falling).clamp_min(0) * (2.0 / (points[2:] - points[:-2]))[:, None]
    return weights.float()


class SpectralLoss(nn.Module):
    """Multi-resolution STFT and log-mel distances over whole utterances.

    Frames whose centre lies beyond an utterance's length are masked, so padded
    batches give the same value as the utterances would alone. No parameters.
    """

    def __init__(self, sample_rate, resolutions=((512, 128), (1024, 256), (2048, 512)), mel=(1024, 256, 80)):
        super().__init__()
        self.resolutions, self.mel = tuple(resolutions), tuple(mel)
        for fft in {fft for fft, _ in self.resolutions} | {self.mel[0]}:
            self.register_buffer(f"window_{fft}", torch.hann_window(fft), persistent=False)
        self.register_buffer("filters", mel_filterbank(sample_rate, self.mel[0], self.mel[2]), persistent=False)

    def log_mel(self, wave):
        fft, hop, _ = self.mel
        return torch.log((self.filters @ magnitude(wave, fft, hop, getattr(self, f"window_{fft}"))).clamp_min(1e-5))

    def forward(self, predicted, target, lengths):
        total = predicted.new_zeros((), dtype=torch.float32)
        for fft, hop in self.resolutions:
            window = getattr(self, f"window_{fft}")
            p, t = magnitude(predicted, fft, hop, window), magnitude(target, fft, hop, window)
            mask = (torch.arange(p.shape[-1], device=p.device)[None] * hop < lengths[:, None]).float()[:, None]
            logarithmic = ((p.clamp_min(1e-5).log() - t.clamp_min(1e-5).log()).abs() * mask).sum() / (mask.sum() * p.shape[1]).clamp_min(1)
            # Spectral convergence per utterance; the floor keeps silent input finite.
            convergence = (((p - t).square() * mask).sum((1, 2)) + 1e-9).sqrt() / (t.square() * mask).sum((1, 2)).sqrt().clamp_min(1.0)
            total = total + logarithmic + convergence.mean()
        hop = self.mel[1]
        p, t = self.log_mel(predicted), self.log_mel(target)
        mask = (torch.arange(p.shape[-1], device=p.device)[None] * hop < lengths[:, None]).float()[:, None]
        mel = ((p - t).abs() * mask).sum() / (mask.sum() * p.shape[1]).clamp_min(1)
        return total / len(self.resolutions), mel


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, channels=(1, 32, 96, 192, 384)):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(weight_norm(nn.Conv2d(a, b, (5, 1), (3, 1), (2, 0))) for a, b in zip(channels[:-1], channels[1:]))
        self.output = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, wave):
        wave = F.pad(wave, (0, (-wave.shape[-1]) % self.period))
        x = wave.reshape(wave.shape[0], 1, -1, self.period)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.output(x), features


class SpectrogramDiscriminator(nn.Module):
    def __init__(self, fft, hop, channels=32):
        super().__init__()
        self.fft, self.hop = fft, hop
        self.register_buffer("window", torch.hann_window(fft), persistent=False)
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.output = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, wave):
        x = magnitude(wave, self.fft, self.hop, self.window).transpose(1, 2)[:, None]   # [B,1,frames,bins]
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.output(x), features


class Discriminators(nn.Module):
    """Training-only waveform critics. Never included in an inference export."""

    def __init__(self, periods=(2, 3, 5, 7, 11), resolutions=((512, 128), (1024, 256), (2048, 512))):
        super().__init__()
        self.networks = nn.ModuleList([PeriodDiscriminator(p) for p in periods]
                                      + [SpectrogramDiscriminator(fft, hop) for fft, hop in resolutions])

    def forward(self, wave):
        return [network(wave) for network in self.networks]


def discriminator_loss(real, fake):
    return sum((r-1).square().mean() + f.square().mean()
               for (r, _), (f, _) in zip(real, fake)) / len(real)


def generator_gan_loss(real, fake):
    adversarial = sum((f-1).square().mean() for f, _ in fake) / len(fake)
    matching = sum(F.l1_loss(f, r.detach())
                   for (_, rs), (_, fs) in zip(real, fake) for r, f in zip(rs, fs))
    matching = matching / sum(len(fs) for _, fs in fake)
    return adversarial, matching


# --- StyleTTS 2 losses and critics (architecture "styletts") -----------------
# Ported from StyleTTS 2 (https://github.com/yl4579/StyleTTS2, MIT, Copyright (c)
# 2023 Aaron (Yinghao) Li): losses.py and Modules/discriminators.py. See
# voca/LICENSE-styletts2. Training-only; nothing here ships in inference.pt.

class MelConvergenceLoss(nn.Module):
    """StyleTTS 2's ``MultiResolutionSTFTLoss``: spectral convergence
    ||Y - X||_1 / ||Y||_1 of normalized log-mel spectrograms at three
    resolutions. (Their torchaudio MelSpectrogram, 128 HTK mels, is replaced by
    the Slaney filterbank above.)"""

    def __init__(self, sample_rate, resolutions=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)), mels=128):
        super().__init__()
        self.resolutions = resolutions
        for fft, _, window in resolutions:
            self.register_buffer(f"window_{fft}", F.pad(torch.hann_window(window), ((fft - window) // 2, (fft - window + 1) // 2)), persistent=False)
            self.register_buffer(f"filters_{fft}", mel_filterbank(sample_rate, fft, mels), persistent=False)

    def forward(self, predicted, target, second=None):
        """``second``: another take (fresh source noise) of the first ``len(second)`` rows. Those rows then
        count as half the spectral energy distance (Gritsenko et al. 2020, "A Spectral Energy Distance for
        Parallel Speech Synthesis"), d(take, target) + d(second, target) - d(take, second): a plain distance
        to one recording is least for a compromise no real take sounds like, which sounded metallic there.
        Returns the loss and how far apart the two takes are (0 without a second)."""
        total, takes = 0.0, 0.0
        for fft, hop, _ in self.resolutions:
            window, filters = getattr(self, f"window_{fft}"), getattr(self, f"filters_{fft}")
            x, y = (self.normalized(wave, fft, hop, window, filters) for wave in (predicted, target))
            distance = (y - x).abs().sum((1, 2))
            if second is not None:
                k = len(second)
                z = self.normalized(second, fft, hop, window, filters)
                apart = (x[:k] - z).abs().sum((1, 2))
                takes = takes + (apart.sum() / y[:k].abs().sum()).detach() / len(self.resolutions)
                distance = torch.cat((0.5 * (distance[:k] + (y[:k] - z).abs().sum((1, 2)) - apart), distance[k:]))
            total = total + distance.sum() / y.abs().sum()
        return total / len(self.resolutions), takes

    @staticmethod
    def normalized(wave, fft, hop, window, filters):
        return (torch.log(1e-5 + filters @ magnitude(wave, fft, hop, window).square()) + 4) / 4


class DiscriminatorP(nn.Module):
    """HiFi-GAN period discriminator, StyleTTS 2 sizes."""

    def __init__(self, period, kernel=5, stride=3):
        super().__init__()
        self.period = period
        channels = (1, 32, 128, 512, 1024)
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(a, b, (kernel, 1), (stride, 1), padding=(2, 0))) for a, b in zip(channels, channels[1:])]
                                   + [weight_norm(nn.Conv2d(1024, 1024, (kernel, 1), 1, padding=(2, 0)))])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, wave):
        x = F.pad(wave[:, None], (0, (-wave.shape[-1]) % self.period), "reflect")
        x = x.view(x.shape[0], 1, -1, self.period)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1), features


class SpecDiscriminator(nn.Module):
    """StyleTTS 2's linear-spectrogram discriminator (one resolution of the MRD)."""

    def __init__(self, fft, hop, window):
        super().__init__()
        self.fft, self.hop = fft, hop
        self.register_buffer("window", F.pad(torch.hann_window(window), ((fft - window) // 2, (fft - window + 1) // 2)), persistent=False)
        self.discriminators = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.out = weight_norm(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, wave):
        x = magnitude(wave, self.fft, self.hop, self.window).transpose(1, 2)[:, None]   # [B, 1, frames, bins]
        features = []
        for conv in self.discriminators:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.out(x)
        features.append(x)
        return x.flatten(1), features


class ComplexSpecDiscriminator(nn.Module):
    """DAC's complex multi-band spectrogram discriminator at one resolution (github.com/descriptinc/descript-audio-codec,
    MIT, dac/model/discriminator.py; see voca/LICENSE-dac): the real and imaginary parts of the STFT, so phase
    coherence and not only loudness, each of ``BANDS`` (shares of the Nyquist) by its own convolutions.
    StyleTTS 2's (``SpecDiscriminator``) sees only magnitudes. Each feature is one layer's, all bands
    together, so feature matching weighs a layer as it does the other critics'."""

    BANDS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

    def __init__(self, fft, channels=32):
        super().__init__()
        self.fft = fft
        self.register_buffer("window", torch.hann_window(fft), persistent=False)
        bins = fft // 2 + 1
        self.bands = [(int(low * bins), int(high * bins)) for low, high in zip(self.BANDS, self.BANDS[1:])]
        layers = ((2, channels, (3, 9), 1, (1, 4)), *[(channels, channels, (3, 9), (1, 2), (1, 4))] * 3, (channels, channels, (3, 3), 1, (1, 1)))
        self.convs = nn.ModuleList(nn.ModuleList(weight_norm(nn.Conv2d(i, o, k, stride=s, padding=p)) for i, o, k, s, p in layers) for _ in self.bands)
        self.out = weight_norm(nn.Conv2d(channels, 1, 3, 1, 1))

    def forward(self, wave):
        spectrum = torch.view_as_real(torch.stft(wave.float(), self.fft, self.fft // 4, window=self.window, pad_mode="constant", return_complex=True))
        x = spectrum.permute(0, 3, 2, 1)      # [B, 2, frames, bins]
        bands = [x[..., low:high] for low, high in self.bands]
        features = []
        for layer in range(len(self.convs[0])):
            bands = [F.leaky_relu(convs[layer](band), 0.1) for convs, band in zip(self.convs, bands)]
            features.append(torch.cat(bands, dim=-1))
        x = self.out(features[-1])
        features.append(x)
        return x.flatten(1), features


class SpeechModelDiscriminator(nn.Module):
    """StyleTTS 2's SLM critic head over all hidden states of a frozen speech model (``speech_model``)."""

    def __init__(self, hidden=768, layers=13, channels=64):
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(hidden * layers, channels, 1))
        self.convs = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels * 2, 5, padding=2)),
                                    weight_norm(nn.Conv1d(channels * 2, channels * 4, 5, padding=2)),
                                    weight_norm(nn.Conv1d(channels * 4, channels * 4, 5, padding=2))])
        self.conv_post = weight_norm(nn.Conv1d(channels * 4, 1, 3, padding=1))

    def forward(self, states):
        """All hidden states [B, 13, frames, 768] -> per-frame scores."""
        x = self.pre(states.transpose(-1, -2).flatten(1, 2))
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
        return self.conv_post(x).flatten(1)


@functools.cache
def speech_model(device):
    """Frozen mHuBERT-147 (utter-project; StyleTTS 2 uses WavLM-base-plus, which is trained on English only; this
    covers yap's languages), loaded once per device. Kept outside every module so it is in no state dict,
    checkpoint or optimizer. 768-wide, 12 layers, as WavLM-base-plus."""
    from transformers import AutoModel
    return AutoModel.from_pretrained("utter-project/mHuBERT-147").to(device).eval().requires_grad_(False)


def resample(wave, source, target):
    """Band-limited resampling by truncating (or zero-padding) the spectrum; differentiable."""
    size = wave.shape[-1] * target // source
    return torch.fft.irfft(torch.fft.rfft(wave.float())[..., :size // 2 + 1], size) * (size / wave.shape[-1])


def slm_states(wave, sample_rate):
    """All 13 hidden states of 24 kHz audio, as [B, 13, frames, 768]."""
    wave = resample(wave, sample_rate, 16000)
    wave = (wave - wave.mean(-1, keepdim=True)) / (wave.var(-1, keepdim=True, unbiased=False) + 1e-7).sqrt()     # as mHuBERT's feature extractor
    states = speech_model(wave.device)(input_values=wave, output_hidden_states=True).hidden_states
    return torch.stack(states, dim=1)


def slm_feature_loss(predicted, target, sample_rate):
    """StyleTTS 2's ``WavLMLoss.forward``: L1 between the speech model's layers of recording
    and output. Also returns both states: the model is frozen, so the critics reuse
    them instead of running it again on the same audio (same values, same gradient)."""
    with torch.no_grad():
        real = slm_states(target, sample_rate)
    fake = slm_states(predicted, sample_rate)
    return (real - fake).abs().mean((0, 2, 3)).sum(), {"real": real, "fake": fake}


def relativistic_loss(real, fake):
    """StyleTTS 2's truncated pointwise relativistic LS loss (TPRLS); the generator
    side calls it with the arguments swapped."""
    total = 0.0
    for r, f in zip(real, fake):
        # The median is a constant here (StyleTTS 2 differentiates it, which puts most of the gradient on the one
        # score it picks: 91% on 1% of the samples on one clip, Codex).
        median = torch.median(r - f).detach()
        below = (r < f + median).float()   # a masked mean, not boolean indexing: no GPU wait
        spread = ((((r - f) - median).square()) * below).sum() / below.sum().clamp_min(1)
        total = total + 0.04 - F.relu(0.04 - spread)
    return total


class ProsodyCritic(nn.Module):
    """Judges how a sentence is delivered, not how it sounds: per token its length and the mean and slope of its pitch
    and the mean of its energy (``delivery``), with the tokens' text features and the voice. Text-conditioned
    prosody discriminators improved naturalness in StyleTTS-ZS (Li et al. 2025). Per-token scores and one for the
    sentence. Durations are dequantized with uniform noise on both sides, so that continuous predictions
    are not told apart from whole recorded frames by that alone."""

    def __init__(self, vocabulary, speakers, channels=128):
        super().__init__()
        self.tokens = nn.Embedding(vocabulary, channels, padding_idx=0)
        add_lexical_embeddings(self, channels)
        self.voices = nn.Embedding(speakers, channels)
        self.input = weight_norm(nn.Conv1d(4, channels, 1))
        self.convs = nn.ModuleList(weight_norm(nn.Conv1d(channels, channels, 5, padding=2 * d, dilation=d)) for d in (1, 2, 4, 8))
        self.score, self.sentence = weight_norm(nn.Conv1d(channels, 1, 1)), weight_norm(nn.Linear(channels, 1))

    @staticmethod
    def delivery(durations, soft, prosody, index, frame_mask):
        """[B, N, 4] log length, pitch, pitch slope and energy per token, from integer ``durations`` [B, N] (with the
        gradient of ``soft``, their continuous prediction, where given) and prosody [B, F, 3] (log pitch, voicing, energy)."""
        length = durations + (soft - soft.detach()) if soft is not None else durations.float()
        relative = position_in_token(durations, index, torch.arange(index.shape[1], device=index.device)[None])[..., 0] - 0.5
        weight = frame_mask.float()
        pitch, energy = prosody[..., 0], prosody[..., 2]

        def tokens(value):
            return torch.zeros_like(length).scatter_add(1, index, value * weight)
        count = tokens(torch.ones_like(pitch)).clamp_min(1)
        mean = tokens(pitch) / count
        spread = tokens(relative.square()).clamp_min(1e-2)
        slope = tokens(relative * (pitch - frames_of(mean[..., None], index)[..., 0])) / spread
        length = (length + torch.rand_like(length) - 0.5).clamp_min(0)
        return torch.stack((torch.log1p(length), 4 * mean, slope, tokens(energy) / count), dim=-1)

    def forward(self, tokens, lexical, speakers, delivery, mask):
        """-> scores [B, N + 1]: every token's and the whole sentence's, and their mask."""
        x = embed_tokens(self, tokens, lexical) + self.voices(speakers)[:, None] + self.input(delivery.transpose(1, 2)).transpose(1, 2)
        x = (x * mask[..., None]).transpose(1, 2)
        for conv in self.convs:
            x = (x + F.leaky_relu(conv(x), 0.1)) * mask[:, None]
        pooled = (x * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1)
        scores = torch.cat((self.score(x)[:, 0], self.sentence(pooled)), dim=1)
        return scores, torch.cat((mask, mask.any(1, keepdim=True)), dim=1).float()

    def discriminator_loss(self, real, fake, mask):
        (r, m), (f, _) = real, fake
        return masked_mean((1 - r).square(), m * mask[:, None]) + masked_mean(f.square(), m * mask[:, None])

    @staticmethod
    def generator_loss(fake, mask):
        f, m = fake
        return masked_mean((1 - f).square(), m * mask[:, None])


class StyleCritics(nn.Module):
    """StyleTTS 2's critics: multi-period (2, 3, 5, 7, 11), multi-resolution
    spectrogram (magnitudes) and the speech-model head, plus DAC's on complex spectrograms
    (``ComplexSpecDiscriminator``) and ours on the delivery (``ProsodyCritic``). LSGAN with TPRLS and feature matching."""

    def __init__(self, sample_rate, vocabulary, speakers):
        super().__init__()
        self.sample_rate = sample_rate
        self.prosody = ProsodyCritic(vocabulary, speakers)
        self.mpd = nn.ModuleList(DiscriminatorP(p) for p in (2, 3, 5, 7, 11))
        self.mrd = nn.ModuleList(SpecDiscriminator(*r) for r in ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)))
        self.complex_mrd = nn.ModuleList(ComplexSpecDiscriminator(fft) for fft in (512, 1024, 2048))
        self.wd = SpeechModelDiscriminator()

    def waveform_critics(self):
        return (*self.mpd, *self.mrd, *self.complex_mrd)

    def discriminator_backward(self, real, fake, own, valid, slm, short):
        """Backward of the critics' loss, one critic at a time: they share nothing, so the gradients are the same and
        only one critic's activations are alive at once (all of them held 85 GB on a 31-clip batch). Returns the
        loss (detached).

        ``fake``: the recordings ``real`` reconstructed; ``own``: the model's own speech (its own timing and melody), in
        shorter windows, of which the rows ``valid`` (1 or 0) count; ``short``: the recordings in windows of that length,
        counted alike, so that length is not a cue. ``slm``: the speech-model states {"real", "fake", "own", "short"}
        of these clips, to also train its head (StyleTTS 2 does every 10th step), or None."""
        weight = torch.cat((torch.ones_like(valid[:1]).expand(len(fake)), valid))    # every fake row weighs the same

        def rejected(fake_scores, own_scores):
            return masked_mean(torch.cat((fake_scores.square().mean(1), own_scores.square().mean(1))), weight)

        def accepted(scores, short_scores):
            return masked_mean(torch.cat(((1 - scores).square().mean(1), (1 - short_scores).square().mean(1))), weight)
        total = 0.0
        for critic in self.waveform_critics():
            r, f, o, e = (critic(wave)[0] for wave in (real, fake, own, short))
            loss = accepted(r, e) + rejected(f, o) + relativistic_loss([r], [f])
            loss.backward()
            total = total + loss.detach()
        if slm is not None:
            loss = accepted(self.wd(slm["real"]), self.wd(slm["short"])) + rejected(*(self.wd(slm[key].detach()) for key in ("fake", "own")))
            loss.backward()
            total = total + loss.detach()
        return total

    def generator_gradients(self, real, fake, own, valid, scale, matching_weight, own_scale):
        """The waveform critics' loss for the generator, one critic at a time: the gradients of ``scale`` x (adversarial
        with TPRLS + ``matching_weight`` x feature matching) at the reconstruction ``fake`` and of ``own_scale`` x the
        adversarial term at the model's own speech ``own`` (``valid`` rows; no recording to match). Returns
        (adversarial, matching, own adversarial, gradient at fake, gradient at own)."""
        fake_leaf, own_leaf = (wave.detach().requires_grad_() for wave in (fake, own))
        adversarial, matching, own_adversarial = (torch.zeros((), device=fake.device) for _ in range(3))
        gradients = [torch.zeros_like(fake), torch.zeros_like(own)]
        for critic in self.waveform_critics():
            with torch.no_grad():
                real_score, real_features = critic(real)
            score, features = critic(fake_leaf)
            own_score = critic(own_leaf)[0]
            match = sum((a - b).abs().mean() for a, b in zip(real_features, features))
            adv = (1 - score).square().mean() + relativistic_loss([score], [real_score])
            own_adv = masked_mean((1 - own_score).square().mean(1), valid)
            for gradient, g in zip(gradients, torch.autograd.grad(scale * (adv + matching_weight * match) + own_scale * own_adv, (fake_leaf, own_leaf))):
                gradient += g
            adversarial, matching, own_adversarial = adversarial + adv.detach(), matching + match.detach(), own_adversarial + own_adv.detach()
        return adversarial, matching, own_adversarial, *gradients

    def speech_model_losses(self, valid, slm):
        """The speech-model head's loss for the generator: at the reconstruction and at the model's own speech."""
        return (1 - self.wd(slm["fake"])).square().mean(), masked_mean((1 - self.wd(slm["own"])).square().mean(1), valid)


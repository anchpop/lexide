import functools
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


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

    def forward(self, predicted, target):
        total = 0.0
        for fft, hop, _ in self.resolutions:
            window, filters = getattr(self, f"window_{fft}"), getattr(self, f"filters_{fft}")
            x, y = ((torch.log(1e-5 + filters @ magnitude(wave, fft, hop, window).square()) + 4) / 4 for wave in (predicted, target))
            total = total + (y - x).abs().sum() / y.abs().sum()
        return total / len(self.resolutions)


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


class WavLMDiscriminator(nn.Module):
    """StyleTTS 2's SLM critic head over all hidden states of a frozen WavLM."""

    def __init__(self, hidden=768, layers=13, channels=64):
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(hidden * layers, channels, 1))
        self.convs = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels * 2, 5, padding=2)),
                                    weight_norm(nn.Conv1d(channels * 2, channels * 4, 5, padding=2)),
                                    weight_norm(nn.Conv1d(channels * 4, channels * 4, 5, padding=2))])
        self.conv_post = weight_norm(nn.Conv1d(channels * 4, 1, 3, padding=1))

    def forward(self, states):
        x = self.pre(states)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
        return self.conv_post(x).flatten(1)


@functools.cache
def wavlm(device):
    """Frozen microsoft/wavlm-base-plus, loaded once per device. Kept outside every
    module so it is in no state dict, checkpoint or optimizer."""
    from transformers import AutoModel
    return AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval().requires_grad_(False)


def resample(wave, source, target):
    """Band-limited resampling by truncating (or zero-padding) the spectrum; differentiable."""
    size = wave.shape[-1] * target // source
    return torch.fft.irfft(torch.fft.rfft(wave.float())[..., :size // 2 + 1], size) * (size / wave.shape[-1])


def slm_states(wave, sample_rate):
    """All 13 WavLM hidden states of 24 kHz audio, as [B, 13 * 768, frames]."""
    states = wavlm(wave.device)(input_values=resample(wave, sample_rate, 16000), output_hidden_states=True).hidden_states
    return torch.stack(states, dim=1).transpose(-1, -2).flatten(1, 2)


def slm_feature_loss(predicted, target, sample_rate):
    """StyleTTS 2's ``WavLMLoss.forward``: L1 between the WavLM layers of recording and output."""
    with torch.no_grad():
        real = wavlm(target.device)(input_values=resample(target, sample_rate, 16000), output_hidden_states=True).hidden_states
    fake = wavlm(predicted.device)(input_values=resample(predicted, sample_rate, 16000), output_hidden_states=True).hidden_states
    return sum((r - f).abs().mean() for r, f in zip(real, fake))


def relativistic_loss(real, fake):
    """StyleTTS 2's truncated pointwise relativistic LS loss (TPRLS); the generator
    side calls it with the arguments swapped."""
    total = 0.0
    for r, f in zip(real, fake):
        median = torch.median(r - f)
        spread = (((r - f) - median).square())[r < f + median].mean()
        total = total + 0.04 - F.relu(0.04 - spread)
    return total


class StyleCritics(nn.Module):
    """StyleTTS 2's critics: multi-period (2, 3, 5, 7, 11), multi-resolution
    spectrogram, and the WavLM head. LSGAN with TPRLS and feature matching."""

    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.mpd = nn.ModuleList(DiscriminatorP(p) for p in (2, 3, 5, 7, 11))
        self.mrd = nn.ModuleList(SpecDiscriminator(*r) for r in ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)))
        self.wd = WavLMDiscriminator()

    def outputs(self, wave):
        return [critic(wave) for critic in (*self.mpd, *self.mrd)]

    def discriminator_loss(self, real, fake, slm):
        """``slm``: also train the WavLM head (StyleTTS 2 does every 10th step)."""
        real_out, fake_out = self.outputs(real), self.outputs(fake)
        r, f = [o for o, _ in real_out], [o for o, _ in fake_out]
        loss = sum((1 - a).square().mean() + b.square().mean() for a, b in zip(r, f)) + relativistic_loss(r, f)
        if slm:
            with torch.no_grad():
                real_states, fake_states = slm_states(real, self.sample_rate), slm_states(fake, self.sample_rate)
            loss = loss + (1 - self.wd(real_states)).square().mean() + self.wd(fake_states).square().mean()
        return loss

    def generator_loss(self, real, fake):
        """Returns (MPD + MRD adversarial with TPRLS and 2x feature matching, SLM adversarial)."""
        with torch.no_grad():
            real_out = self.outputs(real)
        fake_out = self.outputs(fake)
        r, f = [o for o, _ in real_out], [o for o, _ in fake_out]
        matching = 2 * sum((a - b).abs().mean() for (_, ra), (_, fa) in zip(real_out, fake_out) for a, b in zip(ra, fa))
        gan = sum((1 - b).square().mean() for b in f) + matching + relativistic_loss(f, r)
        return gan, (1 - self.wd(slm_states(fake, self.sample_rate))).square().mean()

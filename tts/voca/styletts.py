"""StyleTTS 2 / Kokoro architecture at tiny-voca scale (``architecture="styletts"``).

Ported from StyleTTS 2 (github.com/yl4579/StyleTTS2, MIT: models.py,
Modules/istftnet.py) and Kokoro (github.com/hexgrad/kokoro, Apache-2.0:
kokoro/modules.py, kokoro/istftnet.py); see voca/LICENSE-styletts2. Trained
from scratch on our phonemes, no pretrained weights.

    phonemes -> transformer text encoder (ours, shared with TinyVoca)
        -> [bert_encoder] -> prosody predictor (LSTMs with AdaLayerNorm on the
           prosodic style): per word prominence / contour / excursion, then
           durations, then per frame pitch / voicing / energy
        -> AdaIN residual decoder on text features + F0 + energy (acoustic style)
        -> iSTFTNet generator with the harmonic NSF source -> 24 kHz waveform

Changes from the originals:

* No learned aligner, JDC pitch net, PL-BERT, style diffusion or two-stage
  training. Durations come from forced alignments; pitch, voicing and energy
  targets are ours. The duration projection's sigmoid bins get one more
  output, the pause-presence logit of pause-capable tokens (TinyVoca's rule).
* Everything runs at our acoustic frame rate (hop 256). StyleTTS 2 aligns text
  at half its mel rate and upsamples x2 in the F0/energy branches and the last
  decoder block; our durations are whole frames at the full rate, so those
  upsamplers (and the decoder's stride-2 F0/energy convolutions) are gone.
* A word-level head (``WordProsody``, after Codex's proposal and hierarchical
  prosody TTS) predicts each word occurrence's prominence within its phrase,
  contour shape, excursion and phrase-final movement; durations and the
  F0/energy branch are conditioned on them, and every frame knows its place
  inside its phone (StyleTTS 2 repeats a token state and leaves that to the LSTM).
* The F0 branch predicts log(F0/reference) and a voicing logit. The decoder
  sees F0 / reference (0 when unvoiced) instead of raw Hz, the same curve at unit scale.
* Inference uses one stored style per voice: the running mean of the training
  clips' styles. The style encoders are training-only (voca.train).
* The generator's rates multiply to 256 instead of 300 (configs/styletts.json: 4 x 4 x
  iSTFT hop 16, which a blind A/B heard as even with 8 x 8 x hop 4 and synthesizes 2.5x
  faster; Kokoro: 10 x 6 x 5). It upsamples by nearest-neighbour interpolation and a
  convolution, not transposed convolutions, whose frame-periodic response put
  4-8 kHz bursts at fixed positions of the 256-sample frame in pauses. Its NSF
  source trains end to end (Kokoro's runs under no_grad, so its mixer never
  learns), choosing each sine's amplitude and phase and the noise level per band,
  with the phase integrated at the audio rate.
"""

import math
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from .config import Config
from .data import SHAPES
from .model import Voca, length_mask, position_in_token

DURATION_ENCODER_DROPOUT, DURATION_DROPOUT = 0.2, 0.5     # StyleTTS 2's, in training only: between the encoder's LSTMs, before the lengths
WORD_SEMITONES = 6.0                     # the word head's pitch outputs are in units of this many semitones
WORD_FEATURES = 1 + len(SHAPES) + 2      # word_features: prominence, shape probabilities, excursion, final movement

# run_lstm passes each direction's weights separately; compacting these small tensors per call is cheap.
warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk")

def run_lstm(lstm, x, lengths):
    """One-layer bidirectional ``lstm`` over [B, T, C] that stops at each row's
    length, as over a packed sequence: the backward direction reads every row
    from its own last step, and padded steps come out zero. Unlike packing it
    needs no host-side lengths and no length-dependent shapes (CUDA graphs)."""
    if lstm.num_layers != 1 or not lstm.bidirectional or not lstm.batch_first or lstm.proj_size:
        raise ValueError("run_lstm expects a one-layer, batch-first, bidirectional LSTM")
    step = torch.arange(x.shape[1], device=x.device)
    valid = step < lengths.to(x.device)[:, None]
    backwards = torch.where(valid, lengths.to(x.device)[:, None] - 1 - step, step)   # reverses each row's valid part

    def reverse(v):
        return v.gather(1, backwards[..., None].expand_as(v))

    def direction(v, suffix):
        weights = [getattr(lstm, f"{name}_l0{suffix}") for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
        h = v.new_zeros(1, v.shape[0], lstm.hidden_size)
        return torch.lstm(v, (h, h), weights, True, 1, 0.0, lstm.training, False, True)[0]
    return torch.cat((direction(x, ""), reverse(direction(reverse(x), "_reverse"))), dim=-1) * valid[..., None]


def istft(spectrum, fft, hop, window):
    """``torch.istft`` (centred, one-sided) as overlap-add of windowed frames.
    torch.istft checks the window envelope on the host, a GPU wait that CUDA
    graphs cannot capture; the Hann window at these hops never fails that check."""
    count = spectrum.shape[-1]
    frames = torch.fft.irfft(spectrum, fft, dim=1) * window[:, None]
    size = fft + hop * (count - 1)

    def overlap_add(x):   # trimmed before dividing: the envelope is 0 at the very edges
        return F.fold(x, (1, size), (1, fft), stride=(1, hop)).flatten(1)[:, fft // 2:size - fft // 2]
    return overlap_add(frames) / overlap_add(window.square()[None, :, None].expand(1, fft, count))


def frames_of(values, index):
    """values [B, N, C] per token, index [B, F] owning token -> [B, F, C] (``t_en @ alignment``)."""
    return values.gather(1, index[..., None].expand(-1, -1, values.shape[-1]))


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s, mask=None):
        """``mask`` [B, 1, T]: normalize over those frames only and zero the others, as if each row came unpadded."""
        gamma, beta = self.fc(s)[..., None].chunk(2, dim=1)
        if mask is None:
            return (1 + gamma) * self.norm(x) + beta
        x, count = x.float(), mask.sum(-1, keepdim=True).clamp_min(1)
        mean = (x * mask).sum(-1, keepdim=True) / count
        variance = ((x - mean).square() * mask).sum(-1, keepdim=True) / count
        return ((1 + gamma) * ((x - mean) / (variance + self.norm.eps).sqrt()) + beta) * mask


class AdaLayerNorm(nn.Module):
    """Layer norm over channels of [B, T, C] with a style-predicted scale and shift."""

    def __init__(self, style_dim, channels):
        super().__init__()
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        gamma, beta = self.fc(s)[:, None].chunk(2, dim=-1)
        return (1 + gamma) * F.layer_norm(x, x.shape[-1:]) + beta


class AdainResBlk1d(nn.Module):
    """StyleTTS 2's decoder / F0 / energy block (without its x2 upsampling option)."""

    def __init__(self, dim_in, dim_out, style_dim, dropout=0.0):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1, self.norm2 = AdaIN1d(style_dim, dim_in), AdaIN1d(style_dim, dim_out)
        self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)) if dim_in != dim_out else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, s, mask=None):
        y = self.conv1(self.dropout(F.leaky_relu(self.norm1(x, s, mask), 0.2)))
        y = self.conv2(self.dropout(F.leaky_relu(self.norm2(y, s, mask), 0.2)))
        return (y + (x if self.conv1x1 is None else self.conv1x1(x))) * math.sqrt(0.5)


def small_weight_norm(conv, std=0.01):
    """``weight_norm(conv)`` with weights drawn from N(0, ``std``), HiFi-GAN's generator initialization.
    They are drawn first: once parametrized, ``conv.weight`` is recomputed from the
    magnitude and direction, and initializing it changes nothing."""
    nn.init.normal_(conv.weight, 0.0, std)
    return weight_norm(conv)


class AdaINResBlock1(nn.Module):
    """Kokoro's generator block: dilated convolutions with AdaIN and Snake activations."""

    def __init__(self, channels, kernel, dilations, style_dim):
        super().__init__()
        self.convs1 = nn.ModuleList(small_weight_norm(nn.Conv1d(channels, channels, kernel, dilation=d, padding=d * (kernel - 1) // 2))
                                    for d in dilations)
        self.convs2 = nn.ModuleList(small_weight_norm(nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)) for _ in dilations)
        self.adain1 = nn.ModuleList(AdaIN1d(style_dim, channels) for _ in dilations)
        self.adain2 = nn.ModuleList(AdaIN1d(style_dim, channels) for _ in dilations)
        self.alpha1 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations)
        self.alpha2 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations)

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            y = n1(x, s)
            y = c1(y + (1 / a1) * torch.sin(a1 * y).square())   # Snake
            y = n2(y, s)
            x = x + c2(y + (1 / a2) * torch.sin(a2 * y).square())
        return x


class DurationEncoder(nn.Module):
    """LSTMs over tokens, each followed by AdaLayerNorm; the style is appended to every layer's input.
    With ``distinct`` set, that many LSTMs are looped to the full depth; each pass keeps its own norm."""

    def __init__(self, style_dim, channels, layers, distinct=0):
        super().__init__()
        distinct = distinct or layers
        if layers % distinct:
            raise ValueError(f"{layers} layers cannot loop {distinct} distinct LSTMs evenly")
        self.lstms = nn.ModuleList(nn.LSTM(channels + style_dim, channels // 2, batch_first=True, bidirectional=True) for _ in range(distinct))
        self.norms = nn.ModuleList(AdaLayerNorm(style_dim, channels) for _ in range(layers))

    def forward(self, x, style, lengths, mask):
        s = style[:, None].expand(-1, x.shape[1], -1)
        x = torch.cat((x, s), dim=-1) * mask[..., None]
        for i, norm in enumerate(self.norms):
            x = F.dropout(run_lstm(self.lstms[i % len(self.lstms)], x, lengths), DURATION_ENCODER_DROPOUT, self.training)
            x = torch.cat((norm(x, style), s), dim=-1) * mask[..., None]
        return x


class WordProsody(nn.Module):
    """Per word occurrence: a prominence score (ranked within its phrase),
    contour-shape logits (``voca.data.SHAPES``), pitch excursion and phrase-final
    movement (both in ``WORD_SEMITONES``), from the token states pooled over the
    word and a convolution across neighbouring words. Its outputs, as
    ``word_features``, condition the durations and the F0/energy branches."""

    OUTPUTS = 1 + len(SHAPES) + 2

    def __init__(self, inputs, channels=48):
        super().__init__()
        self.input = nn.Linear(inputs, channels)
        self.context = nn.Conv1d(channels, channels, 5, padding=2)
        self.output = nn.Linear(channels, self.OUTPUTS)

    def forward(self, x, words):
        """x [B, N, inputs] per token -> [B, W, OUTPUTS]."""
        keep = words.mask[..., None].to(x.dtype)
        h = F.gelu(self.input(words.pool(x))) * keep
        h = h + F.gelu(self.context(h.transpose(1, 2)).transpose(1, 2)) * keep
        return self.output(h) * keep


def word_features(outputs, words):
    """``WordProsody`` outputs -> the [B, W, WORD_FEATURES] the predictor is conditioned on:
    prominence standardized within the phrase, shape probabilities, excursion, and
    the final movement on phrase-final words (0 elsewhere)."""
    outputs = outputs.float()
    return torch.cat((words.standardize(outputs[..., 0])[..., None], outputs[..., 1:-2].softmax(-1) * words.mask[..., None],
                      outputs[..., -2:-1], outputs[..., -1:] * words.final[..., None]), dim=-1)


def target_features(targets, words):
    """``voca.data.word_prosody`` targets [B, W, 4] -> the same features and where they were measured, ([B, W, F], [B, W, F])."""
    valid = torch.isfinite(targets) & words.mask[..., None]
    t = targets.nan_to_num()
    shape = F.one_hot(t[..., 1].long().clamp(0, len(SHAPES) - 1), len(SHAPES)).float()
    values = torch.cat((t[..., :1], shape, t[..., 2:] / WORD_SEMITONES), dim=-1)
    return values, torch.cat((valid[..., :1], valid[..., 1:2].expand_as(shape), valid[..., 2:]), dim=-1)


class ProsodyPredictor(nn.Module):
    """StyleTTS 2's prosody predictor. Durations: ``max_duration`` sigmoid bins
    (frames = their sum) plus our pause-presence logit. Per frame: log(F0 /
    reference), voicing logit (F0 branch) and normalized log energy (N branch).

    Ours: a word-level head (``WordProsody``) reads the text and the duration
    encoder's states; its features (or, while training, the recording's) are
    spread back to the word's tokens and from there to its frames, and every
    frame also gets its position inside its phone (``position_in_token``)."""

    def __init__(self, c):
        super().__init__()
        d, s = c.style_prosody_channels, c.style_dim
        self.text_encoder = DurationEncoder(s, d, c.style_prosody_layers, c.style_prosody_distinct)
        self.words = WordProsody(d + d + s)
        self.lstm = nn.LSTM(d + s + WORD_FEATURES, d // 2, batch_first=True, bidirectional=True)
        self.duration_proj = nn.Linear(d, c.style_max_duration + 1)
        self.shared = nn.LSTM(d + s + WORD_FEATURES + 3, d // 2, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList([AdainResBlk1d(d, d, s, c.dropout), AdainResBlk1d(d, d // 2, s, c.dropout), AdainResBlk1d(d // 2, d // 2, s, c.dropout)])
        self.N = nn.ModuleList([AdainResBlk1d(d, d, s, c.dropout), AdainResBlk1d(d, d // 2, s, c.dropout), AdainResBlk1d(d // 2, d // 2, s, c.dropout)])
        self.F0_proj = nn.Conv1d(d // 2, 2, 1)
        self.N_proj = nn.Conv1d(d // 2, 1, 1)

    def forward(self, text, style, lengths, mask, words, teacher=None):
        """text [B, N, d] -> token states [B, N, d + s + WORD_FEATURES] (to be repeated per frame), duration
        logits [B, N, bins + 1], word outputs [B, W, WordProsody.OUTPUTS]. ``teacher``: (``voca.data.word_prosody``
        targets [B, W, 4], [B] whether to use them): where measured, the recording's word features replace the predicted ones."""
        d = self.text_encoder(text, style, lengths, mask)
        outputs = self.words(torch.cat((text, d), dim=-1), words)
        features = word_features(outputs.detach(), words)    # the word head learns only from its own targets
        if teacher is not None:
            values, measured = target_features(teacher[0], words)
            features = torch.where(measured & teacher[1][:, None, None], values, features)
        states = torch.cat((d, words.spread(features).to(d.dtype)), dim=-1)
        timing = F.dropout(run_lstm(self.lstm, states, lengths), DURATION_DROPOUT, self.training)
        return states, self.duration_proj(timing), outputs

    def F0Ntrain(self, frames, style, lengths):
        """Repeated token states with positions [B, F, d + s + WORD_FEATURES + 3] -> [B, F, 3] pitch, voicing
        logit, energy. Each row's are the same whatever it is padded to."""
        x = run_lstm(self.shared, frames, lengths).transpose(1, 2)
        mask = length_mask(lengths, frames.shape[1])[:, None].float()
        f0, n = x, x
        for block in self.F0:
            f0 = block(f0, style, mask)
        for block in self.N:
            n = block(n, style, mask)
        return torch.cat((self.F0_proj(f0), self.N_proj(n)), dim=1).transpose(1, 2)


class SourceModule(nn.Module):
    """The generator's excitation, after hn-NSF (Wang & Yamagishi) as in Kokoro: sines at F0
    and its overtones, plus Gaussian noise, through tanh.

    Ours: the phase is integrated at the audio rate, and the mix is the model's. hn-NSF and Kokoro
    fix it: equal sines gated by voicing, the overtones at random start phases (a phase the generator
    cannot know is no use to it: eng-style-voices-v3's mixer turned them down to 0-14% of the
    fundamental), and white noise at two levels. Here ``gains`` predicts, per frame from the
    generator's input and the voicing, each sine's amplitude and phase (as its sine and cosine
    parts) and the log level of the noise in each of the bands between ``BAND_EDGES``, all
    relative to ``level`` and interpolated linearly to the audio rate. It starts as equal sines
    in phase that follow the voicing, and flat noise, at about Kokoro's initial levels."""

    BAND_EDGES = (1500, 3000, 6000)     # Hz
    TAPS = 65

    def __init__(self, channels, sample_rate, hop, harmonics=9, level=0.02, voiced_noise=0.002):
        super().__init__()
        self.sample_rate, self.hop, self.level = sample_rate, hop, level
        self.register_buffer("orders", torch.arange(1, harmonics + 1, dtype=torch.float32), persistent=False)
        self.register_buffer("bands", self.filterbank(), persistent=False)
        self.gains = nn.Conv1d(channels + 1, 2 * harmonics + len(self.bands), 3, padding=1)    # [features | voicing] -> gains
        nn.init.zeros_(self.gains.weight)
        nn.init.zeros_(self.gains.bias)
        with torch.no_grad():
            self.gains.weight[:harmonics, -1, 1] = 1
            self.gains.weight[2 * harmonics:, -1, 1] = math.log(voiced_noise / level)

    def filterbank(self):
        """Band-pass filters [bands, 1, TAPS] (differences of windowed-sinc low-passes) that add up to an impulse."""
        n = torch.arange(self.TAPS, dtype=torch.float64) - self.TAPS // 2
        window = torch.hamming_window(self.TAPS, periodic=False, dtype=torch.float64)
        low = [torch.sinc(2 * edge / self.sample_rate * n) * window for edge in self.BAND_EDGES]
        low = torch.stack([torch.zeros_like(n), *(h / h.sum() for h in low), (n == 0).double()])
        return low.diff(dim=0).float()[:, None]

    def forward(self, f0, voicing, x):
        """f0 [B, F] in Hz (continuous), voicing [B, F] in [0, 1] (a label or the predicted probability),
        the generator's input x [B, C, F] -> source [B, F * hop]."""
        cycles = f0[..., None] * self.orders / self.sample_rate      # [B, F, H] per sample, constant within a frame
        advance = cycles * self.hop % 1
        start = (advance.cumsum(1) - advance) % 1                     # whole cycles dropped: exact in float32 at any length
        within = cycles[:, :, None] * torch.arange(self.hop, device=f0.device)[:, None]
        phase = 2 * math.pi * (start[:, :, None] + within).flatten(1, 2).transpose(1, 2)
        noise = F.conv1d(torch.randn_like(phase[:, :1]), self.bands, padding=self.TAPS // 2)
        gains = F.interpolate(self.gains(torch.cat((x, voicing[:, None]), dim=1)), scale_factor=self.hop, mode="linear")
        sine, cosine, noise_gain = gains.split((len(self.orders), len(self.orders), len(self.bands)), dim=1)
        return torch.tanh(self.level * ((sine * phase.sin() + cosine * phase.cos()).sum(1) + (noise_gain.exp() * noise).sum(1)))


class Upsample(nn.Module):
    """Nearest-neighbour upsampling by ``rate``, then a convolution spanning about
    two input frames. A constant input comes out constant whatever the weights;
    Kokoro's transposed convolution sums a different pair of kernel taps at each
    output position, a pattern repeating at the input rate that it must learn to
    cancel (Pons et al. 2021, "Upsampling artifacts in neural audio synthesis")."""

    def __init__(self, channels_in, channels_out, rate):
        super().__init__()
        self.rate = rate
        self.conv = small_weight_norm(nn.Conv1d(channels_in, channels_out, 2 * rate + 1, padding=rate))

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.rate, mode="nearest"))


class AntiAlias(nn.Module):
    """A fixed low-pass on each coefficient stream [B, C, T] before a projection that keeps every
    ``stride``-th frame: a windowed sinc at 0.9 of the decimated Nyquist, 8 * stride + 1 taps,
    in float32. Without it the source's harmonics fold back: at hop 16 the coarse injection
    runs at 375 frames/s, and a 147 Hz vowel got a 228 Hz (375 - 147) line 3 dB below its fundamental."""

    def __init__(self, stride):
        super().__init__()
        n = torch.arange(8 * stride + 1, dtype=torch.float64) - 4 * stride
        cutoff = 0.9 / (2 * stride)                     # cycles per input frame
        taps = 2 * cutoff * torch.sinc(2 * cutoff * n) * torch.hamming_window(len(n), periodic=False, dtype=torch.float64)
        self.register_buffer("taps", (taps / taps.sum()).float()[None, None], persistent=False)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = F.pad(x.float(), (self.taps.shape[-1] // 2,) * 2, mode="reflect")
            return F.conv1d(x, self.taps.expand(x.shape[1], 1, -1), groups=x.shape[1])


class Generator(nn.Module):
    """Kokoro's iSTFTNet generator: upsampling with AdaIN resblocks, the harmonic
    source's STFT injected at every scale, and a small inverse STFT writing
    magnitude and phase. Ours: nearest-neighbour upsampling and a convolution
    (``Upsample``) instead of transposed convolutions, a source trained with
    the rest (``SourceModule``), and the source's coefficients low-passed
    before every strided injection (``AntiAlias``)."""

    def __init__(self, c):
        super().__init__()
        rates, channels, s = c.generator_rates, c.generator_channels, c.style_dim
        self.fft, self.hop, self.frame, self.kernels = c.generator_fft, c.generator_hop, c.hop, len(c.generator_resblock_kernels)
        self.m_source = SourceModule(channels, c.sample_rate, c.hop)
        self.ups, self.resblocks, self.noise_convs, self.noise_res, self.anti_alias = (nn.ModuleList() for _ in range(5))
        for i, u in enumerate(rates):
            ch = channels // 2 ** (i + 1)
            self.ups.append(Upsample(channels // 2 ** i, ch, u))
            self.resblocks.extend(AdaINResBlock1(ch, k, (1, 3, 5), s) for k in c.generator_resblock_kernels)
            if i + 1 < len(rates):
                stride = math.prod(rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(self.fft + 2, ch, 2 * stride, stride, padding=(stride + 1) // 2))
                self.noise_res.append(AdaINResBlock1(ch, 7, (1, 3, 5), s))
                self.anti_alias.append(AntiAlias(stride))
            else:
                self.noise_convs.append(nn.Conv1d(self.fft + 2, ch, 1))
                self.noise_res.append(AdaINResBlock1(ch, 11, (1, 3, 5), s))
                self.anti_alias.append(nn.Identity())
        self.conv_post = small_weight_norm(nn.Conv1d(ch, self.fft + 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.register_buffer("window", torch.hann_window(self.fft), persistent=False)

    def excitation(self, f0, voicing, x):
        """The NSF source [B, F * hop] for f0 [B, F] Hz, voicing [B, F] in [0, 1] and the generator input
        x [B, C, F] (it draws noise); ``render`` turns it into the waveform."""
        with torch.autocast(x.device.type, enabled=False):
            return self.m_source(f0.float(), voicing.float(), x.float())

    def render(self, x, s, source, delay=None):
        """The waveform from the generator input x [B, C, F] and the ``excitation``. ``delay``
        (a [1] tensor of samples, under one acoustic frame): render as if the source and the
        input came that much later, and undo the delay. An ideal generator gives the same
        waveform at every delay; whatever follows its own output grid instead (images of the
        voice around its synthesis frame rate) does not."""
        if delay is not None:
            source = delayed(source, delay)
            a = delay.float() / self.frame                 # the input moves a fraction of a frame, interpolated
            x = (1 - a) * x + a * F.pad(x, (1, 0), mode="replicate")[..., :-1]
        with torch.autocast(x.device.type, enabled=False):    # the source and its spectrum in float32 also under bf16
            spectrum = torch.stft(source, self.fft, self.hop, window=self.window, return_complex=True)
        # Real and imaginary parts, not magnitude and angle: they scale with the source, so the learned
        # noise gain controls what reaches the generator (an angle stays full-strength noise at any level).
        har = torch.cat((spectrum.real, spectrum.imag), dim=1)
        for i, (up, anti_alias, noise_conv, noise_res) in enumerate(zip(self.ups, self.anti_alias, self.noise_convs, self.noise_res)):
            x = up(F.leaky_relu(x, 0.1))
            if i == len(self.ups) - 1:
                x = self.reflection_pad(x)
            x = x + noise_res(noise_conv(anti_alias(har)), s)
            x = sum(block(x, s) for block in self.resblocks[i * self.kernels:(i + 1) * self.kernels]) / self.kernels
        x = self.conv_post(F.leaky_relu(x))
        bins = self.fft // 2 + 1
        x = x.float()   # the inverse STFT in float32 also under bf16 autocast
        spectrum = torch.polar(torch.exp(x[:, :bins]), torch.sin(x[:, bins:]))
        wave = istft(spectrum, self.fft, self.hop, self.window)
        return wave if delay is None else delayed(wave, -delay)


def delayed(wave, delay):
    """wave [B, S] moved ``delay`` samples later (earlier if negative; a [1] tensor), the edge sample repeated."""
    index = (torch.arange(wave.shape[-1], device=wave.device) - delay).clamp(0, wave.shape[-1] - 1)
    return wave[:, index]


class Decoder(nn.Module):
    """StyleTTS 2's AdaIN decoder: text features with F0 and energy, re-injected before every block."""

    def __init__(self, c):
        super().__init__()
        h, s = c.decoder_channels, c.style_dim
        self.reference = c.pitch_reference_hz
        self.encode = AdainResBlk1d(c.channels + 2, h, s)
        self.decode = nn.ModuleList([AdainResBlk1d(h + 2 + 64, h, s) for _ in range(3)] + [AdainResBlk1d(h + 2 + 64, c.generator_channels, s)])
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, 3, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, 3, padding=1))
        self.asr_res = weight_norm(nn.Conv1d(c.channels, 64, 1))
        self.generator = Generator(c)

    def features(self, text, condition, style):
        """text [B, F, C], condition [B, F, 3] (log pitch, voiced, energy) -> generator input [B, G, F]."""
        f0 = self.F0_conv((torch.exp(condition[..., 0]) * condition[..., 1])[:, None])
        n = self.N_conv(condition[:, None, :, 2])
        asr = text.transpose(1, 2)
        x = self.encode(torch.cat((asr, f0, n), dim=1), style)
        residual = self.asr_res(asr)
        for block in self.decode:
            x = block(torch.cat((x, residual, f0, n), dim=1), style)
        return x



class ResBlk(nn.Module):
    """StyleTTS 2's 2-D residual block with 'half' downsampling (style encoder)."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        self.downsample_res = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 2, 1, groups=dim_in))
        self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)) if dim_in != dim_out else None

    def forward(self, x):
        shortcut = x if self.conv1x1 is None else self.conv1x1(x)
        if shortcut.shape[-1] % 2:
            shortcut = torch.cat((shortcut, shortcut[..., -1:]), dim=-1)
        residual = self.conv2(F.leaky_relu(self.downsample_res(self.conv1(F.leaky_relu(x, 0.2))), 0.2))
        return (F.avg_pool2d(shortcut, 2) + residual) * math.sqrt(0.5)


class StyleEncoder(nn.Module):
    """StyleTTS 2's style encoder: a normalized mel [B, 1, mels, frames] (>= 80
    frames) -> style vector. Training-only here."""

    def __init__(self, c):
        super().__init__()
        dim = c.style_encoder_channels
        blocks = [spectral_norm(nn.Conv2d(1, dim, 3, 1, 1))]
        for _ in range(4):
            out = min(dim * 2, c.style_encoder_max)
            blocks.append(ResBlk(dim, out))
            dim = out
        blocks += [nn.LeakyReLU(0.2), spectral_norm(nn.Conv2d(dim, dim, 5, 1, 0)), nn.AdaptiveAvgPool2d(1), nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim, c.style_dim)

    def forward(self, mel):
        return self.unshared(self.shared(mel).flatten(1))


class Reading(nn.Module):
    """A distribution over a sentence's delivery: token features [B, N, inputs] and the voice -> mean and log standard deviation [B, Delivery.SIZE]."""

    def __init__(self, inputs, voices, channels=96):
        super().__init__()
        self.input, self.voices = nn.Linear(inputs, channels), nn.Embedding(voices, channels)
        self.convs = nn.ModuleList(nn.Conv1d(channels, channels, 5, padding=2 * d, dilation=d) for d in (1, 2, 4))
        self.attention, self.output = nn.Linear(channels, 1), nn.Linear(channels, 2 * Delivery.SIZE)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, speakers, mask):
        keep = mask[:, None]
        x = (self.input(x) + self.voices(speakers)[:, None]).transpose(1, 2) * keep
        for conv in self.convs:
            x = (x + F.gelu(conv(x))) * keep
        share = self.attention(x.transpose(1, 2)).float().masked_fill(~mask[..., None], -torch.inf).softmax(1).to(x.dtype)
        mean, spread = self.output((x * share.transpose(1, 2)).sum(-1)).float().chunk(2, dim=-1)
        return mean, 2 * torch.tanh(spread / 2)


class Delivery(nn.Module):
    """How a sentence is said beyond its words: a vector drawn anew for every sentence, which the whole prosody
    predictor reads (StyleTTS 2's prosodic style, ours drawn by a conditional VAE instead of diffusion). Ships:
    the ``prior``, what deliveries suit a text in a voice, and the projection to the predictor's style. Training draws
    from the posterior, which has heard the recording (``StyleTrainingSystem.heard``): one text has several
    valid deliveries, and a predictor that has to guess among them learns their average."""

    SIZE = 16

    def __init__(self, c):
        super().__init__()
        self.prior = Reading(c.style_prosody_channels, len(c.speakers))
        self.voices = nn.Embedding(len(c.speakers), c.style_dim)
        self.project = nn.Sequential(nn.Linear(self.SIZE + c.style_dim, c.style_dim), nn.GELU(), nn.Linear(c.style_dim, c.style_dim))

    def style(self, z, speakers):
        return self.project(torch.cat((z.to(self.voices.weight.dtype), self.voices(speakers)), dim=-1))

    @staticmethod
    def draw(reading, variety=1.0, generator=None):
        mean, spread = reading
        return mean + variety * spread.exp() * torch.randn(mean.shape, generator=generator, device=mean.device)


def reading_divergence(posterior, prior):
    """KL(posterior || prior) of two diagonal Gaussians (mean, log standard deviation), in nats per sentence [B]."""
    (a, s), (b, t) = posterior, prior
    return (t - s + ((2 * s).exp() + (a - b).square()) / (2 * (2 * t).exp()) - 0.5).sum(-1)


def recorded_delivery(durations, prosody, index, frame_mask):
    """[B, N, 5] per token, how a recording says it: log length, whether it is said at all, the mean and slope of
    its log pitch and its mean energy, from prosody [B, F, 3] (log pitch, voicing, energy)."""
    relative = position_in_token(durations, index, torch.arange(index.shape[1], device=index.device)[None])[..., 0] - 0.5
    weight = frame_mask.float()

    def tokens(value):
        return torch.zeros(durations.shape, device=value.device).scatter_add(1, index, value * weight)
    count = tokens(torch.ones_like(weight)).clamp_min(1)
    mean = tokens(prosody[..., 0]) / count
    slope = tokens(relative * (prosody[..., 0] - frames_of(mean[..., None], index)[..., 0])) / tokens(relative.square()).clamp_min(1e-2)
    return torch.stack((torch.log1p(durations.float()), (durations > 0).float(), 4 * mean, slope, tokens(prosody[..., 2]) / count), dim=-1)


class StyleVoca(Voca):
    PARAMETER_BUDGET = 30_000_000

    def __init__(self, config=Config()):
        """With ``phoneme_encoder_layers`` set, a phoneme encoder pretrained on text
        (voca.phonebert) feeds the prosody predictor through ``bert_encoder``, as in
        Kokoro; without one, the predictor reads the text encoder."""
        super().__init__(config)
        c = config
        self.bert = None
        if c.phoneme_encoder_layers:
            from .phonebert import PhoneEncoder, encoder_config
            self.bert = PhoneEncoder(encoder_config(c.phoneme_encoder_width, c.phoneme_encoder_heads, c.phoneme_encoder_feedforward,
                                                    c.dropout), c.phoneme_encoder_layers, c.phoneme_encoder_distinct, c.front_layers)
        self.bert_encoder = nn.Linear(c.phoneme_encoder_width if self.bert else c.channels, c.style_prosody_channels)
        self.predictor = ProsodyPredictor(c)
        self.decoder = Decoder(c)
        self.delivery = Delivery(c)
        # How far each frame is through its phone, and how long the phone is: what the decoder must know to place a
        # closure, its release and the rest, and cannot see in a code repeated on every frame. Starts at zero.
        self.frame_place = nn.Linear(2, c.channels, bias=False)
        nn.init.zeros_(self.frame_place.weight)
        # Per voice: the mean acoustic style of the training clips, what inference uses.
        self.register_buffer("style", torch.zeros(len(c.speakers), c.style_dim))

    def prosody_text(self, hidden, tokens, lexical, mask):
        """What the prosody predictor reads of the text."""
        return self.bert_encoder(hidden if self.bert is None else self.bert(tokens, lexical, mask))

    def decoder_text(self, hidden, tokens, durations, index, frame):
        """The text a frame speaks, [B, F, C]: its token's state and, for phones, where in it the frame is
        (``position_in_token``: 0 its start, 1 its end) and the phone's log length. ``index`` [B, F] is each frame's
        token, ``frame`` [B, F] its number in the whole utterance, so that a training window gets the places it has at synthesis."""
        place = position_in_token(durations, index, frame)
        place = torch.stack((2 * place[..., 0] - 1, place[..., 2]), dim=-1) * self.phone_tokens[tokens].gather(1, index)[..., None]
        return frames_of(hidden, index) + self.frame_place(place.to(self.frame_place.weight.dtype))

    def melody(self, states, durations, index, style, lengths):
        """Token states already repeated over whole utterances (``frames_of``) -> [B, F, 3] pitch, voicing logit, energy."""
        frame = torch.arange(index.shape[1], device=index.device)[None]
        frames = torch.cat((states, position_in_token(durations, index, frame).to(states.dtype)), dim=-1)
        return self.predictor.F0Ntrain(frames, style, lengths).float()

    @torch.no_grad()
    def observe_style(self, style, speakers, momentum=0.01):
        """Running mean of the training clips' styles (exponential, like batch norm's)."""
        total = torch.zeros_like(self.style).index_add_(0, speakers, style.float())
        count = torch.zeros(len(self.style), device=style.device).index_add_(0, speakers, torch.ones_like(speakers, dtype=torch.float))
        seen = (count > 0)[:, None]
        self.style.copy_(torch.where(seen, self.style.lerp(total / count.clamp_min(1)[:, None], momentum), self.style))

    def predict_durations(self, logits, tokens, mask, speed=1.0):
        frames = torch.round(torch.sigmoid(logits[..., :-1].float()).sum(-1) / speed).long()
        return self.gate_pauses(frames, logits[..., -1], tokens, mask)

    @torch.inference_mode()
    def synthesize(self, tokens, lexical, lengths, speakers, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, pitch_jitter=0.0,
                   voiced_vowels=True, comma_pause=0.0, stop_pause=0.0, take=0, variety=1.0):
        """Free-running synthesis with the stored mean acoustic style and a delivery drawn with the seed ``take`` (``variety``
        times as far from the text's most typical one). One utterance at a time: the decoder's instance norms would see a
        batch's padding."""
        self.check_controls(speed, pitch_shift, pitch_range)
        if len(lengths) != 1:
            raise ValueError("StyleVoca synthesizes one utterance at a time")
        hidden, mask = self.encode(tokens, lexical, lengths, speakers)
        acoustic_style = self.style[speakers]
        text = self.prosody_text(hidden, tokens, lexical, mask)
        generator = torch.Generator(tokens.device).manual_seed(take)
        prosodic_style = self.delivery.style(self.delivery.draw(self.delivery.prior(text, speakers, mask), variety, generator), speakers)
        states, logits, _ = self.predictor(text, prosodic_style, lengths, mask, self.words(tokens, mask))
        durations = self.pause_floors(self.predict_durations(logits, tokens, mask, speed), tokens, lengths, mask, comma_pause, stop_pause)
        index, frame_mask, frame_lengths = self.align(durations, mask)
        prosody = self.melody(frames_of(states, index), durations, index, prosodic_style, frame_lengths)
        prosody = self.remove_pitch_spikes(prosody)
        if voiced_vowels:
            prosody = self.voice_vowels(prosody, tokens, index, frame_mask)
        prosody = self.widen_pitch(prosody, frame_mask, pitch_range)
        condition = self.prosody_condition(prosody)
        pitch = condition[..., 0] + pitch_shift
        if pitch_jitter > 0:
            pitch = pitch + self.pitch_wobble(pitch.shape, pitch_jitter, seed, pitch.device)
        condition = torch.cat((pitch[..., None], condition[..., 1:]), dim=-1)
        text = self.decoder_text(hidden, tokens, durations, index, torch.arange(index.shape[1], device=index.device)[None])
        features = self.decoder.features(text, condition, acoustic_style)
        with torch.random.fork_rng(devices=[tokens.device] if tokens.device.type == "cuda" else []):
            torch.manual_seed(seed)     # the source draws its noise
            source = self.decoder.generator.excitation(self.frequency(pitch), condition[..., 1], features)
        wave = self.decoder.generator.render(features, acoustic_style, source)
        return wave, frame_lengths * self.config.hop, {"durations": durations, "prosody": prosody, "frame_mask": frame_mask,
                                                       "frame_lengths": frame_lengths,
                                                       "generator": (features, acoustic_style, source)}   # voca.evaluate re-renders it

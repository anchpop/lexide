"""The first architecture (``architecture="tinyvoca"``, from the tiny-voca recipe): a feed-forward transformer with a pitch-driven source-filter head.

All weights start random. Tensor conventions: hidden [B,T,C], frame features
[B,T], waveform [B,S]. Acoustic frame i is centred on sample ``i * hop``.

    phoneme + stress/tone/accent + voice -> transformer encoder -> duration predictor
        -> repeat tokens by duration (supervised durations while training)
        -> frame transformer A -> pitch / voicing / energy
        -> frame transformer B -> filter parameters
        -> harmonic pulse train (from pitch) and noise, each shaped by its
           time-varying filter in the STFT domain -> 24 kHz waveform

The waveform head has no neural upsampling network. The oscillator supplies the
periodicity that a tiny decoder cannot render reliably on its own; the network
only has to predict slowly varying spectral envelopes (complex cepstra), which
is what keeps the voice from sounding buzzy or unstable at this size.
"""

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config
from .text import ALWAYS_VOICED, LEXICAL, PHONE_IDS, PHRASE_BREAKS, SYMBOL_TO_ID, SYMBOLS


class UtteranceTooLong(ValueError):
    """The utterance needs more frames than ``max_frames``: split the input."""


def add_lexical_embeddings(module, channels):
    """One embedding per parallel token feature (``voca.text.LEXICAL``) on ``module``.
    Every feature but stress embeds "none" as zero, so a token without it (a
    language without tone, a pause without a part of speech, a masked word)
    gets nothing added."""
    for name, size in LEXICAL.items():
        setattr(module, name, nn.Embedding(size, channels, padding_idx=None if name == "stress" else 0))


def embed_tokens(module, tokens, lexical):
    """tokens [B, N], lexical [B, N, len(LEXICAL)] -> [B, N, C]."""
    return module.tokens(tokens) + sum(getattr(module, name)(lexical[..., i]) for i, name in enumerate(LEXICAL))


PHONE_TABLE = torch.zeros(len(SYMBOLS), dtype=torch.bool)
PHONE_TABLE[sorted(PHONE_IDS)] = True
BREAK_TABLE = torch.zeros(len(SYMBOLS), dtype=torch.bool)
BREAK_TABLE[[SYMBOL_TO_ID[p] for p in PHRASE_BREAKS]] = True


@dataclass
class Words:
    """The word occurrences of a padded batch, with room for one per token (W = N)."""
    index: torch.Tensor     # [B, N] each token's word, -1 for tokens that are not phones
    members: torch.Tensor   # [B, W, N] 1 where token n belongs to word w (pooling and broadcasting are matrix products)
    mask: torch.Tensor      # [B, W] real words
    phrase: torch.Tensor    # [B, W] the phrase-break tokens before the word: equal within an intonation phrase
    final: torch.Tensor     # [B, W] the last word of its phrase
    same: torch.Tensor      # [B, W, W] two real words of the same phrase

    def pool(self, x):
        """[B, N, C] per token -> [B, W, C] the mean over each word's phones."""
        return (self.members.to(x.dtype) @ x) / self.members.sum(-1, keepdim=True).clamp_min(1).to(x.dtype)

    def spread(self, x):
        """[B, W, C] per word -> [B, N, C] on each word's phones, 0 on other tokens."""
        return self.members.transpose(1, 2).to(x.dtype) @ x

    def standardize(self, x):
        """[B, W] -> z-scores within each phrase (0 for a word alone in its phrase)."""
        same = self.same.to(x.dtype)
        count = same.sum(-1).clamp_min(1)
        mean = (same @ x[..., None])[..., 0] / count
        variance = (same @ (x - mean).square()[..., None])[..., 0] / count
        return (x - mean) * torch.rsqrt(variance + 1e-4) * self.mask


def word_layout(tokens, mask, phones=PHONE_TABLE, breaks=BREAK_TABLE):
    """Word occurrences from the tokens alone, as ``voca.text.tokenize`` numbers
    them: a word is a maximal run of phones, because break tokens always
    separate words. Phrases end at ``PHRASE_BREAKS`` punctuation. ``phones`` and
    ``breaks`` are per-symbol tables on the tokens' device."""
    phone = phones[tokens] & mask
    start = phone & ~F.pad(phone[:, :-1], (1, 0))
    index = torch.where(phone, start.long().cumsum(1) - 1, -1)
    size = tokens.shape[1]
    members = index[:, None, :] == torch.arange(size, device=tokens.device)[None, :, None]
    words = members.any(-1)
    before = (breaks[tokens] & mask).long().cumsum(1)
    phrase = torch.where(members, before[:, None, :], -1).amax(-1)
    following = F.pad(phrase[:, 1:], (0, 1), value=-1)
    same = (phrase[:, :, None] == phrase[:, None, :]) & words[:, :, None] & words[:, None, :]
    return Words(index, members, words, phrase, words & (following != phrase), same)


def position_in_token(durations, index, frame):
    """[B, F, 3] for frames ``frame`` [B, F] owned by tokens ``index`` [B, F]: where
    the frame's centre sits inside its token (and 1 minus that) and the token's log length."""
    owned = durations.gather(1, index).clamp_min(1)
    start = durations.cumsum(1).gather(1, index) - durations.gather(1, index)
    relative = (frame - start + 0.5) / owned
    return torch.stack((relative, 1 - relative, torch.log1p(owned.float()) / 4), dim=-1)


def length_mask(lengths, size=None):
    size = int(lengths.max()) if size is None else size
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


def rotary_table(size, width, device):
    inverse = 1.0 / (10000 ** (torch.arange(0, width, 2, device=device, dtype=torch.float32) / width))
    angle = torch.arange(size, device=device, dtype=torch.float32)[:, None] * inverse[None]
    return angle.cos(), angle.sin()


def rotate(x, table):
    cos, sin = table[0].to(x.dtype), table[1].to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Self-attention with rotary (relative) positions: no absolute position
    table, so utterances longer than those seen in training stay in range."""

    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, mask, table):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rotate(q, table), rotate(k, table), v, attn_mask=mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.channels)
        self.attention = Attention(c.channels, c.heads, c.dropout)
        self.norm2 = nn.LayerNorm(c.channels)
        self.depthwise = nn.Conv1d(c.channels, c.channels, c.kernel, padding=c.kernel // 2, groups=c.channels)
        self.expand = nn.Linear(c.channels, c.feedforward)
        self.project = nn.Linear(c.feedforward, c.channels)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, mask, table):
        keep = mask[..., None]
        x = (x + self.dropout(self.attention(self.norm1(x), mask, table))) * keep
        # Mask before the convolution so padded positions cannot leak back in.
        y = self.depthwise((self.norm2(x) * keep).transpose(1, 2)).transpose(1, 2)
        y = self.project(self.dropout(F.gelu(self.expand(y))))
        return (x + self.dropout(y)) * keep


class Stack(nn.Module):
    """``layers`` blocks deep. With ``distinct`` set, only that many blocks exist and
    they are looped (0, 1, .., distinct-1, 0, 1, ..) to the full depth, each pass
    told its depth by a learned embedding: same compute, fewer parameters."""

    def __init__(self, c, layers, distinct=0):
        super().__init__()
        distinct = distinct or layers
        if layers % distinct:
            raise ValueError(f"{layers} layers cannot loop {distinct} distinct blocks evenly")
        self.layers = layers
        self.blocks = nn.ModuleList(Block(c) for _ in range(distinct))
        self.depth = nn.Parameter(torch.zeros(layers, c.channels)) if distinct < layers else None
        self.norm = nn.LayerNorm(c.channels)
        self.width = c.channels // c.heads

    def forward(self, x, mask):
        table = rotary_table(x.shape[1], self.width, x.device)
        for i in range(self.layers):
            if self.depth is not None:
                x = x + self.depth[i]
            x = self.blocks[i % len(self.blocks)](x, mask, table)
        return self.norm(x) * mask[..., None]


class Front(nn.Module):
    """``layers`` blocks before a text encoder, added to its input through a projection that starts at zero, so a model
    with a front starts as the one without. The phonemes-only model learns there what yap's hints told the original
    (voca.distill)."""

    def __init__(self, c, layers):
        super().__init__()
        self.stack = Stack(c, layers)
        self.out = nn.Linear(c.channels, c.channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, mask):
        return x + self.out(self.stack(x, mask)) * mask[..., None]


class Predictor(nn.Module):
    def __init__(self, c, outputs):
        super().__init__()
        p = c.predictor_channels
        self.conv1 = nn.Conv1d(c.channels, p, 3, padding=1)
        self.conv2 = nn.Conv1d(p, p, 3, padding=1)
        self.norm1, self.norm2 = nn.LayerNorm(p), nn.LayerNorm(p)
        self.drop = nn.Dropout(c.dropout)
        self.out = nn.Linear(p, outputs)

    def forward(self, x, mask):
        x = self.conv1((x * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm1(F.gelu(x))) * mask[..., None]
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm2(F.gelu(x)))
        return self.out(x) * mask[..., None]


def upsample(values, hop):
    """[B, T, D] frame values -> [B, T*hop, D], linear between frame centres i*hop."""
    frames = values.shape[1]
    knots = torch.cat((values, values[:, -1:]), dim=1)
    position = torch.arange(frames * hop, device=values.device, dtype=values.dtype) / hop
    low = position.floor().long()
    weight = (position - low)[None, :, None]
    return knots[:, low] * (1 - weight) + knots[:, low + 1] * weight


def phase_track(f0, sample_rate, hop, phase=None):
    """Oscillator phase [B, T*hop] (radians, float64, wrapped) and instantaneous
    frequency for f0 [B, T] in Hz, interpolated between frames in the log domain."""
    frequency = torch.exp(upsample(torch.log(f0.double().clamp_min(1.0))[..., None], hop)[..., 0])
    theta = 2 * math.pi * torch.cumsum(frequency / sample_rate, dim=1)
    if phase is not None:
        theta = theta + phase.double()[:, None]
    return torch.remainder(theta, 2 * math.pi), frequency


def pulse_train(f0, sample_rate, hop, limit=0.47, phase=None):
    """Band-limited pulse train [B, T*hop] with unit power for f0 [B, T] in Hz.

    Frame values sit at samples i*hop and are interpolated in the log domain.
    The sum of all harmonics below ``limit * sample_rate`` has the closed form
    sin((K+1/2)x) / (2 sin(x/2)) - 1/2; the top harmonic fades in and out so the
    signal stays continuous while the pitch glides. Float64 keeps the phase exact.
    """
    theta, frequency = phase_track(f0, sample_rate, hop, phase)
    harmonics = (limit * sample_rate / frequency).clamp_min(1.0)
    whole = torch.floor(harmonics)
    denominator = 2 * torch.sin(theta / 2)
    tiny = denominator.abs() < 1e-7
    core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denominator), denominator) - 0.5)
    wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
    return (wave * torch.sqrt(2.0 / harmonics)).float()


def filtered(source, response, fft, hop, window):
    """Time-varying filtering: response [B, frames, bins] multiplies the STFT."""
    spectrum = torch.stft(source, fft, hop, window=window, center=True, pad_mode="constant", return_complex=True)
    if spectrum.shape[-1] != response.shape[1]:
        raise ValueError("filter frames do not match the source")
    return torch.istft(spectrum * response.transpose(1, 2), fft, hop, window=window, center=True, length=source.shape[-1])


def cepstrum_response(causal, anticausal, fft):
    """Complex frequency response exp(FFT(cepstrum)). Causal coefficients give a
    minimum-phase filter; anticausal ones add a maximum-phase part (glottal pulse)."""
    buffer = causal.new_zeros(*causal.shape[:-1], fft)
    buffer[..., :causal.shape[-1]] = causal
    if anticausal is not None:
        buffer[..., fft - anticausal.shape[-1]:] = anticausal.flip(-1)
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return torch.polar(torch.exp(spectrum.real.clamp(-14, 6)), spectrum.imag)


def stft_frames(lengths, steps, count):
    """For ``count`` STFT frames on a grid of ``steps`` per acoustic frame (signal
    padded by half a window): which frames touch audio, and the acoustic frame
    each belongs to. Past an utterance's end the owner is held at its last frame,
    so a row's result is the same batched or alone."""
    frame = torch.arange(count, device=lengths.device)[None]
    owner = torch.minimum(frame, (steps * lengths - 1).clamp_min(0)[:, None]) // steps
    return frame < (steps * lengths + 3)[:, None], owner


def take_frames(values, owner):
    return values.gather(1, owner[..., None].expand(-1, -1, values.shape[-1]))


class PostFilterBlock(nn.Module):
    """ConvNeXt block over STFT frames: depthwise conv, then a pointwise MLP."""

    def __init__(self, channels, kernel):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.up = nn.Linear(channels, 3 * channels)
        self.down = nn.Linear(3 * channels, channels)

    def forward(self, x, mask):
        y = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        return (x + self.down(F.gelu(self.up(self.norm(y))))) * mask[..., None]


class PostFilter(nn.Module):
    """Small network that corrects the synthesized spectrum: per STFT bin a gain
    and a phase rotation, from the bin magnitudes and the frame's acoustic
    state. It starts as the identity (zero-initialized output)."""

    def __init__(self, c):
        super().__init__()
        self.config = c
        bins = c.synth_fft // 2 + 1
        self.input = nn.Linear(bins, c.postfilter_channels)
        self.condition = nn.Linear(c.channels, c.postfilter_channels)
        self.blocks = nn.ModuleList(PostFilterBlock(c.postfilter_channels, c.kernel) for _ in range(c.postfilter_layers))
        self.output = nn.Linear(c.postfilter_channels, 2 * bins)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("window", torch.hann_window(c.synth_fft), persistent=False)

    def forward(self, wave, hidden, lengths):
        """wave [B, S] (zero past each utterance), hidden [B, T, C] acoustic frames -> wave [B, S].

        Runs at the STFT frame rate (synth_hop), so it costs little next to an
        audio-rate network. Frames past an utterance's end are masked and its last
        acoustic frame is held, so the result is the same batched or alone."""
        c = self.config
        steps, samples = c.hop // c.synth_hop, wave.shape[-1]
        spectrum = torch.stft(F.pad(wave, (0, c.synth_fft // 2)), c.synth_fft, c.synth_hop, window=self.window,
                              center=True, pad_mode="constant", return_complex=True)
        mask, owner = stft_frames(lengths, steps, spectrum.shape[-1])
        level = (torch.log(spectrum.abs().transpose(1, 2) + 1e-5) + 5) / 5
        condition = self.condition(take_frames(hidden, owner))
        x = (self.input(level) + condition) * mask[..., None]
        for block in self.blocks:
            x = block(x, mask)
        gain, rotation = self.output(x).chunk(2, dim=-1)
        spectrum = spectrum * torch.polar(torch.exp(gain.clamp(-6, 3)), math.pi * torch.tanh(rotation)).transpose(1, 2)
        return torch.istft(spectrum, c.synth_fft, c.synth_hop, window=self.window, center=True,
                           length=samples + c.synth_fft // 2)[:, :samples]


class SourceFilter(nn.Module):
    """Harmonic-plus-noise synthesis controlled per frame by two linear heads,
    then a small post-filter network.

    Harmonics above a predicted per-frame cutoff (classic "maximum voiced
    frequency") get random phase: real voices turn aperiodic in the upper
    bands, and perfectly coherent high harmonics are what sounds buzzy."""

    def __init__(self, c):
        super().__init__()
        self.config = c
        self.harmonic_steps, self.noise_steps = c.hop // c.synth_hop, c.hop // c.noise_hop
        # Outputs: causal + anticausal cepstrum, one gain per sub-frame, the cutoff logit.
        self.harmonic = nn.Linear(c.channels, c.cep_causal + c.cep_anti + self.harmonic_steps + 1)
        self.noise = nn.Linear(c.channels, c.noise_cep + self.noise_steps)
        for head, level in ((self.harmonic, -1.5), (self.noise, -2.5)):   # start quiet and spectrally flat
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            head.bias.data[0] = level
        self.harmonic.bias.data[-1] = 2.0   # cutoff starts high: ~88% of the harmonic band stays periodic
        self.postfilter = PostFilter(c)
        bins = torch.arange(c.synth_fft // 2 + 1, dtype=torch.float32) * c.sample_rate / c.synth_fft
        self.register_buffer("bin_hz", bins, persistent=False)

        def lifter(size, first):
            scale = 1.0 / (1.0 + torch.arange(first, first + size, dtype=torch.float32) / 4)
            if first == 0:
                scale[0] = 2.0      # coefficient 0 is the log gain and needs range
            return scale
        self.register_buffer("causal_lifter", lifter(c.cep_causal, 0), persistent=False)
        self.register_buffer("anti_lifter", lifter(c.cep_anti, 1), persistent=False)
        self.register_buffer("noise_lifter", lifter(c.noise_cep, 0), persistent=False)
        self.register_buffer("synth_window", torch.hann_window(c.synth_fft), persistent=False)
        self.register_buffer("noise_window", torch.hann_window(c.noise_fft), persistent=False)

    @staticmethod
    def _take(values, index):
        return values.gather(1, index[..., None].expand(-1, -1, values.shape[-1]))

    def _subframes(self, frames, lengths, steps):
        """Sub-frame j uses frame j // steps; indices past an utterance's end
        repeat its last sub-frame so padding never changes real audio. Two extra
        sub-frames cover the half window of zeros appended to the source."""
        sub = torch.arange(steps * frames + 3, device=lengths.device)[None].expand(len(lengths), -1)
        return torch.minimum(sub, (steps * lengths - 1).clamp_min(0)[:, None])

    @staticmethod
    def _shape(source, response, fft, hop, window):
        # Half a window of trailing zeros: the frames around an utterance's end are
        # then the same whether it stands alone or is padded inside a batch.
        return filtered(F.pad(source, (0, fft // 2)), response, fft, hop, window)[:, :source.shape[-1]]

    def forward(self, hidden, f0, lengths, noise=None, phase=None):
        c = self.config
        frames = hidden.shape[1]
        samples = frames * c.hop
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            audible = length_mask(lengths * c.hop, samples)
            last = (lengths - 1).clamp_min(0)[:, None]
            # Harmonic branch: envelope interpolated between frames, one gain per sub-frame.
            h = self.harmonic(hidden)
            split = c.cep_causal + c.cep_anti
            sub = self._subframes(frames, lengths, self.harmonic_steps)
            here, following = sub // self.harmonic_steps, torch.minimum((sub + 1) // self.harmonic_steps, last)
            causal = 0.5 * (self._take(h[..., :c.cep_causal], here) + self._take(h[..., :c.cep_causal], following)) * self.causal_lifter
            anti = 0.5 * (self._take(h[..., c.cep_causal:split], here) + self._take(h[..., c.cep_causal:split], following)) * self.anti_lifter
            gains = h[..., split:split + self.harmonic_steps]
            causal = torch.cat((causal[..., :1] + gains.flatten(1).gather(1, sub)[..., None], causal[..., 1:]), dim=-1)
            harmonic_response = cepstrum_response(causal, anti, c.synth_fft)
            # Above the cutoff the harmonics take a random phase per STFT bin and
            # sub-frame. The phases come from the noise stream with every other
            # sample negated (its spectrum mirrored), so they are reproducible from
            # the seed, independent of padding, and uncorrelated with the noise branch.
            cutoff = torch.sigmoid(0.5 * (h[..., -1:].gather(1, here[..., None]) + h[..., -1:].gather(1, following[..., None])))
            periodic = torch.sigmoid((cutoff * c.harmonic_limit * c.sample_rate - self.bin_hz) / c.aperiodic_width_hz)
            if noise is None:
                noise = torch.randn(hidden.shape[0], samples, device=hidden.device)
            sign = 1 - 2 * (torch.arange(samples, device=hidden.device) % 2)
            scramble = torch.stft(F.pad(noise * audible * sign, (0, c.synth_fft // 2)), c.synth_fft, c.synth_hop,
                                  window=self.synth_window, center=True, pad_mode="constant", return_complex=True)
            scramble = torch.polar(torch.ones_like(scramble.real), scramble.angle()).transpose(1, 2)
            harmonic_response = harmonic_response * (periodic + (1 - periodic) * scramble)
            # Noise branch: one envelope per frame, four gains per frame for sharp consonants.
            n = self.noise(hidden)
            sub = self._subframes(frames, lengths, self.noise_steps)
            shape = self._take(n[..., :c.noise_cep], sub // self.noise_steps) * self.noise_lifter
            shape = torch.cat((shape[..., :1] + n[..., c.noise_cep:].flatten(1).gather(1, sub)[..., None], shape[..., 1:]), dim=-1)
            noise_response = cepstrum_response(shape, None, c.noise_fft)
            # Hold the last real pitch value through padding: the pitch glide inside an
            # utterance's final frame must not depend on what the batch was padded with.
            f0 = f0.float().gather(1, torch.minimum(torch.arange(frames, device=f0.device)[None], last))
            pulses = pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit, phase) * audible
            wave = (self._shape(pulses, harmonic_response, c.synth_fft, c.synth_hop, self.synth_window)
                    + self._shape(noise * audible, noise_response, c.noise_fft, c.noise_hop, self.noise_window)) * audible
            return self.postfilter(wave, hidden, lengths) * audible


class SpectralGenerator(nn.Module):
    """iSTFT waveform generator at the STFT frame rate (after Kokoro's iSTFTNet
    decoder, shrunk): a ConvNeXt stack over frames reads the harmonic source's
    spectrum and the acoustic state, and writes each bin as

        harmonic source x complex gain + noise x complex gain + complex addition

    then an inverse STFT gives the waveform. The harmonic term keeps the
    oscillator's pitch-coherent phase, the noise term gives breath and frication,
    and the addition lets it create what neither source has. No fixed cepstral
    envelopes, and no audio-rate network, so it stays cheap on CPU.
    """

    def __init__(self, c):
        super().__init__()
        self.config = c
        bins = c.synth_fft // 2 + 1
        self.input = nn.Linear(bins, c.istft_channels)
        self.condition = nn.Linear(c.channels, c.istft_channels)
        self.blocks = nn.ModuleList(PostFilterBlock(c.istft_channels, c.kernel) for _ in range(c.istft_layers))
        self.output = nn.Linear(c.istft_channels, 6 * bins)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.output.bias.data[:bins] = -1.5             # harmonic gain: start quiet
        self.output.bias.data[2 * bins:3 * bins] = -3.0  # noise gain: quieter
        self.register_buffer("window", torch.hann_window(c.synth_fft), persistent=False)

    def _spectrum(self, wave):
        c = self.config
        return torch.stft(F.pad(wave, (0, c.synth_fft // 2)), c.synth_fft, c.synth_hop, window=self.window,
                          center=True, pad_mode="constant", return_complex=True).transpose(1, 2)

    def forward(self, hidden, f0, lengths, noise=None, phase=None):
        c = self.config
        frames = hidden.shape[1]
        samples = frames * c.hop
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            audible = length_mask(lengths * c.hop, samples)
            last = (lengths - 1).clamp_min(0)[:, None]
            f0 = f0.float().gather(1, torch.minimum(torch.arange(frames, device=f0.device)[None], last))
            if noise is None:
                noise = torch.randn(hidden.shape[0], samples, device=hidden.device)
            harmonic = self._spectrum(pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit, phase) * audible)
            noisy = self._spectrum(noise * audible)
            mask, owner = stft_frames(lengths, c.hop // c.synth_hop, harmonic.shape[1])
            level = (torch.log(harmonic.abs() + 1e-5) + 5) / 5
            x = (self.input(level) + self.condition(take_frames(hidden, owner))) * mask[..., None]
            for block in self.blocks:
                x = block(x, mask)
            h_gain, h_turn, n_gain, n_turn, real, imag = self.output(x).float().chunk(6, dim=-1)
            spectrum = (harmonic * torch.polar(torch.exp(h_gain.clamp(-10, 4)), math.pi * torch.tanh(h_turn))
                        + noisy * torch.polar(torch.exp(n_gain.clamp(-10, 4)), math.pi * torch.tanh(n_turn))
                        + torch.complex(real, imag)) * mask[..., None]
            wave = torch.istft(spectrum.transpose(1, 2), c.synth_fft, c.synth_hop, window=self.window, center=True,
                               length=samples + c.synth_fft // 2)[:, :samples]
            return wave * audible


class NSFBlock(nn.Module):
    """Dilated convolutions over the waveform, residual. Dense rather than
    depthwise-separable: at audio rate the dense kernels are ~6x faster on CPU."""

    def __init__(self, channels, dilations):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv1d(channels, channels, 3, dilation=d, padding=d) for d in dilations)

    def forward(self, x, condition, audible):
        x = (x + condition) * audible
        for layer in self.layers:
            x = (x + F.gelu(layer(x))) * audible
        return x


class NeuralSourceFilter(nn.Module):
    """Neural source-filter waveform model (after Wang & Yamagishi's NSF).

    The source is deterministic signal processing: sines at the first harmonics
    of the predicted pitch, the band-limited pulse train, and white noise. The
    filter is a small stack of dilated convolutions running at the audio rate,
    conditioned on the acoustic state, that turns the source into speech. It
    filters the waveform directly, so there is no frame-wise STFT filtering of a
    pulse train (whose overlap-add sounds metallic) and no fixed harmonic or
    noise envelope: periodicity, breathiness and pulse shape are all learned.
    """

    def __init__(self, c):
        super().__init__()
        self.config = c
        sources = c.nsf_harmonics + 2
        self.source = nn.Conv1d(sources, c.nsf_channels, 1)
        self.condition = nn.Linear(c.channels, c.nsf_channels * c.nsf_blocks)
        dilations = [2 ** i for i in range(c.nsf_layers)]
        self.blocks = nn.ModuleList(NSFBlock(c.nsf_channels, dilations) for _ in range(c.nsf_blocks))
        self.output = nn.Conv1d(c.nsf_channels, 1, 1)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)
        self.postfilter = PostFilter(c) if c.postfilter_layers else None
        self.register_buffer("orders", torch.arange(1, c.nsf_harmonics + 1, dtype=torch.float64), persistent=False)

    def forward(self, hidden, f0, lengths, noise=None, phase=None):
        c = self.config
        frames = hidden.shape[1]
        samples = frames * c.hop
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            audible = length_mask(lengths * c.hop, samples)[:, None].float()
            last = (lengths - 1).clamp_min(0)[:, None]
            # Hold the last real frame through padding, so interpolation inside an
            # utterance's final frame never depends on what the batch was padded with.
            held = torch.minimum(torch.arange(frames, device=f0.device)[None], last)
            f0 = f0.float().gather(1, held)
            hidden = hidden.gather(1, held[..., None].expand(-1, -1, hidden.shape[-1]))
            theta, frequency = phase_track(f0, c.sample_rate, c.hop, phase)
            # Harmonics above the band limit are silent (no aliasing as pitch rises).
            below = (frequency[..., None] * self.orders < c.harmonic_limit * c.sample_rate).float()
            sines = (torch.sin(theta[..., None] * self.orders).float() * below).transpose(1, 2)
            pulses = pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit, phase)[:, None]
            if noise is None:
                noise = torch.randn(hidden.shape[0], samples, device=hidden.device)
            x = self.source(torch.cat((sines, pulses, noise[:, None]), dim=1) * audible)
            conditions = upsample(self.condition(hidden), c.hop).transpose(1, 2).chunk(c.nsf_blocks, dim=1)
            for block, condition in zip(self.blocks, conditions):
                if self.training:
                    # Audio-rate activations are large; recompute them in the backward pass.
                    x = torch.utils.checkpoint.checkpoint(block, x, condition, audible, use_reentrant=False)
                else:
                    x = block(x, condition, audible)
            wave = self.output(x)[:, 0] * audible[:, 0]
            if self.postfilter is not None:
                wave = self.postfilter(wave, hidden, lengths) * audible[:, 0]
            return wave


def pitch_features(f0, reference):
    """[..., 17] encoding of F0 in Hz: log(F0/reference) and sines of it at 8 scales,
    so a frame-rate network can place harmonics precisely."""
    x = torch.log(f0.float() / reference)[..., None]
    angle = x * (math.pi * 2.0 ** torch.arange(8, device=f0.device))
    return torch.cat((x, angle.sin(), angle.cos()), dim=-1)


class Vocos(nn.Module):
    """Vocos-style head (Siuzdak 2023): a ConvNeXt stack at the acoustic frame
    rate writes every bin's log magnitude and phase angle of a 1024/256 STFT,
    then an inverse STFT. No oscillator and no source: the waveform's pulse
    shape and phase are free, and learned from the spectral losses and the
    critics. F0 is an input because the decoder is not always conditioned on
    the recorded pitch the target audio has.
    """

    def __init__(self, c):
        super().__init__()
        self.config = c
        bins = c.n_fft // 2 + 1
        self.condition = nn.Linear(c.channels, c.istft_channels)
        self.pitch = nn.Linear(17, c.istft_channels)
        self.norm_in = nn.LayerNorm(c.istft_channels)
        self.blocks = nn.ModuleList(PostFilterBlock(c.istft_channels, c.kernel) for _ in range(c.istft_layers))
        self.norm = nn.LayerNorm(c.istft_channels)
        self.output = nn.Linear(c.istft_channels, 2 * bins)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)
        self.output.bias.data[:bins] = -2.0      # start quiet
        self.register_buffer("window", torch.hann_window(c.n_fft), persistent=False)

    def forward(self, hidden, f0, lengths, noise=None, phase=None):
        c = self.config
        frames = hidden.shape[1]
        samples = frames * c.hop
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            # Frames centred on i*hop, plus three past the end for the trailing half window.
            mask, owner = stft_frames(lengths, 1, frames + 3)
            x = self.condition(take_frames(hidden.float(), owner)) + self.pitch(pitch_features(take_frames(f0.float()[..., None], owner)[..., 0], c.pitch_reference_hz))
            x = self.norm_in(x) * mask[..., None]
            for block in self.blocks:
                x = block(x, mask)
            level, angle = self.output(self.norm(x)).chunk(2, dim=-1)
            spectrum = torch.polar(torch.exp(level.clamp(max=7.0)), angle) * mask[..., None]
            wave = torch.istft(spectrum.transpose(1, 2), c.n_fft, c.hop, window=self.window, center=True,
                               length=samples + c.n_fft // 2)[:, :samples]
            return wave * length_mask(lengths * c.hop, samples)


class PitchPredictive(nn.Module):
    """Frame-wise autoregressive head with pitch prediction (after Valin et al.'s
    FARGAN, 2024), at 24 kHz.

    Audio is written 64 samples at a time by three GRUs. Each step sees its
    conditioning, the previous 64 samples, white noise, and the "pitch
    prediction": the output one pitch period back (repeated when the period is
    shorter than the step, fractional lags interpolated). The network then only
    has to describe how this period differs from the last one, which is how real
    voices vary from cycle to cycle; an oscillator-driven head repeats an
    idealized pulse instead. Inputs and output are normalized by a predicted
    per-step gain.

    Sequential over steps, so it is trained on crops (voca.revocode), not by
    voca.train on whole utterances. ``history`` primes the lookback buffer with
    the recording before a training crop; at inference it starts silent.
    """

    HISTORY = 1024      # samples of lookback (longest period at 50 Hz: 480)

    def __init__(self, c):
        super().__init__()
        self.config = c
        s, h, k = c.fargan_subframe, c.fargan_channels, c.fargan_condition
        if c.hop % s:
            raise ValueError("fargan_subframe must divide the hop")
        self.steps = c.hop // s
        self.frame = nn.Linear(c.channels + 17, 256)
        self.context = nn.Conv1d(256, 256, 3, padding=1)
        self.subframes = nn.Linear(256, self.steps * k)
        self.gain = nn.Linear(k, 1)
        nn.init.zeros_(self.gain.weight)
        nn.init.constant_(self.gain.bias, math.log(0.3))
        self.mix = nn.Linear(k + 3 * s, h)
        self.grus = nn.ModuleList(nn.GRUCell(h + k, h) for _ in range(3))
        self.gates = nn.ModuleList(nn.Linear(h, h) for _ in range(3))
        self.skip = nn.Linear(4 * h + s, h)
        self.output = nn.Linear(h, s)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden, f0, lengths, noise=None, phase=None, history=None):
        c = self.config
        s = c.fargan_subframe
        batch, frames = hidden.shape[:2]
        samples = frames * c.hop
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            last = (lengths - 1).clamp_min(0)[:, None]
            held = torch.minimum(torch.arange(frames, device=f0.device)[None], last)
            f0 = f0.float().gather(1, held)
            hidden = hidden.gather(1, held[..., None].expand(-1, -1, hidden.shape[-1]))
            # Zero past the end, so the context convolution sees the same padding batched or alone.
            x = F.gelu(self.frame(torch.cat((hidden, pitch_features(f0, c.pitch_reference_hz)), dim=-1))) * length_mask(lengths, frames)[..., None]
            x = F.gelu(self.context(x.transpose(1, 2)).transpose(1, 2))
            conditions = self.subframes(x).view(batch, frames * self.steps, -1)
            gains = torch.exp(self.gain(conditions).clamp(-12, 2))
            # Pitch period in samples at each step's centre (log-F0 interpolated between frames).
            centre = torch.arange(frames * self.steps, device=f0.device) * s + s // 2
            log_f0 = upsample(torch.log(f0)[..., None], c.hop)[:, centre, 0]
            periods = c.sample_rate * torch.exp(-log_f0)
            if noise is None:
                noise = torch.randn(batch, samples, device=hidden.device)
            audible = length_mask(lengths * c.hop, samples).float()
            buffer = hidden.new_zeros(batch, self.HISTORY) if history is None else history.float()
            states = [hidden.new_zeros(batch, c.fargan_channels) for _ in self.grus]
            offsets = torch.arange(s, device=f0.device, dtype=torch.float32)
            outputs = []
            for j in range(frames * self.steps):
                gain, period, condition = gains[:, j], periods[:, j:j + 1], conditions[:, j]
                # One period back, wrapped into the past while the period is shorter than the step.
                position = self.HISTORY + offsets - (torch.floor(offsets / period) + 1) * period
                low = position.floor()
                weight = position - low
                low = low.long().clamp(0, self.HISTORY - 1)
                high = (low + 1).clamp(max=self.HISTORY - 1)
                predicted = ((buffer.gather(1, low) * (1 - weight) + buffer.gather(1, high) * weight) / gain).clamp(-20, 20)
                previous = (buffer[:, -s:] / gain).clamp(-20, 20)
                y = torch.tanh(self.mix(torch.cat((condition, previous, predicted, noise[:, j * s:(j + 1) * s]), dim=-1)))
                layers = [y]
                for k, (gru, gate) in enumerate(zip(self.grus, self.gates)):
                    states[k] = gru(torch.cat((layers[-1], condition), dim=-1), states[k])
                    layers.append(states[k] * torch.sigmoid(gate(states[k])))
                y = torch.tanh(self.output(torch.tanh(self.skip(torch.cat((*layers, predicted), dim=-1))))) * gain
                y = y * audible[:, j * s:(j + 1) * s]
                buffer = torch.cat((buffer[:, s:], y), dim=1)
                outputs.append(y)
            return torch.cat(outputs, dim=1)


VOCODERS = {"nsf": NeuralSourceFilter, "istft": SpectralGenerator, "source_filter": SourceFilter, "vocos": Vocos,
            "fargan": PitchPredictive}


class Voca(nn.Module):
    """What both architectures share: phoneme, lexical-feature and voice
    embeddings, the transformer text encoder, the token tables, and the
    synthesis-time rules (pause gating and floors, voiced vowels, pitch range and wobble)."""

    def __init__(self, config):
        super().__init__()
        self.config = c = config
        self.tokens = nn.Embedding(c.vocab_size, c.channels, padding_idx=0)
        add_lexical_embeddings(self, c.channels)
        self.voices = nn.Embedding(len(c.speakers), c.channels)
        self.front = Front(c, c.front_layers) if c.front_layers else None
        self.encoder = Stack(c, c.encoder_layers)
        self.register_buffer("phone_tokens", PHONE_TABLE.clone(), persistent=False)
        self.register_buffer("phrase_breaks", BREAK_TABLE.clone(), persistent=False)
        voiced = torch.zeros(c.vocab_size, dtype=torch.bool)
        voiced[[SYMBOL_TO_ID[v] for v in ALWAYS_VOICED]] = True
        self.register_buffer("voiced_tokens", voiced, persistent=False)
        # Punctuation that gets an inference-time minimum pause: 1 phrase-internal, 2 sentence end.
        kind = torch.zeros(c.vocab_size, dtype=torch.long)
        kind[[SYMBOL_TO_ID[p] for p in (",", ";", ":", "—")]] = 1
        kind[[SYMBOL_TO_ID[p] for p in (".", "?", "!", "…")]] = 2
        self.register_buffer("pause_kind", kind, persistent=False)

    def encode(self, tokens, lexical, lengths, speakers):
        if (tokens.ndim != 2 or lexical.shape != (*tokens.shape, len(LEXICAL)) or lengths.shape != (tokens.shape[0],)
                or speakers.shape != lengths.shape):
            raise ValueError(f"expected tokens[B,N], lexical[B,N,{len(LEXICAL)}], lengths[B], speakers[B]")
        # Value checks wait for the GPU; training data was validated when it was loaded.
        if tokens.shape[1] > self.config.max_tokens or not self.training and bool(((lengths < 1) | (lengths > tokens.shape[1])).any()):
            raise ValueError("invalid token length")
        mask = length_mask(lengths, tokens.shape[1])
        x = (embed_tokens(self, tokens, lexical) + self.voices(speakers)[:, None]) * mask[..., None]
        if self.front is not None:
            x = self.front(x, mask)
        return self.encoder(x, mask), mask

    def words(self, tokens, mask):
        return word_layout(tokens, mask, self.phone_tokens, self.phrase_breaks)

    def align(self, durations, token_mask, frames=None):
        """Integer durations (zero is allowed for pause tokens) -> the token each
        frame belongs to [B, F], the frame mask and the frame counts. ``frames``:
        the longest utterance's frame count when the caller knows it (no GPU wait)."""
        if durations.shape != token_mask.shape:
            raise ValueError("invalid durations")
        lengths = durations.sum(1)
        if not self.training:
            if bool((durations < 0).any()) or bool((durations[~token_mask] != 0).any()):
                raise ValueError("invalid durations")
            if bool((lengths < 1).any()):
                raise ValueError("utterance has no frames")
            if bool((lengths > self.config.max_frames).any()):
                raise UtteranceTooLong("utterance exceeds max_frames; split the input")
        size = int(lengths.max()) if frames is None else frames
        frame = torch.arange(size, device=durations.device)[None].expand(durations.shape[0], size).contiguous()
        index = torch.searchsorted(durations.cumsum(1), frame, right=True).clamp(max=durations.shape[1] - 1)
        return index, frame < lengths[:, None], lengths

    def gate_pauses(self, frames, presence, tokens, mask):
        """Every phoneme is audible for at least one frame. A pause-capable token
        (edge, punctuation, word boundary) lasts its predicted length when the
        presence logit says there is a pause, and vanishes otherwise."""
        present = self.phone_tokens[tokens] | (presence > 0)
        return torch.where(present, frames.clamp_min(1), torch.zeros_like(frames)) * mask

    def pause_floors(self, durations, tokens, lengths, mask, comma_pause, stop_pause):
        """Minimum pauses (seconds) at punctuation inside the utterance. The
        teacher barely pauses at commas, and learners need the phrasing. A run of marks (``?!``, ``--``) shares
        one minimum, on its first mark; the marks before EOS get none: trailing silence is just latency. A dash
        gets one only where the model pauses there: the same mark joins words (``pimple-faced``)."""
        kind = self.pause_kind[tokens]
        kind = torch.where((tokens == SYMBOL_TO_ID["—"]) & (durations == 0), 0, kind) * mask
        marked = kind > 0
        first = marked & ~F.pad(marked[:, :-1], (1, 0))
        run = first.long().cumsum(1) - 1
        size = tokens.shape[1]
        slot = torch.where(marked, run, size)     # tokens that are no mark: a slot of their own
        total = torch.zeros(len(tokens), size + 1, device=tokens.device, dtype=durations.dtype).scatter_add(1, slot, durations)
        stops = torch.zeros_like(total).scatter_add(1, slot, (kind == 2).long())
        seconds = torch.where(stops > 0, stop_pause, comma_pause)
        minimum = torch.ceil(seconds * self.config.sample_rate / self.config.hop).long()
        position = torch.arange(size, device=tokens.device)
        last_phone = torch.where(mask & self.phone_tokens[tokens], position, -1).amax(1)
        final = position > last_phone[:, None]
        deficit = (minimum - total).clamp_min(0).gather(1, slot)
        return durations + torch.where(first & ~final, deficit, 0).to(durations.dtype)

    def voice_vowels(self, prosody, tokens, owner, frame_mask):
        """The pitch tracker missed creaky vowels in the recordings, and the
        voicing predictor learned to drop whole vowels (heard as a whispered
        word). Frames of the inventory's always-voiced phones (in English, every
        vowel; a language's choice, see scripts/build_inventory.py) are kept voiced."""
        vowel = self.voiced_tokens[tokens].gather(1, owner) & frame_mask
        return torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(4.0), prosody[..., 1])[..., None],
                          prosody[..., 2:]), dim=-1)

    def remove_pitch_spikes(self, prosody, tokens, owner):
        """Replace isolated peaks above 9 semitones, or 3 within one always-voiced phone."""
        p = prosody[..., 0]
        left, mid, right = p[:, :-2], p[:, 1:-1], p[:, 2:]
        vowel = self.voiced_tokens[tokens].gather(1, owner)[:, 1:-1]
        inside = vowel & (owner[:, :-2] == owner[:, 2:])
        limit = torch.where(inside, 3., 9.) * (math.log(2) / 12)
        spike = (mid - left > limit) & (mid - right > limit)
        correction = F.pad(torch.where(spike, (left + right) / 2 - mid, 0.), (1, 1))
        return torch.cat(((p + correction[:, :p.shape[1]])[..., None], prosody[..., 1:]), dim=-1)

    @staticmethod
    def check_controls(speed, pitch_shift, pitch_range):
        if not math.isfinite(speed) or not 0.25 <= speed <= 4 or not math.isfinite(pitch_shift) or not 0 <= pitch_range <= 3:
            raise ValueError("speed must be between 0.25 and 4, pitch_shift finite and pitch_range between 0 and 3")

    @staticmethod
    def prosody_condition(prediction):
        """What the decoder (and the source's voicing) is conditioned on when it is not teacher-forced.
        Voicing is the probability, not a decision: where the predictor is unsure (voiced
        consonants, vowel-fricative boundaries) the source crossfades harmonics and noise
        instead of cutting the harmonics and raising the noise 30 dB within a frame."""
        return torch.stack((prediction[..., 0].clamp(-2.0, 1.5), torch.sigmoid(prediction[..., 1]),
                            prediction[..., 2].clamp(-4.5, 3.0)), dim=-1)

    def frequency(self, log_f0):
        c = self.config
        return (c.pitch_reference_hz * torch.exp(log_f0.float())).clamp(c.f0_min, c.f0_max)

    @staticmethod
    def widen_pitch(prosody, frame_mask, pitch_range):
        """Scale the predicted pitch movement around each utterance's voiced mean.
        Regression predicts a flatter contour than the recordings; values above 1
        restore range (1.0 leaves the prediction untouched)."""
        if isinstance(pitch_range, (int, float)) and pitch_range == 1.0:
            return prosody
        voiced = ((prosody[..., 1] > 0) & frame_mask).to(prosody.dtype)
        mean = (prosody[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
        pitch = prosody[..., 0] + (prosody[..., 0] - mean) * (pitch_range - 1)
        return torch.cat((pitch[..., None], prosody[..., 1:]), dim=-1) * frame_mask[..., None]

    @staticmethod
    def pitch_wobble(shape, semitones, seed, device):
        """Small random pitch movement for the oscillator only (log-F0 units).
        Recorded pitch has cycle-to-cycle irregularity that a predicted contour
        lacks; two thirds of the wobble is slow drift, one third frame-level."""
        generator = torch.Generator().manual_seed(seed + 7919)
        fast = torch.randn(shape, generator=generator)
        slow = F.avg_pool1d(torch.randn(shape[0], 1, shape[1] + 16, generator=generator), 17, stride=1)[:, 0]
        slow = slow / slow.std().clamp_min(1e-6)
        return ((2 * slow + fast) / 5 ** 0.5 * semitones * math.log(2) / 12).to(device)


class TinyVoca(Voca):
    PARAMETER_BUDGET = 8_000_000    # tiny-voca kept 5M; post-filter or spectral generator add 1-3M

    def __init__(self, config=Config()):
        super().__init__(config)
        c = config
        # Outputs per token: log(1 + frames) and, for pause-capable tokens, a
        # "has frames at all" logit. A pause is absent more often than present, so
        # one regression would average both cases into a pause that is too short.
        self.duration = Predictor(c, 2)
        self.frame_voices = nn.Embedding(len(c.speakers), c.channels)
        self.frame_position = nn.Linear(3, c.channels)
        self.prosody_stack = Stack(c, c.prosody_layers)
        # Outputs: continuous log(F0/reference), voiced logit, normalized log RMS energy.
        self.prosody_head = nn.Linear(c.channels, 3)
        self.prosody_projection = nn.Linear(3, c.channels)
        self.acoustic_stack = Stack(c, c.acoustic_layers)
        self.vocoder = VOCODERS[c.waveform](c)

    def expand(self, hidden, durations, token_mask, speakers, frames=None):
        """Repeat token states by integer durations and add each frame's position inside its token."""
        index, mask, lengths = self.align(durations, token_mask, frames)
        features = position_in_token(durations, index, torch.arange(index.shape[1], device=hidden.device)[None]).to(hidden.dtype)
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + self.frame_position(features) + self.frame_voices(speakers)[:, None]
        return expanded * mask[..., None], mask, lengths

    def predict_prosody(self, frames, frame_mask):
        hidden = self.prosody_stack(frames, frame_mask)
        return hidden, self.prosody_head(hidden) * frame_mask[..., None]

    def decode(self, prosody_hidden, condition, frame_mask):
        return self.acoustic_stack((prosody_hidden + self.prosody_projection(condition)) * frame_mask[..., None], frame_mask)

    def predict_durations(self, hidden, mask, tokens, speed=1.0):
        output = self.duration(hidden, mask).float()
        frames = torch.round(torch.expm1(output[..., 0].clamp(0, math.log1p(400))) / speed).long()
        return self.gate_pauses(frames, output[..., 1], tokens, mask)

    @torch.inference_mode()
    def synthesize(self, tokens, lexical, lengths, speakers, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, pitch_jitter=0.0,
                   voiced_vowels=True, comma_pause=0.0, stop_pause=0.0, take=0):
        """Free-running synthesis: predicted durations, pitch, voicing and energy (``take``: there is only one delivery)."""
        self.check_controls(speed, pitch_shift, pitch_range)
        hidden, mask = self.encode(tokens, lexical, lengths, speakers)
        durations = self.pause_floors(self.predict_durations(hidden, mask, tokens, speed), tokens, lengths, mask, comma_pause, stop_pause)
        frames, frame_mask, frame_lengths = self.expand(hidden, durations, mask, speakers)
        prosody_hidden, prosody = self.predict_prosody(frames, frame_mask)
        if voiced_vowels:
            prosody = self.voice_vowels(prosody, tokens, self.align(durations, mask)[0], frame_mask)
        prosody = self.widen_pitch(prosody, frame_mask, pitch_range)
        acoustic = self.decode(prosody_hidden, self.prosody_condition(prosody), frame_mask)
        # One noise stream per row, drawn at the row's own length, so batching or
        # padding never changes a row's audio.
        noise = nn.utils.rnn.pad_sequence([torch.randn(int(n) * self.config.hop, generator=torch.Generator().manual_seed(seed + row))
                                           for row, n in enumerate(frame_lengths)], batch_first=True)
        pitch = prosody[..., 0] + pitch_shift
        if pitch_jitter > 0:
            pitch = pitch + self.pitch_wobble(pitch.shape, pitch_jitter, seed, pitch.device)
        wave = self.vocoder(acoustic, self.frequency(pitch), frame_lengths, noise=noise.to(hidden.device))
        return wave, frame_lengths * self.config.hop, {"durations": durations, "prosody": prosody, "frame_mask": frame_mask,
                                                       "acoustic": acoustic, "frame_lengths": frame_lengths}


def build_model(config):
    """The model a configuration describes (``config.architecture``)."""
    if config.architecture == "styletts":
        from .styletts import StyleVoca
        return StyleVoca(config)
    return TinyVoca(config)

"""StyleTTS 2 / Kokoro architecture at tiny-voca scale (``architecture="styletts"``).

Ported from StyleTTS 2 (github.com/yl4579/StyleTTS2, MIT: models.py,
Modules/istftnet.py) and Kokoro (github.com/hexgrad/kokoro, Apache-2.0:
kokoro/modules.py, kokoro/istftnet.py); see voca/LICENSE-styletts2. Trained
from scratch on our phonemes, no pretrained weights.

    phonemes -> transformer text encoder (ours, shared with TinyVoca)
        -> [bert_encoder] -> prosody predictor (LSTMs with AdaLayerNorm on the
           prosodic style): durations, then per frame pitch / voicing / energy
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
* The F0 branch predicts log(F0/reference) and a voicing logit. The decoder
  sees F0 / reference (0 when unvoiced) instead of raw Hz, the same curve at unit scale.
* Inference uses one stored style per voice: the running mean of the training
  clips' styles. The style encoders are training-only (voca.train).
* The generator's rates multiply to 256 instead of 300 (default 8 x 8 x iSTFT hop 4,
  Kokoro: 10 x 6 x 5).
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from .config import Config
from .model import Voca


def run_lstm(lstm, x, lengths):
    """Bidirectional LSTM over [B, T, C] that stops at each row's length."""
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    return nn.utils.rnn.pad_packed_sequence(lstm(packed)[0], batch_first=True, total_length=x.shape[1])[0]


def frames_of(values, index):
    """values [B, N, C] per token, index [B, F] owning token -> [B, F, C] (``t_en @ alignment``)."""
    return values.gather(1, index[..., None].expand(-1, -1, values.shape[-1]))


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        gamma, beta = self.fc(s)[..., None].chunk(2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


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

    def forward(self, x, s):
        y = self.conv1(self.dropout(F.leaky_relu(self.norm1(x, s), 0.2)))
        y = self.conv2(self.dropout(F.leaky_relu(self.norm2(y, s), 0.2)))
        return (y + (x if self.conv1x1 is None else self.conv1x1(x))) * math.sqrt(0.5)


class AdaINResBlock1(nn.Module):
    """Kokoro's generator block: dilated convolutions with AdaIN and Snake activations."""

    def __init__(self, channels, kernel, dilations, style_dim):
        super().__init__()
        self.convs1 = nn.ModuleList(weight_norm(nn.Conv1d(channels, channels, kernel, dilation=d, padding=d * (kernel - 1) // 2)) for d in dilations)
        self.convs2 = nn.ModuleList(weight_norm(nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)) for _ in dilations)
        for conv in (*self.convs1, *self.convs2):
            nn.init.normal_(conv.weight, 0.0, 0.01)
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
    """LSTMs over tokens, each followed by AdaLayerNorm; the style is appended to every layer's input."""

    def __init__(self, style_dim, channels, layers):
        super().__init__()
        self.lstms = nn.ModuleList(nn.LSTM(channels + style_dim, channels // 2, batch_first=True, bidirectional=True) for _ in range(layers))
        self.norms = nn.ModuleList(AdaLayerNorm(style_dim, channels) for _ in range(layers))

    def forward(self, x, style, lengths, mask):
        s = style[:, None].expand(-1, x.shape[1], -1)
        x = torch.cat((x, s), dim=-1) * mask[..., None]
        for lstm, norm in zip(self.lstms, self.norms):
            x = torch.cat((norm(run_lstm(lstm, x, lengths), style), s), dim=-1) * mask[..., None]
        return x


class ProsodyPredictor(nn.Module):
    """StyleTTS 2's prosody predictor. Durations: ``max_duration`` sigmoid bins
    (frames = their sum) plus our pause-presence logit. Per frame: log(F0 /
    reference), voicing logit (F0 branch) and normalized log energy (N branch)."""

    def __init__(self, c):
        super().__init__()
        d, s = c.style_prosody_channels, c.style_dim
        self.text_encoder = DurationEncoder(s, d, c.style_prosody_layers)
        self.lstm = nn.LSTM(d + s, d // 2, batch_first=True, bidirectional=True)
        self.duration_proj = nn.Linear(d, c.style_max_duration + 1)
        self.shared = nn.LSTM(d + s, d // 2, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList([AdainResBlk1d(d, d, s, c.dropout), AdainResBlk1d(d, d // 2, s, c.dropout), AdainResBlk1d(d // 2, d // 2, s, c.dropout)])
        self.N = nn.ModuleList([AdainResBlk1d(d, d, s, c.dropout), AdainResBlk1d(d, d // 2, s, c.dropout), AdainResBlk1d(d // 2, d // 2, s, c.dropout)])
        self.F0_proj = nn.Conv1d(d // 2, 2, 1)
        self.N_proj = nn.Conv1d(d // 2, 1, 1)

    def forward(self, text, style, lengths, mask):
        """text [B, N, d] -> token states [B, N, d + s] (to be repeated per frame), duration logits [B, N, bins + 1]."""
        d = self.text_encoder(text, style, lengths, mask)
        return d, self.duration_proj(run_lstm(self.lstm, d, lengths))

    def F0Ntrain(self, frames, style, lengths):
        """Repeated token states [B, F, d + s] -> [B, F, 3] pitch, voicing logit, energy."""
        x = run_lstm(self.shared, frames, lengths).transpose(1, 2)
        f0, n = x, x
        for block in self.F0:
            f0 = block(f0, style)
        for block in self.N:
            n = block(n, style)
        return torch.cat((self.F0_proj(f0), self.N_proj(n)), dim=1).transpose(1, 2)


class SourceModuleHnNSF(nn.Module):
    """Harmonic source of hn-NSF (Wang & Yamagishi) as in Kokoro: sines at F0 and
    its first harmonics (phase integrated at the frame rate, then interpolated),
    noise where unvoiced, merged by one linear layer and tanh."""

    def __init__(self, sample_rate, upsample_scale, harmonics=8, sine_amp=0.1, noise_std=0.003, voiced_threshold=10):
        super().__init__()
        self.sample_rate, self.upsample_scale, self.sine_amp, self.noise_std, self.voiced_threshold = (
            sample_rate, upsample_scale, sine_amp, noise_std, voiced_threshold)
        self.register_buffer("orders", torch.arange(1, harmonics + 2, dtype=torch.float32), persistent=False)
        self.l_linear = nn.Linear(harmonics + 1, 1)

    def forward(self, f0):
        """f0 [B, S] in Hz at the audio rate, 0 where unvoiced -> [B, S, 1]."""
        with torch.no_grad():
            rad = (f0[..., None] * self.orders / self.sample_rate) % 1
            initial = torch.rand(f0.shape[0], len(self.orders), device=f0.device)
            initial[:, 0] = 0                       # random start phase for the overtones only
            rad[:, 0] = rad[:, 0] + initial
            rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear")
            phase = torch.cumsum(rad, dim=-1) * 2 * math.pi
            phase = F.interpolate(phase * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            voiced = (f0 > self.voiced_threshold).float()[..., None]
            noise = (voiced * self.noise_std + (1 - voiced) * self.sine_amp / 3) * torch.randn_like(phase)
            sines = torch.sin(phase) * self.sine_amp * voiced + noise
        return torch.tanh(self.l_linear(sines))


class Generator(nn.Module):
    """Kokoro's iSTFTNet generator: transposed-convolution upsampling with
    AdaIN resblocks, the harmonic source's STFT injected at every scale, and a
    small inverse STFT writing magnitude and phase."""

    def __init__(self, c):
        super().__init__()
        rates, channels, s = c.generator_rates, c.generator_channels, c.style_dim
        self.fft, self.hop, self.kernels = c.generator_fft, c.generator_hop, len(c.generator_resblock_kernels)
        self.m_source = SourceModuleHnNSF(c.sample_rate, c.hop)
        self.ups, self.resblocks, self.noise_convs, self.noise_res = (nn.ModuleList() for _ in range(4))
        for i, u in enumerate(rates):
            ch = channels // 2 ** (i + 1)
            self.ups.append(weight_norm(nn.ConvTranspose1d(channels // 2 ** i, ch, 2 * u, u, padding=u // 2)))
            self.resblocks.extend(AdaINResBlock1(ch, k, (1, 3, 5), s) for k in c.generator_resblock_kernels)
            if i + 1 < len(rates):
                stride = math.prod(rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(self.fft + 2, ch, 2 * stride, stride, padding=(stride + 1) // 2))
                self.noise_res.append(AdaINResBlock1(ch, 7, (1, 3, 5), s))
            else:
                self.noise_convs.append(nn.Conv1d(self.fft + 2, ch, 1))
                self.noise_res.append(AdaINResBlock1(ch, 11, (1, 3, 5), s))
        self.conv_post = weight_norm(nn.Conv1d(ch, self.fft + 2, 7, 1, padding=3))
        for conv in (*self.ups, self.conv_post):
            nn.init.normal_(conv.weight, 0.0, 0.01)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.register_buffer("window", torch.hann_window(self.fft), persistent=False)

    def forward(self, x, s, f0):
        """x [B, C, F], style [B, S], f0 [B, F] Hz (0 unvoiced) -> wave [B, F * hop]."""
        with torch.no_grad():
            source = self.m_source(f0.repeat_interleave(self.m_source.upsample_scale, dim=-1))[..., 0]
            spectrum = torch.stft(source, self.fft, self.hop, window=self.window, return_complex=True)
            har = torch.cat((spectrum.abs(), spectrum.angle()), dim=1)
        for i, (up, noise_conv, noise_res) in enumerate(zip(self.ups, self.noise_convs, self.noise_res)):
            x = up(F.leaky_relu(x, 0.1))
            if i == len(self.ups) - 1:
                x = self.reflection_pad(x)
            x = x + noise_res(noise_conv(har), s)
            x = sum(block(x, s) for block in self.resblocks[i * self.kernels:(i + 1) * self.kernels]) / self.kernels
        x = self.conv_post(F.leaky_relu(x))
        bins = self.fft // 2 + 1
        spectrum = torch.polar(torch.exp(x[:, :bins]), torch.sin(x[:, bins:]))
        return torch.istft(spectrum, self.fft, self.hop, window=self.window)


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

    def forward(self, text, condition, source_f0, style):
        return self.generator(self.features(text, condition, style), style, source_f0)


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


class StyleVoca(Voca):
    PARAMETER_BUDGET = 30_000_000

    def __init__(self, config=Config(), bert=None):
        """``bert``: optional pretrained phoneme encoder, ``(tokens, stress, mask) ->
        [B, N, phoneme_encoder_width]`` (voca.phonebert.PhoneEncoder). As in Kokoro
        its features reach the prosody predictor through ``bert_encoder``; without
        one, the predictor reads the text encoder."""
        super().__init__(config)
        c = config
        self.bert = bert
        self.bert_encoder = nn.Linear(c.phoneme_encoder_width or c.channels, c.style_prosody_channels)
        self.predictor = ProsodyPredictor(c)
        self.decoder = Decoder(c)
        # Per voice: the mean [acoustic | prosodic] style of the training clips, what inference uses.
        self.register_buffer("style", torch.zeros(len(c.speakers), 2 * c.style_dim))

    def predict_tokens(self, hidden, tokens, stress, mask, lengths, prosodic_style):
        text = hidden if self.bert is None else self.bert(tokens, stress, mask)
        return self.predictor(self.bert_encoder(text), prosodic_style, lengths, mask)

    @torch.no_grad()
    def observe_style(self, style, speakers, momentum=0.01):
        """Running mean of the training clips' styles (exponential, like batch norm's)."""
        for voice in speakers.unique():
            self.style[voice].lerp_(style[speakers == voice].mean(0).float(), momentum)

    def predict_durations(self, logits, tokens, mask, speed=1.0):
        frames = torch.round(torch.sigmoid(logits[..., :-1].float()).sum(-1) / speed).long()
        return self.gate_pauses(frames, logits[..., -1], tokens, mask)

    @torch.inference_mode()
    def synthesize(self, tokens, stress, lengths, speakers, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, pitch_jitter=0.0,
                   voiced_vowels=True, comma_pause=0.0, stop_pause=0.0):
        """Free-running synthesis with the stored mean style. One utterance at a
        time: the instance norms would see a batch's padding."""
        self.check_controls(speed, pitch_shift, pitch_range)
        if len(lengths) != 1:
            raise ValueError("StyleVoca synthesizes one utterance at a time")
        hidden, mask = self.encode(tokens, stress, lengths, speakers)
        acoustic_style, prosodic_style = self.style[speakers].chunk(2, dim=-1)
        states, logits = self.predict_tokens(hidden, tokens, stress, mask, lengths, prosodic_style)
        durations = self.pause_floors(self.predict_durations(logits, tokens, mask, speed), tokens, lengths, mask, comma_pause, stop_pause)
        index, frame_mask, frame_lengths = self.align(durations, mask)
        prosody = self.predictor.F0Ntrain(frames_of(states, index), prosodic_style, frame_lengths)
        if voiced_vowels:
            prosody = self.voice_vowels(prosody, tokens, index, frame_mask)
        prosody = self.widen_pitch(prosody, frame_mask, pitch_range)
        condition = self.hard_prosody(prosody)
        pitch = condition[..., 0] + pitch_shift
        if pitch_jitter > 0:
            pitch = pitch + self.pitch_wobble(pitch.shape, pitch_jitter, seed, pitch.device)
        condition = torch.cat((pitch[..., None], condition[..., 1:]), dim=-1)
        with torch.random.fork_rng(devices=[tokens.device] if tokens.device.type == "cuda" else []):
            torch.manual_seed(seed)     # the NSF source draws its start phases and noise
            wave = self.decoder(frames_of(hidden, index), condition, self.frequency(pitch) * condition[..., 1], acoustic_style)
        return wave, frame_lengths * self.config.hop, {"durations": durations, "prosody": prosody, "frame_mask": frame_mask,
                                                       "frame_lengths": frame_lengths}

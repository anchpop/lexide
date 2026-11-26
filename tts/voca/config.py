from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

# 6: the StyleTTS generator upsamples by interpolation and trains its source (noise gain, mixer).
# 7: the generator reads its source spectrum as real and imaginary parts (was magnitude and angle).
# 8: the generator low-passes its source coefficients before each strided injection (no aliased source lines).
# 9: soft voicing: the source and the decoder take the voicing probability, not a 0/1 decision.
# 10: the source's sines (amplitude and phase) and noise bands are predicted per frame, not fixed.
# 5: aperiodic harmonic cutoff + post-filter. 2-3: byte tokens + learned aligner.
# 11: how a sentence is delivered is drawn from the text's distribution (voca.styletts.Delivery), not one stored style per voice.
# 12: the source's sines (amplitude and phase) and noise bands are predicted per frame, not fixed.
# 13: the decoder is told each frame's place in its phone (StyleVoca.frame_place).
CHECKPOINT_FORMAT = 13


def check_format(state, what):
    """Refuse a checkpoint written by another model format."""
    if state.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(f"{what} is format {state.get('format_version')}, this code reads format {CHECKPOINT_FORMAT} "
                         "(see voca.config.CHECKPOINT_FORMAT); retrain it")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    # Acoustic features (see ``acoustic_config`` for their version).
    sample_rate: int = 24000
    hop: int = 256
    n_fft: int = 1024
    n_mels: int = 80
    f0_min: float = 50.0
    f0_max: float = 600.0
    speakers: tuple[str, ...] = ("achernar",)
    hints: bool = True      # reads what only yap knows of the text (``voca.text.HINTS``), not just phonemes and punctuation
    front_layers: int = 0   # layers in front of both text encoders (``model.Front``): the phonemes-only model's, distilled from a hinted one (voca.distill)
    # Size of voca.text.SYMBOLS, which comes from the inventory file.
    vocab_size: int = field(default_factory=lambda: len(__import__("voca.text", fromlist=["SYMBOLS"]).SYMBOLS))
    # Text encoder and frame-level transformer.
    channels: int = 192
    heads: int = 4
    feedforward: int = 640
    kernel: int = 7
    encoder_layers: int = 4
    prosody_layers: int = 2
    acoustic_layers: int = 5
    dropout: float = 0.1
    predictor_channels: int = 128
    # Source-filter waveform head: harmonic branch and noise branch STFT sizes,
    # causal/anticausal complex-cepstrum orders.
    synth_fft: int = 512
    synth_hop: int = 128
    noise_fft: int = 256
    noise_hop: int = 64
    cep_causal: int = 80
    cep_anti: int = 16
    noise_cep: int = 32
    harmonic_limit: float = 0.47
    aperiodic_width_hz: float = 400.0     # softness of the periodic/aperiodic harmonic cutoff
    # "istft": frame-rate spectral generator + inverse STFT (SpectralGenerator).
    # "nsf": neural source-filter waveform model (NeuralSourceFilter).
    # "source_filter": cepstral harmonic-plus-noise synthesis + post-filter (runs/eng-gemini).
    waveform: str = "nsf"
    nsf_channels: int = 32
    nsf_blocks: int = 2
    nsf_layers: int = 8             # dilations 1..128 per block
    nsf_harmonics: int = 8
    # "vocos": ConvNeXt + 1024/256 iSTFT writing magnitude and phase (Vocos); uses istft_channels/layers.
    # "fargan": frame-wise autoregressive with pitch prediction (PitchPredictive), trained by voca.revocode.
    istft_channels: int = 256
    istft_layers: int = 6
    fargan_subframe: int = 64
    fargan_channels: int = 192
    fargan_condition: int = 128
    postfilter_channels: int = 192
    postfilter_layers: int = 4      # 0: none (runs/eng-nsf was trained without)
    # "tinyvoca": frame transformers + one of the waveform heads above (TinyVoca).
    # "styletts": StyleTTS 2 / Kokoro prosody predictor, AdaIN decoder and
    # iSTFTNet generator, conditioned on a style vector (voca.styletts.StyleVoca).
    architecture: str = "tinyvoca"
    style_dim: int = 128                  # each of the acoustic and the prosodic half
    style_prosody_channels: int = 192
    style_prosody_layers: int = 3
    style_prosody_distinct: int = 0       # looped duration encoder: distinct LSTMs repeated to the depth (0: none shared)
    style_max_duration: int = 64          # frames; duration = sum of this many sigmoids
    decoder_channels: int = 448
    generator_channels: int = 320          # ~8M generator; 224 ~4M, 144 ~2M
    generator_rates: tuple[int, ...] = (8, 8)   # x generator_hop = hop
    generator_resblock_kernels: tuple[int, ...] = (3, 7, 11)
    generator_fft: int = 16
    generator_hop: int = 4
    style_encoder_channels: int = 64      # training-only style encoders
    style_encoder_max: int = 512
    crop_frames: int = 256                # training window for the decoder (StyleTTS 2 trains on crops)
    # A pretrained phoneme encoder (voca.phonebert) feeding the prosody predictor;
    # 0 layers: none. voca.train --phoneme-encoder fills these from its checkpoint.
    phoneme_encoder_layers: int = 0
    phoneme_encoder_distinct: int = 0     # looped: this many distinct blocks repeated to the full depth (0: none shared)
    phoneme_encoder_width: int = 256
    phoneme_encoder_heads: int = 4
    phoneme_encoder_feedforward: int = 1024
    max_tokens: int = 512
    max_frames: int = 3000
    mel_mean: float = -5.0
    mel_scale: float = 5.0
    energy_mean: float = -4.0
    energy_scale: float = 2.0
    pitch_reference_hz: float = 200.0

    def __post_init__(self):
        from .text import SYMBOLS
        if self.vocab_size != len(SYMBOLS):
            raise ValueError(f"vocab_size must equal the phoneme inventory size {len(SYMBOLS)}")
        if self.channels % self.heads or (self.channels // self.heads) % 2:
            raise ValueError("channels must be divisible by heads with an even head width")
        if not self.speakers or len(set(self.speakers)) != len(self.speakers):
            raise ValueError("speaker names must be unique and nonempty")
        for fft, hop in ((self.synth_fft, self.synth_hop), (self.noise_fft, self.noise_hop)):
            if fft != 4 * hop or self.hop % hop:
                raise ValueError("synthesis windows need 4x overlap and a hop dividing the acoustic hop")
        if self.cep_causal + self.cep_anti >= self.synth_fft // 2 or self.noise_cep >= self.noise_fft // 2:
            raise ValueError("cepstrum orders must be well below half the synthesis FFT size")
        if min(self.encoder_layers, self.prosody_layers, self.acoustic_layers, self.kernel) < 1 or not self.kernel % 2:
            raise ValueError("layer counts must be positive and the kernel odd")
        if min(self.mel_scale, self.energy_scale) <= 0 or not 0 < self.harmonic_limit < 0.5:
            raise ValueError("invalid normalization scale or harmonic limit")
        if self.architecture not in ("tinyvoca", "styletts"):
            raise ValueError("architecture must be tinyvoca or styletts")
        if math.prod(self.generator_rates) * self.generator_hop != self.hop or self.generator_fft != 4 * self.generator_hop:
            raise ValueError("generator rates times its iSTFT hop must equal the hop, with a 4x-overlap iSTFT")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{key: tuple(v) if isinstance(v, list) else v for key, v in value.items()})

    @property
    def fingerprint(self):
        return _digest(self.to_dict())

    @property
    def acoustic_config(self):
        """Settings that determine stored wave/mel/prosody tensors. Tokens and
        durations are versioned separately by the annotation layer.
        Feature version 3 added ``pitch_check``, a second pitch tracker's track."""
        fields = ("sample_rate", "hop", "n_fft", "n_mels", "f0_min", "f0_max")
        return {**{key: getattr(self, key) for key in fields}, "feature_version": 3,
                "pitch_reference_hz": 200, "log_floor": 1e-5}

    @property
    def acoustic_fingerprint(self):
        return _digest(self.acoustic_config)

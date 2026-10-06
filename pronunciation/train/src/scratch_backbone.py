"""Locally initialized acoustic Conformer; no language inputs or downloaded weights.

The valid 400-sample STFT and centered stride-two subsampler put frame i at
200 + 320*i samples, exactly the existing CTC/VAD grid. Real convolutional
DFT buffers keep the raw-waveform frontend portable to ONNX (no complex ops).
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ScratchConfig:
    hidden_size: int = 256
    num_hidden_layers: int = 16
    heads: int = 4
    ff_size: int = 1024
    kernel_size: int = 31
    dropout: float = 0.1
    specaugment: bool = True
    final_dropout: float = 0.0
    model_type: str = "scratch_conformer"


class MelFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        import torchaudio
        phase = 2 * torch.pi * torch.arange(201)[:, None] * torch.arange(400)[None] / 400
        window = torch.hann_window(400)
        self.register_buffer("dft", torch.cat([phase.cos(), -phase.sin()])[:, None] * window)
        self.register_buffer("mel", torchaudio.functional.melscale_fbanks(
            201, 0, 8000, 128, 16000, norm="slaney", mel_scale="slaney"))

    def forward(self, waveform, lengths):
        # Both normalizations exclude padding; callers may pass arbitrary pad values.
        valid = torch.arange(waveform.shape[1], device=waveform.device)[None] < lengths[:, None]
        x = waveform.float().masked_fill(~valid, 0)
        mean = x.sum(-1, keepdim=True) / lengths[:, None]
        x = (x - mean).masked_fill(~valid, 0)
        x = x * torch.rsqrt(x.square().sum(-1, keepdim=True) / lengths[:, None] + 1e-7)
        # Explicit fp32: the log-energy floor is below fp16's useful range.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = F.conv1d(x[:, None], self.dft, stride=160)
            power = spectrum[:, :201].square() + spectrum[:, 201:].square()
            mel = (power.transpose(1, 2) @ self.mel).clamp_min(1e-5).log()
            mel_lengths = (lengths - 400) // 160 + 1
            mask = torch.arange(mel.shape[1], device=x.device)[None] < mel_lengths[:, None]
            weight = mask[:, :, None]
            # One scalar per clip preserves the spectral envelope of isolated
            # vowels; per-bin CMVN would erase stationary formant differences.
            count = mel_lengths[:, None, None] * mel.shape[-1]
            mean = (mel * weight).sum((1, 2), keepdim=True) / count
            centered = (mel - mean) * weight
            var = centered.square().sum((1, 2), keepdim=True) / count
            return centered * torch.rsqrt(var + 1e-5), mask


def specaugment(x, mask):
    """Two independent time/frequency masks per clip, sampled on the device."""
    b, t, f = x.shape
    lengths = mask.sum(1)
    for _ in range(2):
        widths = (torch.rand(b, device=x.device) * (lengths.float() * 0.05).clamp(max=20)).long()
        starts = (torch.rand(b, device=x.device) * (lengths - widths).clamp(min=1)).long()
        pos = torch.arange(t, device=x.device)[None]
        time = (pos >= starts[:, None]) & (pos < (starts + widths)[:, None])
        widths = torch.randint(0, 17, (b,), device=x.device)
        starts = (torch.rand(b, device=x.device) * (f - widths)).long()
        pos = torch.arange(f, device=x.device)[None]
        freq = (pos >= starts[:, None]) & (pos < (starts + widths)[:, None])
        x = x.masked_fill(time[:, :, None] | freq[:, None], 0)
    return x


class RotaryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.dim = cfg.hidden_size // cfg.heads
        self.dropout = cfg.dropout
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.register_buffer("freq", 10000 ** (-torch.arange(0, self.dim, 2).float() / self.dim))

    def forward(self, x, mask):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, t, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        angles = torch.arange(t, device=x.device).float()[:, None] * self.freq
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
        def rotate(z):
            a, b = z[..., 0::2], z[..., 1::2]
            return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
        y = F.scaled_dot_product_attention(rotate(q), rotate(k), v,
            attn_mask=mask[:, None, None], dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class ConformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_size
        def ff():
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.ff_size), nn.SiLU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.ff_size, d), nn.Dropout(cfg.dropout))
        self.ff1, self.ff2 = ff(), ff()
        self.attention = RotaryAttention(cfg)
        self.conv_norm = nn.LayerNorm(d)
        self.pointwise_in = nn.Linear(d, 2*d)
        self.depthwise = nn.Conv1d(d, d, cfg.kernel_size, padding=cfg.kernel_size//2, groups=d)
        # Per-frame LayerNorm, not BatchNorm: inference is independent of padding/batchmates.
        self.depth_norm = nn.LayerNorm(d)
        self.pointwise_out = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        valid = mask[:, :, None]
        x = (x + 0.5 * self.ff1(x)) * valid
        x = (x + self.dropout(self.attention(x, mask))) * valid
        y = F.glu(self.pointwise_in(self.conv_norm(x)), dim=-1) * valid
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        y = self.pointwise_out(F.silu(self.depth_norm(y)))
        x = (x + self.dropout(y)) * valid
        return self.norm(x + 0.5 * self.ff2(x)) * valid


class ScratchConformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = ScratchConfig(**(config or {}))
        c = self.config
        if c.hidden_size % c.heads or (c.hidden_size // c.heads) % 2 or c.kernel_size % 2 != 1:
            raise ValueError("RoPE needs an even head dimension; convolution needs an odd kernel")
        self.frontend = MelFrontend()
        self.subsample = nn.Conv1d(128, c.hidden_size, 3, stride=2, padding=1)
        self.layers = nn.ModuleList(ConformerBlock(c) for _ in range(c.num_hidden_layers))
        self._grad_ckpt = False

    def gradient_checkpointing_enable(self, **kwargs):
        self._grad_ckpt = True

    def gradient_checkpointing_disable(self, **kwargs):
        self._grad_ckpt = False

    def _get_feat_extract_output_lengths(self, lengths):
        return (lengths - 400) // 320 + 1

    def forward(self, input_values, attention_mask=None, output_hidden_states=False, **kwargs):
        lengths = (attention_mask.sum(-1) if attention_mask is not None else
                   torch.full((input_values.shape[0],), input_values.shape[1], device=input_values.device, dtype=torch.long))
        x, mel_mask = self.frontend(input_values, lengths)
        if self.training and self.config.specaugment:
            x = specaugment(x, mel_mask)
        x = self.subsample(x.transpose(1, 2)).transpose(1, 2)
        mask = torch.arange(x.shape[1], device=x.device)[None] < self._get_feat_extract_output_lengths(lengths)[:, None]
        x = x * mask[:, :, None]
        states = [x] if output_hidden_states else None
        for layer in self.layers:
            x = (checkpoint(layer, x, mask, use_reentrant=False) if self._grad_ckpt and self.training
                 else layer(x, mask))
            if states is not None:
                states.append(x)
        return SimpleNamespace(last_hidden_state=x, hidden_states=tuple(states) if states is not None else None)

    def save_pretrained(self, save_dir, **kwargs):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path / "scratch_backbone.pt")

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        saved = torch.load(Path(path) / "scratch_backbone.pt", map_location="cpu", weights_only=True)
        model = cls(saved["config"])
        model.load_state_dict(saved["state_dict"])
        return model

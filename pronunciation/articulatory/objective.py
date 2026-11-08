"""Audio-only objectives and envelope features; no decoder participates here."""
import math

import torch
from torch import nn


def mel_bank(size, device, bands=80):
    hz = torch.linspace(0, 8000, size//2+1, device=device)
    edges = 700 * (torch.exp(torch.linspace(0, math.log(1+8000/700), bands+2,
                                          device=device))-1)
    rising = (hz[None]-edges[:-2, None])/(edges[1:-1]-edges[:-2])[:, None]
    falling = (edges[2:, None]-hz[None])/(edges[2:]-edges[1:-1])[:, None]
    bank = torch.minimum(rising, falling).clamp_min(0)
    # Small FFTs cannot resolve 80 independent bands; empty bands stay zero.
    return bank / bank.sum(-1, keepdim=True).clamp_min(1e-8)


class AudioLoss(nn.Module):
    def __init__(self, target, kind='linear'):
        super().__init__()
        self.kind = kind
        self.targets = {}
        for size in (128, 256, 512, 1024):
            window = torch.hann_window(size, device=target.device)
            self.register_buffer(f'window_{size}', window)
            bank = mel_bank(size, target.device)
            self.register_buffer(f'mel_{size}', bank)
            spectrum = torch.stft(target, size, size//4, window=window,
                                  return_complex=True).abs()
            self.targets[size] = spectrum if kind == 'linear' else bank @ spectrum

    def forward(self, prediction, sizes=(256, 512, 1024)):
        losses = []
        for size in sizes:
            a = torch.stft(prediction, size, size//4,
                           window=getattr(self, f'window_{size}'), return_complex=True).abs()
            if self.kind == 'mel':
                a = getattr(self, f'mel_{size}') @ a
            b = self.targets[size]
            if self.kind == 'linear':
                log_error = ((a+1e-4).log()-(b+1e-4).log()).abs().mean()
            else:
                floor = b.max().clamp_min(1e-8)*1e-3
                log_error = (a.clamp_min(floor).log()-b.clamp_min(floor).log()).abs().mean()
            losses.append((a-b).norm()/b.norm().clamp_min(1e-6) + log_error)
        return torch.stack(losses).mean()


def envelope(spectrum, bank, lifter=16):
    """Log-mel cepstral smoothing, with level and linear log-frequency tilt removed."""
    values = (spectrum @ bank.T).clamp_min(1e-7).log()
    n = values.shape[-1]
    basis = torch.cos(math.pi/n*(torch.arange(n, device=values.device)+.5)[:, None]
                      * torch.arange(lifter, device=values.device)[None])
    basis[:, 0] *= 1/math.sqrt(n)
    basis[:, 1:] *= math.sqrt(2/n)
    values = (values @ basis) @ basis.T
    tilt = torch.linspace(-1, 1, n, device=values.device)
    values = values-values.mean(-1, keepdim=True)
    return values-(values @ tilt)[..., None]*tilt/tilt.square().sum()

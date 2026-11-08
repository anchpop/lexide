"""Small differentiable tube synthesizer. SI lengths; cross-sections in cm².

Impedances are normalized by rho*c/(1 cm²). Transfer matrices map downstream
pressure/volume velocity to upstream values. Far-field pressure is proportional
to the time derivative of oral + nasal volume velocity (co-located outlets).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from maeda import NAMES, contours, geometry

SR, HOP, FFT = 16000, 160, 1024


def multiply(left, right):
    a, b, c, d = left
    e, f, g, h = right
    return a*e+b*g, a*f+b*h, c*e+d*g, c*f+d*h


def tube(area, length, frequencies):
    """Lossy cylinder: boundary-layer attenuation + damped yielding-wall loss.

    sqrt(f)/radius approximates viscothermal loss; a low-frequency wall term
    broadens low poles. This is passive phenomenology, not tissue mechanics.
    """
    f = frequencies.clamp_min(1.)
    radius = torch.sqrt(area[..., None] * 1e-4 / math.pi)
    attenuation = .00003 * torch.sqrt(f) / radius + .6 / (1 + (f / 700)**2)
    gamma = attenuation + 2j * math.pi * f / 350.
    gl = gamma * torch.as_tensor(length, device=area.device)[..., None]
    z = 1 / area[..., None]
    a, s = torch.cosh(gl), torch.sinh(gl)
    return a, z*s, s/z, a


def radiation(area, f):
    ka = 2 * math.pi * f / 350 * torch.sqrt(area[..., None] * 1e-4 / math.pi)
    return (.5 * ka**2 / (1 + .5 * ka**2) + .61j * ka / (1 + .61*ka)) / area[..., None]


def transfer(areas, length, coupling, f):
    """Return glottal and constriction pressure-source far-field responses.

    Nasal neck + fixed 12 cm nasal branch is a shunt at section 16. Frication
    uses a pressure source with an absorbing upstream termination: downstream
    response only, a deliberate v1 approximation (no back-cavity resonance).
    """
    shape = areas.shape[:-1] + (len(f),)
    one = torch.ones(shape, device=areas.device, dtype=torch.complex64)
    zero = torch.zeros_like(one)
    identity = (one, zero, zero, one)
    nasal = identity
    for area in (1.4, 2.4, 3.2, 2.6, 1.8, 1.):
        nasal = multiply(nasal, tube(torch.full_like(coupling, area), .02, f))
    nasal = multiply(tube(coupling, .008, f), nasal)
    zn = radiation(torch.ones_like(coupling), f)
    na, nb, nc, nd = nasal
    nasal_pressure = na*zn + nb
    yn = (nc*zn + nd) / nasal_pressure
    zlip = radiation(areas[..., -1], f)
    suffix = identity
    fric_responses = []
    nasal_ratio = zero
    for i in range(39, -1, -1):
        suffix = multiply(tube(areas[..., i], length / 40, f), suffix)
        if i == 16:
            a, b, c, d = suffix
            nasal_ratio = (a*zlip+b) / nasal_pressure
            suffix = (a, b, c+yn*a, d+yn*b)
        a, b, c, d = suffix
        outlet = 1 + nasal_ratio if i <= 16 else 1
        fric_responses.append(outlet / (a*zlip+b))
    a, b, c, d = suffix
    oral = 1 / (c*zlip+d)
    glottal = oral * (1 + nasal_ratio)
    weights = torch.softmax(-5 * areas, dim=-1)
    fric = (torch.stack(fric_responses[::-1], -2) * weights[..., None]).sum(-2)
    # Far-field differentiation, normalized at 1 kHz. Zero DC.
    radiate = 1j * f / 1000
    return glottal * radiate, fric * radiate


def interpolate(x, samples):
    position = torch.arange(samples, device=x.device) / HOP
    lo = position.long().clamp_max(len(x)-1)
    hi = (lo+1).clamp_max(len(x)-1)
    fraction = (position-lo).clamp_max(1)[:, None]
    return x[lo]*(1-fraction) + x[hi]*fraction


class Synth(nn.Module):
    def __init__(self, closures=False):
        super().__init__()
        self.closures = closures
        self.register_buffer('frequencies', torch.fft.rfftfreq(FFT, 1/SR))
        self.register_buffer('window', torch.hann_window(FFT))

    def forward(self, articulation, source, f0, voiced, samples, scale=1., noise=None):
        areas, length, coupling, _ = geometry(articulation, scale, self.closures)
        h, hf = transfer(areas, length, coupling, self.frequencies)
        if self.closures:
            # No free downstream noise bypass at an oral seal. This is a flow
            # gate, not a pressure reservoir or a model of release bursts.
            hf = hf * ((areas.amin(-1) - .001) / .05).clamp(0, 1)[:, None]
        gains = 4 * source[..., :3].sigmoid()
        tilt = 1. + 2.5 * source[..., 3].sigmoid()
        # Source phase integrates given pitch. No acoustic reference enters here.
        pitch = interpolate(f0[:, None], samples)[:, 0]
        phase = 2 * math.pi * torch.cumsum(pitch / SR, 0)
        harmonics = torch.arange(1, 81, device=f0.device)
        freq = f0[:, None] * harmonics
        bins = freq / (SR / FFT)
        lo = bins.floor().long().clamp(0, FFT//2-1)
        frac = (bins - lo).clamp(0, 1)
        magnitude = h.abs()
        response = magnitude.gather(1, lo)*(1-frac) + magnitude.gather(1, lo+1)*frac
        amplitude = (response / harmonics**tilt[:, None] * (freq < SR/2)
                     * gains[:, :1] * voiced[:, None])
        signal = (torch.sin(phase[:, None]*harmonics) * interpolate(amplitude, samples)).sum(-1)
        if noise is None:
            noise = torch.randn(2, samples, device=f0.device)
        noise_fft = torch.stft(noise, FFT, HOP, window=self.window, return_complex=True)
        filters = torch.stack((h.abs()*gains[:, 1:2], hf.abs()*gains[:, 2:3]))
        filters = F.interpolate(filters.transpose(1, 2), size=noise_fft.shape[-1],
                                mode='linear', align_corners=True)
        filtered = torch.istft(noise_fft*filters, FFT, HOP, window=self.window, length=samples)
        return .1 * (signal + filtered.sum(0))


def smoothness(articulation):
    # Mean squared physical-control velocity, units of normalized range / second.
    return (articulation.sigmoid().diff(dim=0) * 100).square().mean()


# Relaxed posture: every control mid-range except a closed velum.
REST = torch.tensor([.5]*(len(NAMES)-1) + [.12])


def effort_weights(target):
    """100 Hz centered 20 ms energy: up to 20× rest pressure in quiet frames.

    Audio, not aligned first/last phones, defines silence. Frames below 1% of
    the clip's peak energy get extra pressure; speech retains unit weight.
    """
    energy = F.avg_pool1d(F.pad(target.square()[None, None], (HOP, HOP)),
                          2*HOP, stride=HOP)[0, 0]
    relative = energy / energy.max().clamp_min(1e-12)
    return 1 + 19*(1-relative/.01).clamp(0, 1)


def effort(articulation, weights=None):
    distance = (articulation.sigmoid() - REST.to(articulation)).square().mean(-1)
    return (distance if weights is None else distance*weights).mean()

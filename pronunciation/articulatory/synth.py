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


# Yielding wall, cgs per cm² of wall: resistance, mass (VTCalcs defaults).
WALL_RESISTANCE, WALL_MASS, RHO_C = 1600., 1.5, 39.9


def tube(area, length, frequencies):
    """Lossy cylinder: boundary-layer attenuation and a yielding wall.

    sqrt(f)/radius approximates viscothermal loss. The wall is a distributed
    shunt (mass + resistance per perimeter), which raises and damps low poles.
    """
    f = frequencies.clamp_min(1.)
    area = area[..., None]
    radius = torch.sqrt(area * 1e-4 / math.pi)
    k = 2j * math.pi * f / 350.
    wall = 2 * torch.sqrt(math.pi * area) * RHO_C / (WALL_RESISTANCE + 2j * math.pi * f * WALL_MASS)
    q = torch.sqrt(1 + wall / (k / 100 * area))
    gl = (.00003 * torch.sqrt(f) / radius + k * q) * torch.as_tensor(length, device=area.device)[..., None]
    z = 1 / (area * q)
    a, s = torch.cosh(gl), torch.sinh(gl)
    return a, z*s, s/z, a


def radiation(area, f):
    """Piston in a baffle, low-ka series resistance + mass (VTCalcs RL circuit)."""
    return 2 * math.pi * (f / 35000)**2 + 16j * f / (3 * 35000 * torch.sqrt(math.pi * area[..., None]))


def transfer(areas, length, coupling, f, junction=None, glottis=None, jets=None, resistance=None):
    """Far-field responses to the glottal flow source and to turbulence noise.

    Length is the whole tract (equal sections) or one value per section. Nasal
    neck + fixed 12 cm nasal branch is a shunt before section `junction`,
    round(0.4*N) by default (16 for N=40).
    Coupling must be positive; use 1e-6 cm² for an effectively closed port.
    glottis is the glottal area in cm². The tract then ends in the glottal
    slit and the trachea, which damps the formants; None is a hard wall.
    jets holds the noise pressure amplitude at each section, with an absorbing
    upstream termination: downstream response only. Loading it with the back
    cavity as well did not fit clean audio any better.
    resistance is the series loss that steady flow adds to each section, then
    to the nasal port: sound passes poorly through a narrow gap with air
    streaming through it.
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
    if resistance is not None:
        na, nb = na + resistance[..., -1:]*nc, nb + resistance[..., -1:]*nd
    nasal_pressure = na*zn + nb
    yn = (nc*zn + nd) / nasal_pressure
    zlip = radiation(areas[..., -1], f)
    suffix = identity
    fric = nasal_ratio = zero
    sections = areas.shape[-1]
    if junction is None:
        junction = round(.4 * sections)
    lengths = torch.as_tensor(length, device=areas.device)
    if lengths.ndim < areas.ndim:
        lengths = (lengths / sections)[..., None].expand(areas.shape)
    for i in range(sections-1, -1, -1):
        suffix = multiply(tube(areas[..., i], lengths[..., i], f), suffix)
        if i == junction:
            a, b, c, d = suffix
            nasal_ratio = (a*zlip+b) / nasal_pressure
            suffix = (a, b, c+yn*a, d+yn*b)
        a, b, c, d = suffix
        if jets is not None:
            fric = fric + jets[..., i, None] * (1 + nasal_ratio if i <= junction else 1) / (a*zlip+b)
        if resistance is not None:  # upstream of the jet it makes
            suffix = a + resistance[..., i, None]*c, b + resistance[..., i, None]*d, c, d
            a, b, c, d = suffix
    glottal = (1 + nasal_ratio) / (c*zlip+d)
    if glottis is not None:
        # Kinetic resistance at 8 cmH2O and the inertance of a 3 mm slit, then a 2.5 cm² trachea.
        glottal = (1 + nasal_ratio) / (c*zlip+d + (a*zlip+b) / ((.107 + 2j*math.pi*f*.3/35000)/glottis[..., None] + .4))
    # Far-field differentiation, normalized at 1 kHz. Zero DC.
    radiate = 1j * f / 1000
    return glottal * radiate, fric * radiate


# Rate (per second, per cm² of orifice) at which the pressure behind a closure
# follows its supply: a 37 m/s jet at 8 cmH2O into the 8 cm³ that the walls yield.
EQUALISE = 460.


def aerodynamics(areas, coupling, source, junction):
    """Slow flow from the lungs through glottis, pharynx, then mouth and nose side by side.

    source: raw lung pressure (0..2, 1 = 8 cmH2O) and glottal area (0..0.5 cm²)
    per frame. Sound is not free: it follows from these and the tract. Each
    orifice drops pressure by (flow/area)², and anything under 0.1 mm² is shut.
    The pressure behind the mouth and nose outlets lags its steady value by
    the time the walls take to fill, so the voice dies some tens of ms into a
    closure and a release lets the stored pressure out as a burst. Voicing
    grows as the 1.5 power of the drop across adducted folds. Noise at the
    glottis and in every section is flow³/area^2.5.

    Returns voicing and aspiration amplitudes [frames], noise [frames, sections],
    the glottal area and the flow resistance of each section and of the port
    [frames, sections+1] (jet speed over sound speed, per area).
    """
    lung, glottis = 2*source[..., 0].sigmoid(), .5*source[..., 1].sigmoid() + 1e-6
    areas, coupling = (areas-.001).clamp_min(0) + 1e-6, (coupling-.001).clamp_min(0) + 1e-6
    narrowest = lambda a: a.pow(-4).sum(-1).pow(-.25)
    mouth = narrowest(areas[..., junction:])
    up, out = (glottis.pow(-2) + narrowest(areas[..., :junction]).pow(-2)).rsqrt(), mouth + coupling
    steady = lung * up**2 / (up**2 + out**2)
    # p[t] = p[t-1] e^-x + steady (1 - e^-x), solved in closed form over the frames.
    x = (EQUALISE*HOP/SR * (up + out)).double()
    filled, total = -torch.expm1(-x), x.cumsum(-1)
    end = (torch.logcumsumexp((steady.double().clamp_min(1e-30) * filled).log() + total, -1) - total).exp()
    before = F.pad(end[..., :-1], (1, 0))
    oral = (steady + (before - steady) * filled/x).to(areas.dtype).clamp_min(0)  # mean over the frame
    drop = (lung - oral).clamp_min(0)
    across = drop * (up/glottis)**2
    behind, front = (up * (drop+1e-12).sqrt())[..., None], (oral+1e-12).sqrt()[..., None]
    flow = torch.cat((behind.expand(*up.shape, junction), (mouth[..., None]*front).expand(*up.shape, areas.shape[-1]-junction)), -1)
    return (2 * across**1.5 * ((.12-glottis)/.03).sigmoid(), glottis.sqrt() * across**1.5, 4 * flow**3 * areas.pow(-2.5),
            glottis, 37/350 * torch.cat((flow / areas**2, front / coupling[..., None]), -1))


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
        self.register_buffer('harmonics', torch.arange(1, 81))
        mel = torch.log1p(self.frequencies/700) / math.log1p(SR/2/700)
        self.register_buffer('basis', torch.cos(math.pi * mel[:, None] * torch.arange(8)))

    def at_harmonics(self, magnitude, f0):
        """[frames, bins] magnitudes read at the 80 harmonics of f0; zero above Nyquist."""
        freq = f0[:, None] * self.harmonics
        bins = freq / (SR / FFT)
        lo = bins.floor().long().clamp(0, FFT//2-1)
        frac = (bins - lo).clamp(0, 1)
        return (magnitude.gather(1, lo)*(1-frac) + magnitude.gather(1, lo+1)*frac) * (freq < SR/2)

    def render(self, amplitude, filters, f0, samples, noise=None):
        """Sinusoids at the harmonics of f0 with [frames, 80] amplitudes, plus one
        white-noise stream through each [frames, bins] magnitude filter."""
        # Source phase integrates given pitch. No acoustic reference enters here.
        pitch = interpolate(f0[:, None], samples)[:, 0]
        phase = 2 * math.pi * torch.cumsum(pitch / SR, 0)
        signal = (torch.sin(phase[:, None]*self.harmonics) * interpolate(amplitude, samples)).sum(-1)
        if noise is None:
            noise = torch.randn(len(filters), samples, device=f0.device)
        noise_fft = torch.stft(noise, FFT, HOP, window=self.window, return_complex=True)
        filters = F.interpolate(filters.transpose(1, 2), size=noise_fft.shape[-1],
                                mode='linear', align_corners=True)
        filtered = torch.istft(noise_fft*filters, FFT, HOP, window=self.window, length=samples)
        return .1 * (signal + filtered.sum(0))

    def forward(self, articulation, source, f0, voiced, samples, scale=1., noise=None, colour=None):
        areas, length, coupling, _ = geometry(articulation, scale, self.closures)
        return self.sound(areas, length, coupling, source, f0, voiced, samples, noise, colour)

    def spectra(self, areas, length, coupling, source, colour=None, junction=None):
        """Harmonic envelope before its tilt, noise envelope [frames, bins] and the tilt.

        source: see `aerodynamics`.
        colour: [3, 8] cosine coefficients over the mel axis for the static
        log spectra of voice, breath and hiss: one speaker's source, noise and
        level, not a per-frame envelope.
        """
        if junction is None:
            junction = round(.4 * areas.shape[-1])
        voice, breath, jets, glottis, resistance = aerodynamics(areas, coupling, source, junction)
        glottal, frication = transfer(areas, length, coupling, self.frequencies, junction, .002+glottis, jets, resistance)
        colours = (1., 1., 1.) if colour is None else (colour @ self.basis.T).exp()
        glottal = .03 * glottal.abs()
        # Tilt: an abducted glottis loses its upper harmonics.
        return (voice[:, None]*colours[0]*glottal, breath[:, None]*colours[1]*glottal + .03*colours[2]*frication.abs(),
                1 + 2.5*glottis/(glottis+.1))

    def sound(self, areas, length, coupling, source, f0, voiced, samples, noise=None, colour=None, junction=None):
        harmonic, aperiodic, tilt = self.spectra(areas, length, coupling, source, colour, junction)
        amplitude = self.at_harmonics(harmonic, f0) / self.harmonics**tilt[:, None] * voiced[:, None]
        return self.render(amplitude, aperiodic[None], f0, samples, noise)


def levels(areas, coupling, source, junction=16):
    """Voicing, aspiration and frication levels as render.py reads them: 4*sigmoid of each."""
    voice, breath, jets, *_ = aerodynamics(areas, coupling, source, junction)
    return (torch.stack((voice, breath, jets.sum(-1)), -1) / 4).clamp(1e-4, 1-1e-4).logit()


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

"""Differentiable PB1 Maeda geometry, not an independent Gaussian-tube model.

Port of semi_polar, lam, convert_scale and sagittal_to_area in VTCalcs
(sensein/VocalTractModels, revision 59b4b99c53ab2da9ad3e6df5bfe5ca2ba0c91cea).
Copyright 2013–2014 Satrajit Ghosh and Shinji Maeda; Apache-2.0, see
maeda_data/LICENSE. The original PB1 specification is vendored unchanged.
Modified here: batched torch math, bounded controls, two-region speaker scaling,
volume-preserving equal-length resampling, and explicit numerical floors.

The seven factors are standard-deviation scores, bounded to +/-3 by sigmoid;
velum is separate. PB1 is one speaker, not universal anatomy. Independent factor
combinations are NOT all anatomically valid: like lam(), we block tongue/wall
crossings and negative lip dimensions. This is piecewise differentiable, not a
collision-free biomechanical model. Acoustic wall loss/nasal anatomy stay in synth.
"""
from functools import lru_cache
import math
from pathlib import Path

import torch

NAMES = ('jaw', 'tongue_position', 'tongue_shape', 'tongue_tip',
         'lip_aperture', 'lip_protrusion', 'larynx', 'velum')
# Exact VTCalcs src/Vtcalcs.c vowelpar[11][7] / arcb.m presets, not invented IPA gold.
# The source labels ah and aa are ambiguous for IPA /a/; report both.
PRESETS = {
    'iy': [.5, -2., 1., -2., 1., -1., 0.],
    'ah': [-1.5, .5, 0., -.5, .5, -.5, 0.],
    'aa': [-1.5, 2., 0., -.5, .5, -.5, 0.],
    'uw': [.5, 2., 1.5, -2., -1., 1.5, 0.],
}


def preset(name, device='cpu', dtype=torch.float32):
    """Raw controls for a source-labelled vowel, with nearly closed nasal port."""
    p = torch.tensor(PRESETS[name], device=device, dtype=dtype)
    return torch.cat(((p / 6 + .5).logit(), p.new_tensor([-8.])))


@lru_cache(None)
def _tables(device, dtype):
    text = (Path(__file__).parent / 'maeda_data/pb1_spec.dat').read_text()
    def numbers(section, label):
        body = section.split(label, 1)[1].split('**', 1)[0]
        return torch.tensor([float(x) for x in body.split()], device=device, dtype=dtype)
    def stats(name, factors):
        section = text.split(f'*** Statistical data for {name} ***')[1]
        mean = numbers(section, '** Means of observed variables **')
        sd = numbers(section, '** Standard deviations **')
        load = numbers(section, '** Factor patterns (loadings), A **')
        return mean / tek, sd / tek, load.reshape(len(mean), -1)[:, :factors]
    # PB1 uses the same TEK scale for VT and lips (TEKlip=0).
    tek = numbers(text, '*** Scale factor (from cm to TEK unit) for VT and Lips ***')[0]
    spec = numbers(text, ' (M1, M2, M3, DL, OMEGA, THETA, IX0, IY0)')
    m1, m2, m3 = [int(x) for x in spec[:3]]
    dl, omega, theta = [float(x) for x in spec[3:6]]
    origins, normals = [], []
    omega, theta = math.radians(omega), math.radians(theta)
    for i in range(m1 + m2 + m3):
        angle = omega if i < m1 else omega + theta * min(i + 1 - m1, m2)
        if i < m1:
            distance, direction = dl * (m1-i-1), omega-math.pi/2
        elif i < m1 + m2:
            distance, direction = 0., 0.
        else:
            distance, direction = dl * (i+1-m1-m2), angle+math.pi/2
        origins.append([distance*math.cos(direction), distance*math.sin(direction)])
        normals.append([math.cos(angle), math.sin(angle)])
    wall = text.split('*** Statistical data for WAL ***')[1]
    wall = numbers(wall, '** Means of observed variables **') / tek
    lipsection = text.split('*** Statistical data for LIP ***')[1]
    incisor = (numbers(lipsection, '** Meams of upper lip position (x, y) **') - spec[6:8]) / tek
    ab = numbers(text, '*** Alpha-beta sagittal-area conversion coeffifients ***').reshape(-1, 3)
    # Source iniva_tng=6 (zero based); segment i uses j=i+6-3.
    return (stats('TNG', 4), stats('LIP', 3), stats('LRX', 2), wall,
            torch.tensor(origins, device=device, dtype=dtype)[6:],
            torch.tensor(normals, device=device, dtype=dtype)[6:], incisor, ab[4:31, 1:])


def _native(raw):
    """29 paired contour vertices in corrected cm, and 28 segment areas."""
    if raw.shape[-1] != len(NAMES):
        raise ValueError('Maeda requires eight controls; old nine-control trajectories are incompatible')
    tng, lip, lrx, wall, origin, normal, incisor, ab = _tables(raw.device, raw.dtype)
    p = 6 * raw[..., :7].sigmoid() - 3
    def deform(stats, indices):
        mean, sd, loading = stats
        return mean + sd * (p[..., indices] @ loading.T)
    # Soft minimum: a hard clamp zeroes the gradient of any section touching the
    # wall, so a fit that once pressed the tongue to the palate could not leave.
    tongue = wall - torch.nn.functional.softplus(wall - deform(tng, [0, 1, 2, 3])[..., 1:], beta=20)
    lips = deform(lip, [0, 4, 5]).clamp_min(0)
    larynx = deform(lrx, [0, 6])
    inside = origin + tongue[..., None]*normal
    outside = (origin + wall[..., None]*normal).expand_as(inside)
    inner0, outer0 = larynx[..., 1:3], larynx[..., 3:5]
    upper = (incisor + raw.new_tensor([0., .8])).expand_as(inner0)
    lower = upper - torch.stack((torch.zeros_like(lips[..., 2]), lips[..., 2]), -1)
    # Source floors each of the two half-lip lengths to .01 cm.
    protrusion = lips[..., 1].clamp_min(.02/1.1)
    extension = torch.stack((-protrusion, torch.zeros_like(protrusion)), -1)
    inner = torch.cat((inner0[..., None, :], ((inner0+inside[..., 0, :])/2)[..., None, :],
                       inside, lower[..., None, :], (lower+extension)[..., None, :]), -2)
    outer = torch.cat((outer0[..., None, :], ((outer0+outside[..., 0, :])/2)[..., None, :],
                       outside, upper[..., None, :], (upper+extension)[..., None, :]), -2)
    # Mirror source's left-facing profile; translate the glottal midpoint to zero.
    glottis = (inner0 + outer0)[..., None, :] / 2
    transform = raw.new_tensor([-1.1, 1.1])  # source size_correction=1.10
    inner, outer = (inner-glottis)*transform, (outer-glottis)*transform
    mid = (inner+outer)/2
    lengths = torch.linalg.vector_norm(mid.diff(dim=-2), dim=-1).clamp_min(1e-6)
    # Sum two triangle cross products instead of Heron's sqrt: finite at closure.
    def cross(a, b):
        return (a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]).abs()
    a, b, c, d = inner[..., :-1, :], inner[..., 1:, :], outer[..., 1:, :], outer[..., :-1, :]
    quad = .5 * (cross(b-a, c-a) + cross(c-a, d-a))
    width = quad[..., :-1] / lengths[..., :-1]
    oral = 1.4 * ab[:, 0] * width.pow(ab[:, 1])  # source's documented 40% ad-hoc correction
    lip_area = math.pi/4 * lips[..., 2] * lips[..., 3] * 1.1**2
    return outer, inner, lengths, torch.cat((oral, lip_area[..., None]), -1)


def _scaled_native(raw, scale):
    """Continuous local similarity along the native midline, before resampling.

    PB1 vertices 2..9 are straight pharyngeal cuts; 10..20 turn toward
    the palate. Join across vertices 11..13 (neutral sections 15.6..18.5,
    near the nasal junction). Integrate segment scale along the midline;
    scale paired offsets about it, so neither contour has a seam. Areas
    retain PB1's empirical conversion, scaled by mean squared local scale.
    """
    outer, inner, lengths, areas = _native(raw)
    scale = torch.as_tensor(scale, device=raw.device, dtype=raw.dtype)
    if scale.ndim == 0 or scale.shape[-1] != 2:
        return (outer*scale[..., None, None], inner*scale[..., None, None],
                lengths*scale[..., None], areas*scale[..., None]**2)
    blend = ((torch.arange(29, device=raw.device, dtype=raw.dtype)-11)/2).clamp(0, 1)
    local = scale[..., :1]*(1-blend) + scale[..., 1:]*blend
    left, right = local[..., :-1], local[..., 1:]
    segment = (left+right)/2
    mid = (outer+inner)/2
    increments = mid.diff(dim=-2)*segment[..., None]
    center = torch.cat((torch.zeros_like(mid[..., :1, :]), increments.cumsum(-2)), -2)
    outer = center + (outer-mid)*local[..., None]
    inner = center + (inner-mid)*local[..., None]
    return outer, inner, lengths*segment, areas*(left.square()+left*right+right.square())/3


def _resample(raw, scale, closures):
    outer, inner, lengths, native_areas = _scaled_native(raw, scale)
    edges = torch.cat((torch.zeros_like(lengths[..., :1]), lengths.cumsum(-1)), -1)
    total = edges[..., -1]
    target = total[..., None] * torch.linspace(0, 1, 41, device=raw.device, dtype=raw.dtype)
    # Integrate piecewise-constant native areas over each equal-length section.
    overlap = (torch.minimum(target[..., 1:, None], edges[..., None, 1:])
               - torch.maximum(target[..., :-1, None], edges[..., None, :-1])).clamp_min(0)
    areas = (overlap * native_areas[..., None, :]).sum(-1) / (total[..., None]/40)
    # As in appro_area_function, retain exact outlet area for rounded vowels.
    areas = torch.cat((areas[..., :-1], native_areas[..., -1:]), -1)
    # 40 paired landmarks from glottis to outlet, spaced by midline arclength,
    # not source table index. Areas above describe 40 CELLS on this same tract.
    positions = total[..., None] * torch.linspace(0, 1, 40, device=raw.device, dtype=raw.dtype)
    index = torch.searchsorted(edges.contiguous(), positions.contiguous(), right=True).sub(1).clamp(0, 27)
    fraction = (positions-edges.gather(-1, index)) / lengths.gather(-1, index)
    def sample(points):
        idx = index[..., None].expand(index.shape+(2,))
        return points.gather(-2, idx)*(1-fraction[..., None]) + points.gather(-2, idx+1)*fraction[..., None]
    outer, inner = sample(outer), sample(inner)
    floor = .001 if closures else .0001
    areas = areas.clamp_min(floor)
    gap = torch.linalg.vector_norm(outer-inner, dim=-1)
    return areas, total/100, .001+1.8*raw[..., 7].sigmoid()**3, gap, outer, inner


def geometry(raw, scale=1., closures=False):
    """40 areas (cm²), length (m), nasal coupling (cm²), gaps (cm).

    Scale is isotropic if scalar, otherwise (..., 2) = (pharynx, oral).
    Batched isotropic scales must not have trailing dimension 2.
    """
    return _resample(raw, scale, closures)[:4]


def contours(raw, scale=1., closures=False):
    """40×2 outer/inner landmarks, cm; +x facing, glottal midpoint at origin."""
    return _resample(raw, scale, closures)[4:]

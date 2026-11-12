"""Area-function input straight from MRI outlines, with no articulatory model.

Tissue outlines are closed loops; the airway is what lies between the
wall-side loops and the tongue-side loop along gridlines normal to the wall.
synth.transfer takes the resulting per-section gaps (after a gap→area
conversion) and lengths directly.
"""
import numpy as np
import torch
from torch.nn import functional as F

import mri_fit
import usc

# Gridlines per landmark-delimited wall piece: pharynx (glottis → velum
# height), velar bridge, hard palate. One more at the teeth, then the lips.
GRID = (14, 6, 14)


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def gridlines(path, counts, smooth=.6, margin=1.):
    """Origins and unit normals along a glottis→teeth wall path, cm.

    path holds 100 points per landmark-delimited piece and counts the
    gridlines per piece, so an index is the same anatomical place in every
    speaker. The path is smoothed so that neighbouring normals fan out like a
    semi-polar grid instead of following outline wiggles. The tongue must lie
    to the right of travel. Origins sit margin behind the wall.
    """
    step = path.diff(dim=0).norm(dim=-1).mean()
    radius = int(3*smooth/step)
    kernel = torch.exp(-.5*(torch.arange(-radius, radius+1)*step/smooth)**2).to(path)
    # Reflect through the endpoints so that smoothing does not shorten the path.
    padded = torch.cat([2*path[:1]-path[1:radius+1].flip(0), path, 2*path[-1:]-path[-radius-1:-1].flip(0)])
    path = F.conv1d(padded.T[:, None], (kernel/kernel.sum())[None, None])[:, 0].T
    position = torch.cat([i*100 + torch.arange(n)*100/n for i, n in enumerate(counts)]
                         + [torch.tensor([len(path)-1.])]).to(path)
    lo = position.long().clamp_max(len(path)-2)
    tangent = F.normalize(path[lo+1]-path[lo], dim=-1)
    normal = torch.stack((tangent[:, 1], -tangent[:, 0]), -1)
    return path[lo] + (position-lo)[:, None]*(path[lo+1]-path[lo]) - margin*normal, normal


def hits(origin, normal, loop):
    """Distances [frames, gridlines, edges] at which gridlines cross closed polylines; inf if never."""
    edge = (loop.roll(-1, 1) - loop)[:, None]
    offset = loop[:, None] - origin[None, :, None]
    denominator = cross(normal[None, :, None], edge)
    t, s = cross(offset, edge)/denominator, cross(offset, normal[None, :, None])/denominator
    return torch.where((s >= 0) & (s <= 1) & (t > 0), t, torch.inf)


def gaps(origin, normal, walls, tongue, reach=6.):
    """Midsagittal gap and midline point per gridline, cm.

    The gap runs from the last wall-side crossing to the first tongue-side
    crossing. Overlapping outlines (contact) give zero; a gridline that
    misses the tongue gives its reach.
    """
    inner = hits(origin, normal, tongue).amin(-1).clamp_max(reach)
    outer = torch.cat([hits(origin, normal, w) for w in walls], -1)
    before = torch.where(outer <= inner[..., None], outer, -torch.inf).amax(-1)
    overlap = ((outer > inner[..., None]) & (outer < inner[..., None]+.5)).any(-1)
    outer = torch.where(before.isfinite(), before, torch.where(overlap, inner, torch.zeros_like(inner)))
    return inner-outer, origin + ((inner+outer)/2)[..., None]*normal


def wall(utterance, frames):
    """Mean wall path over some frames: one fixed grid per recording.

    The glottis is taken at the height of the lowest epiglottis point, which
    is where the tracked outlines stop resolving the larynx.
    """
    pieces = [[], [], []]
    for i in frames:
        a = utterance.articulators[i]
        base = a['epiglottis'][a['epiglottis'][:, 1].argmin()]
        start = np.linalg.norm(a['arytenoid']-base, axis=1).argmin()
        pharynx = usc.below_height(a['pharyngeal_wall'], a['hard_palate'][-1, 1])
        pieces[0].append(usc.resample(np.vstack([a['arytenoid'][start:], pharynx]), 24))
        pieces[1].append(np.stack([pharynx[-1], a['hard_palate'][-1]]))
        pieces[2].append(usc.resample(a['hard_palate'][::-1], 24))
    return torch.tensor(np.concatenate([usc.resample(np.mean(p, 0), 100) for p in pieces]), dtype=torch.float32)


def measure(utterance):
    """Per tracked frame: 36 gaps and section lengths (cm) from glottis to lips, and the velum–wall gap."""
    origin, normal = gridlines(wall(utterance, np.linspace(0, len(utterance.frames)-1, 100).astype(int)), GRID)
    tongue, back, top = [torch.tensor(np.stack([s[i] for s in utterance.segments]), dtype=torch.float32) for i in range(3)]
    gap, mid = gaps(origin, normal, [back, top], tongue)
    count = range(len(utterance.frames))
    upper, lower, velum = [torch.tensor(np.stack([usc.resample(utterance.articulators[i][k], 24) for i in count]),
                                        dtype=torch.float32) for k in ('upper_lip', 'lower_lip', 'velum')]
    nearest = (upper[:, :, None]-lower[:, None]).norm(dim=-1).flatten(1).argmin(1)
    # Outline steps (epiglottis, vallecula) zigzag the raw midline; smooth before measuring length.
    mid[:, 1:-1] = (mid[:, :-2] + 2*mid[:, 1:-1] + mid[:, 2:])/4
    mid = torch.cat([mid, ((upper[count, nearest//24] + lower[count, nearest % 24])/2)[:, None]], 1)
    step = mid.diff(dim=1).norm(dim=-1)
    return dict(gap=torch.cat([gap, mri_fit.minimum_gap(upper, lower)[:, None]], 1).numpy(),
                length=((F.pad(step, (1, 0)) + F.pad(step, (0, 1)))/2).numpy(),
                port=torch.cdist(velum, back).amin((1, 2)).numpy(), origin=origin.numpy(), normal=normal.numpy())

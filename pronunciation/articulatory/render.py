"""Midsagittal animation of fitted trajectories, with the synthesized audio.

    scripts/py-linux.sh articulatory/render.py out/network/epoch-3/<stem>-fit.npz

Writes <stem>-fit.mp4 next to the npz. The head outline is decoration; the
tract walls, lips, velum port, voicing and frication cues are the model's state.
"""
import subprocess
import sys
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep
import torch

from synth import contours, geometry

FPS = 30
BG, INK, DIM, FAINT = '#07090c', '#d8dde3', '#5b6570', '#26303a'
WALL, TONGUE, NASAL, VOICE, HISS = '#7fe3f0', '#ff7a95', '#b48cff', '#ffc861', '#f2f5f7'
MONO = 'DejaVu Sans Mono'


def spline(points, closed=False, n=300):
    points = np.asarray(points, float)
    if closed:
        points = np.vstack((points, points[:1]))
    tck, _ = splprep(points.T, s=0, per=closed, k=min(3, len(points) - 1))
    return np.array(splev(np.linspace(0, 1, n), tck)).T


# Profile facing +x in the tract's cm frame: glottis at the origin, lips near x=9.
HEAD = spline([(9.3, 12.0), (10.7, 12.9), (9.4, 15.8), (9.1, 17.0), (9.5, 18.0),
               (9.0, 21.0), (3.0, 25.6), (-6.8, 22.5), (-8.6, 17.0), (-7.6, 12.2),
               (-5.4, 9.2), (-4.8, 4.0), (-5.0, -3.0)])
CHIN = spline([(9.0, 8.6), (8.4, 7.2), (6.4, 6.5), (3.6, 5.6), (2.8, 2.0), (3.0, -3.0)])
EYE = spline([(7.0, 17.0), (7.9, 17.25), (8.3, 17.0), (7.9, 16.8)], closed=True, n=60)
BROW = spline([(6.4, 18.2), (7.6, 18.5), (8.8, 18.2)], n=40)
EAR = spline([(-1.0, 17.4), (0.0, 17.8), (0.3, 16.0), (-0.2, 13.8), (-1.0, 13.6),
              (-1.4, 15.2)], closed=True, n=80)
# Nasal passage from the velopharyngeal port to the nostril.
NOSE = spline([(-0.3, 9.0), (0.2, 10.6), (1.8, 12.0), (4.5, 12.6), (7.5, 12.7), (9.6, 12.3)])
VERTEBRAE = [(-2.8, y) for y in np.arange(-2.5, 9.5, 1.2)]
MOUTH = np.array([8.9, 10.5])
ROOT = 12  # inner sections below this are larynx/pharynx wall, above it tongue


def frame_state(data, t):
    i = min(int(t * data['control_hz']), len(data['f0']) - 1)
    art = torch.tensor(data['articulation'][i:i+1])
    extra = [True] if data.get('closures', False) else []
    outer, inner = contours(art, torch.as_tensor(data['scale']), *extra)
    coupling = geometry(art, torch.as_tensor(data['scale']), *extra)[2]
    source = 4 / (1 + np.exp(-data['source'][i, :3]))
    # Seat the tract so its neutral lips meet the drawn mouth.
    rest = contours(torch.zeros_like(art), torch.as_tensor(data['scale']))
    shift = MOUTH - ((rest[0][0, -1] + rest[1][0, -1]) / 2).numpy()
    return dict(outer=outer[0].numpy() + shift, inner=inner[0].numpy() + shift,
                coupling=float(coupling[0]),
                voicing=float(source[0] * data['voiced'][i]), hiss=float(source[2]), index=i)


def draw(ax, data, state, t, duration):
    ax.clear()
    ax.set_xlim(-11, 13.5), ax.set_ylim(-6.5, 28.5), ax.set_aspect('equal'), ax.axis('off')
    for curve in (HEAD, CHIN, EYE, BROW, EAR):
        ax.plot(*curve.T, color=WALL, lw=.8, alpha=.28)
    for x, y in VERTEBRAE:
        ax.add_patch(plt.Rectangle((x - .9, y), 1.8, .9, fill=False, ec=WALL, lw=.5, alpha=.15))
    outer, inner = state['outer'], state['inner']
    # Lips join the face outline to the tract's moving ends.
    upper, lower = outer[-1], inner[-1]
    for points in ((HEAD[0], upper + [.35, .2], upper), (lower, lower + [.35, -.2], CHIN[0])):
        ax.plot(*spline(points, n=60).T, color=TONGUE, lw=1.6)
    # Tongue body: the inner contour closed along the floor of the mouth.
    tongue = inner[ROOT:-3]
    floor = [tongue[-1] + [-.2, -1.4], (5.5, 7.4), (2.6, 6.2), tongue[0] + [.3, -.3]]
    ax.fill(*np.vstack((tongue, floor)).T, color=TONGUE, alpha=.14, lw=0)
    ax.plot(*tongue.T, color=TONGUE, lw=1.8)
    ax.plot(*inner[-4:].T, color=TONGUE, lw=1.6)  # lower incisors to lip
    ax.plot(*inner[:ROOT + 1].T, color=WALL, lw=1.2, alpha=.7)
    ax.plot(*outer.T, color=WALL, lw=1.6)
    # Velum hangs from the palate; it swings forward and down as the port opens.
    open_ = min(state['coupling'] / 1.8, 1) ** .5
    attach = outer[26]
    tip = outer[18] + [.45, -.15] + open_ * np.array([1.0, -.9])
    ax.plot(*spline([attach, (attach + tip) / 2 + [.1, -.45], tip], n=40).T,
            color=NASAL, lw=3, solid_capstyle='round')
    ax.plot(*NOSE.T, color=NASAL, lw=1 + 3 * open_, alpha=.12 + .7 * open_)
    # Voicing glows at the glottis; frication sprays at the narrowest gap.
    glottis = (outer[0] + inner[0]) / 2
    v = min(state['voicing'] / 2, 1)
    for r, a in ((.9, .08), (.55, .18), (.28, .5)):
        ax.add_patch(plt.Circle(glottis, r * (.4 + v), color=VOICE, alpha=a * v, lw=0))
    gap = np.linalg.norm(outer - inner, axis=1)
    narrow = int(np.argmin(gap[ROOT:])) + ROOT
    h = min(state['hiss'] / 2, 1)
    if h > .05:
        rng = np.random.default_rng(state['index'])
        spray = (outer[narrow] + inner[narrow]) / 2 + rng.normal(0, .35, (24, 2)) * [1, .5]
        ax.scatter(*spray.T, s=2, color=HISS, alpha=.8 * h, lw=0)
    ax.text(-10.6, 27.6, 'LEXIDE / ARTICULATORY', color=DIM, fontsize=8, family=MONO)
    ax.text(13.0, 27.6, str(data.get('label', '')).upper(), color=VOICE, fontsize=9,
            family=MONO, ha='right')
    ax.text(-10.6, 26.6, str(data.get('sentence', '')), color=INK, fontsize=10, family=MONO)
    timeline(ax, data, t, duration)


def timeline(ax, data, t, duration):
    x0, x1, y = -10.6, 13.0, -5.2
    scale = (x1 - x0) / duration
    ax.plot([x0, x1], [y + .9, y + .9], color=FAINT, lw=.6)
    bounds = data['boundaries']
    for phone, a, b in zip(data['phonemes'], bounds[:-1], bounds[1:]):
        current = a <= t < b
        ax.plot([x0 + a * scale] * 2, [y + .6, y + 1.2], color=FAINT, lw=.6)
        ax.text(x0 + (a + b) / 2 * scale, y, phone, ha='center', family=MONO,
                fontsize=13 if current else 9, color=INK if current else DIM,
                fontweight='bold' if current else 'normal')
    ax.plot([x0 + t * scale] * 2, [y - .4, y + 1.6], color=VOICE, lw=1)


def main(path):
    path = Path(path)
    data = dict(np.load(path, allow_pickle=True))
    wav = path.with_suffix('.wav')
    duration = len(data['f0']) / data['control_hz']
    fig = plt.figure(figsize=(7.2, 9.6), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1), facecolor=BG)
    with tempfile.TemporaryDirectory() as tmp:
        for n in range(int(duration * FPS)):
            t = n / FPS
            draw(ax, data, frame_state(data, t), t, duration)
            fig.savefig(f'{tmp}/{n:05d}.png', dpi=100, facecolor=BG)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                        '-i', f'{tmp}/%05d.png', '-i', str(wav), '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest',
                        str(path.with_suffix('.mp4'))], check=True)
    print(path.with_suffix('.mp4'))


if __name__ == '__main__':
    main(sys.argv[1])

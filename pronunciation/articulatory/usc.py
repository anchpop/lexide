"""USC75 / Annot16 reader. Data remain outside the source tree.

Lim et al. 2021, doi:10.1038/s41597-021-00976-x: 2.4 mm pixels.
Tracks: centered pixel coordinates, y up, frameNo one-based. Video: 84²,
left-facing. Returned curves are cm, right-facing, y up. Annot16 is research
only / no redistribution; generated overlays must remain local as well.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.io import loadmat

PIXEL_CM = .24
ROOT = Path(__file__).resolve().parents[1] / '.work/usc75'
ANNOT = ROOT / '75SpeakerAnnot16-extracted/75SpeakerAnnot16'
VIDEOS = ROOT / 'dataset_2drt_video_only-extracted'


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams']


def video_frames(path):
    stream = next(s for s in probe(path) if s['codec_type'] == 'video')
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                                   '-f', 'rawvideo', '-pix_fmt', 'gray', '-'])
    return np.frombuffer(raw, np.uint8).reshape(-1, stream['height'], stream['width'])


def from_pixels(points):
    """Image pixel centers → physical right-facing cm coordinates."""
    return (42 - np.asarray(points)) * PIXEL_CM


def to_pixels(points):
    return 42 - np.asarray(points) / PIXEL_CM


def resample(points, n=48):
    """Piecewise linear arclength sampling; no spline distortion."""
    points = np.asarray(points)
    arc = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    positions = np.linspace(0, arc[-1], n)
    return np.stack([np.interp(positions, arc, points[:, i]) for i in range(2)], -1)


def intervals(path):
    raw = path.read_bytes()
    text = raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig')
    tier = re.split(r'name\s*=\s*"phones"', text, flags=re.IGNORECASE)[1]
    # Stop at the next tier instead of accidentally reading later word labels.
    tier = re.split(r'\n\s*item \[\d+\]:', tier)[0]
    return [(float(a), float(b), p) for a, b, p in re.findall(
        r'intervals \[\d+\]:\s+xmin = ([\d.e+-]+)\s+xmax = ([\d.e+-]+)\s+text = "([^"]*)"', tier)]


@dataclass
class Utterance:
    stem: str
    video: Path  # MP4 contains both matching MRI frames and synchronized audio.
    fps: float
    frames: np.ndarray
    times: np.ndarray
    phones: np.ndarray
    segments: list
    articulators: list
    phone_intervals: list[tuple[float, float, str]]
    discarded_track_frames: int = 0

    @cached_property
    def audio(self):
        """Lazy (float32 samples × channels, sample_rate), on the MP4 clock.

        Preserve source rate/channels; sample zero is the first decoded audio
        sample. USC75 streams start at time zero, as does their video timeline;
        audio can slightly outlast video, so retain its full source duration.
        """
        stream = next(s for s in probe(self.video) if s['codec_type'] == 'audio')
        assert float(stream.get('start_time', 0)) == 0
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(self.video),
                                       '-map', '0:a:0', '-f', 'f32le', '-'])
        samples = np.frombuffer(raw, dtype='<f4').reshape(-1, int(stream['channels']))
        return samples, int(stream['sample_rate'])


# (closed segment loop, its local anatomical label). Vertex counts vary by
# template/recording, so absolute offsets cannot identify an articulator.
ARTICULATORS = {'epiglottis': (0, 1), 'tongue': (0, 2), 'lower_lip': (0, 4),
                'chin': (0, 5), 'arytenoid': (1, 1), 'pharyngeal_wall': (1, 2),
                'hard_palate': (2, 1), 'velum': (2, 2), 'upper_lip': (2, 5)}


def alignment_path(stem):
    speaker = stem.split('_')[0]
    path = ANNOT / speaker / 'alignment' / f'{stem}_text.TextGrid'
    if not path.exists():
        # Annot16 drops the numeric task prefix on sub022 grandfather/northwind.
        alternate = re.sub(r'_2drt_\d+_', '_2drt_', stem)
        path = path.with_name(f'{alternate}_text.TextGrid')
    return path


def load(stem):
    speaker = stem.split('_')[0]
    tracks = loadmat(ANNOT / speaker / 'track' / f'{stem}_track.mat',
                     simplify_cells=True)['trackdata']
    video = VIDEOS / speaker / '2drt/video' / f'{stem.lower()}_video.mp4'
    stream = next(s for s in probe(video) if s['codec_type'] == 'video')
    assert (stream['width'], stream['height']) == (84, 84)
    fps = float(Fraction(stream['avg_frame_rate']))
    # Some automatic tracks extend past the encoded video or assign different
    # contours to the same frame number. Neither is an unambiguous observation:
    # exclude out-of-video tracks and ALL occurrences of duplicated frame IDs.
    count = len(tracks)
    numbers = np.array([t['frameNo'] for t in tracks])
    unique, counts = np.unique(numbers, return_counts=True)
    valid = (numbers >= 1) & (numbers <= int(stream['nb_frames'])) & ~np.isin(numbers, unique[counts > 1])
    tracks = [t for t, keep in zip(tracks, valid) if keep]
    discarded = count-len(tracks)
    frames = np.array([t['frameNo'] - 1 for t in tracks], dtype=int)
    # Some tracks omit initial video frames (e.g. sub013 vcv1 r2 starts at 2).
    # frameNo, never track-array position, is the authoritative video mapping.
    assert np.all(np.diff(frames) > 0) and frames[0] >= 0
    assert frames[-1] < int(stream['nb_frames']), (stem, frames[-1], stream['nb_frames'])
    times = frames / fps
    phones = np.full(len(tracks), '', dtype='<U32')
    phone_intervals = intervals(alignment_path(stem))
    for a, b, phone in phone_intervals:
        phones[(times >= a) & (times < b)] = phone
    segments = [[np.asarray(s['v']) * [-PIXEL_CM, PIXEL_CM]
                 for s in t['contours']['segment'][:3]] for t in tracks]
    articulators = [{name: segments[j][s][np.asarray(t['contours']['segment'][s]['i']) == label]
                     for name, (s, label) in ARTICULATORS.items()} for j, t in enumerate(tracks)]
    return Utterance(stem, video, fps, frames, times, phones, segments, articulators, phone_intervals, discarded)


# Within the named anatomical label: zero-based, half-open visible surfaces.
# Tongue excludes the ventral tip return (last 3 of 20 vertices); lip mucosa
# excludes external skin. Posterior wall is also clipped to hard-palate/velum
# junction height below: PB1 does not model the superior nasopharyngeal wall.
REGIONS = {'tongue': (0, 17), 'pharyngeal_wall': (0, None),
           'hard_palate': (0, None), 'lower_lip': (0, 5), 'upper_lip': (5, None)}


def below_height(points, height):
    """Clip inferior→superior wall at velum-root height, excluding nasopharynx."""
    above = np.flatnonzero(points[:, 1] >= height)
    if not len(above):
        return points
    i = above[0]
    assert i > 0, 'no oral pharyngeal wall below the hard-palate/velum junction'
    fraction = (height-points[i-1, 1]) / (points[i, 1]-points[i-1, 1])
    return np.vstack([points[:i], points[i-1] + fraction*(points[i]-points[i-1])])


def curves(utterance, indices, n=32):
    def surface(i, name, a, b):
        articulators = utterance.articulators[i]
        points = articulators[name][a:b]
        if name == 'pharyngeal_wall':
            points = below_height(points, articulators['hard_palate'][-1, 1])
        return resample(points, n)
    return {name: np.stack([surface(i, name, a, b) for i in indices])
            for name, (a, b) in REGIONS.items()}


def demographics(speaker):
    return json.loads((VIDEOS / 'metafile_public_20210129.json').read_text())[speaker]['demographic']

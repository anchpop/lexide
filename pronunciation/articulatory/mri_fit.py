"""MRI geometry ceiling test, not acoustic inversion or a native-gesture prior.

Run with py-linux.sh -I articulatory/mri_fit.py. All outputs stay local because
Annot16 prohibits redistribution. Fit registration/anatomy only on r1 recordings;
r2 recordings are held out for both PCA training and global Maeda parameters.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import maeda
import usc

OUT = Path(__file__).resolve().parent / 'out/mri'
DEVICE = 'cuda'
N = 32
# Broad candidate surfaces. Crop to projections of observed endpoints, not to a
# tiny model subset: the same visible extent then enters both directed distances.
MODEL_REGIONS = {'tongue': (1, 7, 27), 'pharyngeal_wall': (0, 2, 15),
                 'hard_palate': (0, 15, 28), 'lower_lip': (1, 27, 29),
                 'upper_lip': (0, 27, 29)}
COLORS = ('#0891b2', '#c026d3', '#d97706')


def tensor(a):
    return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)


def projections(points, curve):
    a, delta = curve[:, :-1], curve.diff(dim=1)
    fraction = ((points[:, :, None] - a[:, None]) * delta[:, None]).sum(-1)
    fraction = (fraction / delta.square().sum(-1)[:, None].clamp_min(1e-10)).clamp(0, 1)
    projected = a[:, None] + fraction[..., None] * delta[:, None]
    squared = (points[:, :, None] - projected).square().sum(-1)
    return squared, fraction


def directed(points, curve):
    return projections(points, curve)[0].amin(-1).clamp_min(1e-12).sqrt().mean(-1)


def distance(a, b):
    """Symmetric mean point-to-polyline distance in cm, one value per frame."""
    return (directed(a, b) + directed(b, a)) / 2


def sample_curve(curve, positions):
    lengths = curve.diff(dim=1).norm(dim=-1)
    arc = torch.cat([lengths[:, :1] * 0, lengths.cumsum(-1)], -1)
    indices = torch.searchsorted(arc.contiguous(), positions.contiguous(), right=True)
    indices = (indices - 1).clamp(0, curve.shape[1] - 2)
    fraction = (positions - arc.gather(1, indices)) / lengths.gather(1, indices).clamp_min(1e-8)
    gather = indices[..., None].expand(-1, -1, 2)
    return curve.gather(1, gather) + fraction[..., None] * curve.diff(dim=1).gather(1, gather)


def uniform(curve):
    length = curve.diff(dim=1).norm(dim=-1).sum(-1, keepdim=True)
    return sample_curve(curve, length * torch.linspace(0, 1, N, device=curve.device))


def minimum_gap(a, b):
    """Exact 2D polyline minimum, including crossings between vertices."""
    nearest = torch.minimum(projections(a, b)[0].amin((1, 2)),
                            projections(b, a)[0].amin((1, 2))).sqrt()
    p, r = a[:, :-1, None], a.diff(dim=1)[:, :, None]
    q, s = b[:, None, :-1], b.diff(dim=1)[:, None]
    def cross(x, y):
        return x[..., 0]*y[..., 1] - x[..., 1]*y[..., 0]
    denominator = cross(r, s)
    valid = denominator.abs() > 1e-8
    divisor = torch.where(valid, denominator, torch.ones_like(denominator))
    t, u = cross(q-p, s)/divisor, cross(q-p, r)/divisor
    intersects = (valid & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)).any((1, 2))
    return torch.where(intersects, torch.zeros_like(nearest), nearest)


def visible(curve, target):
    """Linearly sample model between nearest projections of MRI endpoints."""
    squared, fractions = projections(target[:, [0, -1]], curve)
    indices = squared.argmin(-1)
    lengths = curve.diff(dim=1).norm(dim=-1)
    arc = torch.cat([lengths[:, :1] * 0, lengths.cumsum(-1)], -1)
    ends = arc.gather(1, indices) + lengths.gather(1, indices) * fractions.gather(2, indices[..., None])[..., 0]
    positions = ends[:, :1] + (ends[:, 1:] - ends[:, :1]) * torch.linspace(0, 1, N, device=curve.device)
    return sample_curve(curve, positions)


def anatomy(global_):
    # No uniform scale gauge: geometric mean of region factors is exactly one.
    ratio = .5 * global_[4].tanh()
    return torch.stack([ratio.exp(), (-ratio).exp()])


def register(contours, global_):
    angle = global_[1]
    rotation = torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()]).reshape(2, 2)
    return tuple(p @ rotation.T * global_[0].exp() + global_[2:4] for p in contours)


def model(raw, global_):
    art = torch.cat([raw, torch.zeros_like(raw[:, :1])], -1)
    return register(maeda._scaled_native(art, anatomy(global_))[:2], global_)


def public_regions(raw, global_, targets):
    """Audit public contours() at the exact physical extents scored natively.

    Project MRI endpoints onto the native candidate segments, map those fractions
    to native midline arc, then cut the public 40-point polyline at those SAME
    arc coordinates. Do not refit controls, registration, or visible endpoints.
    Public interpolation can cut native corners; this measures that difference.
    """
    art = torch.cat([raw, torch.zeros_like(raw[:, :1])], -1)
    native = maeda._scaled_native(art, anatomy(global_))
    native_global = register(native[:2], global_)
    public_global = register(maeda.contours(art, anatomy(global_)), global_)
    lengths = native[2]
    arc = torch.cat([lengths[:, :1]*0, lengths.cumsum(-1)], -1)
    result = {}
    for name, (side, a, b) in MODEL_REGIONS.items():
        squared, fractions = projections(targets[name][:, [0, -1]], native_global[side][:, a:b])
        indices = squared.argmin(-1)
        ends = arc.gather(1, indices+a) + lengths.gather(1, indices+a) * fractions.gather(2, indices[..., None])[..., 0]
        ends = ends / arc[:, -1:] * 39
        positions = torch.arange(40, device=raw.device).expand(len(raw), -1)
        positions = positions.clamp(min=ends.amin(-1, keepdim=True), max=ends.amax(-1, keepdim=True))
        index = positions.long().clamp(0, 38)
        fraction = positions-index
        gather = index[..., None].expand(-1, -1, 2)
        curve = public_global[side]
        bounded = curve.gather(1, gather) + fraction[..., None]*curve.diff(dim=1).gather(1, gather)
        result[name] = uniform(bounded)
    return result


def regions(raw, global_, targets):
    contours = model(raw, global_)
    return {name: visible(contours[side][:, a:b], targets[name])
            for name, (side, a, b) in MODEL_REGIONS.items()}


def errors(raw, global_, targets):
    prediction = regions(raw, global_, targets)
    return torch.stack([distance(prediction[k], targets[k]) for k in targets], -1)


def initialize(targets):
    # Two anatomical anchors: inferior front/back wall midpoint, lip midpoint.
    glottis = (targets['tongue'][:, 0] + targets['pharyngeal_wall'][:, 0]) / 2
    lip = (targets['upper_lip'].mean(1) + targets['lower_lip'].mean(1)) / 2
    o, i = maeda._scaled_native(torch.zeros(1, 8, device=DEVICE), 1.)[:2]
    source_glottis = (i[0, 7] + o[0, 2]) / 2
    source_lip = (o[0, -1] + i[0, -1]) / 2
    source = source_lip - source_glottis
    target = (lip - glottis).mean(0)
    scale = target.norm() / source.norm()
    angle = torch.atan2(target[1], target[0]) - torch.atan2(source[1], source[0])
    rotation = torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()]).reshape(2, 2)
    shift = glottis.mean(0) - source_glottis @ rotation.T * scale
    return torch.stack([scale.log(), angle, shift[0], shift[1], angle * 0])


def fit(targets, steps, global_=None, starts=('rest', 'iy', 'aa', 'uw'), refine=0):
    train_global = global_ is None
    base = initialize(targets) if train_global else global_.detach()
    runs, best = [], None
    n = len(next(iter(targets.values())))
    for start in starts:
        initial = torch.zeros(7, device=DEVICE) if start == 'rest' else maeda.preset(start, DEVICE)[:7]
        raw = initial.repeat(n, 1).requires_grad_()
        global_fit = base.clone().requires_grad_(train_global)
        optimizer = torch.optim.Adam([raw, global_fit] if train_global else [raw], lr=.045)
        history = []
        for step in range(steps + refine):
            optimizer.zero_grad()
            losses = errors(raw, global_fit, targets)
            loss = losses.mean()
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(raw.grad).all():
                raise RuntimeError('nonfinite MRI fit')
            optimizer.step()
            if step == steps:
                for group in optimizer.param_groups:
                    group['lr'] = .012
            if step % 100 == 0 or step == steps + refine - 1:
                history.append([step, float(loss.detach()) * 10])
        with torch.no_grad():
            losses = errors(raw, global_fit, targets).mean(-1)
            mean = float(losses.mean())
            runs.append(dict(start=start, mean_mm=10 * mean, history_mm=history))
            if best is None:
                best = (raw.detach().clone(), global_fit.detach().clone(), losses.clone())
            elif train_global:
                if mean < float(best[2].mean()):
                    best = (raw.detach().clone(), global_fit.detach().clone(), losses.clone())
            else:
                improve = losses < best[2]
                best[0][improve] = raw.detach()[improve]
                best[2][improve] = losses[improve]
        print('fit', 'train' if train_global else 'fixed', start, runs[-1], flush=True)
    return best[0], best[1], runs


def summary(values):
    values = np.asarray(values)
    return dict(n=len(values), median_mm=float(np.median(values)), p90_mm=float(np.quantile(values, .9)))


def dataset(speaker, max_frames, stems=None):
    targets = {name: [] for name in usc.REGIONS}
    ids, frames, phones, split = [], [], [], []
    for path in sorted((usc.ANNOT / speaker / 'track').glob('*_track.mat')):
        stem = path.stem.removesuffix('_track')
        task = int(stem.split('_')[2])
        if task > 10 or (stems is not None and stem not in stems):
            continue
        u = usc.load(stem)
        candidates = np.flatnonzero(~np.isin(u.phones, ['', 'sil', 'sp', 'spn']))
        if not len(candidates):
            continue
        selected = candidates[np.linspace(0, len(candidates)-1, min(max_frames, len(candidates))).astype(int)]
        # Ensure rare target consonants are represented; at most eight added
        # centers per recording. This is a coverage sample, not prevalence data.
        basephones = np.char.rstrip(np.char.upper(u.phones), '012')
        for phone in ['IY', 'AA', 'UW', 'K', 'T', 'S', 'M', 'NG']:
            pool = np.flatnonzero(basephones == phone)
            if len(pool) and not np.any(basephones[selected] == phone):
                selected = np.r_[selected, pool[len(pool)//2]]
        selected = np.sort(selected)
        c = usc.curves(u, selected)
        for k in c:
            targets[k].extend(c[k])
        ids.extend([stem] * len(selected)); frames.extend(u.frames[selected])
        phones.extend(u.phones[selected]); split.extend([stem.endswith('_r2')] * len(selected))
    return {k: tensor(np.array(v)) for k, v in targets.items()}, np.array(ids), np.array(frames), np.array(phones), np.array(split)


def manual_check(speaker):
    from PIL import Image
    u = usc.load(speaker + '_2drt_17_topic1')
    frames = usc.video_frames(u.video)
    rows = []
    # Ground truth names cover complete annotated curves, independently of the
    # shorter mucosal surfaces used for Maeda. Do not compare lip skin to Maeda.
    full = ('tongue', 'pharyngeal_wall', 'hard_palate', 'lower_lip', 'upper_lip', 'velum')
    for path in sorted((usc.ANNOT / 'hand_ground_truth/ground_truth_json').glob(speaker + '*.json')):
        frame = int(path.stem.split('-')[-1]) - 1
        jpg = usc.ANNOT / 'hand_ground_truth/extracted_frames_jpg' / path.name.replace('_track_', '_video.mp4_').replace('.json', '.jpg')
        image = np.asarray(Image.open(jpg).convert('L'), float)
        candidate = np.arange(max(0, frame-2), min(len(frames), frame+3))
        mae = np.abs(frames[candidate].astype(float)-image).mean((1, 2))
        truth = json.loads(path.read_text())
        track_index = int(np.searchsorted(u.frames, frame))
        assert u.frames[track_index] == frame
        measures = {}
        for name in full:
            manual = tensor(usc.resample(usc.from_pixels(truth[name]), N)[None])
            tracked = tensor(usc.resample(u.articulators[track_index][name], N)[None])
            measures[name] = float(distance(manual, tracked)[0]) * 10
        rows.append(dict(frame=int(frame), jpeg_best_offset=int(candidate[mae.argmin()]-frame),
                         jpeg_mae=float(mae[candidate == frame][0]), errors_mm=measures))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 9), constrained_layout=True)
    for ax, row in zip(axes.flat, rows[:6]):
        frame = row['frame']; track_index = int(np.searchsorted(u.frames, frame))
        path = usc.ANNOT / 'hand_ground_truth/ground_truth_json' / f'{u.stem}_track_frame-{frame+1}.json'
        truth = json.loads(path.read_text())
        ax.imshow(frames[frame], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        for j, name in enumerate(full):
            ax.plot(*np.asarray(truth[name]).T, color='white', lw=1.,
                    label='manual JSON' if j == 0 else None)
            ax.plot(*usc.to_pixels(u.articulators[track_index][name]).T,
                    color=COLORS[2], ls='--', lw=.8, label='automatic track' if j == 0 else None)
        ax.set(xlim=(5, 75), ylim=(75, 5), title=f'{speaker} video frame {frame} | JPEG offset {row["jpeg_best_offset"]}')
        ax.legend(fontsize=7, loc='lower left')
    fig.savefig(OUT / f'{speaker}-manual.png', dpi=150)
    plt.close(fig)
    return rows


def distal_velum(velum):
    """Tip and next posterior-surface vertex, excluding the attached root.

    Annot16 orders the oral-facing surface first, then returns along the nasal
    surface. Lowest y identifies the free tip in these upright sagittal tracks.
    This heuristic is saved separately from the requested full-contour minimum.
    """
    tip = velum[..., 1].argmin(-1)
    index = (tip[:, None] + torch.arange(2, device=velum.device)).clamp_max(velum.shape[1]-1)
    return velum.gather(1, index[..., None].expand(-1, -1, 2))


def port_measure(speaker):
    rows = []
    for path in sorted((usc.ANNOT / speaker / 'track').glob('*_track.mat')):
        stem = path.stem.removesuffix('_track')
        if int(stem.split('_')[2]) > 10:
            continue
        u = usc.load(stem)
        velum = tensor(np.stack([s['velum'] for s in u.articulators]))
        wall = tensor(np.stack([s['pharyngeal_wall'] for s in u.articulators]))
        # Both directions include segment interior projections, rather than a
        # sparse point-pair minimum. Still an in-plane proxy, not port area.
        with torch.no_grad():
            mm = minimum_gap(velum, wall).cpu().numpy() * 10
            tip_mm = minimum_gap(distal_velum(velum), wall).cpu().numpy() * 10
        for i, (phone, value) in enumerate(zip(u.phones, mm)):
            rows.append((stem, int(u.frames[i]), float(u.times[i]), phone, float(value), float(tip_mm[i])))
    ids, frames, times, phones, mm, tip_mm = zip(*rows)
    phones = np.array(phones); mm = np.array(mm); tip_mm = np.array(tip_mm)
    nasal = np.isin(np.char.lower(phones), ['m', 'n', 'ng', 'ŋ'])
    oral = ~nasal & ~np.isin(phones, ['', 'sil', 'sp', 'spn'])
    # Near-nasal oral frames: +/-150 ms, within the same recording only.
    near = np.zeros(len(mm), bool)
    ids = np.array(ids); times = np.array(times)
    for stem in np.unique(ids):
        indexes = np.flatnonzero(ids == stem)
        nasal_times = times[indexes[nasal[indexes]]]
        if len(nasal_times):
            near[indexes] = np.min(np.abs(times[indexes, None]-nasal_times[None]), 1) <= .15
    np.savez_compressed(OUT / f'{speaker}-port.npz', utterance=ids, frame=frames,
                        time=times, phone=phones, port_mm=mm, distal_port_mm=tip_mm)
    from scipy.stats import mannwhitneyu
    auc = float(mannwhitneyu(mm[nasal], mm[oral]).statistic / (nasal.sum()*oral.sum()))
    distal = {'nasal': summary(tip_mm[nasal]), 'oral': summary(tip_mm[oral]),
              'oral_near_nasal_150ms': summary(tip_mm[oral & near]),
              'oral_far_from_nasal': summary(tip_mm[oral & ~near]),
              'nasal_minus_oral_median_mm': float(np.median(tip_mm[nasal])-np.median(tip_mm[oral])),
              'nasal_auc_larger_is_nasal': float(mannwhitneyu(tip_mm[nasal], tip_mm[oral]).statistic / (nasal.sum()*oral.sum())),
              'definition': 'lowest-y velum vertex (free-tip heuristic) and next posterior vertex to full pharyngeal wall',
              'by_phone': {p: summary(tip_mm[phones == p]) for p in np.unique(phones) if p}}
    return {'distal_tip': distal, 'nasal_auc_larger_is_nasal': auc,
            'by_phone': {p: summary(mm[phones == p]) for p in np.unique(phones) if p},
            'nasal': summary(mm[nasal]), 'oral': summary(mm[oral]),
            'oral_near_nasal_150ms': summary(mm[oral & near]),
            'oral_far_from_nasal': summary(mm[oral & ~near]),
            'nasal_minus_oral_median_mm': float(np.median(mm[nasal])-np.median(mm[oral])),
            'definition': 'exact 2D polyline minimum including crossings, velum segment2 label2 to posterior wall segment1 label2; an in-plane distance proxy, not area'}


def draw(ax, image, target, outer, inner, title):
    ax.set_facecolor('#1a1a19')
    if image is not None:
        ax.imshow(image[:, ::-1], cmap='gray', vmin=0, vmax=255,
                  extent=(-9.96, 10.20, -9.96, 10.20), interpolation='nearest')
    for j, p in enumerate(target.values()):
        ax.plot(*p.T, color=COLORS[2], lw=.8, ls='--', label='MRI visible surfaces' if j == 0 else None)
    ax.plot(*outer.T, color=COLORS[0], lw=1.6, label='Maeda outer')
    ax.plot(*inner.T, color=COLORS[1], lw=1.6, ls='-.', label='Maeda inner')
    ax.set(xlim=(-7, 8), ylim=(-8, 9), aspect='equal', title=title, xlabel='cm (right-facing)', ylabel='cm')
    ax.legend(loc='lower right', fontsize=7)


def contact(speaker, data, raw, global_, ids, frames, phones, test):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    chosen = []
    for phone in ['IY', 'AA', 'UW', 'K', 'T', 'S', 'M', 'NG']:
        indexes = np.flatnonzero(test & (np.char.rstrip(np.char.upper(phones), '012') == phone))
        if len(indexes):
            chosen.append(indexes[len(indexes)//2])
    fig, axes = plt.subplots(2, 4, figsize=(16, 10), constrained_layout=True)
    cache = {}
    with torch.no_grad():
        outer, inner = [p.cpu().numpy() for p in model(raw, global_)]
    for ax, index in zip(axes.flat, chosen):
        if ids[index] not in cache:
            cache[ids[index]] = usc.video_frames(usc.load(ids[index]).video)
        target = {k: v[index].cpu().numpy() for k, v in data.items()}
        draw(ax, cache[ids[index]][frames[index]], target, outer[index], inner[index],
             f'{speaker} {phones[index]} | {ids[index].split("_2drt_")[1]} f{frames[index]}')
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    fig.savefig(OUT / f'{speaker}-contact.png', dpi=150)
    plt.close(fig)


def clip(speaker, global_, steps):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    # Six seconds of a held-out VCV recording, retaining exact video clock.
    u = usc.load(speaker + '_2drt_01_vcv1_r2')
    first = int(2 * u.fps); video_indices = np.arange(first, int(8 * u.fps), 4)
    selected = np.searchsorted(u.frames, video_indices)
    assert np.array_equal(u.frames[selected], video_indices)
    targets = {k: tensor(v) for k, v in usc.curves(u, selected).items()}
    raw, _, runs = fit(targets, steps, global_, starts=('rest', 'iy', 'aa', 'uw'))
    with torch.no_grad():
        outer, inner = [v.cpu().numpy() for v in model(raw, global_)]
    frames = usc.video_frames(u.video)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    with tempfile.TemporaryDirectory() as tmp:
        for j, frame in enumerate(selected):
            target = {k: v[j].cpu().numpy() for k, v in targets.items()}
            for ax in axes:
                ax.clear()
            draw(axes[0], frames[u.frames[frame]], target, outer[j], inner[j], f'{speaker} MRI + fit | {u.times[frame]:.2f}s {u.phones[frame]}')
            draw(axes[1], None, {}, outer[j], inner[j], 'Maeda geometry only | same cm scale')
            fig.savefig(f'{tmp}/{j:05d}.png', dpi=100)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', str(u.fps/4),
                        '-i', f'{tmp}/%05d.png', '-ss', str(first/u.fps), '-i', str(u.video),
                        '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', '-shortest', str(OUT / f'{speaker}-comparison.mp4')], check=True)
    plt.close(fig)
    np.savez_compressed(OUT / f'{speaker}-clip.npz', frame=u.frames[selected], time=u.times[selected],
                        raw=raw.cpu().numpy(), global_=global_.cpu().numpy(), outer=outer, inner=inner)
    return runs


def comparison_plot(result):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    speakers = result['speakers']
    fig, axes = plt.subplots(len(speakers), 2, figsize=(10, 4*len(speakers)),
                             squeeze=False, constrained_layout=True, facecolor='#1a1a19')
    for row, (speaker, entry) in enumerate(speakers.items()):
        for col, statistic in enumerate(['median_mm', 'p90_mm']):
            ax = axes[row, col]
            ax.set_facecolor('#1a1a19')
            ax.tick_params(colors='#eeeeee')
            for spine in ax.spines.values():
                spine.set_color('#666666')
            values = [entry['pca_test'][str(k)][statistic] for k in range(1, 11)]
            maeda_error = entry['maeda_test']['tongue'][statistic]
            ax.plot(range(1, 11), values, color=COLORS[0], marker='o', ms=5, lw=1.5, label='Speaker PCA')
            ax.axhline(maeda_error, color=COLORS[1], ls='--', lw=1.5, label='Maeda (7 controls)')
            ax.text(10, values[-1], f'  {values[-1]:.2f}', va='center', fontsize=9, color='#eeeeee')
            ax.text(10, maeda_error, f'  {maeda_error:.2f}', va='center', fontsize=9, color='#eeeeee')
            ax.set(xlabel='PCA components (k)', ylabel='Tongue curve distance (mm)',
                   title=f'{speaker} | {statistic.removesuffix("_mm")}', xlim=(.5, 11.5), ylim=(0, 8))
            ax.set_xticks(range(1, 11)); ax.legend(frameon=False, fontsize=9, labelcolor='#eeeeee')
            for text in (ax.xaxis.label, ax.yaxis.label, ax.title):
                text.set_color('#eeeeee')
            ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT / 'pca-comparison.png', dpi=150)
    plt.close(fig)


def run(args):
    OUT.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    result = {'config': vars(args), 'pixel_mm': usc.PIXEL_CM*10,
              'control_names': maeda.NAMES[:7],
              'global_parameter_names': ['log_similarity_scale', 'angle_radians', 'tx_cm', 'ty_cm', 'raw_anatomy_ratio'],
              'scale_source': 'https://www.nature.com/articles/s41597-021-00976-x',
              'metric': 'per-frame symmetric mean point-to-polyline distance; 32 uniform arclength samples each side; mm',
              'anatomical_labels_segment_and_label': usc.ARTICULATORS,
              'mri_regions_within_label_zero_based_half_open': usc.REGIONS,
              'pharyngeal_wall_limit': 'clip at y of last hard-palate vertex (velum root); exclude superior nasopharyngeal wall',
              'model_candidate_regions_zero_based_half_open': MODEL_REGIONS,
              'model_endpoint_policy': 'crop candidate to closest projections of MRI endpoints; evaluate same visible extent',
              'excluded': ['chin', 'epiglottis', 'arytenoid', 'ventral tongue return', 'external lip skin', 'velum (no geometric control in PB1)'],
              'split': 'r1 train, r2 held-out recording; repeated prompts shared but no shared frames', 'speakers': {}}
    if args.resume:
        result['speakers'] = json.loads((OUT / 'metrics.json').read_text())['speakers']
    for speaker in args.speakers:
        if args.resume and speaker in result['speakers']:
            print('retaining completed speaker', speaker, flush=True)
            continue
        data, ids, frames, phones, test = dataset(speaker, args.max_frames)
        train_data = {k: v[~test] for k, v in data.items()}
        test_data = {k: v[test] for k, v in data.items()}
        train_raw, global_, train_runs = fit(train_data, args.steps, refine=args.refine)
        test_raw, _, test_runs = fit(test_data, args.steps, global_, refine=args.refine)
        raw = torch.empty((len(test), 7), device=DEVICE)
        raw[~test], raw[test] = train_raw, test_raw
        with torch.no_grad():
            err = errors(raw, global_, data).cpu().numpy()*10
            fitted = {k: v.cpu().numpy() for k, v in regions(raw, global_, data).items()}
        # PCA learns only training-utterance tongue points, no test mean/rotation.
        train = train_data['tongue'].flatten(1)
        mean = train.mean(0)
        _, singular, basis = torch.linalg.svd(train - mean, full_matrices=False)
        centered = test_data['tongue'].flatten(1) - mean
        pca = {}; pca_errors = []; reconstructed = []
        for k in range(1, 11):
            pred = (mean + centered @ basis[:k].T @ basis[:k]).reshape(-1, N, 2)
            e = distance(uniform(pred), test_data['tongue']).cpu().numpy()*10
            pca[str(k)] = summary(e); pca_errors.append(e); reconstructed.append(pred.cpu().numpy())
        manual = manual_check(speaker)
        entry = dict(demographics=usc.demographics(speaker),
                     train_frames=int((~test).sum()), test_frames=int(test.sum()),
                     train_utterances=list(np.unique(ids[~test])), test_utterances=list(np.unique(ids[test])),
                     train_phones=dict(Counter(phones[~test])), test_phones=dict(Counter(phones[test])),
                     maeda_test={k: summary(err[test, j]) for j, k in enumerate(data)},
                     maeda_train={k: summary(err[~test, j]) for j, k in enumerate(data)},
                     pca_test=pca, global_parameters=global_.cpu().tolist(),
                     anatomy_scales=anatomy(global_).cpu().tolist(),
                     controls_near_rails_fraction=((6*test_raw.sigmoid()-3).abs() > 2.85).float().mean(0).cpu().tolist(),
                     train_optimization=train_runs, test_optimization=test_runs,
                     manual_tracker={k: summary([r['errors_mm'][k] for r in manual]) for k in manual[0]['errors_mm']},
                     manual_frames=manual, velopharyngeal_port=port_measure(speaker))
        np.savez_compressed(OUT / f'{speaker}-fit.npz', utterance=ids, frame=frames, phone=phones,
                            test=test, raw=raw.detach().cpu().numpy(), global_=global_.cpu().numpy(),
                            errors_mm=err, pca_errors_mm=pca_errors, pca_reconstruction=reconstructed,
                            pca_mean=mean.cpu().numpy(), pca_basis=basis.cpu().numpy(),
                            pca_singular_values=singular.cpu().numpy(),
                            **{f'target_{k}': v.cpu().numpy() for k, v in data.items()},
                            **{f'fit_{k}': v for k, v in fitted.items()})
        contact(speaker, data, raw, global_, ids, frames, phones, test)
        if not args.no_clip:
            entry['clip_optimization'] = clip(speaker, global_, args.steps)
        result['speakers'][speaker] = entry
        (OUT / 'metrics.json').write_text(json.dumps(result, indent=2))
        print(speaker, json.dumps(entry['maeda_test']), 'PCA', json.dumps(pca), flush=True)
    comparison_plot(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speakers', nargs='+', default=['sub013', 'sub022'])
    parser.add_argument('--max-frames', type=int, default=80)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--refine', type=int, default=150)
    parser.add_argument('--no-clip', action='store_true')
    parser.add_argument('--resume', action='store_true', help='retain completed speakers in metrics.json after interruption')
    run(parser.parse_args())

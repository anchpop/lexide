"""Direct low-rate trajectory fitting, with structured experiment settings."""
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from objective import AudioLoss
from synth import NAMES, SR, Synth, effort, effort_weights, smoothness

OUT = Path(__file__).parent / 'out'
# Neutral PB1 initialization; old Gaussian codebooks are not compatible.
DEFAULT = dict(loss='linear', closures=True, initialization='neutral', updates=1000,
               seed=7, learning_rate=.04, smoothness=.002, effort=1.)


def expand(raw, frames):
    return F.interpolate(raw.T[None], size=frames, mode='linear', align_corners=True)[0].T


def fit(rows, data_dir, destination, config):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'config.json').write_text(json.dumps(config, indent=2))
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    synth = Synth(closures=config['closures']).cuda()
    summary = []
    for row in rows:
        stem = Path(row['file']).stem
        data = np.load(data_dir / (stem+'.npz'))
        target, f0, voiced = [torch.tensor(data[k], device='cuda') for k in ('audio', 'f0', 'voiced')]
        frames, knots = len(f0), len(f0)//4+1
        noise = torch.randn(2, len(target), device='cuda')
        assert config['initialization'] == 'neutral'
        neutral = torch.zeros((knots, len(NAMES)), device='cuda')
        neutral[:, -1] = -2.
        articulation = torch.nn.Parameter(neutral)
        source = torch.nn.Parameter(torch.tensor([0., -7., -7., 0.], device='cuda').repeat(knots, 1))
        anatomy = torch.nn.Parameter(torch.zeros((), device='cuda'))
        parameters = [articulation, source, anatomy]
        optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'])
        objective = AudioLoss(target, config['loss'])
        weights = effort_weights(target)
        curve, start = [], time.monotonic()
        for step in range(config['updates']+1):
            art, src = expand(articulation, frames), expand(source, frames)
            scale = .85 + .3 * anatomy.sigmoid()
            output = synth(art, src, f0, voiced, len(target), scale, noise)
            sizes = (128, 256) if step < 100 else (256, 512, 1024)
            reconstruction = objective(output, sizes)
            regularizers = dict(smoothness=config['smoothness']*smoothness(art),
                                effort=config.get('effort', 0)*effort(art, weights))
            loss = reconstruction + sum(regularizers.values())
            assert torch.isfinite(loss), (stem, step)
            if step % 100 == 0 or step == config['updates']:
                with torch.no_grad():
                    full = float(objective(output))
                item = dict(step=step, objective=float(loss.detach()), full_audio_loss=full,
                            seconds=time.monotonic()-start,
                            **{k: float(v.detach()) for k, v in regularizers.items()})
                curve.append(item)
                print(destination.name, stem, item, flush=True)
                (destination / f'{stem}-curve.json').write_text(json.dumps(curve, indent=2))
            if step == 0:
                sf.write(destination / f'{stem}-initial.wav', output.detach().cpu().numpy(), SR, subtype='FLOAT')
            if step < config['updates']:
                optimizer.zero_grad()
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in parameters)
                torch.nn.utils.clip_grad_norm_(parameters, 10.)
                optimizer.step()
        sf.write(destination / f'{stem}-fit.wav', output.detach().cpu().numpy(), SR, subtype='FLOAT')
        sf.write(destination / f'{stem}-original.wav', data['audio'], SR, subtype='FLOAT')
        np.savez(destination / f'{stem}-fit.npz', articulation=art.detach().cpu().numpy(),
                 source=src.detach().cpu().numpy(), scale=float(scale.detach()), f0=data['f0'],
                 voiced=data['voiced'], phone=data['phone'], relative=data['relative'],
                 phonemes=data['phonemes'], boundaries=data['boundaries'],
                 samples=len(target), sample_rate=SR, control_hz=100, closures=config['closures'])
        summary.append(dict(file=stem, sentence=row['sentence'], initial=curve[0]['full_audio_loss'],
                            final=curve[-1]['full_audio_loss'], seconds=curve[-1]['seconds']))
    (destination / 'fit-summary.json').write_text(json.dumps(summary, indent=2))


def main():
    rows = json.loads((OUT / 'ablation-data/clips.json').read_text())
    fit(rows, OUT / 'ablation-data', OUT / 'fit', DEFAULT)


if __name__ == '__main__':
    main()

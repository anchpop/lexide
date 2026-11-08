"""Train phone→tract/source with audio and weak posture/gesture regularizers."""
import json
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf
import torch

from evaluate import score_files
from network import ControlNet, GestureLoss
from objective import AudioLoss
from prepare import MODEL, OUT
from synth import SR, Synth, effort, effort_weights, smoothness

DATA = OUT / 'corpus'
DEST = OUT / 'network-anatomy'
CONFIG = dict(seed=43, epochs=19, learning_rate=.001, smoothness=.002, effort=1.,
              saturation=.1, gesture=.1,
              phone_embedding=32, language_embedding=4, speaker_embedding=8,
              context_channels=96, dilations=[1, 3, 9], kernel=5)


def load(row):
    with np.load(DATA / (Path(row['file']).stem+'.npz')) as data:
        return {k: data[k].copy() for k in data.files}


def inputs(data, row, speakers):
    # This is the complete inference input boundary: no audio or optimized controls.
    return dict(phone=torch.tensor(data['phone'], device='cuda'),
        relative=torch.tensor(data['relative'], device='cuda'),
        language=torch.tensor(0, device='cuda'),
        speaker=torch.tensor(speakers.index(row['voice']), device='cuda'),
        f0=torch.tensor(data['f0'], device='cuda'),
        voiced=torch.tensor(data['voiced'], device='cuda'))


@torch.no_grad()
def evaluate(net, synth, rows, speakers, epoch, out=DEST):
    net.eval()
    destination = out / f'epoch-{epoch}'
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        data = load(row)
        stem = Path(row['file']).stem
        args = inputs(data, row, speakers)
        art, source, scale = net(**args)
        generator = torch.Generator(device='cuda').manual_seed(193)
        noise = torch.randn(2, len(data['audio']), device='cuda', generator=generator)
        waveform = synth(art, source, args['f0'], args['voiced'], len(data['audio']), scale, noise)
        assert torch.isfinite(waveform).all()
        sf.write(destination / (stem+'-fit.wav'), waveform.cpu().numpy(), SR, subtype='FLOAT')
        sf.write(destination / (stem+'-original.wav'), data['audio'], SR, subtype='FLOAT')
        np.savez(destination / (stem+'-fit.npz'), articulation=art.cpu().numpy(),
            source=source.cpu().numpy(), scale=scale.cpu().numpy(), f0=data['f0'], voiced=data['voiced'],
            phone=data['phone'], relative=data['relative'], phonemes=data['phonemes'],
            boundaries=data['boundaries'], samples=len(data['audio']), sample_rate=SR,
            control_hz=100, closures=synth.closures, language='eng', speaker=row['voice'],
            sentence=row['sentence'])
    (destination / 'clips.json').write_text(json.dumps(rows, indent=2))
    metric = score_files(rows, DATA, destination)
    print('HELDOUT', epoch, metric['edits'], metric['reference_count'], metric['per'], flush=True)
    net.train()
    return metric


def main():
    DEST.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])
    config = CONFIG | dict(audio_loss='linear', closures=True)
    (DEST / 'config.json').write_text(json.dumps(config, indent=2))
    rows = json.loads((DATA / 'clips.json').read_text())
    train = [r for r in rows if r['split'] == 'train']
    heldout = [r for r in rows if r['split'] == 'heldout']
    assert not {r['sentence'] for r in train} & {r['sentence'] for r in heldout}
    assert len(train) >= 1980 and len(heldout) >= 2
    speakers = sorted({r['voice'] for r in train})
    assert all(r['voice'] in speakers for r in heldout)
    vocabulary = json.loads((MODEL / 'frame_matrix.json').read_text())['heads']['phone']['labels']
    net = ControlNet(len(vocabulary), len(speakers)).cuda()
    synth = Synth(config['closures']).cuda()
    gestures = GestureLoss(vocabulary).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    # RAM caches avoid re-reading small npz files each epoch; no GPU corpus cache.
    cached = {r['file']: load(r) for r in train}
    progress = dict(config=config, training_clips=len(train), heldout_clips=len(heldout),
                    speakers=speakers, epochs=[], curve=[], metrics={})
    progress['metrics']['0'] = evaluate(net, synth, heldout, speakers, 0)
    start = time.monotonic()
    step = 0
    for epoch in range(1, config['epochs']+1):
        random.shuffle(train)
        losses, contributions = [], []
        epoch_start = time.monotonic()
        for row in train:
            data = cached[row['file']]
            args = inputs(data, row, speakers)
            target = torch.tensor(data['audio'], device='cuda')
            art, source, scale = net(**args)
            output = synth(art, source, args['f0'], args['voiced'], len(target), scale)
            audio_loss = AudioLoss(target, config['audio_loss'])(output)
            # Past |3| a sigmoid control stops learning; keep it where gradients live.
            saturation = (art.abs()-3).clamp_min(0).square().mean()
            terms = dict(audio=audio_loss,
                smoothness=config['smoothness']*smoothness(art),
                effort=config['effort']*effort(art, effort_weights(target)),
                saturation=config['saturation']*saturation)
            gesture = gestures(art, scale, args['phone'], args['relative'])
            terms.update({name: config['gesture']*value for name, value in zip(
                gestures.names, gesture, strict=True)})
            loss = sum(terms.values())
            assert torch.isfinite(loss), (epoch, row['file'])
            optimizer.zero_grad()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step()
            losses.append(float(loss.detach()))
            contributions.append([float(v.detach()) for v in terms.values()])
            step += 1
            if step % 100 == 0:
                item = dict(step=step, epoch=epoch, mean_loss=float(np.mean(losses[-100:])),
                            seconds=time.monotonic()-start,
                            contributions=dict(zip(terms, np.mean(contributions[-100:], axis=0))))
                progress['curve'].append(item)
                (DEST / 'progress.json').write_text(json.dumps(progress, indent=2))
                print('TRAIN', item, flush=True)
        progress['epochs'].append(dict(epoch=epoch, steps=step, mean_loss=float(np.mean(losses)),
            seconds=time.monotonic()-start, epoch_seconds=time.monotonic()-epoch_start,
            contributions=dict(zip(terms, np.mean(contributions, axis=0)))))
        torch.save(dict(state_dict=net.state_dict(), optimizer=optimizer.state_dict(), config=config,
            epoch=epoch, steps=step, vocabulary=vocabulary, languages=['eng'], speakers=speakers,
            data_provenance=json.loads((DATA / 'provenance.json').read_text())), DEST / f'epoch-{epoch}.pt')
        progress['metrics'][str(epoch)] = evaluate(net, synth, heldout, speakers, epoch)
        (DEST / 'progress.json').write_text(json.dumps(progress, indent=2))
    # A declared final-epoch comparison, not best-of-test checkpoint selection.
    print('DONE', step, 'updates', time.monotonic()-start, 'seconds', flush=True)


if __name__ == '__main__':
    main()

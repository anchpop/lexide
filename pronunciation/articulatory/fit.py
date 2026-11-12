"""Direct fit of one clip: audio → per-frame mouth, lung pressure and glottis.

Free harmonic and noise envelopes are first fitted through the renderer alone.
The tract and source are fitted to those envelopes, then refined on the
waveform. No phone information enters.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from maeda import NAMES, geometry
from objective import AudioLoss, mel_bank
from synth import FFT, SR, Synth, effort, effort_weights, levels, smoothness

OUT = Path(__file__).parent / 'out'
SMOOTHNESS, EFFORT = 1e-4, .1
KNOT = 20  # frames per lung-pressure knot: it moves slowly


class Maeda(nn.Module):
    """Seven factors and velum per frame; pharynx and oral scale for the clip."""
    junction = 16

    def __init__(self, frames):
        super().__init__()
        raw = torch.zeros(frames, len(NAMES))
        raw[:, -1] = -2.
        self.raw = nn.Parameter(raw)
        self.scale = nn.Parameter(torch.zeros(2))

    def anatomy(self):
        return .8 + .4*self.scale.sigmoid()

    def forward(self):
        return geometry(self.raw, self.anatomy(), True)[:3]

    def penalty(self, weights):
        return SMOOTHNESS*smoothness(self.raw) + EFFORT*effort(self.raw, weights)


def vocode(synth, audio, f0, noise, steps=600):
    """Free per-frame harmonic and noise envelopes through the renderer: what any tract could at best say."""
    bank = mel_bank(FFT, audio.device)
    unmel = bank / bank.sum(0).clamp_min(1e-8)
    envelopes = nn.Parameter(torch.full((len(f0), 2, 80), -4., device=audio.device))
    optimizer = torch.optim.Adam([envelopes], lr=.05)
    objective = AudioLoss(audio)
    for _ in range(steps):
        harmonic, aperiodic = (envelopes @ unmel).exp().unbind(1)
        loss = objective(synth.render(synth.at_harmonics(harmonic, f0), aperiodic[None], f0, len(audio), noise))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return harmonic.detach(), aperiodic.detach()


def fit(audio, f0, voiced, tract=None, steps=(400, 1000), log=None):
    """Returns the tract, raw source [frames, 2], speaker colour and the waveform."""
    frames = len(f0)
    synth = Synth(closures=True).to(audio.device)
    tract = (tract or Maeda(frames)).to(audio.device)
    noise = torch.randn(1, len(audio), device=audio.device)
    lung = nn.Parameter(torch.zeros(frames//KNOT + 2, device=audio.device))
    # The folds start together where there is pitch and apart elsewhere.
    glottis = nn.Parameter(2 - 4*voiced.round())
    colour = nn.Parameter(torch.zeros(3, 8, device=audio.device))
    parameters = [*tract.parameters(), lung, glottis, colour]
    targets = vocode(synth, audio, f0, noise)
    bank = mel_bank(FFT, audio.device)
    objective, weights = AudioLoss(audio), effort_weights(audio)
    number = (synth.frequencies / f0[:, None]).clamp_min(1.)

    def source():
        # Recording level is unknown, so the clip's peak pressure is 8 cmH2O by definition.
        return torch.stack((F.interpolate((lung-lung.max())[None, None], size=frames, mode='linear', align_corners=True)[0, 0],
                            glottis), -1)

    for count, waveform in zip(steps, (False, True)):
        optimizer = torch.optim.Adam(parameters, lr=.03)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, count, .003)
        for step in range(count):
            if waveform:
                output = synth.sound(*tract(), source(), f0, voiced, len(audio), noise, colour, tract.junction)
                loss = objective(output, (128, 256) if step < 100 else (256, 512, 1024))
            else:
                harmonic, aperiodic, tilt = synth.spectra(*tract(), source(), colour, tract.junction)
                loss = sum((((model @ bank.T).clamp_min(1e-5).log() - (target @ bank.T).clamp_min(1e-5).log()).abs().mean(-1)
                            * weight).mean() for model, target, weight in (
                                (harmonic / number**tilt[:, None], targets[0], voiced), (aperiodic, targets[1], 1.)))
            if log and step % 100 == 0:
                print(log, 'waveform' if waveform else 'envelope', step, round(float(loss.detach()), 4), flush=True)
            # Blowing is effort too, where nothing at all is heard.
            loss = (loss + tract.penalty(weights) + SMOOTHNESS*smoothness(glottis)
                    + 20*EFFORT*((weights > 19)*source()[:, 0].sigmoid().square()).mean())
            assert torch.isfinite(loss), step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.)
            optimizer.step()
            schedule.step()
    with torch.no_grad():
        return tract, source(), colour.detach(), synth.sound(*tract(), source(), f0, voiced, len(audio), noise, colour, tract.junction)


def main():
    destination = OUT / 'fit'
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    for row in json.loads((OUT / 'ablation-data/clips.json').read_text()):
        stem = Path(row['file']).stem
        data = np.load(OUT / 'ablation-data' / (stem+'.npz'))
        audio, f0, voiced = [torch.tensor(data[k], device='cuda') for k in ('audio', 'f0', 'voiced')]
        tract, source, colour, output = fit(audio, f0, voiced, log=stem)
        sf.write(destination / f'{stem}-fit.wav', output.cpu().numpy(), SR, subtype='FLOAT')
        sf.write(destination / f'{stem}-original.wav', data['audio'], SR, subtype='FLOAT')
        with torch.no_grad():
            areas, _, coupling = tract()
            np.savez(destination / f'{stem}-fit.npz', articulation=tract.raw.cpu().numpy(),
                     source=levels(areas, coupling, source).cpu().numpy(), lung_glottis=source.cpu().numpy(),
                     colour=colour.cpu().numpy(), scale=tract.anatomy().cpu().numpy(), f0=data['f0'],
                     voiced=data['voiced'], phone=data['phone'], relative=data['relative'],
                     phonemes=data['phonemes'], boundaries=data['boundaries'], sentence=row['sentence'],
                     samples=len(audio), sample_rate=SR, control_hz=100, closures=True)


if __name__ == '__main__':
    main()

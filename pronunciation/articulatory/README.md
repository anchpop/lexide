# Articulatory synthesis

A differentiable vocal-tract synthesizer whose mouth is moved by gradient
descent or by a small network, trained only on audio. Mouth geometry → tube
areas → frequency-domain chain-matrix acoustics (with a nasal side branch) →
harmonic + noise source → waveform. The loss is a multi-resolution STFT
distance to the source recording.

Two ways to drive the mouth:

- **Direct fit** (`fit.py`): optimize one clip's trajectories against its
  audio. This is audio → articulation inversion. Keep it free of
  phoneme-specific priors: it is the path that could show a learner what
  *they* did, so it must not be pulled toward native gestures.
- **Network** (`train.py`, `network.py`): phonemes + timing + language +
  speaker + source F0 → controls. Speaker sets anatomy (pharynx and oral
  scale) and source character only, never articulation.

## Files

- `maeda.py`: Maeda's articulatory model (PB1 tables from VTCalcs,
  Apache-2.0, see `maeda_data/`). Seven factors plus velum, resampled to 40
  sections, two-part speaker anatomy scale.
- `synth.py`: tube acoustics, source, smoothness and effort (rest-posture)
  priors.
- `objective.py`: audio losses. `evaluate.py`: phone edit distance between our
  small phonetizer's free decode of the resynthesis and of the original. This
  is the metric throughout. It is never a training loss.
- `prepare.py`: Chirp3-HD English TTS clips with CTC alignment and Praat F0
  (`ablation` = 10 dev clips, `corpus` = 2017 train + 20 held-out
  unseen-sentence clips).
- `render.py`: midsagittal animation with audio.
- `sanity.py`, `test_articulatory.py`: acoustic sanity and regression tests.

## Running

```sh
export LEXIDE_DATA_VENV=~/.venv-articulatory LD_LIBRARY_PATH=/run/opengl-driver/lib
cd /data/coding/lexide/pronunciation
scripts/py-linux.sh articulatory/prepare.py ablation   # or: corpus
scripts/py-linux.sh articulatory/fit.py                # direct fits → out/fit
scripts/py-linux.sh articulatory/train.py              # network → out/network-anatomy
scripts/py-linux.sh articulatory/render.py out/network-anatomy/epoch-19/<stem>-fit.npz
scripts/py-linux.sh articulatory/test_articulatory.py
```

The venv holds CUDA torch, numpy, scipy, soundfile, praat-parselmouth,
onnxruntime and matplotlib (`requirements.txt`). Outputs go to `out/`, which is
excluded via `.git/info/exclude`.

## What we found (Nov 2026)

Phone edits are counted against the phonetizer's reading of the original
audio. Lower is better.

| Setup | Edits |
|---|---|
| Direct fit, heuristic Gaussian-tube geometry, 500 steps | 95/179 |
| + constrictions allowed to close | 84/179 |
| + codebook init, 2000 steps | 74/179 |
| Direct fit, heuristic geometry + effort prior, 1000 steps | 96/179 |
| Direct fit, Maeda geometry, same settings | 132/179 |
| Direct fit, Maeda + silence-weighted effort | 147/179 |
| Network, heuristic geometry, 3 epochs | 279/378 held-out |
| Network, Maeda + gesture loss + two-part anatomy, 19 epochs | 229/378 held-out |

- **Gradients can steer a tube model.** Fits are intelligible to the
  phonetizer, but they sound muffled and robotic.
- **The audio loss cannot tell human movement from impostor movement.**
  Fitted controls slammed rail to rail. The tongue hugged the palate, nasals
  never lowered the velum, /k g/ never raised the dorsum, and the mouth
  thrashed in silence. An effort prior (distance from rest) fixes the
  thrashing for little accuracy. Log-mel loss was worse than linear plus
  log-linear STFT.
- **Maeda's model gives real tongue shapes but fits audio worse** than the
  free-form geometry. It still does not make movements human on its own.
- **Phoneme-class gesture hinges** (bilabial closure, nasal velum, alveolar and
  velar constriction), used in the network only, took the network from 0% to
  95%+ on most gesture checks. Velars still fail: the network keeps a near-lip
  constriction that undercuts the dorsal one. These are hand-written rules,
  not evidence of human motion.
- **Per-speaker anatomy is learnable from audio.** The learned pharynx scale
  separates all 30 voices by published gender (female 0.85–0.95, male
  0.98–1.13). The male:female ratio is about 1.18 for the pharynx and 1.06 for
  the oral cavity, matching real anatomy. Nine female oral scales sit on the
  0.85 lower bound, so the range should widen.
- Watch for sigmoid saturation in network outputs: one control once drifted to
  raw −7.7 and froze. The saturation penalty prevents it.

## Next

Use the USC 75-speaker real-time MRI corpus (Lim et al. 2021, CC BY) and the
Annot-16 articulator tracks (16 speakers, research use only, do not
redistribute), downloaded to `pronunciation/.work/usc75/`. The plan:

1. Test whether Maeda can represent real MRI tongue shapes. If it cannot,
   build a factor model from the MRI.
2. Derive real control trajectories and per-speaker anatomy.
3. Train the network with articulatory supervision in place of the gesture
   hinges.

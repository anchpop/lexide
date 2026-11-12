# Articulatory synthesis

A differentiable vocal-tract synthesizer whose mouth is moved by gradient
descent or by a small network, trained only on audio. Mouth geometry → tube
areas → frequency-domain chain-matrix acoustics (with a nasal side branch) →
harmonic + noise source → waveform. Voicing and noise are not free: they
follow from lung pressure, the glottal opening and the air the tract lets
through. The loss is a multi-resolution STFT distance to the source recording.

Two ways to drive the mouth:

- **Direct fit** (`fit.py`): optimize one clip's per-frame mouth, glottis and
  lung pressure against its audio, starting from a fit to free spectral
  envelopes. This is audio → articulation inversion. Keep it free of
  phoneme-specific priors: it is the path that could show a learner what
  *they* did, so it must not be pulled toward native gestures.
- **Network** (`train.py`, `network.py`): phonemes + timing + language +
  speaker + source F0 → controls. Speaker sets anatomy (pharynx and oral
  scale) and source character only, never articulation.

## Files

- `maeda.py`: Maeda's articulatory model (PB1 tables from VTCalcs,
  Apache-2.0, see `maeda_data/`). Seven factors plus velum, resampled to 40
  sections, two-part speaker anatomy scale.
- `synth.py`: tube acoustics (chain matrices, yielding wall, baffled-piston
  radiation, open-glottis termination; preset formants within 10 Hz of
  VTCalcs), aerodynamic source (`aerodynamics`) with a static per-speaker
  colour, smoothness and effort (rest-posture) priors.
- `mri_tract.py`: tract gaps and section lengths measured straight from MRI
  outlines along gridlines, for the tube solver, with no Maeda fit in between.
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

**Coordinate compatibility:** contours now use a rigid upper-incisor/palate
anchor, not the current glottal midpoint. Tube lengths, areas and acoustics are
unchanged apart from floating-point roundoff. Older MRI registrations, fitted
controls and contour-supervised checkpoints were learned in a moving frame;
refit/re-register them before evaluating with the new geometry. A single global
translation cannot repair that frame-dependent change. The original inversion
benchmark remains unmodified; its acoustics diagnosis uses a local legacy-module
snapshot under `out/acoustics-diagnosis/`.

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

## Clean audio against EMA (Nov 2026)

Direct fits to clean, non-scanner recordings, scored against EMA coils
(BY2014: one French speaker, CC BY-SA; USC-TIMIT EMA: one English speaker,
CC BY). Coils are registered to the model per speaker on separate utterances.
Scripts and results are in `out/clean-inversion/`.

- **The audio-only fit does not move like the speaker on clean audio either.**
  Its tongue points sit as far from the coils as a rest posture does, and its
  trajectories correlate about 0.2 with them. Letting the model follow the
  coils directly reaches 0.7–0.8, so Maeda can make the movements.
- **The audio loss is flat, not wrong.** A fit held near the coils reaches the
  same audio loss as the free fit, with about three times the correlation.
  Started at the coil-following mouth and left free, the fit walks away within
  200 steps and ends at an equal or worse loss. Different starting postures
  end in different mouths with similar losses.
- **Reshaping the loss does not fix this.** A smoothed-envelope loss, an
  envelope loss with level, tilt and broad colour removed, and a coarse-to-fine
  schedule all leave the coil-following mouth and none ranks it first.
- **The coil-following mouth itself sounds no better than a rest posture**
  through our acoustics. Five fleshpoints mapped onto one speaker's factor model
  do not fix constriction areas, the pharynx or the nasal port well enough, so
  it is a movement reference, not an acoustic one.
- **A phone-conditioned starting mouth helps.** Starting the English fits from
  the network's prediction gives a lower audio loss and twice the correlation of
  a neutral start, though the fit still degrades the prediction.
- The yielding wall and radiation fix took the ten TTS clips from 147/179 to
  125/179 edits with nothing else changed.

## Real mouths through the synthesizer (Nov 2026)

Forward direction only: traced MRI outlines in, audio out, scored on two
held-out speakers of the USC 75-speaker corpus. Scripts, numbers, wavs and
figures are in `out/forward/`. Annot-16 is research-only, so nothing derived
from it is in the source tree.

- **Measure the tract from the outlines, not through a Maeda fit.** Gridline
  gaps with Maeda's own α/β and the old source already beat the Maeda-fit
  pipeline, most of all on vowels. Their vowel formants sit near those of the
  recordings: F2 closely, F1 a little low.
- **The audio loss is the wrong thing to calibrate the tract with.** Learning
  α, β and an outline offset end-to-end lowers the loss and corrects F1, but it
  does so by making the tube narrow and insensitive to the mouth. F2 collapses
  toward a neutral tube and fewer phones survive. Left unbounded, it shrinks the
  areas until wall losses flatten every formant. Keep Maeda's values until there
  is a formant-level target.
- **A source that reads level, voicing and tilt from the audio helps a
  little.** Praat calls a large share of the vowel frames in scanner audio
  unvoiced, and the old source is silent there.
- **The renderer is not the limit.** Free per-frame envelopes through the same
  harmonic-plus-noise renderer and the same F0 are read nearly as well as the
  originals.
- **The outlines are.** A network with no physics, from every traced outline to
  spectral envelopes, is hardly more intelligible than the tube model, even on
  speakers it trained on.
- **The true mouth now beats other real mouths.** It scores better than the
  mouths of nearly all other recordings and than itself shifted in time, with
  the minimum at zero shift. It still loses to a free fit: started at the
  truth, a fit walks away and ends lower, and a fit from a neutral mouth does
  not find it.

## What the tube can do on clean audio (Nov 2026)

One area function per 10 ms frame, fitted to the ten TTS dev clips with
nothing between the parameters and the tube: no knots, no effort prior. Each
fit starts from a fit of the tube's own envelopes to free harmonic and noise
envelopes (the renderer's ceiling), then refines on the waveform. Scripts,
wavs, figures and per-class numbers are in `out/physics-ceiling/`.

| Tract, per frame | Old source | + open glottis, speaker colour |
|---|---|---|
| 40 free areas, length, nasal port | 15/179 | 9/179 |
| Maeda's seven factors + velum | 47/179 | 19/179 |
| Free envelopes through the renderer, no tube | | 5/179 |

On 20 held-out clips from 15 voices the right-hand column reads 9/378,
24/378 and 2/378.

- **The tube is not the limit.** Free areas are read nearly as well as the
  renderer allows, and Maeda's eight controls lose a tenth of the phones, not
  the 70% of earlier fits. `fit.py`, unchanged but for the new source, goes
  from 125/179 to 52/179; per-frame controls from an envelope start do the
  rest.
- **Start from envelopes.** The same free tube from a neutral start ends at
  28/179. Fitting the tube to smooth target envelopes first roughly halves
  the errors, and it is what makes Maeda fits work.
- **Two physics changes paid, on every phone class.** A glottis that opens
  into the trachea (one area per frame) damps the formants, which a hard wall
  leaves too sharp: 0.4 dB of log-mel error on vowels with free areas, 0.7 dB
  with Maeda. A static colour per speaker (eight cosine coefficients each for
  voice, breath and hiss) absorbs the glottal spectrum and the level between
  the three sources: 0.5 dB on every class with free areas, about 1 dB with
  Maeda. Free areas can stand in for much of either; Maeda's cannot, and the
  two together take its vowels from 52 to 65 of 73 kept and its stops from 34
  to 41 of 43.
- **Three did not, and are not in `synth.py`.** Noise loaded by the back
  cavity as well as the front, a maxillary sinus on the nasal branch and
  sound radiated by the walls during closure each changed the log-mel error
  of every class by 0.2 dB or less with the others in place.
- **Phone identity is in the areas.** Giving each phone the tract of another
  while keeping its source leaves under a fifth of the phones readable;
  flattening the source while keeping the tract leaves half to two thirds.
- **Free areas are a filter, not a mouth.** They close near the larynx for
  stops, never open the port for nasals and put constrictions anywhere. Maeda's
  factors put /i/ at the palate with a wide pharynx and /s/ behind the teeth,
  but stops close wherever is convenient and nasals are still made with a
  narrow mouth and a shut port. Intelligibility now costs little anatomy; the
  audio still does not say which closure or whether the velum is down.
- An MRI-derived factor space for the mouth fits about as well as Maeda at the
  same size and nearly as well as free areas at twice it, with the same blind
  spots.

## Sound from air flow (Nov 2026)

The free source gains are gone. A frame now has a mouth, a glottal opening
and a lung pressure that moves only every 200 ms. `synth.aerodynamics` works
out voicing, aspiration, frication and release bursts from the air these let
through: steady orifice flow from the lungs through glottis and pharynx, then
mouth and nose side by side, with the pressure behind a closure building over
tens of ms. Flow through a narrow gap also damps the sound passing it.
`fit.py` is the per-frame fit with envelope start, plus a rest-posture prior.
Same clips and scoring as above. Scripts, wavs, figures, videos and numbers
are in `out/aero/`.

| Tract, per frame | Free source gains | Aerodynamic source |
|---|---|---|
| 40 free areas, length, nasal port | 9/179 | 11/179 |
| Maeda's seven factors + velum | 19/179 | 36/179 |

On the 20 held-out clips: free areas 9/378 → 7/378, Maeda 24/378 → 39/378.
The MRI-derived mouth space pays about what Maeda does.

- **Silence no longer needs a closure.** With the old gains the Maeda fits
  kept the mouth sealed through 93% of non-speech. Now lung pressure falls to
  1 cmH2O there against 4 in speech (the clip's peak is 8 by definition), the
  mouth is sealed in 14% of silent frames and sits near its rest posture.
- **The mouth alone carries half the phones.** With lung pressure and glottis
  held flat over the clip, Maeda fits still get half read, two thirds of the
  fricatives among them. Over half of /p t k/ release with a burst that
  nothing but the stored pressure makes.
- **Free areas pay nothing for it; Maeda and the MRI space pay.** Their
  errors roughly double. A closure is now heard, and theirs are in the wrong
  place: /ɡ/ is read as /b/ or /d/, /n/ as /m/.
- **Each tract has one closure for everything.** Stops are shut about as
  often as before (80%). Maeda now shuts the lips for /t d n/ and /k ɡ ŋ/ as
  well, about 10% of which close at the right place, fewer than when it used
  the tongue blade for all. The MRI space still uses the tongue blade, and so
  gets half of /t d n/ and few of the others.
- **The loss does not care where.** Moving every such closure to its textbook
  place with open lips and refining both mouths ends 0.4% higher in loss, less
  than two runs of the same fit differ, and most of the moved closures stay.
  Soft floors, so that shut lips still feel a gradient, changed nothing.
- **The nasal port stays where it starts.** Started shut, it stays shut: a
  murmur through a 0.05 cm² leak at the lips is a real sound and enough air
  for the folds. With the loss that flow causes in such a gap an open port is
  ten times louder than that leak and twice as loud as a 0.1 cm² one, yet the
  fit takes the leak. Started from the phone network's mouth (ten held-out
  clips), the port stays open on 79% of nasals with the same loss and edits.
  Started half open everywhere, it stays open on most vowels too.
- **Real mouths are no closer.** On the clean EMA recordings the same fit now
  reads far better (BY2014 27/121 edits, was 84/121; USC-TIMIT F1 53/136, was
  109/136), but its movements correlate with the coils as before (0.25 and
  0.22, was 0.21 and 0.21; coil-following ceiling 0.79 and 0.70) and its
  tongue points are further off (6.4 and 7.9 mm, was 4.9 and 5.9, a rest
  posture 4.8 and 5.8). Started at the coil-following mouth it walks away to
  the same loss (0.34 and 0.42). Held at that mouth, the sound is far worse
  (loss 1.6–1.7 against 1.0).
- **Smoother is not more human here.** Ten times the smoothness slows the
  controls from 11 to 4.5 ranges per second, leaves the closures where they
  were and costs phones (52/179 against 33/179 before the flow loss was
  added). The rest-posture prior costs none.
- The network takes the two source numbers but has not been retrained. Old
  checkpoints keep their mouths; their source heads were reset
  (`out/aero/migrate.py`), so their audio means nothing until then.

## Next

1. Give the fit a prior for what the audio leaves open: where a closure is
   and whether the port is open. Search will not find them, since the loss is
   flat there. The prior has to be about how mouths move, not about the phone,
   or the direct fit can no longer show a learner what they did: the
   MRI-derived mouth space with a model of its real trajectories (USC
   75-speaker corpus, Lim et al. 2021, CC BY, in `pronunciation/.work/usc75/`;
   its Annot-16 tracks are research use only, do not redistribute). Judge it
   by closure place, the port and the EMA benchmark, not by phone edits.
2. Retrain the network with the aerodynamic source and a colour per speaker.
   Its mouth is the only start so far that keeps the port open on nasals.
3. In Maeda fits over half of a fricative's noise is aspiration through a
   narrowed tract, not frication at the constriction (free areas: under a
   tenth). Check the two noise levels against flow measurements before
   trusting which one a fit picks.

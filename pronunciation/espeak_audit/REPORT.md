# Where espeak goes wrong, measured on real voices

*Audit built overnight 2026-05-31. Question posed: don't fix anything yet —
just find, empirically, where espeak's transcription diverges from what real
speakers actually produce.*

## TL;DR

espeak emits a **broad, citation-form** transcription. On real audio it is
systematically wrong in two ways that matter for an IPA model, plus one place
where it's right and we can now prove it automatically:

1. **English flapping — espeak misses it entirely.** Intervocalic /t,d/ are
   realized as a voiced flap **[ɾ]**. Measured on CK (human, Tatoeba): voiced
   fraction of the closure **1.0** intervocalically vs **0.3** elsewhere, with
   **no release burst** (−27 dB vs −15 dB). Google TTS (Algieba) shows the same
   (voiced 0.6 vs 0.0). espeak writes /t/ or /d/ every time.
2. **English /u/-fronting — espeak marks it back, it isn't.** espeak's `uː`
   measures **F2 ≈ 1646 Hz** for CK and **≈ 1281 Hz** for Google TTS — fronted
   to ~[ʉ]. A back /u/ would sit near 850–1000 Hz.
3. **French nasalization — espeak is RIGHT, and we can confirm it from the
   signal.** Nasal vowels show F1-bandwidth widening (oral **99 Hz** → nasal
   **200–287 Hz**) and the A1-P0 drop (**+3.3 → −1.5 dB**) that are the textbook
   acoustic signatures of nasal coupling. This reproduces *automatically* the
   hand-analysis from the original French chat.

The headline methodological result, which validates the whole approach:

> **Neither vad-clean NOR Allosaurus catches the flap — only the acoustics do.**
> vad-clean's greedy reading is *character-for-character identical to espeak*
> (it was trained on espeak labels). Allosaurus — a CMU phone recognizer trained
> on ~2000 languages, which never saw espeak or our data — transcribes the flaps
> as plain `t` too. Every *model* inherits broadness. Only direct acoustic
> measurement (closure voicing + burst) recovers the narrow truth. This is
> exactly why a model can't be allowed to vote on its own future training labels.

## Architecture (and the anti-circularity guardrail)

```
text ──(espeak FORK, broad/citation)──▶ phoneme sequence ──(vocab.json)──▶ ids
audio ─(vad-clean on Modal, T4 fp16)──▶ CTC log-probs ─(torchaudio.forced_align)─▶ per-phone time spans
spans + audio ──(parselmouth, LOCAL)──▶ acoustic measurements  ◀── THE ARBITER
                                          (+ Allosaurus, independent cross-check)
```

- **The model only places time boundaries.** Where the /t/ region *is* in time
  is an acoustic-landmark task (the constriction is there regardless of label).
  It gets **no vote** on what the phone *is*. The verdict comes from acoustics,
  which are independent of both espeak and the model.
- This sidesteps the ouroboros: if v1's *likelihood* picked pronunciation
  variants for v2's labels, v1's errors become v2's targets. We never do that.
- Bonus independent signal: forced-aligning espeak's broad sequence makes
  **over-specification visible** — a phone espeak posits that isn't in the audio
  gets squeezed to ~0 frames with a poor alignment score (no model judgment of
  identity needed). Careful read/TTS speech here rarely deletes (CK: 0 candidates,
  Schedar: 6 low-rate), as expected; this will matter on connected speech.

## Findings in detail

### 1. Flapping (English) — `out/figs/*flap_voicing.png`
| context | n | voiced frac (median) | burst hi−lo dB (median) |
|---|---|---|---|
| CK intervocalic /t,d/ | 33 | **1.00** | **−27.4** (no burst) |
| CK elsewhere | 185 | 0.30 | −14.7 |
| Algieba intervocalic | 25 | **0.60** | −16.5 |
| Algieba elsewhere | 116 | 0.00 | −10.1 |

42% (CK) / 48% (Algieba) of intervocalic tokens are clearly flap-like (short +
voiced). espeak: 0% flaps (writes /t,d/). Allosaurus: 0% flaps (writes t/tʰ).

*Robustness check:* re-measured on the **raw** CTC span (centered on the stop
core, before window dilation, so no bleed from neighbouring vowels) the contrast
holds and sharpens — CK intervocalic voiced **1.0** vs **0.22** elsewhere;
Algieba **0.67** vs **0.0**. The dilated window did not manufacture the effect.

### 2. /u/-fronting (English) — `out/figs/eng_tatoeba_CK.vowelspace.png`
espeak `uː`: CK F2 **1646 Hz** (n=50), Algieba F2 **1281 Hz** (n=18). In the
vowel chart `uː` lands mid-front near `ə`, nowhere near the genuinely back
`ʊ/o/ɔ`. `ʊ` stays backer (CK F2 1128) as expected.

### 3. Vowel reduction (English)
Unstressed vowels shorten (CK: stressed **80 ms** → unstressed **60 ms**, real
midpoint-boundary durations). espeak partially captures this via `ə/ɐ/ᵻ`, so
it's a softer miss than flapping or /u/-fronting.

### 4. Nasalization (French) — `out/figs/fra_*nasal_b1.png`
| vowel | n | B1 (Hz) | A1−P0 (dB) |
|---|---|---|---|
| oral reference | 393 | 99 | +3.3 |
| ɑ̃ | 40 | 287 | −1.4 |
| ɔ̃ | 24 | 227 | −1.5 |
| ɛ̃ | 9 | 197 | +1.4 |
| œ̃ | 6 | 220 | −0.8 |

Allosaurus independently renders these as oral-vowel + `ŋ` (e.g. `ɔ̃`, `o ŋ`),
agreeing that nasality is present. **espeak is correct here** — and we can now
say so from the signal.

### 5. Schwa vs ø (French)
For this TTS voice, `ə` (F2 1483) stays distinct from `ø` (F2 1180, but only
n=4). The `ə≡ø` *merge* the original chat found was **speaker-specific**; it does
not appear in this voice. Per-speaker anchoring matters — this is why we never
compare to textbook Hz.

## Limitations (honest)
- **CTC alignment is peaky** (~1 frame/phone). Fixed for measurement by dilating
  windows into blank gaps, and for duration by midpoint-boundary segmentation.
  Raw spans are kept only as the over-specification signal.
- **A1−P0 is skipped for F1 < 400 Hz** (F1 sits in the P0 band → confounded).
- **Neural recognizers inherit broadness** — Allosaurus is a useful *cross-check*
  for nasality but is NOT a reliable narrow arbiter (missed every flap). Trust
  the DSP.
- **Coverage**: 60 clips × 3 voices, one human English speaker (CK dominates
  English Tatoeba), TTS voices are "careful" (little connected-speech reduction).

## What this implies for labels (non-circular)
1. **Rule-governed allophony → deterministic rules on espeak's output.** Flapping
   (intervocalic /t,d/ → [ɾ]) and /u/-fronting are regular AmE processes. Apply
   allophonic *rules* to espeak's broad string to get a narrow citation form —
   cheap, and independent of any model. Optionally acoustic-verify.
2. **Speaker-specific / variant cases → adjudicate with acoustics, never model
   likelihood.** The feature extractors here (closure voicing, burst tilt, B1,
   A1−P0, formants) run at scale and are the legitimate arbiter.
3. **Allosaurus** is worth keeping as an independent nasality/segment cross-check,
   not as ground truth.

## Reproduce
```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python3
# Modal apps already deployed: espeak-audit-aligner (vad-clean), allosaurus
$PY espeak_audit/run_audit.py --lang eng --voice CK --source tatoeba --n 60
$PY espeak_audit/analyze.py --tag eng_tatoeba_CK --lang eng
$PY espeak_audit/allosaurus_check.py eng_tatoeba_CK eng eng 25
$PY espeak_audit/plots.py
```
Files: `phonetics.py` (DSP arbiter), `modal_aligner.py` (Modal/T4 aligner),
`run_audit.py` (espeak→align→measure driver), `analyze.py` (detectors),
`allosaurus_check.py` (independent check), `plots.py`. Outputs in `out/`.

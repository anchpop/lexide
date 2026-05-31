# Flap relabeling — per-token acoustic narrowing PoC

Proof of concept for the general pattern that takes espeak from broad→narrow on a
per-token basis: **espeak broad label → forced-align → per-token acoustic
measurement → confidence-gated relabel → narrower label.** Flapping (American
English intervocalic /t,d/ → tap [ɾ]) is the first phenomenon; the structure
generalises to any phenomenon whose per-token signal is acoustically separable.

## Why flapping first
The 4-source audit showed intervocalic /t,d/ flapping is per-token **separable**:
`voiced_frac` is ~bimodal (CK: 68% clearly flap / 17% clearly stop / 15%
ambiguous). A true [t] has a voiceless closure + burst; a flap is voiced
throughout with no burst — so `voiced_frac` (fraction of the closure that's
voiced) reads the same physical event a human ear judges. **Ear-validated**: on a
10-clip spot check spanning the confidence range, the detector's calls matched
listening.

## The detector (`flap_relabel.py::classify`)
Intervocalic /t,d/ only (vowel before AND after, across word boundaries — "that
it's" flaps). 
- `/t/`: `voiced_frac ≥ 0.60 and burst < 5 dB` → **[ɾ]** (high conf if vf≥0.75);
  `vf ≤ 0.35` → keep [t]; else ambiguous → keep [t].
- `/d/`: voicing can't separate [d] from [ɾ] (both voiced) → uses duration
  (`≤45 ms and voiced`). Conservative; under-flaps /d/ on purpose (cost is low —
  [d] and [ɾ] are acoustically close).

**Confidence gate:** only confident flaps are relabeled; ambiguous tokens and all
stops keep the broad symbol. This bounds injected label noise — the labels become
*strictly narrower where the signal is unambiguous, identical elsewhere*.

## Applied result (`flap_apply.py`, English audit clips)
| source | clips | intervoc /t,d/ | → [ɾ] |
|---|---|---|---|
| tts (Algieba) | 240 | 79 | 19 (24%) |
| fleurs (pooled) | 240 | 289 | 48 (17%) |
| tatoeba (CK, human) | 240 | 125 | 20 (16%) |
| **total** | 720 | 493 | **87 (18%)** |

Pimsleur is **excluded**: its English manifest contains mixed-language
instructional rows (Arabic/Korean text read by the en-us voice), so its phoneme
streams aren't English and its flap counts would be meaningless.

**Genuine drop-in.** Output `out/<tag>.phonemes_flapped.jsonl` starts from the
real `phonemes.jsonl` rows (preserving `source`/`stress`/`whisper_*`/
`duration_sec`) and rewrites the **`phonemes`** field that training
(`StressDataset`) actually reads — the narrowed sequence — keeping the broad
original under **`phonemes_espeak`**. It only relabels a clip whose audit-measured
sequence EXACTLY matches that clip's `phonemes.jsonl` sequence (so a position is
never misattributed); non-matching clips are skipped + counted (0 skipped here —
all 720 aligned).

## Limitations / honesty
- Demonstrated on the 720 clean-English audit clips (measurements already
  computed). It is NOT yet applied to the uploaded training labels.
- `/d/` is under-flapped (duration path is shakier than /t/'s voicing path).
- Thresholds are first-guess + ear-validated on a *small* sample; a blind ~25-token
  calibration would give a real accuracy number before trusting at scale.
- The "intervocalic" filter is crude (catches some onset /t/ across a boundary);
  voiced_frac correctly rejects those, but a stress-aware context rule is cleaner.

## To ship at full scale
Run `run_audit`'s align+measure over ALL eng clips (the cost step — Modal align +
parselmouth over 111k clips, plus the long-clip perf fix), point `flap_apply` at
those segments, then merge `phonemes_flapped` into `phonemes.jsonl` and re-upload.
The relabel itself is free (`ɾ` is already in the vocab).

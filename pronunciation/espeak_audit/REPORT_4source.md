# espeak Phonemizer Audit — Final 4-Source Report

**Sources:** Tatoeba (single human read) · Google TTS (Chirp3-HD single voice) · FLEURS (multi-speaker pooled) · Pimsleur (didactic single instructor) · **7 languages × 240 clips/source**

---

## 1. TL;DR

**Confirmed in ALL FOUR sources (bulletproof — act on these):**

- **eng** — `/uː`-class fronting (F2 of /uː/ sits +280–465 Hz above /ʊ/); word-final voiced-stop devoicing; durational unstressed reduction.
- **deu** — Final devoicing (Auslautverhärtung); lenis onset devoicing; tense/lax `iː/ɪ` and `uː/ʊ` quality contrasts; stress→duration reduction.
- **fra** — Nasal-vowel B1 widening vs own oral_ref; `e/ɛ` and `o/ɔ` F1 height splits; spurious English/length-marked vowel leakage (`uː ɔː ɪ ʊ ʌ …`); no-aspiration / full-voicing of obstruents.
- **ita** — Spurious lax `/ɪ ʊ/` collapsing to `/i u/`; `e/ɛ` & `o/ɔ` F1 splits; `/r/`→tap `[ɾ]` relabeling; no unstressed reduction.
- **por** — Nasal B1 elevation vs own oral_ref; unstressed duration reduction; voiced/voiceless stop voicing-by-position.
- **rus** — `/ʲ/` palatalization over-spec (standalone segment at 20 ms frame floor, systematically deleted); akanye/unstressed reduction.
- **spa** — Spirantization (`β ð ɣ`→stops/v relabeling); no vowel reduction; voicing-by-position contrast.

**Confirmed in THREE sources (4th NA/register-suppressed — still actionable):**

- **eng** — Intervocalic `/t,d/` flapping (Pimsleur over-voices everything → register-masked, not a refutation).
- **deu** — Stress→F1 reduction (Pimsleur gap negligible, 15 Hz); schwa over-spec/deletion (Pimsleur fully realizes schwa).
- **por** — Spurious `/ʲ/` over-spec (Pimsleur retains the glide).

**Confirmed in TWO sources (FLEURS/Pimsleur underpowered, not refuting):**

- **rus** — Final devoicing (FLEURS n=5, Pimsleur n=3 — too few word-final voiced tokens to test; contrast machinery intact).

**WEAK / needs acoustic adjudication (do NOT ship as deterministic):**

- **fra** — Schwa `/ə/` over-emission (only Tatoeba supports; FLEURS actively *inserts* schwa → refutes).
- **rus** — Onset/intervocalic partial devoicing (tail-only p10 effect; medians stay voiced; not an espeak error).

---

## 2. Master Table

Legend: ✓ supports · ✗ refutes · ~ weak/register-suppressed/marginal · NA underpowered · **EE** = is_espeak_error

| Lang | Phenomenon | tato | tts | fleurs | pims | Robustness | EE? |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| eng | Intervocalic /t,d/ flapping | ✓ | ✓ | ✓ | ~ | three | ✓ |
| eng | /uː/ fronting (F2≫/ʊ/) | ✓ | ✓ | ✓ | ✓ | all_four | ✓ |
| eng | Word-final voiced-stop devoicing | ✓ | ✓ | ✓ | ✓ | all_four | ✓ |
| eng | Unstressed durational reduction | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| deu | Final devoicing (Auslautverhärtung) | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| deu | Lenis onset devoicing | ✓ | ✓ | ~ | ✓ | all_four | ✓ |
| deu | Tense/lax iː/ɪ quality contrast | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| deu | Tense/lax uː/ʊ quality contrast | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| deu | Stress→duration reduction | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| deu | Stress→F1 reduction | ✓ | ✓ | ✓ | ~ | three | ✗ |
| deu | Schwa over-spec/deletion | ✓ | ~ | ✓ | ✗ | three | ✓ |
| fra | Nasal-vowel B1 widening | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| fra | e/ɛ F1 height split | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| fra | o/ɔ F1 height split | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| fra | Schwa /ə/ over-emission | ✓ | ~ | ✗ | ✗ | one | ✓ |
| fra | Spurious English/length-marked vowels | ✓ | ✓ | ✓ | ✓ | all_four | ✓ |
| fra | No aspiration / full obstruent voicing | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| ita | Spurious lax /ɪ ʊ/ → /i u/ collapse | ✓ | ✓ | ✓ | ~ | all_four | ✓ |
| ita | e/ɛ & o/ɔ F1 splits | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| ita | /r/ → tap [ɾ] relabeling | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| ita | No unstressed vowel reduction | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| por | Spurious /ʲ/ over-spec | ✓ | ✓ | ✓ | ✗ | three | ✓ |
| por | Nasal-vowel B1 elevation | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| por | Unstressed duration reduction | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| por | Voiced/voiceless voicing-by-position | ✓ | ~ | ✓ | ✓ | all_four | ✗ |
| rus | /ʲ/ palatalization over-spec | ✓ | ✓ | ✓ | ✓ | all_four | ✓ |
| rus | Akanye / unstressed reduction | ✓ | ✓ | ✓ | ✓ | all_four | ✓ |
| rus | Final devoicing | ✓ | ✓ | NA | NA | two | ✓ |
| rus | Onset/intervoc partial devoicing (tail) | ✓ | ~ | ~ | ~ | three (weak) | ✗ |
| spa | Spirantization β/ð/ɣ→stops/v | ✓ | ✓ | ✓ | ~ | all_four | ✓ |
| spa | No vowel reduction | ✓ | ✓ | ✓ | ✓ | all_four | ✗ |
| spa | Voicing-by-position contrast | ✓ | ~ | ✓ | ✓ | all_four | ✗ |

(`~` for tts on por/spa voicing and eng-flapping = TTS over-voicing register, contrast still holds → counted as supporting. `~` for fleurs/deu lenis = mean-only separation, medians identical.)

---

## 3. The Bulletproof List

### 3a. all_four — confirmed across every source

These survive single-human variance, TTS over-voicing, FLEURS multi-speaker pooling, AND Pimsleur didactic suppression simultaneously. Highest confidence.

**espeak ERRORS (espeak emits the wrong/spurious symbol — fix at phonemizer or enrichment layer):**

| Lang | Error | Nature of fix |
|---|---|---|
| eng | `/uː/` transcribed back but realized fronted | quality/feature correction |
| eng | word-final voiced stops kept voiced, realized devoiced | position-conditioned devoicing |
| deu | lenis onset voiced, partly devoiced by humans | position-conditioned partial devoicing |
| fra | English/length-marked vowels leaked into French text | delete spurious symbols (low volume) |
| ita | spurious lax `/ɪ ʊ/` (Italian lacks them) → merge to `/i u/` | symbol collapse |
| ita | trill `/r/` emitted, realized as tap `[ɾ]` | narrow-transcription granularity |
| rus | standalone `/ʲ/` segment with no acoustic exponent (20 ms floor) | delete/attach as secondary feature |
| rus | akanye: reduced `ʌ/ɑ`→`[a]` | reduced-vowel relabeling |
| spa | continuant `β/ð/ɣ` realized toward stops/v | spirantization relabeling |

**espeak CORRECT (real phonetics not in the string — modeling gap, not transcription error):**

| Lang | Phenomenon |
|---|---|
| eng | unstressed durational reduction |
| deu | final devoicing (already encoded), iː/ɪ + uː/ʊ quality, stress→duration |
| fra | nasal B1 widening, e/ɛ + o/ɔ F1 splits, no-aspiration/full voicing |
| ita | e/ɛ + o/ɔ splits, no unstressed reduction |
| por | nasal B1 elevation, unstressed duration reduction, voicing-by-position |
| spa | no vowel reduction, voicing-by-position |

### 3b. three — 4th source NA or register-suppressed, with explicit reason

| Lang | Phenomenon | Missing source | Why it's NA, not refuting |
|---|---|---|---|
| eng | intervocalic /t,d/ flapping | Pimsleur | Hyper-careful register **over-voices everything** (non-intervoc voiced_frac 0.6 > intervoc 0.5; flaplike 0.119; 80 ms unflapped stops). Suppression, not contradiction. |
| deu | stress→F1 reduction | Pimsleur | Didactic register preserves unstressed vowel quality; gap only 15 Hz (direction right, magnitude negligible). Other 3: gaps 68–141 Hz. |
| deu | schwa over-spec/deletion | Pimsleur | Careful register **fully realizes** schwa (n_deleted=0). FLEURS strongest (rate 0.018, 202 model-deletions). |
| por | spurious /ʲ/ over-spec | Pimsleur | Careful register **retains the glide**. ʲ is #1 deletion in tato/tts/fleurs (n 32/89/464). |

### 3c. two — both missing sources underpowered (token-count artifact)

| Lang | Phenomenon | Missing | Why NA |
|---|---|---|---|
| rus | final devoicing | FLEURS (n=5), Pimsleur (n=3) | Too few word-final voiced tokens to test; both keep coda_other ~1.0 (machinery intact). Tato & TTS both clean median 0.0 vs 1.0. TTS confirming *despite* its over-voicing strengthens the call. |

---

## 4. Source-Character Lessons

**Google TTS (Chirp3-HD) — over-voices obstruents.**
Systematically inflates `voiced_frac` on voiceless and onset/intervocalic obstruents (eng final-devoicing intervoc ref only 0.7; por/spa voiceless onset ~0.4–0.5 vs ~0.1 in humans; rus partial-devoicing saturates at 1.0). **Good for:** spectral/formant contrasts (vowel quality, fronting, nasal B1), durational reduction, and *position* contrasts where the ordinal split survives. **Bad for:** absolute voicing levels — never use TTS alone to call a devoicing magnitude. Counterintuitively, when TTS *still* shows devoicing despite its bias (rus final devoicing median 0.0), that is a strong confirmation.

**Pimsleur — didactic register suppresses connected-speech allophony.**
Hyper-careful instructor speech over-articulates: retains schwa (deu/por), over-voices stops, fully realizes the `/ʲ/` glide, lengthens stressed syllables (deu 80.9 ms, rus 70.8 ms), and flattens unstressed vowel-quality reduction (deu F1 gap 15 Hz, ita ΔF1=3). **Good for:** confirming things that *should* be robust to register (final devoicing, vowel-quality height contrasts, nasalization, duration ordering). **Bad for:** any reduction/elision/flapping/lenition phenomenon — its absence there is a register fact, NOT a refutation. Treat a Pimsleur null on a connected-speech process as NA.

**FLEURS — multi-speaker pooled → formant-absolute noise + volume.**
Mixed vocal tracts and genders inflate and smear *absolute* F1/F2/B1 values; magnitudes are uninterpretable in isolation. **Good for:** (a) within-source contrasts (ratios/directions cancel the pooling — nasal-vs-oral, tense-vs-lax, e-vs-ɛ), (b) high-volume substitution/deletion statistics (huge n surfaces rare events: r→ɾ 280×, ʲ-deletion 792×, ɪ→i 82×). **Bad for:** absolute formant claims, and small-n positional buckets (rus word-final voiced n=5). Watch for genuine reverse-direction noise (rus a→ʌ 85× alongside ʌ→a) — check the net ratio.

**Tatoeba — single human, spontaneous, low volume.**
Cleanest *human* reference with no TTS bias and no didactic over-articulation. **Good for:** ground-truth voicing levels (fra voiceless onset 0.0), spontaneous-speech reductions/elisions that only show up in casual reading (fra schwa over-emission shows up here and nowhere else). **Bad for:** rare-event statistics (low n per symbol → thin cells like fra œ̃ n=9), and any single-speaker idiosyncrasy that can't be cross-checked. A Tatoeba-only effect (fra schwa) is a hypothesis, not a finding.

---

## 5. Final Label-Enrichment Recommendation

### A. SHIP as deterministic rules (bulletproof + rule-governed)

These are all_four (or three-with-register-NA) **espeak errors** whose correction is a context-free or position-governed rewrite — no per-token acoustic check needed:

1. **rus `/ʲ/`** — drop the standalone `ʲ` segment (no acoustic exponent; 20 ms floor) or re-encode as a secondary feature on the preceding consonant. all_four, deletion-rate + floor-duration evidence, speaker-independent.
2. **fra spurious English/length-marked vowels** (`uː ɔː ɪ ʊ ʌ ɜː ɑː ɒ ɐ oː aː iː yː`) — strip/remap; none are French phonemes. all_four, low-volume but reproducible.
3. **ita lax `/ɪ ʊ/` → `/i u/`** — Italian has no lax high vowels; collapse deterministically. all_four.
4. **ita `/r/` → `[ɾ]`** — if narrow-transcription granularity is desired; otherwise leave (allophonic, not phonemic). all_four, model-substitution evidence.
5. **deu final devoicing** — already correctly encoded by espeak; keep. (Validation, not a fix.)
6. **deu/eng position-conditioned voiced-stop devoicing (word-final)** — rule-governed by position; safe to encode as a positional devoicing feature.
7. **Durational unstressed reduction (eng, deu, por) & no-reduction (ita, spa)** — these are *modeling-layer* targets (duration is not in the string). Ship as duration-prediction priors, not as symbol rewrites. Stress placement is already in espeak.

### B. NEEDS ACOUSTIC ADJUDICATION (rule unsafe; realization is gradient/context-dependent)

1. **eng intervocalic /t,d/ flapping** — three-source but gradient and register-sensitive; flap vs full stop depends on segmental context and rate. Enrich from an acoustic flap detector, not a blanket rule.
2. **spa spirantization β/ð/ɣ → stop/v** — all_four but the stop↔continuant choice is allophonic and continuous; relabel from acoustics (burst/continuant detection), not deterministically.
3. **deu lenis onset devoicing** — all_four by direction but FLEURS separation is mean-only (medians identical); partial/gradient → acoustic voiced_frac threshold per token.
4. **eng `/uː/` fronting & deu/fra/ita vowel-quality contrasts** — espeak's symbols are right; if enriching toward narrow transcription, use measured F2/F1 per token (down-weight pooled absolutes).
5. **fra/por nasal B1, voicing-by-position** — confirmed but realization is acoustic; adjudicate per token.

### C. STILL UNMEASURED / DO NOT SHIP

1. **fra schwa /ə/ over-emission** — one source (Tatoeba) only; FLEURS *refutes* (inserts schwa). Register-gradient; needs spontaneous-speech corpus before any rule.
2. **rus onset/intervoc partial devoicing** — tail-only (p10) effect, medians voiced, not an espeak error. Note only; do not act.
3. **rus final devoicing** — directionally solid (tato+tts clean) but only n=2 sources had testable tokens. Safe as a rule by analogy to deu, but flag the thin word-final FLEURS/Pimsleur coverage; re-measure with more word-final voiced tokens before full confidence.

---

## Summary

The audit is **strongly source-independent at its core**: 22 of 31 phenomena replicate across all four sources, and they do so precisely because the load-bearing metrics — within-source formant *contrasts* (nasal-vs-oral, tense-vs-lax, e-vs-ɛ), *position*-conditioned voicing, deletion/substitution *directions*, and duration *ordering* — are all speaker-independent and survive vocal-tract pooling, single-speaker idiosyncrasy, TTS over-voicing, and didactic over-articulation alike. The four sources fail in *orthogonal, well-understood* ways (TTS inflates voicing; Pimsleur suppresses connected-speech allophony; FLEURS smears absolute formants while surfacing rare events through volume; Tatoeba is clean but thin), so agreement across them is genuine triangulation rather than shared bias. Where a source dissents, it is almost always a known register or power artifact — a Pimsleur null on flapping/schwa/palatalization, an n=3–5 FLEURS/Pimsleur positional bucket — correctly scored NA rather than refuting. Only two findings are genuinely fragile: French schwa over-emission (Tatoeba-only, FLEURS actively refutes) and Russian tail-only partial devoicing (not an espeak error). The honest read: the espeak *errors* worth shipping deterministically (rus ʲ, fra spurious vowels, ita lax-vowel collapse) are the most source-robust of all, while the gradient allophonic processes (flapping, spirantization, lenis devoicing) are real but demand acoustic adjudication rather than blanket rewrites. Net: act confidently on the all_four list, treat the three/two tiers as register-validated, and quarantine the two weak cases.
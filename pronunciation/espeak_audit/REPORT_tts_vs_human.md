# TTS-vs-Human espeak Audit: Cross-Speaker Verification Pass

## 1. TL;DR

This pass re-ran every phenomenon from the original **TTS-only** espeak audit against a matched **human** speaker (Tatoeba) in all 7 languages, and adversarially re-checked the numbers (n, within-speaker contrasts, artifact confounds). The point was to find out where the TTS-only conclusions were real linguistics versus synthesizer artifacts, and where the TTS speaker quietly *hid* a real effect.

Headline outcome:

- **The TTS-only audit generalised well for vowel-quality and stress/reduction structure.** Every mid-vowel split (French e/ɛ, o/ɔ; Italian e/ɛ, o/ɔ), the tense/lax separations (German iː/ɪ, uː/ʊ), Russian /ɨ/–/i/ backness, French/Spanish "no reduction", and the stress-duration / akanye patterns held up in **both** speakers.
- **Confirmed robust espeak errors (act on these):** Italian spurious /ɪ/ and /ʊ/ tense-lax splits, Italian /r/→tap, French uvular /ʊ/ and schwa over-emission, Portuguese spurious palatal /ʲ/, and (substitution-only but bidirectional) Spanish spirantisation over-application.
- **TTS *hid* real effects (human_only — the TTS-only report MISSED these):** AmE voiced-stop maintained voicing, German lenis onset devoicing, Russian onset + intervocalic voiced-obstruent devoicing leakage. In all of these the Chirp3 TTS **over-voices**, so a TTS-only reading underestimated how much real speakers devoice.
- **TTS artifacts to retract (tts_only / refuted):** French "intervocalic voiceless lenition" (refuted — 0.1 within noise), French final-stop release burst, Portuguese glide w→ʊ vocalization (model-only), Portuguese & Spanish voiceless-stop voicing leakage (raw-span bleed), Spanish tap-vs-trill, Russian dark-/ɫ/ "distinction" (refuted — model collapse, no acoustics).
- **Not measurable in any speaker (drop from the report as findings):** vowel nasalization in eng/deu/ita/spa (empty `nasals` table), English dark-/l/ and aspiration (metric doesn't isolate the feature), Portuguese t/d affrication (no context split).

## 2. Master Table

| Lang | Phenomenon | Presence | TTS vs Human (numbers) | Verdict | espeak err? |
|---|---|---|---|---|---|
| eng | Intervocalic /t,d/ flapping | both | vf 0.5 vs 0.0 (n79/412); burst −15.4/−5.9 — human vf 1.0 vs 0.0 (n125/710); −22.4/−11.6 | supported | yes |
| eng | /uː/ fronting (F2) | both | uː 1363 vs ʊ 1081 (n69/139) — human 1583 vs 1189 (n132/182) | supported | yes |
| eng | Unstressed reduction (dur+F1) | both | ratio 0.85 (60.5/71.1ms) — human 0.75 (60.6/80.3ms) | supported | no |
| eng | Voiced-stop maintained voicing | human_only | onset/intervoc vf 0.5/0.7 (n71/68) — human 1.0/1.0 (n118/130) | supported | no |
| eng | Voiceless onset aspiration | neither | onset −7.8 < coda −2.6 — human −14.1 < −4.7 (metric≠VOT) | supported | no |
| eng | Pre-nasal nasalisation | neither | nasals={} both | supported | no |
| eng | Dark coda /l/ | neither | no coda-l bucket either | supported | no |
| deu | Final devoicing | both | vf med 0.0 (n205) — human 0.0 (n216) | supported | no |
| deu | Onset lenis devoicing | human_only | vf med 0.7 (n249) — human 0.4 (n238) | supported | no |
| deu | Intervocalic voicing maintained | both | med 1.0 (n134) — human 1.0 (n122), both >onset>final | supported | no |
| deu | Voiceless stay voiceless intervoc | both | med 0.1 (n77) — human 0.1 (n78) | supported | no |
| deu | Stress reduction (dur+F1) | both | 60.7→50.4ms F1 527→386 — human 60.3→50.3 F1 500→386 | supported | no |
| deu | F2 centralization unstressed | neither | F2 1626/1615 flat — human 1936/1867 rises | supported | no |
| deu | Coda /r/→[ɐ] | neither | ɐ n=14 (garbage disp.) — human ɐ n=2 | weak | yes |
| deu | Tense/lax via quality | both | iː F2 2348 vs ɪ 2000 — human 2517 vs 2315 | supported | no |
| deu | Overspecification (0-frame phones) | tts_only | 23 deleted (1 degenerate clip) — human 9 | weak | no |
| deu | Nasalization | neither | nasals={} both | supported | no |
| fra | Nasal B1 widening | both | oral 104→nasals 213–268 — human 148→198–286 | supported | no |
| fra | Nasal A1-P0 drop | both | ~4dB TTS — human ~0dB (marginal) | weak | no |
| fra | e vs ɛ split | both | ΔF1 59/ΔF2 120 — human ΔF1 48/ΔF2 101 | supported | no |
| fra | o vs ɔ split | both | ΔF1 137 — human ΔF1 37 (near-merge) | supported | no |
| fra | ø vs œ contrast | both | ΔF1 138 (n18) — human ΔF1 121 (œ n9) | weak | no |
| fra | No unstressed reduction | both | F2 1721→1609 — human 1757→1664 (parallel) | supported | no |
| fra | Stress = duration | both | 72/60ms (1.19×) — human 70/51ms (1.39×) | supported | no |
| fra | No aspiration / voiced fully voiced | both | voiced 1.0 / voiceless 0.2–0.3 — human 1.0 / 0.0 | supported | no |
| fra | Intervoc voiceless lenition | tts_only | 0.3 vs onset 0.2 (within sd) — human 0.0/0.0 | **refuted** | no |
| fra | Final voiceless release burst | tts_only | +12.2dB (n44) — human −2.1dB (n30) | supported | no |
| fra | Uvular /ʊ/ over-spec | both | del 0.009/del7 — human 0.024/del12 | supported | yes |
| fra | Schwa deletion | both | del 0.005/del8 — human 0.036/del11 | supported | yes |
| fra | ə vs ø/œ distinct | both | ə F2 1520 > ø 1363 — human 1635 > 1453 (small) | weak | no |
| fra | Spurious EN vowels | both | all cells n=1–3 | weak | yes |
| ita | /e/ vs /ɛ/ F1 split | both | 193Hz (482.6/675.9) — human 78Hz (513.8/591.4) | supported | no |
| ita | /o/ vs /ɔ/ F1 split | both | 331Hz (530.2/861.4) — human 61Hz (497.5/558.1) | supported | no |
| ita | Spurious /ɪ/ vs /i/ | both | incoherent (F2 ɪ>i, no F1) — human merged | supported | yes |
| ita | Spurious /ʊ/ vs /u/ | both | overlap — human overlap (u n=15 thin) | weak | yes |
| ita | No unstressed reduction | both | dur 70.3→60.2 F2 high — human 100.6→60.7 F2 high | supported | no |
| ita | Intervoc voicing of voiceless | neither | 0.3 (n392) — human 0.2 (n161), true voiced=1.0 | supported | no |
| ita | /r/→tap | both | 44 (top sub) — human 21 (#2 sub) | supported | yes |
| ita | /a/→[ɐ] centralization | human_only | absent from TTS subs — human 12 subs (no formant) | weak | yes |
| ita | Geminate→singleton | both | kː→k 7 — human 10 (CTC peakiness) | weak | no |
| ita | Nasalization | neither | nasals={} both | supported | no |
| por | Spurious /ʲ/ | both | top-del 0.045 (n89), mdel 19 — human 0.031 (n32), mdel 9 | supported | yes |
| por | Unstressed dur reduction | both | 70.4>50.6ms (n1441) — human 70.4>50.6 (n847) | supported | no |
| por | Unstressed spectral reduction | both | F1 500→453,F2 1408→1607 — human flat (sd 159→302) | weak | no |
| por | Nasal-vowel B1 elevation | both | õ/ɐ̃/ẽ B1 2–3× oral — human 1.3–2× oral | supported | no |
| por | Glide w→[ʊ] vocalization | tts_only | w→ʊ 21 (model-only) — human absent, w del 8 | weak | no |
| por | Final/unstr high-vowel reduction | both | ʊ361/ɪ274 + bidir subs — human ʊ242/ɪ119 | weak | no |
| por | t/d→tʃ/dʒ affrication | neither | no stop-by-vowel split either | supported | no |
| por | Voiced-stop intervoc lenition | neither | voiced_frac 1.0, burst −28.5/−29.8 — human 1.0, −27.8/−26.0 | supported | no |
| por | Voiceless intervoc residual voicing | tts_only | 0.3/0.4 — human 0.1/0.0 (clean) | supported | no |
| rus | /ʲ/ over-spec | both | del 0.09 (n700), mdel 100 — human 0.079 (n889), mdel 86 | weak | yes |
| rus | Intervoc voiced leakage | human_only | frac 1.0 (n79) — human 0.8/0.7, p10 0.2 (n121) | supported | no |
| rus | Onset voiced devoicing | human_only | frac 1.0 (n43) — human 0.6, p10 0.0 (n70) | supported | no |
| rus | Final devoicing | both | vl-final 0.0 (n74) — human 0.0 (n87) | supported | no |
| rus | Akanye / reduction | both | unstr F2 +115 — human +53 (TTS reduces more) | supported | no |
| rus | Stress = duration | both | 60.5/50.3 (+10.2) — human 60.4/50.2 (+10.2) | supported | no |
| rus | Stress = quality | both | F1−56/F2+115 — human F1−16/F2+53 | supported | no |
| rus | /ɨ/ vs /i/ backness | both | ΔF2 261 (n151/386) — human ΔF2 385 (n183/428) | supported | no |
| rus | Velarised /ɫ/ vs /l/ | both | ɫ→l 46 (collapse, no acoustic) — human 31 | **refuted** | no |
| rus | Burst tilt v/vl | both | coda vl −1.6 vs v −12.6 — human +3.8 vs −11.3 | supported | no |
| spa | Spirantisation→stop | both | d38/b25/g23 — human d57/b27/g27 (sub-only) | supported | yes |
| spa | No reduction | both | F1 570/573 flat — human 460/447 flat | supported | no |
| spa | Mid/low vowel drift | both | o7/a4 — human o25/a24 (TTS under-shows) | supported | no |
| spa | /s/ voicing + retraction | both | z24/no-retract — human z40/S19/Z9 | supported | no |
| spa | Voiceless stop voicing leakage | tts_only | 0.4–0.5 — human 0.1 (raw-span bleed) | supported(artifact) | no |
| spa | Tap vs trill /r/ | tts_only | r→ɾ 11 — human none in top | weak | no |
| spa | Coarticulatory nasalization | human_only | TTS 0 nasal labels — human ɐ̃11/7, ʊ̃3 (table empty) | weak | no |
| spa | Overspec /ʲ/, /β/ | both | TTS ~0 — human ʲ-del4 (n49)/β-del3 | weak | yes |

## 3. The Three Lists That Matter

### (a) ROBUST espeak errors confirmed in BOTH speakers — act on these
These are genuine espeak transcription errors present in *both* the synthetic and the real speaker, with adequate n. Safe to encode as deterministic corrections:

- **ita: spurious /ɪ/ tense-lax split** — human i and ɪ are acoustically identical (F1 389/F2 2531 vs 369/2500); espeak fabricates a contrast Italian lacks. Strongest of the set.
- **ita: /r/→tap** — r→ɾ top substitution in both (44 / 21), plus ɾ insertions. Trill over-specified.
- **por: spurious palatal /ʲ/** — #1 deleted symbol in both (0.045 / 0.031), corroborated by model deletions (19 / 9). Strongest espeak-error claim overall.
- **fra: uvular /ʊ/ over-emission** — deletion rate + model-deletion count both higher in human (0.024/del12 vs 0.009/del7). espeak emits a full vowel speech reduces.
- **fra: schwa over-emission** — human deletes ~7× more (0.036 vs 0.005). espeak inserts optional schwas real speech drops.
- **spa: spirantisation over-applied** (substitution-only but bidirectional and high-n in both: d38/b25/g23 vs d57/b27/g27).
- **ita: spurious /ʊ/ split** and **eng intervocalic /t,d/ flapping** and **eng /uː/ fronting** — also confirmed in both, with the caveats that ita /ʊ/ rests partly on a thin human /u/ cell (n=15), and the eng pair are weaker copies in TTS.

### (b) HUMAN_ONLY — espeak errors / effects the TTS HID (the TTS-only report MISSED)
In every case Chirp3 TTS **over-voices** or **over-articulates**, so the TTS-only audit underestimated the real effect. These need the human signal to be detected:

- **eng: voiced-stop maintained voicing** — human onset/intervoc vf 1.0/1.0; TTS only 0.5/0.7. TTS under-voices English voiced stops.
- **deu: lenis onset devoicing** — human vf 0.4 (lenis); TTS 0.7 (over-voiced). Real German fortis/lenis lenition.
- **rus: intervocalic voiced-obstruent leakage** — human 0.8 (p10 0.2); TTS pinned at 1.0.
- **rus: onset voiced devoicing** — human 0.6 (p10 0.0); TTS 1.0.
- (weak, substitution-only) **ita: /a/→[ɐ]** and **spa: coarticulatory nasalization** — present only in human model labels, no acoustic backing.

Practical implication: any deterministic rule that assumes the TTS voicing profile is representative is wrong for voiced obstruents — real speakers devoice considerably more.

### (c) TTS_ONLY — claims to RETRACT (TTS-only report OVER-claimed)
Artifacts of Chirp3 synthesis / forced alignment, not real phenomena:

- **fra: intervocalic voiceless lenition — REFUTED.** TTS 0.3 vs onset 0.2 is within sd; human flat 0.0/0.0. Not a phenomenon.
- **rus: velarised /ɫ/ distinction — REFUTED.** No acoustic F2-of-/l/ field exists; the only signal is ɫ→l model *collapse*, which points against a maintained distinction.
- **fra: final voiceless release burst** (+12.2dB vs human −2.1dB) — real in TTS but TTS over-release; human is near-unreleased. Don't encode as French.
- **por: glide w→[ʊ] vocalization** — model-substitution only, absent in human (w deleted instead). Synthesis/segmentation artifact.
- **por & spa: voiceless-stop voicing leakage** — raw midpoint-span bleed into surrounding voicing; human is cleanly voiceless. Artifact, correctly killed.
- **spa: tap-vs-trill /r→ɾ/** — one substitution row, no acoustics, absent in human top-12. Synthesis under-trill tendency.
- **deu: overspecification "0-frame phones"** — driven by 1–2 degenerate clips (whole-clip collapse), an alignment artifact, not espeak.

## 4. Per-Language Notes (supported / weak only)

**English (US).** Three robust findings: intervocalic flapping of /t,d/ (espeak labels plain stops, both speakers voice + suppress the burst intervocalically), /uː/ fronting (F2 well above back anchors in both), and unstressed reduction (shorter + lower F1, huge n). The one missed effect: voiced stops stay fully voiced in the human (1.0) but only half-voiced in TTS — a TTS-only read would wrongly conclude English voiced stops are weakly voiced. Aspiration, nasalization, and dark-/l/ are not measurable from the available metrics.

**German.** Final devoicing, intervocalic voicing maintenance, voiceless-stay-voiceless, stress reduction (dur+F1), and tense/lax via *quality* (duration field is degenerate, correctly excluded) all hold in both. F2 does **not** centralize under destressing in either speaker (reduction is durational/F1 only). The human-only finding is lenis onset devoicing (0.4 vs TTS's over-voiced 0.7). Coda-r→[ɐ] and the overspecification count are weak/artifactual.

**French.** Nasal-vowel B1 widening, both mid-vowel laxing contrasts (e/ɛ robust; o/ɔ real but human nearly merges while TTS over-articulates), no-reduction, stress-as-duration, and no-aspiration all confirmed in both. Two robust espeak over-specifications: uvular /ʊ/ and schwa, both deleted far more by the human. A1-P0 nasal drop, ø/œ, and ə-vs-ø/œ are weak (small n / marginal gaps). **Retract** the intervocalic-lenition claim (refuted) and treat the final release burst as TTS over-release.

**Italian.** Both mid-vowel F1 splits confirmed (TTS over-separates, especially ɔ at 861Hz vs human 558). The two strongest espeak errors: spurious /ɪ/ (human i/ɪ identical) and /r/→tap. /ʊ/ split also an espeak error but weak (thin human /u/). No-reduction confirmed (quality preserved; TTS flattens duration). /a/→[ɐ] is human-only but substitution-only. Geminate→singleton is a CTC-peakiness artifact, not espeak.

**Portuguese (BR).** Spurious /ʲ/ is the headline robust espeak error (#1 deleted in both). Unstressed durational reduction and nasal-vowel B1 elevation confirmed in both. Spectral reduction is weak (human medians barely move; rests on F1 variance broadening). Voiced-stop lenition and t/d affrication are unevaluable (neither). Voiceless intervocalic residual voicing is a TTS-only carryover artifact. Glide vocalization and the final high-vowel "process" are model-only/weak.

**Russian.** Rich and mostly robust: final devoicing, akanye/reduction, stress-by-duration, stress-by-quality, /ɨ/–/i/ backness, and burst spectral tilt (voiceless > voiced, ~11–15dB) all confirmed in both. Two human-only voicing findings (intervocalic leakage, onset devoicing) where TTS sits artificially at 1.0. /ʲ/ over-spec is weak (model-only evidence). **Retract** the velarised-/ɫ/ distinction (refuted: model collapse, no acoustics).

**Spanish (ES).** No-reduction (flat F1/F2, n>1200 each) and spirantisation-over-application confirmed in both. /s/-voicing (s→z) shared; postalveolar retraction (s→S/Z) is human-only. Mid/low vowel drift holds on substitution counts (the cross-speaker Hz sentence in the original note is invalid and should be dropped). Voiceless-stop voicing leakage is a TTS raw-span artifact. Tap-vs-trill, nasalization, and /ʲ//β/ overspec are all weak.

## 5. Updated Recommendation

**Deterministic rules now SAFE (confirmed in both speakers, adequate n):**
- Treat espeak's separate Italian /ɪ/ as a fabrication → merge to /i/. (Strongest.)
- Map espeak Italian trill /r/ → tap /ɾ/ in tap-dominant contexts.
- Drop/optionalize espeak's Portuguese palatal /ʲ/.
- Optionalize espeak's French uvular /ʊ/ and schwa /ə/ (high deletion in real speech).
- Keep espeak's mid-vowel and tense/lax *quality* contrasts (fra e/ɛ o/ɔ; deu iː/ɪ uː/ʊ; ita e/ɛ o/ɔ; rus ɨ/i) — all real in both.
- Keep stress = duration (+~10ms) across deu/fra/por/rus/spa; keep no-reduction for fra/ita/spa; keep akanye for rus.
- Encode AmE intervocalic flapping and /uː/ fronting.
- Italian /ʊ/→/u/ merge: safe but flagged weak (thin human /u/).

**Rules that REQUIRE the human signal (TTS would mislead):**
- Voiced-obstruent voicing/devoicing profiles. TTS over-voices everywhere — use human values for eng voiced-stop voicing, deu lenis onset, rus onset + intervocalic devoicing. Do **not** calibrate voiced_frac thresholds off TTS.

**Do NOT encode (retract / unmeasurable):**
- French intervocalic voiceless lenition (refuted), Russian dark-/ɫ/ distinction (refuted), French final release burst, Portuguese w-vocalization, por/spa voiceless-stop voicing leakage, Spanish tap-vs-trill. All vowel-nasalization findings in eng/deu/ita/spa (empty `nasals` table), English aspiration & dark-/l/, Portuguese t/d affrication — no usable measurement.

## Summary

The TTS-only espeak audit generalised **well for structure, poorly for voicing.** Of the phenomena, the entire vowel-quality and stress/reduction layer — mid-vowel splits, tense/lax quality contrasts, /ɨ/–/i/ backness, akanye, stress-as-duration, and the no-reduction languages — held up in both the synthetic and the real speaker. The genuine espeak errors worth acting on (Italian /ɪ/ and /r/→tap, Portuguese /ʲ/, French /ʊ/ and schwa over-emission, Spanish spirantisation) were all confirmed bidirectionally. The systematic blind spot was **voicing**: Chirp3 TTS over-voices obstruents almost everywhere, so the TTS-only pass *missed* real-speaker devoicing in English voiced stops, German lenis onsets, and Russian onset/intervocalic obstruents — four human_only findings. It also *over-claimed* a cluster of voicing/lenition/release effects that turned out to be raw-span bleed, model-collapse, or synthesizer over-articulation; two (French intervocalic lenition, Russian dark-/ɫ/) are outright refuted. Several named acoustic tests (vowel nasalization in 4 languages, English aspiration and dark-/l/) produced no usable data in either speaker and should be dropped from the findings rather than reported. Net: trust the TTS audit for vowel quality and prosody; never trust it for voiced/voiceless realization without the human anchor.
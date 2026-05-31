# Cross-Language espeak Audit Report

Adversarially-verified phonetic audit of espeak phonemizations against Google Chirp3-HD TTS audio, 7 languages, 240 clips/language. Verdicts: **supported** (holds under recomputation), **weak** (directionally plausible, under-powered or single-channel), **refuted** (contradicted or unbacked). `espeak error?` flags whether espeak's output diverges from the audio (true) or espeak is correct (false).

## 1. TL;DR — strongest CONFIRMED espeak errors

These are the `supported` + `is_espeak_error=true` findings, with high or near-high confidence. They are the divergences worth acting on:

- **English /uː/ GOOSE-fronting (high).** espeak emits a back rounded /uː/; audio is strongly fronted: F2 median 1363 Hz vs canonical back-[u] ~850 Hz, F1 312 Hz, n=69. The broad back symbol understates fronting by ~500 Hz.
- **English intervocalic /t,d/ flapping (high).** espeak keeps full voiceless/voiced stops intervocalically; audio is a voiced weak-burst tap: voiced_frac 0.5 vs 0.0, burst −15.4 vs −5.9 dB (n=79 vs 412). Corroborated by an explicit model ɾ insertion. (The "tap-like duration" sub-argument is a CTC-floor artifact — only voicing + burst carry it.)
- **Russian standalone /ʲ/ palatalization (high).** espeak emits palatalization as its own segment (n=700); audio gives it essentially no span — deletion rate 0.09 (highest in set), every example pinned at the 20.1–20.3 ms aligner frame floor, model deletes it 100×. Coarticulatory, not a phone.
- **Portuguese spurious palatal glide /ʲ/ (high).** espeak inserts a separate /ʲ/ segment; audio drops it — top deletion rate 0.045, top model deletion (n=19), examples at ~20 ms zero-duration frames.
- **French spurious English-style long/lax vowels (high).** espeak emits uː/ɔː/oː/aː/ɜː/ʌ/ɪ/ʊ/ɒ as singletons; each is n≤3 of 4118 segments — phonemizer loanword/letter-name artifacts, not French phonemes.
- **Italian spurious lax high vowels /ɪ/ /ʊ/ (high).** espeak splits tense/lax; audio merges — ɪ F1 350 ≈ i 328 with ɪ F2 *fronter* (2116) not centralized; ʊ 380/1095 ≈ u 348/1029 (n=179/109). Model independently collapses ʊ→u (19), ɪ→i (18).
- **English word-final voiced-stop devoicing (medium).** espeak [+voice] final b/d/g; audio devoiced: voiced_frac median 0.0, mean 0.4 (bimodal/partial), burst −15.7 dB, n=80.

Note the recurring shape: the most reliable confirmed errors are **over-specifications** (espeak emits a segment or feature the audio does not realize) and a few **broad-symbol** cases (GOOSE, flapping). Almost every *quality/length* sub-phoneme split that espeak omits turns out to be **correct** (German tense/lax, French mid-vowels, Russian ɨ/i, Spanish 5-vowel stability).

## 2. Findings table (supported + weak; refuted dropped)

3 findings were **refuted** and excluded: English intervocalic /s/↔/z/ assimilation, French schwa-deletion under-modeling, Spanish tap/trill distinction.

| Language | Phenomenon | espeak → audio | Verdict | espeak error? |
|---|---|---|---|---|
| English | Intervocalic /t,d/ flapping | full stops → voiced tap (vf 0.5 vs 0.0, burst −15.4 vs −5.9 dB, n=79/412) | supported | **yes** |
| English | Word-final voiced-stop devoicing | [+voice] → devoiced (vf med 0.0/mean 0.4, burst −15.7 dB, n=80) | supported | **yes** |
| English | /uː/ GOOSE-fronting | back /uː/ → fronted (F2 1363, F1 312, n=69) | supported | **yes** |
| English | /ʊ/ central-back vs /uː/ | back-ish → matches (F2 1081, n=139) | supported | no |
| English | /oʊ/ GOAT back onset | back o+ʊ → matches (o F2 1030, ʊ 1081, n=68) | supported | no |
| English | Unstressed reduction | stress+schwa → ~11 ms shortening, no clean F1/F2 centralisation | weak | no |
| English | Onset voiceless aspiration | plain voiceless → higher burst tilt (−7.8 vs −17.9 dB) but no VOT | weak | yes (low) |
| English | Pre-nasal nasalisation | oral → NOT TESTABLE (nasals bucket empty) | supported | no |
| English | Dark coda /l/ | single /l/ → NOT TESTABLE (no per-position data) | supported | no |
| German | Final devoicing | b/d/g → p/t/k (vf med 0.0, n=205; no voiced/final bucket) | supported | no |
| German | Voiced stops genuinely voiced | voiced → d 1.00/ɡ 0.90/b 0.77 (intervoc 1.0, n=134) | supported | no |
| German | Coda /r/ → [ɐ] | r/ɾ (112/204) vs ɐ 14×; F1 bimodal, no position data | weak | yes (low) |
| German | Stress reduction (height+dur) | stress → F1 527/dur 60.7 vs 386/50.4 ms (n=976/1070) | supported | no |
| German | Onset voiceless voicing contrast | plain voiceless → 0.3 < voiced 0.7 (n=161/249) | supported | no |
| German | Tense/lax vowel quality | tense vs lax → F1 splits confirmed (iː287/ɪ380 etc.) | supported | no |
| French | Nasal vowel nasalisation | 4 nasals → wide B1 213–268 vs oral 104, A1-P0 drop | supported | no |
| French | œ̃/ɛ̃ merger | distinct → overlap within SD (n=38/21) | weak | **yes** |
| French | Mid-vowel e/ɛ, o/ɔ | distinct → F1 splits maintained (n=205/202; 66/57) | supported | no |
| French | Schwa ə vs ø | stable ə → distinct but spread possibly artefactual | weak | no |
| French | Uvular /ʁ/ | single ʁ → retained (del 0.009, n=338); tap/trill model-only | weak | no |
| French | No aspiration / voicing contrast | plain voiceless → vf 0.0–0.3 vs 1.0 (n 44–214) | supported | no |
| French | Spurious EN long/lax vowels | uː/ɔː/ɪ/ʊ… → each n≤3 of 4118, artifacts | supported | **yes** |
| French | Stress reduction durational | stress → 71.9 vs 60.5 ms, F1/F2 within SD | supported | no |
| Italian | Spurious lax high /ɪ/ /ʊ/ | tense/lax → merged (ɪ 350≈i 328, ʊ≈u) | supported | **yes** |
| Italian | Open/close mid e/ɛ, o/ɔ | marked → F1 split (e 483/ɛ 676; o 530/ɔ 861) | supported | no |
| Italian | No unstressed reduction | full vowels → no schwa drift (per-vowel proof off-file) | weak | no |
| Italian | Voiced/voiceless stops | marked → vf 1.0 vs ~0.3 (n=258/938) | supported | no |
| Italian | Trill /r/ vs tap /ɾ/ | r=152/ɾ=306 → no acoustic arbiter; model r→ɾ=44 hints | weak | no |
| Portuguese | Nasal vowels real | marked → B1 156→236–369, A1-P0 −0.6→−4.8..−9.7 | supported | no |
| Portuguese | /t d/ palatalization before /i/ | tʃ/dʒ → ʃ n=82, ʒ n=138 present, no affrication acoustics | weak | no |
| Portuguese | Coda /l/ → [w] | coda-l → w (1 clip; w→l n=7 counter) | weak | no |
| Portuguese | Glide /w/ → vowel [ʊ] | w → ʊ (top sub n=21, model-based) | supported | **yes** |
| Portuguese | Unstressed reduction = duration | stress → 70.4→50.6 ms (n~1.4k), quality modest | supported | no |
| Portuguese | Spurious palatal glide /ʲ/ | inserts ʲ → dropped (del 0.045 top, model-del 19, ~20 ms) | supported | **yes** |
| Portuguese | Front high i/ɪ split | i vs ɪ → near-merged (F1 364/387, churn) | weak | yes (low) |
| Portuguese | Voiced/voiceless stops robust | b/d/g vs p/t/k → vf 1.0 vs 0.3–0.4 | supported | no |
| Portuguese | Coda /s/ not palatalized to [ʃ] | coda s → sibilant frication (bucket mixes s/ʃ) | weak | no |
| Portuguese | Mid-vowel e too closed | closed e → open ɛ (sub n=10; ɛ→a n=6 open-bias) | weak | yes (low) |
| Portuguese | /ɐ/ vs /a/ | ɐ vs a → distinct (F1 718.9 vs 850.9, ~132 Hz) | weak | no |
| Russian | Standalone /ʲ/ palatalization | emits ʲ (n=700) → no span (del 0.09, all 20 ms floor) | supported | **yes** |
| Russian | Glide /j/ over-inserted | inserts j (n=237) → modest del 0.013, 1 floor example | weak | yes (low) |
| Russian | Trill /r/ → tap [ɾ] | trill /r/ → tap claim only from model r→ɾ=76 | weak | yes (low) |
| Russian | Velarized /ɫ/ → [l] | ɫ → plain l only from model ɫ→l=46, no F2 measure | weak | yes (low) |
| Russian | Akanye / reduction | stressed ɑ vs reduced ʌ/ə → confirmed (dur 60.5/50.3, ɑ F1 654/ʌ 497) | supported | no |
| Russian | Final obstruent devoicing | devoicing → vf med 0.0 (n=74), no voiced/final bucket | supported | no |
| Russian | Stop voicing contrast | voiced/voiceless → vf ~1.0 vs ~0.0–0.2 (n=43–399) | supported | no |
| Russian | /ɨ/ vs /i/ | split → F2 1735 vs 1997 (n=151/386), ɨ backer | supported | no |
| Russian | Reduced /ɪ/ vs /i/ | split → near-identical formants, bidirectional confusion | weak | yes (low) |
| Spanish | Spirantization b/d/g → β/ð/ɣ | approximants → intervoc bucket only n=5; symbol counts off-file | weak | no |
| Spanish | No vowel reduction | none → F1 569.8/573.4, dur 60.4/60.4 (n=1235/1681) | supported | no |
| Spanish | Mid-vowel /e o/ vs [ɛ ɔ] | plain → ɛ F1 520≈e 514; ɔ absent from table | weak | no |
| Spanish | Voiceless stops stay voiceless | p t k → burst −30.7/−29.8 dB closure (n=254/328) | supported | no |
| Spanish | espeak not over-specifying | — → 3 deletions / 6570 segs | supported | no |

## 3. Per-language findings (supported + weak only)

### English (US) — eng_tts_en-US-Chirp3-HD-Algieba
- **Intervocalic /t,d/ flapping (supported, error, high).** Intervocalic stops (n=79) voiced_frac 0.5 / burst −15.4 dB vs non-intervocalic (n=412) 0.0 / −5.9 dB; obstruent block corroborates (voiced/intervocalic vf 0.7), plus one explicit model ɾ insertion. Duration sub-argument discarded as CTC-floor artifact (20.4 vs 20.5 ms).
- **Word-final voiced-stop devoicing (supported, error, medium).** Final voiced stops (n=80) voiced_frac median 0.0 / mean 0.4 (bimodal → partial devoicing) / burst −15.7 dB vs voiced/intervocalic 0.7.
- **/uː/ GOOSE-fronting (supported, error, high).** F2 1363 Hz (p10 913, p90 1921), F1 312 Hz, n=69 — fronted ~500 Hz past back [u], 280 Hz above /ʊ/ (1081).
- **/ʊ/ central-back (supported, no error, medium).** F2 1081 Hz (n=139), well below /uː/; back-ish symbol defensible. Divergence is localized to GOOSE, not FOOT.
- **/oʊ/ GOAT back onset (supported, no error, medium).** o nucleus F2 1030 (n=68), following ʊ 1081 — both back/central; no fronting, internally consistent with the GOOSE call.
- **Unstressed reduction (weak, no error, medium).** Duration real (71.1→60.5 ms, n=996/815) but spectral "centralisation" contradicted: F1 *lower* unstressed (355 vs 434), F2 flat/wrong-way (1408→1505). espeak already marks stress+schwa.
- **Onset voiceless aspiration (weak, error-low).** voiced/onset 0.5 vs voiceless/onset 0.1 confirms voicing; aspiration rests on burst tilt (−7.8 vs −17.9 dB), a crude proxy with no true VOT. Borderline refuted.
- **Pre-nasal nasalisation & dark coda /l/ (supported, not testable).** nasals bucket empty; /l/ has no per-position data — honest "no evidence either way."

### German — deu_tts_de-DE-Chirp3-HD-Fenrir
- **Final devoicing / Auslautverhärtung (supported, no error, high).** voiceless/word_final n=205 voiced_frac med 0.0 (mean 0.1, p90 0.4); no voiced/word_final bucket exists — espeak bakes devoicing into the lexicon, correct by construction.
- **Voiced stops genuinely voiced (supported, no error, high).** Recomputed: d 1.00 (n=193), ɡ 0.90 (n=116), b 0.77 (n=105); intervocalic voiced 1.0 (n=134), onset voiced 0.7. Gradient lenis onset tail, but espeak's voiced call holds.
- **Coda /r/ → [ɐ] (weak, error-low).** ɐ (n=14) splits 10 clean open-central (F1 340–493) + 4 wild outliers (F1 1066–1962); no position field, so under-vocalization untestable. A flag, not a finding.
- **Stress reduction (supported, no error, high).** F1 527 vs 386, dur 60.7 vs 50.4 ms (n=976/1070); F2 flat (1614/1626) — height+duration, not frontness.
- **Onset voicing contrast (supported, no error, high).** voiceless/onset 0.3 (n=161) cleanly between voiced/onset 0.7 and voiceless/intervocalic 0.1 — clean three-way ordering.
- **Tense/lax vowel quality (supported, no error, high).** iː 287/ɪ 380; uː 320 F2 1094 / ʊ 415 F2 1074; eː 380/ɛ 527; oː 395 F2 890 / ɔ 588 F2 1175 (n≥51). Length unmeasurable (~20 ms peaky spans; summary's "0.0" is a rounding artifact).

### French — fra_tts_fr-FR-Chirp3-HD-Schedar
- **Nasal vowel nasalisation (supported, no error, high).** B1 >2× oral_ref for all four nasals (268/236/213/224 vs 104; n 132/88/38/21); A1-P0 valid (all F1>400) and consistently negative.
- **œ̃/ɛ̃ merger (weak, error, medium).** F1 differ ~20 Hz, F2 ~41 Hz — inside within-category SD (182/134) — but n=21/38 small; overlap doesn't prove the specific Parisian merger.
- **Mid-vowel e/ɛ, o/ɔ (supported, no error, high).** e/ɛ ~59 Hz F1 + ~120 Hz F2 (n=205/202); o/ɔ ~137 Hz F1 (n=66/57, F1 only). espeak distinctions warranted.
- **Schwa ə vs ø (weak, no error, medium).** ə F1 median 356 / mean 510 / sd 449 (n=192) vs ø 410 (n=19); spread as likely CTC midpoint/transition artifact as phonetic.
- **Uvular /ʁ/ (weak, no error, low).** Retained (del 0.009, n=338); tap/trill variability is model-sub only (ʁ→ɾ 12, ʁ→r 7), no acoustic arbiter.
- **No aspiration / voicing contrast (supported, no error, high).** voiced_frac 0.0–0.3 voiceless vs 1.0 voiced across positions (n 44–214); aspiration absence asserted not measured.
- **Spurious English long/lax vowels (supported, error, high).** uː/oː/aː/ɜː/ʌ/ʊ n=1, ɔː/ɪ n=3, ɒ n=2 of 4118 — phonemizer artifacts. Low n *is* the evidence.
- **Stress reduction durational (supported, no error, medium).** stressed 71.9 vs 60.5 ms (n=276/1500); F1/F2 within SD — French final lengthening, no spectral reduction.

### Italian — ita_tts_it-IT-Chirp3-HD-Leda
- **Spurious lax high /ɪ/ /ʊ/ (supported, error, high).** i 328/1797, ɪ 350/2116 (F2 *fronter*, not centralized), u 348/1029, ʊ 380/1095 (n=179/109). Model collapses ʊ→u 19, ɪ→i 18. High-vowel F2 generally smeared but near-identical F1 + no centralization carry the merge.
- **Open/close mid e/ɛ, o/ɔ (supported, no error, high).** e 483 (n=618)/ɛ 676 (n=143); o 530 (n=581)/ɔ 861 (n=110) — ~190 and ~330 Hz F1 gaps; ɔ A1-P0 +3.2 valid (F1>400).
- **No unstressed reduction (weak, no error, medium).** Global F1 610/F2 1238 stressed vs F1 513/F2 1507 unstressed (no central drift); dur 70.3/60.2 ms. Decisive per-vowel proof is off-file.
- **Voiced/voiceless stops (supported, no error, high).** voiced vf 1.0 (onset n=89, intervoc n=169); voiceless 0.4/0.3/0.1 (n=281/392/265). ~0.3 attributed to raw-span confound, not lenition.
- **Trill /r/ vs tap /ɾ/ (weak, no error, low).** Counts r=152/ɾ=306 accurate; no trill discriminator; peaky dur backwards (r 20 < ɾ 40 ms). Only model r→ɾ=44 hints espeak over-assigns trill.

### Portuguese (BR) — por_tts_pt-BR-Chirp3-HD-Leda
- **Nasal vowels real (supported, no error, high).** oral_ref B1 156.5 vs nasal 236–369; A1-P0 −0.6 vs −4.8..−9.7 (n 83/27/27). ĩ/ũ correctly excluded.
- **/t d/ palatalization before /i/ (weak, no error, low).** ʃ n=82, ʒ n=138 present but no affrication acoustics — symbol presence doesn't isolate affrication.
- **Coda /l/ → [w] (weak, no error, low).** One clip (bolsa→b o w s); w→l n=7 actually counters; no coda-l-specific stat.
- **Glide /w/ → vowel [ʊ] (supported, error, medium).** w→ʊ n=21 is the single largest substitution; model-based (exact_match 0.317), and "w" spans diphthong glides + coda-l. Strongest single disagreement, consistent with BR vocalic realization.
- **Unstressed reduction = duration (supported, no error, high).** stressed 70.4 ms (n=1441) vs unstressed 50.6 ms (n=1341), ~28% shortening; F1 modest (499.8→453.2), F2 rises — quality claim tempered.
- **Spurious palatal glide /ʲ/ (supported, error, high).** del_rate 0.045 (top), top model deletion n=19, examples at ~20 ms zero-duration frames — converging signals.
- **Front high i/ɪ split (weak, error-low).** F1 ~22 Hz, F2 ~84 Hz, both small vs SD (520–650); subs symmetric (i→ɪ 8, ɪ→i 6); ɪ also high insertions — churn, not clean collapse.
- **Voiced/voiceless stops (supported, no error, high).** vf 1.0 vs 0.3 intervoc (n=166/286), 1.0 vs 0.4 onset (n=54/208) — cleanest in the set.
- **Coda /s/ not palatalized to [ʃ] (weak, no error, low).** coda bucket (burst −10.6, n=175) mixes s with other coda fricatives; high-freq energy fits [s] and [ʃ] alike — no test run.
- **Mid-vowel e too closed (weak, error-low).** e→ɛ n=10 (2nd-largest sub), e/ɛ well separated (476.7/720.9); but ɛ→a n=6 suggests general open-mid arbiter bias.
- **/ɐ/ vs /a/ (weak, no error, low).** F1 718.9 vs 850.9 (~132 Hz, n=265/461), small one-directional subs (ɐ→a 9) — data argues for *keeping* the distinction; "over-distinguished" premise shaky.

### Russian — rus_tts_ru-RU-Chirp3-HD-Fenrir
- **Standalone /ʲ/ palatalization (supported, error, high).** Highest deletion 0.09 on largest n (700); every example at 20.1–20.3 ms aligner floor; model deletes 100 + ʲ→i 17. Aligner-based metric, not CTC artifact — palatalization is coarticulatory.
- **Glide /j/ over-inserted (weak, error-low).** del 0.013 (n=237), 7× lower than ʲ; 1 floor example; model deletes 22 but also inserts 4 — conflates two opposite claims.
- **Trill /r/ → tap [ɾ] (weak, error-low).** Model r→ɾ=76 + 8 ɾ ins only; dur fields all 0.0, no trill/tap acoustic. Single-channel model-only.
- **Velarized /ɫ/ → [l] (weak, error-low).** Model ɫ→l=46 + 8 l ins; no F2/velarization measure present. Inference from relabeling.
- **Akanye / reduction (supported, no error, high).** stressed dur 60.5 vs 50.3 ms (n=851/1205); ɑ F1 654 (n=260) vs reduced ʌ F1 497 (n=384) — symbol-level split real; F1 mean (495/474) weaker but dur + split solid.
- **Final obstruent devoicing (supported, no error, high).** voiceless/word_final vf med 0.0 (n=74), no voiced/word_final bucket; voiced/coda_other 1.0 (n=129), voiced/intervoc 1.0 (n=79). Minority retain voicing (mean 0.2, p90 1.0).
- **Stop voicing contrast (supported, no error, high).** voiced 1.0 (onset 43 / interv 79 / coda 129) vs voiceless 0.2/0.2/0.0 (n=197/114/399); burst also separates.
- **/ɨ/ vs /i/ (supported, no error, high).** ɨ F2 1735 (n=151) vs i 1997 (n=386), ~260 Hz, ɨ backer; F1 negligible (383/342) — F2 is the right cue.
- **Reduced /ɪ/ vs /i/ (weak, error-low).** ɪ 368/2048 vs i 342/1997 nearly coincident; bidirectional model confusion (ɪ→i 19, i→ɪ 11); a1_p0 differs but out-of-range (both F1<400). Genuinely ambiguous.

### Spanish (ES) — spa_tts_es-ES-Chirp3-HD-Sulafat
- **Spirantization b/d/g → β/ð/ɣ (weak, no error, low).** voiced/onset vf 1.0 / weak burst −28.1 dB (n=83) and model ð→d 38 / β→b 25 / ɣ→ɡ 23 (collapse to stops); but voiced/intervocalic bucket only n=5 (dur 301.5 ms implausible), symbol counts off-file — core spirantizing-context claim has almost no within-speaker acoustic support.
- **No vowel reduction (supported, no error, high).** F1 569.8/573.4, dur 60.4/60.4 ms (n=1235/1681), F2 gap 85 Hz ≪ sd 400 — strongest finding in the set.
- **Mid-vowel /e o/ vs [ɛ ɔ] (weak, no error, low).** ɛ F1 520 ≈ e 514 (no height contrast, ɛ n=33); ɛ F2 1465 vs 1910 noisy (sd 503); /ɔ/ absent from table — o-side untested.
- **Voiceless stops stay voiceless (supported, no error, medium).** burst −30.7 (intervoc, n=254) / −29.8 dB (onset, n=328) low-energy closure; ~0.5 voiced_frac is documented span-bleed confound.
- **espeak not over-specifying (supported, no error, high).** 3 deletion candidates / 6570 segments; t 0.006 (n=349), s 0.002 (n=464).

## 4. Cross-cutting patterns

- **Final obstruent devoicing is espeak-correct in both German and Russian.** Both show voiceless/word_final voiced_frac median 0.0 (n=205 DE, 74 RU) with *no voiced/word_final bucket emitted at all* — espeak bakes devoicing into the lexicon, so the audio cannot contradict it. Identical structural shape across the two languages; both high confidence, both NOT errors.
- **Nasal vowels are real and well-encoded in French and Portuguese.** Both confirmed via the same instrument — wide B1 (FR 213–268 vs 104; PT 236–369 vs 156.5) plus a consistent A1-P0 drop, valid because F1>400. espeak's nasal symbols are justified in both; the only nasal *error* candidate is the FR œ̃/ɛ̃ merger (weak).
- **Intervocalic lenition: confirmed for English (flapping), unproven for Spanish (spirantization).** English flapping is a clean, high-confidence error (voiced tap, burst weakening). Spanish spirantization is the *same phonological family* but the within-speaker acoustic bucket collapsed to n=5 — direction right, evidence absent. The model-collapse signal (ð→d, β→b, ɣ→ɣ) is identical in spirit to the EN ɾ insertion, but model-only.
- **Standalone palatal/glide segments are systematically over-specified.** Russian /ʲ/ (del 0.09, n=700) and Portuguese /ʲ/ (del 0.045) are the two cleanest over-specification errors in the entire audit, both pinned to the ~20 ms aligner floor. Russian /j/ and the various glide cases are the weak echoes of the same pattern.
- **Phonemizer loanword leakage.** French emits a whole inventory of English long/lax vowels (uː/ɔː/ɪ/ʊ/ʌ/ɒ…) at n≤3 each — a phonemizer artifact, not phonology. Worth checking other languages for the same letter-name/loanword leakage.
- **Stress → duration, not spectrum, everywhere except where espeak already encodes it segmentally.** EN (~11 ms), FR (71.9/60.5), PT (70.4/50.6, ~28%), DE (60.7/50.4, but DE *also* has a real 141 Hz F1 height effect) all show duration as the robust correlate; Russian akanye and German reduction are the cases where espeak's *segmental* reduction marking (ʌ/ə split) is acoustically justified. Spanish uniquely shows *no* reduction at all (flat F1 and duration).
- **Tense/lax and mid-vowel quality splits espeak makes are overwhelmingly correct** (German 4 pairs, French e/ɛ + o/ɔ, Italian e/ɛ + o/ɔ, Russian ɨ/i). The splits espeak makes that are *wrong* are the ones imported from English (Italian ɪ/ʊ, French long vowels) — i.e. cross-linguistic contamination, not genuine target-language contrasts.
- **Trill-vs-tap and velarization are an evidence dead-zone.** Italian r/ɾ, French ʁ, Russian /r/→ɾ and /ɫ/→l, Spanish ɾ/r — every one of these collapses to "model substitution only, no acoustic arbiter." The acoustic pipeline currently has no trill-period / contact-count / lateral-F2 discriminator, so none of these can be adjudicated.

## 5. Recommendations for non-circular label enrichment

**Rule-governed — fixable with deterministic allophonic rules on espeak output (no model likelihood, no acoustic adjudication needed):**

- **English intervocalic /t,d/ → [ɾ]:** context-conditioned flap rule (V_V, with stress conditioning). Confirmed acoustically (voiced tap + burst weakening), and the trigger context is purely symbolic. High value.
- **English /uː/ fronting:** apply a fronted-GOOSE realization (raise F2 target / retag symbol) for US English. Confirmed broad-symbol error; rule is unconditional for the dialect.
- **Final devoicing (German, Russian):** already correct — no rule needed; do *not* "fix" it. Flag as a positive control.
- **Drop standalone /ʲ/ as a segment (Russian, Portuguese):** fold palatalization into a feature on the adjacent consonant rather than a phone. Confirmed over-specification at the aligner floor; deterministic on symbol context.
- **Strip spurious English-style vowels in French (and audit other langs):** phonemizer post-filter that rejects out-of-inventory symbols (uː/ɔː/ɪ/ʊ/ʌ/ɒ…) appearing at negligible frequency — these are loanword/letter-name leakage, removable by an inventory whitelist per language.
- **Drop spurious Italian lax high vowels /ɪ/ /ʊ/:** merge to /i/ /u/ — confirmed merger (near-identical F1, fronter-not-centralized F2), and the fix is a deterministic symbol remap.

**Speaker/variant-specific — require acoustic adjudication, NEVER model likelihood:**

- **English word-final devoicing (partial/bimodal, mean 0.4):** voicing is gradient and voice-dependent; needs per-token voiced_frac, not a blanket rule.
- **French œ̃/ɛ̃ merger:** Parisian-speaker-specific; under-powered (n=21/38). Needs more speakers + formant adjudication before encoding.
- **German coda /r/ → [ɐ]:** bimodal ɐ realization, no position data — needs position-tagged acoustic re-extraction, not a rule.
- **Portuguese w→ʊ, e→ɛ, coda-l→w, palatalization:** all model-substitution-derived (exact_match as low as 0.317); the "w" symbol conflates glide + coda-l. Need acoustic measures (vowel formants for w→ʊ; affrication/frication spectra for tʃ/dʒ) before acting.
- **All trill/tap/velarization cases (IT r/ɾ, FR ʁ, RU r→ɾ + ɫ→l, ES ɾ/r):** the pipeline must grow a trill-period/contact-count and a lateral-F2 discriminator. Until then these are unadjudicable — do NOT enrich from model substitutions, which are circular (the model was trained on espeak).

**Hard rule throughout:** the espeak-trained CTC model cannot arbitrate espeak. Every model_agreement / substitution count is corroborating signal at best and circular at worst; it must never be the sole basis for a label change. The clean wins above are exactly the cases that have an *independent* acoustic or aligner-based channel (voiced_frac, burst, B1/A1-P0, deletion-rate at the frame floor).

## Summary

The biggest confirmed cross-language findings are a tight cluster of two error types. (1) **Over-specification**: espeak emits segments the audio doesn't realize — Russian /ʲ/ (deletion 0.09, n=700, all at the 20 ms aligner floor) and Portuguese /ʲ/ (0.045) are the cleanest, joined by French's English-vowel leakage (n≤3 of 4118) and Italian's spurious lax /ɪ/ /ʊ/. (2) **Broad-symbol / allophony**: English GOOSE-fronting (F2 1363 vs ~850), English intervocalic flapping (voiced tap, burst −15.4 vs −5.9 dB), and English final devoicing. Conversely, the larger sample firmly *vindicated* most espeak splits: German tense/lax (4 pairs), French/Italian mid-vowels, Russian ɨ/i, and German+Russian final devoicing are all acoustically correct — the splits espeak gets wrong are precisely the ones imported from English, i.e. cross-linguistic contamination.

What changed at n≈4× (240 vs 60 clips): the vowel-formant and voicing buckets crossed into adequately-powered territory (n=69–700 on the headline contrasts), which is what let GOOSE-fronting, the German tense/lax quartet, Russian ɨ/i, and Italian's merge move from suggestive to high-confidence supported. The larger n also exposed structural data gaps that more clips can't fix: empty nasal buckets (EN, DE), no consonant position field (DE coda /r/), the Spanish intervocalic bucket collapsing to n=5, and the complete absence of any trill/tap or lateral-velarization acoustic discriminator. 3 findings were refuted (EN /s/↔/z/, FR schwa-deletion, ES tap/trill), all on the same defect — they rested solely on model substitutions with no acoustic backing. The actionable conclusion: ship deterministic allophonic rules for the symbolically-triggered, acoustically-confirmed cases (flapping, GOOSE, /ʲ/ removal, English-vowel stripping, Italian lax-vowel merge), and quarantine everything trill/tap/velarization until the acoustic pipeline grows an arbiter — never enrich from the espeak-trained model's own likelihood.
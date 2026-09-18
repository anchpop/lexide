//! Edit-distance comparison of supplied readings with the decoded prediction.
use super::{PhonemeAlternative as RawPhonemeAlt, PredictResponse};
use g2p_types::{Language, Phonemized};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

type Reading = Vec<Vec<String>>;

/// One step of the optimal alignment between predicted and expected phoneme
/// sequences. Read in order, the ops reconstruct both sequences and show
/// exactly where they disagree.
///
/// `Sub`/`Extra` carry probabilities sourced from the model's top-k at the
/// predicted position. `expected_prob` is `None` when the expected phoneme
/// wasn't in the model's top-k for that position — i.e. the model
/// effectively assigned zero probability to the correct answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum AlignmentOp {
    /// Same phoneme on both sides.
    Match { phoneme: String, probability: f64 },
    /// Different phoneme — predicted has one thing, expected has another.
    Sub {
        expected: String,
        predicted: String,
        predicted_prob: f64,
        expected_prob: Option<f64>,
    },
    /// Predicted has a phoneme the expected sequence doesn't.
    Extra {
        predicted: String,
        predicted_prob: f64,
    },
    /// Expected has a phoneme the model didn't output. We have no model
    /// position for this gap, so no probability is available.
    Missing { expected: String },
}

/// Expand a raw IPA token into the deployed model's comparable token sequence.
/// Shared by expected readings, predictions, and top-k alternatives.
///
/// Only explicit tie bars split tokens: preserve untied diphthongs/diacritics.
pub fn normalize_phonemes(token: &str, language: Option<Language>) -> Vec<String> {
    token
        .split(['\u{0361}', '\u{035c}'])
        .filter_map(|component| normalize_phoneme(component, language))
        .collect()
}

/// Normalize a single IPA component into a canonical comparable form. Returns
/// `None` for tokens that consist entirely of non-phonemic markers.
///
/// Three layers of cleanup, applied in order:
///
/// 1. **Universal stripping** — remove characters that aren't phonemic in
///    any language: suprasegmental stress (`ˈ`/`ˌ`), syllable boundary
///    (`.`), length marks (`ː`/`ˑ`), the liaison/elision marker (`‿`),
///    ASCII digits (the multilingual wav2vec2 model leaks Mandarin tone
///    numbers like `y5`, `i5`, `a5` into French output), and `^`, an
///    espeak artifact the Russian voice leaves on a word-final palatalized
///    consonant (`царь` → `tsɑrɪ^`) that lexide blacklists at preprocess
///    time, so no model can emit it.
/// 2. **Internal whitespace strip** — `f a ɪ` is already split on
///    whitespace by the caller, but defensive in case a token slipped
///    through with embedded whitespace.
/// 3. **Per-language canonicalization** — collapses phoneme equivalence
///    classes so model conventions and ground-truth conventions agree:
///    for French, `{r, ʀ}` collapse to `ʁ`, `{ts, tɕ, tɕh}` to `t`,
///    `ɥ → y` (the model doesn't reliably distinguish the rounded palatal
///    approximant from `y`).
///
/// Combining diacritics inside the phoneme (e.g. the tilde on `ã`) are NOT
/// touched — those are phonemic (except German non-syllabic U+032F).
fn normalize_phoneme(token: &str, language: Option<Language>) -> Option<String> {
    let stripped: String = token
        .chars()
        .filter(|c| {
            !matches!(*c, 'ˈ' | 'ˌ' | '.' | 'ː' | 'ˑ' | '‿' | '^')
                && !c.is_ascii_digit()
                && !c.is_whitespace()
        })
        .collect();
    let canonical = canonicalize_for_language(&stripped, language);
    (!canonical.is_empty()).then_some(canonical)
}

/// Map a phoneme token onto its canonical form for the given target
/// language. Symmetric across predicted/expected — when both sides go
/// through this, equivalence-class members compare equal.
fn canonicalize_for_language(token: &str, language: Option<Language>) -> String {
    match language {
        // German inventory: 43,267 rows / 1,079,676 tokens in lexide's
        // pronunciation/data/audio/deu/phonemes.jsonl contain neither ʔ nor
        // U+032F. Strip these reference-only marks, not vowels or diphthongs.
        Some(Language::German) => token
            .chars()
            .filter(|c| !matches!(c, 'ʔ' | '\u{032f}'))
            .collect(),
        Some(Language::French) => match token {
            // R variants: ground truth uses ʁ (uvular fricative); the
            // multilingual model emits any of: r (alveolar trill, common
            // across many languages), ʀ (uvular trill, French stage variant),
            // ɾ (alveolar tap, Spanish/Italian-flavored), x (voiceless velar
            // fricative, sometimes emitted for the devoiced uvular allophone
            // /χ/ that occurs phrase-finally in French). All collapse to ʁ.
            "r" | "ʀ" | "ʁ" | "ɾ" | "x" => "ʁ".to_string(),
            // /t/ affricate variants observed in the model output for
            // French. Not phonemic in French; collapse to plain /t/.
            "ts" | "tɕ" | "tɕh" | "tɕʰ" => "t".to_string(),
            // The model conflates the rounded palatal approximant with /y/.
            "ɥ" => "y".to_string(),
            // Front /a/ vs back /ɑ/ is no longer phonemically distinguished
            // in modern Parisian French; both wikipron and the model use
            // them inconsistently. Collapse to a.
            "ɑ" | "a" => "a".to_string(),
            _ => token.to_string(),
        },
        _ => token.to_string(),
    }
}

/// Normalize the model's raw phoneme output AND the parallel top-k
/// alternatives at the same time. Returns the two lists with matching
/// length, parallel by position.
///
/// Empty normalized tokens drop their entire position. A tied affricate
/// expands into component positions; alternatives with the same component
/// count contribute probability only to their corresponding position. Other
/// lengths are omitted, since a per-position top-k cannot represent them.
/// Canonical equivalents at each position merge by summing probability, then
/// sort by descending probability. No probability is renormalized.
fn normalize_with_topk(
    raw_phonemes: &[String],
    raw_top_k: &[Vec<RawPhonemeAlt>],
    language: Option<Language>,
) -> (Vec<String>, Vec<Vec<(String, f64)>>) {
    let mut normalized = Vec::with_capacity(raw_phonemes.len());
    let mut normalized_top_k: Vec<Vec<(String, f64)>> = Vec::with_capacity(raw_phonemes.len());

    for (i, raw) in raw_phonemes.iter().enumerate() {
        let components = normalize_phonemes(raw, language);
        let mut merged = vec![HashMap::<String, f64>::new(); components.len()];
        if let Some(alts) = raw_top_k.get(i) {
            for alt in alts {
                let alt_components = normalize_phonemes(&alt.phoneme, language);
                // Alternatives must span the same number of component positions.
                // Do not credit a single /t/ with a whole /t͡ʃ/, or duplicate an
                // atomic alternative across both positions of a split affricate.
                if alt_components.len() == components.len() {
                    for (position, component) in merged.iter_mut().zip(alt_components) {
                        *position.entry(component).or_default() += alt.probability;
                    }
                }
            }
        }
        normalized.extend(components);
        for position in merged {
            let mut alts: Vec<_> = position.into_iter().collect();
            alts.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            normalized_top_k.push(alts);
        }
    }

    (normalized, normalized_top_k)
}

/// Look up `phoneme`'s probability in a normalized top-k list. Returns
/// `None` when the phoneme wasn't in the top-k — i.e. the model
/// effectively assigned ~0 probability to it.
fn prob_of(phoneme: &str, top_k: &[(String, f64)]) -> Option<f64> {
    top_k
        .iter()
        .find(|(p, _)| p == phoneme)
        .map(|(_, prob)| *prob)
}

/// Levenshtein alignment with per-position probability annotations. Same
/// algorithm as before; ops now carry the model's confidence at the
/// predicted position so the JSONL output can show how close the model
/// was to the correct answer.
/// The first word of `reading` with two or more phonemes that the alignment
/// leaves entirely unheard: every phoneme `Missing`, none matched or
/// substituted. A whole word gone is a letter or word the audio skipped —
/// a voice that silently drops "œ" — however small the edit distance looks
/// beside a long example. One-phoneme words are exempt: the model does
/// swallow a lone schwa.
fn unheard_word<'a>(reading: &'a Reading, ops: &[AlignmentOp]) -> Option<&'a [String]> {
    let mut consumed = ops
        .iter()
        .filter(|op| !matches!(op, AlignmentOp::Extra { .. }));
    for word in reading {
        let heard = consumed
            .by_ref()
            .take(word.len())
            .filter(|op| !matches!(op, AlignmentOp::Missing { .. }))
            .count();
        if word.len() >= 2 && heard == 0 {
            return Some(word);
        }
    }
    None
}

fn align(
    predicted: &[String],
    predicted_top_k: &[Vec<(String, f64)>],
    expected: &[String],
) -> (usize, Vec<AlignmentOp>) {
    let (m, n) = (predicted.len(), expected.len());

    // Helper for the chosen phoneme's prob at predicted position i. Falls
    // back to 1.0 if the position has no top-k (shouldn't happen, but
    // defensive — e.g. for callers in tests that pass empty top-k).
    let pred_prob = |i: usize| -> f64 {
        predicted_top_k
            .get(i)
            .and_then(|alts| alts.first())
            .map(|(_, p)| *p)
            .unwrap_or(1.0)
    };
    // Helper for prob of the expected phoneme at predicted position i.
    let exp_prob_at = |i: usize, exp_ph: &str| -> Option<f64> {
        predicted_top_k
            .get(i)
            .and_then(|alts| prob_of(exp_ph, alts))
    };

    // dp[i][j] = min cost to align predicted[..i] with expected[..j].
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if predicted[i - 1] == expected[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    // Backtrack from (m, n) to (0, 0), preferring match > sub > extra/missing
    // on ties so the alignment is readable. `i-1` is the predicted position
    // we're emitting an op for; we use it to look up probabilities.
    let mut ops: Vec<AlignmentOp> = Vec::with_capacity(m.max(n));
    let (mut i, mut j) = (m, n);
    while i > 0 || j > 0 {
        let here = dp[i][j];
        if i > 0 && j > 0 {
            let cost = if predicted[i - 1] == expected[j - 1] {
                0
            } else {
                1
            };
            if dp[i - 1][j - 1] + cost == here {
                if cost == 0 {
                    ops.push(AlignmentOp::Match {
                        phoneme: predicted[i - 1].clone(),
                        probability: pred_prob(i - 1),
                    });
                } else {
                    ops.push(AlignmentOp::Sub {
                        expected: expected[j - 1].clone(),
                        predicted: predicted[i - 1].clone(),
                        predicted_prob: pred_prob(i - 1),
                        expected_prob: exp_prob_at(i - 1, &expected[j - 1]),
                    });
                }
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i - 1][j] + 1 == here {
            ops.push(AlignmentOp::Extra {
                predicted: predicted[i - 1].clone(),
                predicted_prob: pred_prob(i - 1),
            });
            i -= 1;
            continue;
        }
        // Remaining case: dp[i][j-1] + 1 == here
        ops.push(AlignmentOp::Missing {
            expected: expected[j - 1].clone(),
        });
        j -= 1;
    }
    ops.reverse();
    (dp[m][n], ops)
}

/// The closest accepted reading, with normalized labels and alignment diagnostics.
#[derive(Debug, Clone)]
pub struct PronunciationScore {
    pub predicted: Vec<String>,
    pub expected: Vec<String>,
    pub variant_index: usize,
    pub variants_considered: usize,
    pub edit_distance: usize,
    /// Edit distance divided by max(predicted length, expected length, 1).
    pub edit_distance_ratio: f64,
    pub alignment: Vec<AlignmentOp>,
    pub unheard_word: Option<Vec<String>>,
}

impl PronunciationScore {
    /// Apply the verifier's empty-output, missing-word and mismatch gates.
    pub fn failure_reason(&self, mismatch_threshold: f64) -> Option<String> {
        let dist = self.edit_distance;
        let max_len = self.predicted.len().max(self.expected.len()).max(1);
        let pct = self.edit_distance_ratio * 100.0;
        if self.predicted.is_empty() {
            Some("model returned no phonemes".into())
        } else if let Some(word) = &self.unheard_word {
            Some(format!(
                "word /{}/ not heard ({dist} edits over max-len {max_len} = {pct:.0}%)",
                word.join(" ")
            ))
        } else if self.edit_distance_ratio > mismatch_threshold {
            Some(format!("phoneme mismatch ({dist} edits over max-len {max_len} = {pct:.0}%, best of {} variant(s))", self.variants_considered))
        } else {
            None
        }
    }
}

impl PredictResponse {
    /// Compare supplied readings. Ties keep the first reading; an empty set
    /// returns None. Only segmental labels and word boundaries are scored;
    /// stress, tone and pitch remain available on the supplied targets.
    /// None selects language-independent normalization.
    pub fn score(
        &self,
        expected: &[Phonemized],
        language: Option<Language>,
    ) -> anyhow::Result<Option<PronunciationScore>> {
        let raw = self
            .phonemes
            .iter()
            .map(|p| p.phoneme.clone())
            .collect::<Vec<_>>();
        let topk = self
            .phonemes
            .iter()
            .map(|p| p.top_k.clone())
            .collect::<Vec<_>>();
        let (predicted, topk) = normalize_with_topk(&raw, &topk, language);
        let mut best: Option<PronunciationScore> = None;
        for (variant_index, target) in expected.iter().enumerate() {
            let spans = if target.word_spans.is_empty() {
                vec![(0, target.phonemes.len())]
            } else {
                target.word_spans.clone()
            };
            let mut end = 0;
            let mut reading = Vec::new();
            for (start, stop) in spans {
                anyhow::ensure!(
                    start == end && start <= stop && stop <= target.phonemes.len(),
                    "invalid pronunciation word spans"
                );
                reading.push(
                    target.phonemes[start..stop]
                        .iter()
                        .flat_map(|p| normalize_phonemes(p, language))
                        .collect::<Vec<_>>(),
                );
                end = stop;
            }
            anyhow::ensure!(
                end == target.phonemes.len(),
                "pronunciation word spans do not cover all phonemes"
            );
            let flat = reading.concat();
            let (dist, alignment) = align(&predicted, &topk, &flat);
            if best.as_ref().is_none_or(|best| dist < best.edit_distance) {
                let missing = unheard_word(&reading, &alignment).map(<[String]>::to_vec);
                let ratio = dist as f64 / predicted.len().max(flat.len()).max(1) as f64;
                best = Some(PronunciationScore {
                    predicted: predicted.clone(),
                    expected: flat,
                    variant_index,
                    variants_considered: expected.len(),
                    edit_distance: dist,
                    edit_distance_ratio: ratio,
                    alignment,
                    unheard_word: missing,
                });
            }
        }
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn word(phones: &[&str]) -> Vec<String> {
        phones.iter().map(|p| p.to_string()).collect()
    }
    fn normalize_phonemes(token: &str, language: Language) -> Vec<String> {
        super::normalize_phonemes(token, Some(language))
    }
    fn normalize_with_topk(
        raw: &[String],
        topk: &[Vec<RawPhonemeAlt>],
        language: Language,
    ) -> (Vec<String>, Vec<Vec<(String, f64)>>) {
        super::normalize_with_topk(raw, topk, Some(language))
    }
    fn normalize_phoneme(token: &str, language: Language) -> Option<String> {
        super::normalize_phoneme(token, Some(language))
    }
    #[test]
    fn typed_readings_select_the_best_variant_and_validate_word_spans() {
        let prediction: PredictResponse = serde_json::from_value(serde_json::json!({
            "phonemes": [{"phoneme": "a", "top_k": []}, {"phoneme": "b", "top_k": []}]
        }))
        .unwrap();
        let targets = [
            Phonemized::from_ipa_tokens("a c"),
            Phonemized::from_ipa_tokens("a | b"),
        ];
        let score = prediction.score(&targets, None).unwrap().unwrap();
        assert_eq!(score.variant_index, 1);
        assert_eq!(score.edit_distance, 0);
        assert!(score.failure_reason(0.0).is_none());
        assert!(prediction.score(&[], None).unwrap().is_none());
        let mut invalid = targets[0].clone();
        invalid.word_spans = vec![(0, 99)];
        assert!(prediction.score(&[invalid], None).is_err());
        let missing = prediction
            .score(&[Phonemized::from_ipa_tokens("a b | x y")], None)
            .unwrap()
            .unwrap();
        assert_eq!(missing.unheard_word, Some(word(&["x", "y"])));
        assert!(missing
            .failure_reason(1.0)
            .unwrap()
            .starts_with("word /x y/ not heard"));
    }

    #[test]
    fn tied_affricates_expand_symmetrically_in_every_language() {
        for language in [
            Language::German,
            Language::SpanishEuro,
            Language::Italian,
            Language::French,
            Language::English,
            Language::Russian,
            Language::Hindi,
            Language::PortugueseBrazil,
            Language::Korean,
            Language::ChineseSimplified,
            Language::Japanese,
            Language::Thai,
        ] {
            for (raw, components) in [
                ("ˈt͡ʃː", ["t", "ʃ"]),
                ("t͜ʃ", ["t", "ʃ"]),
                ("d͡ʒ", ["d", "ʒ"]),
                ("d͜ʒ", ["d", "ʒ"]),
                ("t͡s", ["t", "s"]),
                ("t͜s", ["t", "s"]),
                ("d͡z", ["d", "z"]),
                ("d͜z", ["d", "z"]),
            ] {
                let expected = normalize_phonemes(raw, language);
                assert_eq!(expected, word(&components));
                let (predicted, topk) = normalize_with_topk(&word(&[raw]), &[], language);
                assert_eq!(predicted, expected);
                assert_eq!(topk.len(), 2);
                assert_eq!(align(&predicted, &topk, &expected).0, 0);
            }
        }
        assert_eq!(normalize_phonemes("aɪ", Language::German), word(&["aɪ"]));
        assert_eq!(normalize_phonemes("ɪ̯", Language::German), word(&["ɪ"]));
        assert_eq!(normalize_phonemes("ɐ̯", Language::German), word(&["ɐ"]));
        assert_eq!(normalize_phonemes("ɪ̯", Language::SpanishEuro), word(&["ɪ̯"]));
        assert_eq!(
            normalize_phonemes("ʔ", Language::German),
            Vec::<String>::new()
        );
        assert_eq!(normalize_phonemes("ʔ", Language::English), word(&["ʔ"]));
        assert_eq!(normalize_phonemes("ã", Language::German), word(&["ã"]));
    }

    #[test]
    fn expanded_topk_is_component_aligned_without_spurious_atomic_matches() {
        let alt = |phoneme: &str, probability| RawPhonemeAlt {
            phoneme: phoneme.into(),
            probability,
        };
        let raw = word(&["ʔ", "t͡ʃ", "a", "t"]);
        let topk = vec![
            vec![alt("ʔ", 1.0)],
            vec![
                alt("t͡ʃ", 0.5),
                alt("t͜ʃ", 0.2),
                alt("d͡ʒ", 0.1),
                alt("t", 0.2),
            ],
            vec![alt("a", 0.9)],
            vec![alt("t", 0.6), alt("t͡ʃ", 0.4)],
        ];
        let (tokens, normalized) = normalize_with_topk(&raw, &topk, Language::German);
        assert_eq!(tokens, word(&["t", "ʃ", "a", "t"]));
        assert_eq!(normalized.len(), tokens.len());
        assert_eq!(prob_of("t", &normalized[0]), Some(0.7));
        assert_eq!(prob_of("ʃ", &normalized[1]), Some(0.7));
        assert_eq!(prob_of("d", &normalized[0]), Some(0.1));
        assert_eq!(prob_of("ʒ", &normalized[1]), Some(0.1));
        assert_eq!(prob_of("t", &normalized[1]), None);
        assert_eq!(prob_of("a", &normalized[2]), Some(0.9));
        assert_eq!(prob_of("t", &normalized[3]), Some(0.6));
        assert_eq!(prob_of("ʃ", &normalized[3]), None);
    }

    #[test]
    fn systemic_cues_normalize_references_without_hiding_spoken_errors() {
        // Exact examples extracted from the September 14 deployed verification
        // reports under /tmp/lexide-deploy-verify/{deu,spa,ita}-systemic.json.
        let fixtures: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("fixtures/systemic-cues.json")).unwrap();
        for fixture in fixtures {
            let language = match fixture["language"].as_str().unwrap() {
                "deu" => Language::German,
                "spa" => Language::SpanishEuro,
                "ita" => Language::Italian,
                _ => unreachable!(),
            };
            let raw: Vec<String> = serde_json::from_value(fixture["raw"].clone()).unwrap();
            let original: Vec<String> =
                serde_json::from_value(fixture["expected"].clone()).unwrap();
            let expected: Vec<_> = original
                .iter()
                .flat_map(|p| normalize_phonemes(p, language))
                .collect();
            let (heard, topk) = normalize_with_topk(&raw, &[], language);
            assert_eq!(
                expected,
                original
                    .iter()
                    .flat_map(|p| {
                        if language == Language::German {
                            match p.as_str() {
                                "ʔ" => vec![],
                                "ʏ̯" => word(&["ʏ"]),
                                _ => vec![p.clone()],
                            }
                        } else {
                            match p.as_str() {
                                "t͡ʃ" => word(&["t", "ʃ"]),
                                "d͡ʒ" => word(&["d", "ʒ"]),
                                _ => vec![p.clone()],
                            }
                        }
                    })
                    .collect::<Vec<_>>(),
                "{}",
                fixture["text"]
            );
            assert_eq!(
                heard,
                serde_json::from_value::<Vec<String>>(fixture["heard"].clone()).unwrap()
            );
            assert!(
                align(&heard, &topk, &expected).0 > 0,
                "do not erase genuine errors: {}",
                fixture["text"]
            );
        }
    }

    #[test]
    fn normalize_strips_suprasegmentals() {
        let lang = Language::English;
        assert_eq!(normalize_phoneme("ˈa", lang), Some("a".to_string()));
        assert_eq!(normalize_phoneme("ˌb", lang), Some("b".to_string()));
        assert_eq!(normalize_phoneme("iː", lang), Some("i".to_string()));
        assert_eq!(normalize_phoneme(".", lang), None);
        assert_eq!(normalize_phoneme("ˈ", lang), None);
        // Combining diacritics inside the phoneme are preserved.
        assert_eq!(normalize_phoneme("ã", lang), Some("ã".to_string()));
        // espeak's Russian `^` artifact is not a phone anywhere.
        assert_eq!(
            normalize_phoneme("ɪ^", Language::Russian),
            Some("ɪ".to_string())
        );
        assert_eq!(normalize_phoneme("^", Language::Russian), None);
    }

    #[test]
    fn normalize_strips_tone_digits_and_liaison() {
        let lang = Language::French;
        // Mandarin tone digits leak through the multilingual wav2vec2 model.
        assert_eq!(normalize_phoneme("y5", lang), Some("y".to_string()));
        assert_eq!(normalize_phoneme("i5", lang), Some("i".to_string()));
        assert_eq!(normalize_phoneme("a5", lang), Some("a".to_string()));
        // Liaison marker has no phonetic content.
        assert_eq!(normalize_phoneme("‿", lang), None);
        assert_eq!(normalize_phoneme("a‿", lang), Some("a".to_string()));
    }

    #[test]
    fn french_canonicalization() {
        let lang = Language::French;
        assert_eq!(normalize_phoneme("r", lang), Some("ʁ".to_string()));
        assert_eq!(normalize_phoneme("ʁ", lang), Some("ʁ".to_string()));
        assert_eq!(normalize_phoneme("ts", lang), Some("t".to_string()));
        assert_eq!(normalize_phoneme("tɕ", lang), Some("t".to_string()));
        assert_eq!(normalize_phoneme("ɥ", lang), Some("y".to_string()));
        // Non-French langs: no canonicalization, just pass-through.
        assert_eq!(
            normalize_phoneme("r", Language::English),
            Some("r".to_string())
        );
    }

    #[test]
    fn unheard_word_is_one_with_no_phoneme_matched_or_substituted() {
        let reading: Reading = vec![word(&["o", "ʊ"]), word(&["æ", "z"]), word(&["ə"])];
        let missing = |p: &str| AlignmentOp::Missing {
            expected: p.to_string(),
        };
        let matched = |p: &str| AlignmentOp::Match {
            phoneme: p.to_string(),
            probability: 1.0,
        };
        // "œ" (o ʊ) skipped entirely, the rest heard: the first word is unheard.
        let ops = vec![
            missing("o"),
            missing("ʊ"),
            matched("æ"),
            matched("z"),
            matched("ə"),
        ];
        assert_eq!(
            unheard_word(&reading, &ops),
            Some(word(&["o", "ʊ"]).as_slice())
        );
        // A substitution counts as heard; extras don't consume expected phonemes.
        let ops = vec![
            AlignmentOp::Extra {
                predicted: "h".into(),
                predicted_prob: 1.0,
            },
            AlignmentOp::Sub {
                expected: "o".into(),
                predicted: "ɔ".into(),
                predicted_prob: 1.0,
                expected_prob: None,
            },
            missing("ʊ"),
            matched("æ"),
            matched("z"),
            missing("ə"),
        ];
        // The lone schwa is exempt even though it went unheard.
        assert_eq!(unheard_word(&reading, &ops), None);
    }

    fn fake_topk(predicted: &[String]) -> Vec<Vec<(String, f64)>> {
        predicted.iter().map(|p| vec![(p.clone(), 1.0)]).collect()
    }

    #[test]
    fn align_basics() {
        let s = |slice: &[&str]| slice.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let cases: &[(&[&str], &[&str], usize)] = &[
            (&["a", "b", "c"], &["a", "b", "c"], 0),
            (&["a", "b", "c"], &["a", "x", "c"], 1),
            (&["a", "b"], &["a", "b", "c"], 1),
            (&[], &["a", "b"], 2),
        ];
        for (predicted, expected, want_cost) in cases {
            let predicted_v = s(predicted);
            let expected_v = s(expected);
            let topk = fake_topk(&predicted_v);
            assert_eq!(align(&predicted_v, &topk, &expected_v).0, *want_cost);
        }
    }

    #[test]
    fn align_produces_readable_ops() {
        let s = |slice: &[&str]| slice.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // predicted: a b c d e
        // expected:  a x c   e
        // Optimal: match a, sub b→x, match c, extra d, match e  (cost 2)
        let predicted = s(&["a", "b", "c", "d", "e"]);
        let topk = fake_topk(&predicted);
        let (cost, ops) = align(&predicted, &topk, &s(&["a", "x", "c", "e"]));
        assert_eq!(cost, 2);
        let mut reconstructed_predicted = Vec::new();
        let mut reconstructed_expected = Vec::new();
        for op in &ops {
            match op {
                AlignmentOp::Match { phoneme, .. } => {
                    reconstructed_predicted.push(phoneme.clone());
                    reconstructed_expected.push(phoneme.clone());
                }
                AlignmentOp::Sub {
                    expected,
                    predicted,
                    ..
                } => {
                    reconstructed_predicted.push(predicted.clone());
                    reconstructed_expected.push(expected.clone());
                }
                AlignmentOp::Extra { predicted, .. } => {
                    reconstructed_predicted.push(predicted.clone());
                }
                AlignmentOp::Missing { expected } => {
                    reconstructed_expected.push(expected.clone());
                }
            }
        }
        assert_eq!(reconstructed_predicted, vec!["a", "b", "c", "d", "e"]);
        assert_eq!(reconstructed_expected, vec!["a", "x", "c", "e"]);
    }

    #[test]
    fn sub_carries_probabilities() {
        // Predicted "i" with prob 0.6 at position 0; "e" was the model's
        // runner-up at prob 0.3. Expected is "e", so the Sub op should
        // report predicted_prob = 0.6 and expected_prob = Some(0.3).
        let predicted = vec!["i".to_string()];
        let topk = vec![vec![("i".to_string(), 0.6), ("e".to_string(), 0.3)]];
        let expected = vec!["e".to_string()];
        let (_, ops) = align(&predicted, &topk, &expected);
        match &ops[0] {
            AlignmentOp::Sub {
                predicted_prob,
                expected_prob,
                ..
            } => {
                assert!((predicted_prob - 0.6).abs() < 1e-9);
                assert_eq!(*expected_prob, Some(0.3));
            }
            other => panic!("expected Sub, got {other:?}"),
        }
    }

    #[test]
    fn sub_reports_none_when_expected_not_in_topk() {
        let predicted = vec!["i".to_string()];
        let topk = vec![vec![("i".to_string(), 0.95)]]; // only one alt
        let expected = vec!["œ".to_string()];
        let (_, ops) = align(&predicted, &topk, &expected);
        match &ops[0] {
            AlignmentOp::Sub { expected_prob, .. } => {
                assert_eq!(*expected_prob, None);
            }
            other => panic!("expected Sub, got {other:?}"),
        }
    }

    #[test]
    fn normalize_with_topk_merges_canonical_equivalents() {
        // Raw top-k has both `r` and `ʁ`; canonicalization collapses them
        // to `ʁ`, and the merged top-k should have one `ʁ` entry with the
        // summed probability.
        let raw_phonemes = vec!["r".to_string()];
        let raw_topk = vec![vec![
            RawPhonemeAlt {
                phoneme: "r".to_string(),
                probability: 0.4,
            },
            RawPhonemeAlt {
                phoneme: "ʁ".to_string(),
                probability: 0.3,
            },
            RawPhonemeAlt {
                phoneme: "a".to_string(),
                probability: 0.2,
            },
        ]];
        let (norm, norm_topk) = normalize_with_topk(&raw_phonemes, &raw_topk, Language::French);
        assert_eq!(norm, vec!["ʁ".to_string()]);
        assert_eq!(norm_topk[0][0].0, "ʁ");
        assert!((norm_topk[0][0].1 - 0.7).abs() < 1e-9);
        assert_eq!(norm_topk[0][1].0, "a");
    }
}

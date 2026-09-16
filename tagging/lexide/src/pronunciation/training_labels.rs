//! Deterministic preprocessing only, shared with Python's `validate_phonemes`.

use anyhow::{bail, ensure, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const DEFINITION: &str = include_str!("../../data/training_labels.json");

#[derive(Debug, Deserialize)]
struct Definition {
    base_vocab: HashSet<String>,
    vocab_extensions: HashSet<String>,
    token_remap: HashMap<String, String>,
    lang_phoneme_remap: HashMap<String, HashMap<String, String>>,
    token_blacklist: HashSet<String>,
}

/// Remapped phones and their original, still-aligned stress values.
///
/// The stress type is carried opaquely: callers can use integers or their own
/// stress enum without lexide depending on a particular g2p engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingLabels<S> {
    pub phonemes: Vec<String>,
    pub stress: Vec<S>,
}

/// Convert raw g2p phones to the current deterministic preprocessing labels.
///
/// Applies exact language remaps, then global remaps, then accepts vocabulary
/// members (including specials), drops blacklisted tokens with their stress,
/// and errors on unknown tokens. No Unicode normalization, token splitting,
/// language aliases or g2p calls are performed. Use dataset identifiers such as
/// `eng`, `fra`, `ita` and `zho-hans`; unrecognized identifiers apply only the
/// global rules, just like Python preprocessing.
///
/// This is **not** a reconstruction of an arbitrary checkpoint's final labels.
/// It excludes acoustic narrowing, French rhythmic-group stress sidecars (which
/// require sentence text and word spans), and prosody supervision masks. Matching
/// g2p build identity alone does not establish voice/canon or postprocessing
/// equivalence. Choose the appropriate source and label pipeline for the model.
///
/// Unlike Python's legacy zip loop, unequal phone/stress lengths are errors.
/// Unknown tokens are errors rather than a partially accepted output to ignore.
///
/// ```
/// use lexide::pronunciation::training_labels;
/// let labels = training_labels("eng", &["ɐ", ".", "ɪ"], &[1, 0, 2])?;
/// assert_eq!(labels.phonemes, ["ə", "ɪ"]);
/// assert_eq!(labels.stress, [1, 2]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn training_labels<P: AsRef<str>, S: Clone>(
    language: &str,
    phonemes: &[P],
    stress: &[S],
) -> Result<TrainingLabels<S>> {
    static LABELS: OnceLock<Definition> = OnceLock::new();
    LABELS
        .get_or_init(|| {
            serde_json::from_str(DEFINITION).expect("embedded training-label definition")
        })
        .apply(language, phonemes, stress)
}

impl Definition {
    fn apply<P: AsRef<str>, S: Clone>(
        &self,
        language: &str,
        phonemes: &[P],
        stress: &[S],
    ) -> Result<TrainingLabels<S>> {
        ensure!(
            phonemes.len() == stress.len(),
            "training labels require aligned phone/stress lengths: {} phones, {} stress values",
            phonemes.len(),
            stress.len(),
        );
        let remap = self.lang_phoneme_remap.get(language);
        let mut labels = TrainingLabels {
            phonemes: Vec::with_capacity(phonemes.len()),
            stress: Vec::with_capacity(stress.len()),
        };
        for (index, (raw, stress)) in phonemes.iter().zip(stress).enumerate() {
            let raw = raw.as_ref();
            let phone = remap
                .and_then(|table| table.get(raw))
                .map_or(raw, String::as_str);
            let phone = self.token_remap.get(phone).map_or(phone, String::as_str);
            if self.base_vocab.contains(phone) || self.vocab_extensions.contains(phone) {
                labels.phonemes.push(phone.to_owned());
                labels.stress.push(stress.clone());
            } else if !self.token_blacklist.contains(phone) {
                bail!("unknown training-label token {phone:?} (raw {raw:?}, index {index}, language {language:?})");
            }
        }
        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        lang: Option<String>,
        phones: Vec<String>,
        stress: Vec<usize>,
        extra_vocab: Vec<String>,
        expected: (Vec<String>, Vec<usize>, BTreeSet<String>),
    }

    #[test]
    fn frozen_python_conformance() {
        // Captured with the OLD validator and actual cached get_vocab(), not
        // calculated from the definition under test. Python reads this too.
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../data/training_labels_conformance.json"))
                .unwrap();
        for case in fixture.cases {
            let mut definition: Definition = serde_json::from_str(DEFINITION).unwrap();
            definition.base_vocab.extend(case.extra_vocab.clone());
            let language = case.lang.as_deref().unwrap_or("");
            let actual = if case.extra_vocab.is_empty() {
                training_labels(language, &case.phones, &case.stress)
            } else {
                definition.apply(language, &case.phones, &case.stress)
            };
            if case.expected.2.is_empty() {
                let actual = actual.unwrap();
                assert_eq!(actual.phonemes, case.expected.0, "{}", case.name);
                assert_eq!(actual.stress, case.expected.1, "{}", case.name);
            } else {
                assert!(actual.is_err(), "{}", case.name);
            }
            // Also check each token in mixed unknown/valid cases: a whole-row
            // Err alone could hide incorrect remapping of the valid tokens.
            let mut kept = (Vec::new(), Vec::new(), BTreeSet::new());
            for (phone, stress) in case.phones.iter().zip(&case.stress) {
                match definition.apply(language, &[phone], &[*stress]) {
                    Ok(labels) => {
                        kept.0.extend(labels.phonemes);
                        kept.1.extend(labels.stress);
                    }
                    Err(error) => {
                        assert!(error.to_string().contains(&format!("{phone:?}")), "{error}");
                        // Frozen unknown inputs are unmapped; synthetic remap
                        // failures have a separate unit test below.
                        kept.2.insert(phone.clone());
                    }
                }
            }
            assert_eq!(kept, case.expected, "{}", case.name);
        }
    }

    #[test]
    fn language_then_global_then_vocab_then_blacklist() {
        let mut definition: Definition = serde_json::from_str(DEFINITION).unwrap();
        definition.lang_phoneme_remap.insert(
            "synthetic".into(),
            HashMap::from([("raw".into(), "ε".into())]),
        );
        definition.token_blacklist.insert("ɛ".into());
        let labels = definition.apply("synthetic", &["raw"], &[2]).unwrap();
        assert_eq!(labels.phonemes, ["ɛ"]);
        assert_eq!(labels.stress, [2]);
        definition.token_remap.insert("ε".into(), "unknown".into());
        let error = definition
            .apply("synthetic", &["raw"], &[2])
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown") && error.contains("raw"));
    }

    #[test]
    fn mismatched_lengths_fail_in_both_directions() {
        for (phones, stress) in [(vec!["a"], vec![]), (vec![], vec![0])] {
            let error = training_labels("eng", &phones, &stress)
                .unwrap_err()
                .to_string();
            assert!(error.contains("phone/stress lengths"));
        }
    }

    #[test]
    fn stress_can_be_an_external_non_copy_type() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        enum Stress {
            Primary(String),
            None,
        }
        let stress = [Stress::Primary("opaque".into()), Stress::None];
        let labels = training_labels("eng", &["ᵻ", "?"], &stress).unwrap();
        assert_eq!(labels.phonemes, ["ə"]);
        assert_eq!(labels.stress, [stress[0].clone()]);
    }
}

use crate::pos::PartOfSpeech;
use crate::{Lemma, Tokenization};

/// A match found by the discontinuous lemma matcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscontinuousMatch<K = String> {
    /// Index of the pattern that matched
    pub pattern_index: usize,
    /// Matched label
    pub matched_label: K,
    /// Token positions of each matched anchor (in order)
    pub positions: Vec<usize>,
}

/// Constraint on the tokens that appear in a gap between two anchors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GapConstraint {
    /// At least one token in the gap must have this POS
    ContainsPos(PartOfSpeech),
}

/// A single anchor in a discontinuous pattern.
#[derive(Debug, Clone)]
struct Anchor {
    lemma: Lemma,
    /// Constraint on the gap *before* this anchor (between the previous anchor and this one).
    /// Not applicable to the first anchor.
    gap_constraint: Option<GapConstraint>,
}

/// A pattern matcher for discontinuous lemma sequences.
///
/// Matches ordered lemma sequences that may have intervening tokens between them.
/// For example, pattern `["ne", "que"]` will match "ne soit que" because "ne" and "que"
/// appear in order with "soit" between them.
///
/// Supports:
/// - Per-gap maximum distance constraints
/// - Per-gap POS constraints (e.g., "a verb must appear between these anchors")
///
/// This is useful for constructions like French "ne...que" (restriction), "ne...pas"
/// (negation), German "weder...noch", Italian "non...mai", etc.
pub struct DiscontinuousLemmaMatcher<K = String>
where
    K: Clone,
{
    patterns: Vec<(K, Vec<Anchor>)>,
    max_gap: Option<usize>,
}

impl<K: Clone> DiscontinuousLemmaMatcher<K> {
    /// Creates a new discontinuous lemma matcher from simple string patterns.
    ///
    /// # Arguments
    ///
    /// * `patterns` - Labeled patterns: each is a sequence of lemmas that must appear in order
    /// * `max_gap` - Optional maximum number of tokens allowed between consecutive anchors.
    ///   `None` means unlimited gap.
    pub fn new(patterns: &[(K, Vec<&str>)], max_gap: Option<usize>) -> Self {
        let anchor_patterns: Vec<(K, Vec<Anchor>)> = patterns
            .iter()
            .map(|(label, pattern)| {
                (
                    label.clone(),
                    pattern
                        .iter()
                        .map(|&s| Anchor {
                            lemma: Lemma {
                                lemma: s.to_string(),
                            },
                            gap_constraint: None,
                        })
                        .collect(),
                )
            })
            .collect();
        Self {
            patterns: anchor_patterns,
            max_gap,
        }
    }

    /// Creates a new discontinuous lemma matcher with per-gap constraints.
    ///
    /// Each anchor after the first can have an optional [`GapConstraint`] that
    /// constrains the tokens appearing in the gap before it.
    ///
    /// # Arguments
    ///
    /// * `patterns` - Labeled patterns with optional gap constraints per anchor.
    ///   Each anchor is `(lemma, Option<GapConstraint>)`. The constraint on the
    ///   first anchor is ignored.
    /// * `max_gap` - Optional maximum number of tokens allowed between consecutive anchors.
    pub fn with_constraints(
        patterns: &[(K, Vec<(&str, Option<GapConstraint>)>)],
        max_gap: Option<usize>,
    ) -> Self {
        let anchor_patterns: Vec<(K, Vec<Anchor>)> = patterns
            .iter()
            .map(|(label, pattern)| {
                (
                    label.clone(),
                    pattern
                        .iter()
                        .map(|&(s, ref constraint)| Anchor {
                            lemma: Lemma {
                                lemma: s.to_string(),
                            },
                            gap_constraint: constraint.clone(),
                        })
                        .collect(),
                )
            })
            .collect();
        Self {
            patterns: anchor_patterns,
            max_gap,
        }
    }

    /// Finds all occurrences of any pattern in the tokenization.
    ///
    /// For each pattern, tries every occurrence of the first anchor as a starting
    /// point, then greedily matches subsequent anchors. Returns the first valid
    /// match per pattern (if any).
    pub fn find_all(&self, tokenization: &Tokenization) -> Vec<DiscontinuousMatch<K>> {
        let lemmas: Vec<&Lemma> = tokenization.tokens.iter().map(|t| &t.lemma).collect();
        let mut matches = Vec::new();

        'pattern: for (idx, (label, anchors)) in self.patterns.iter().enumerate() {
            if anchors.is_empty() {
                continue;
            }

            // Try each position as a potential start for the first anchor
            for start in 0..lemmas.len() {
                if *lemmas[start] != anchors[0].lemma {
                    continue;
                }

                let mut positions = vec![start];
                let mut matched = true;

                // Try to greedily match remaining anchors
                for anchor_idx in 1..anchors.len() {
                    let prev_pos = *positions.last().unwrap();
                    let search_end = if let Some(max_gap) = self.max_gap {
                        (prev_pos + 1 + max_gap).min(lemmas.len())
                    } else {
                        lemmas.len()
                    };

                    let mut found = false;
                    for i in (prev_pos + 1)..search_end {
                        if *lemmas[i] == anchors[anchor_idx].lemma {
                            // Check gap constraint
                            if let Some(ref constraint) = anchors[anchor_idx].gap_constraint {
                                let gap_tokens = &tokenization.tokens[(prev_pos + 1)..i];
                                if !check_gap_constraint(gap_tokens, constraint) {
                                    continue;
                                }
                            }
                            positions.push(i);
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        matched = false;
                        break;
                    }
                }

                if matched {
                    matches.push(DiscontinuousMatch {
                        pattern_index: idx,
                        matched_label: label.clone(),
                        positions,
                    });
                    continue 'pattern;
                }
            }
        }

        matches
    }

    /// Returns the number of patterns in this matcher.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

fn check_gap_constraint(gap_tokens: &[crate::Token], constraint: &GapConstraint) -> bool {
    match constraint {
        GapConstraint::ContainsPos(pos) => gap_tokens.iter().any(|t| t.pos == *pos),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dep::DependencyRelation;
    use crate::{Text, Token};

    fn make_token_with_pos(lemma: &str, pos: PartOfSpeech) -> Token {
        Token {
            text: Text {
                text: lemma.to_string(),
            },
            whitespace: " ".to_string(),
            pos,
            lemma: Lemma {
                lemma: lemma.to_string(),
            },
            dep: DependencyRelation::Root,
            head: 0,
        }
    }

    fn make_token(lemma: &str) -> Token {
        make_token_with_pos(lemma, PartOfSpeech::Noun)
    }

    fn make_tokenization(lemmas: &[&str]) -> Tokenization {
        Tokenization {
            tokens: lemmas.iter().map(|l| make_token(l)).collect(),
        }
    }

    #[test]
    fn test_basic_discontinuous_match() {
        let patterns = vec![("ne_que".to_string(), vec!["ne", "que"])];
        let matcher = DiscontinuousLemmaMatcher::new(&patterns, None);

        // "je ne sais que faire" — ne...que with "sais" between
        let tokenization = make_tokenization(&["je", "ne", "sais", "que", "faire"]);
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_label, "ne_que");
        assert_eq!(matches[0].positions, vec![1, 3]);
    }

    #[test]
    fn test_contiguous_also_matches() {
        let patterns = vec![("ne_que".to_string(), vec!["ne", "que"])];
        let matcher = DiscontinuousLemmaMatcher::new(&patterns, None);

        let tokenization = make_tokenization(&["ne", "que"]);
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].positions, vec![0, 1]);
    }

    #[test]
    fn test_no_match_wrong_order() {
        let patterns = vec![("ne_que".to_string(), vec!["ne", "que"])];
        let matcher = DiscontinuousLemmaMatcher::new(&patterns, None);

        let tokenization = make_tokenization(&["que", "ne"]);
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_no_match_missing_element() {
        let patterns = vec![("ne_que".to_string(), vec!["ne", "que"])];
        let matcher = DiscontinuousLemmaMatcher::new(&patterns, None);

        let tokenization = make_tokenization(&["ne", "sais", "pas"]);
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_max_gap_allows() {
        let patterns = vec![("ne_que".to_string(), vec!["ne", "que"])];
        let matcher = DiscontinuousLemmaMatcher::new(&patterns, Some(3));

        let tokenization = make_tokenization(&["ne", "soit", "que"]);
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_max_gap_rejects() {
        let patterns = vec![("ne_que".to_string(), vec!["ne", "que"])];
        let matcher = DiscontinuousLemmaMatcher::new(&patterns, Some(1));

        // Gap of 3 — exceeds limit of 1
        let tokenization = make_tokenization(&["ne", "le", "lui", "ai", "que"]);
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_three_element_pattern() {
        let patterns = vec![("ni_ni_ni".to_string(), vec!["ni", "ni", "ni"])];
        let matcher = DiscontinuousLemmaMatcher::new(&patterns, None);

        let tokenization = make_tokenization(&["ni", "toi", "ni", "moi", "ni", "personne"]);
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].positions, vec![0, 2, 4]);
    }

    #[test]
    fn test_multiple_patterns() {
        let patterns = vec![
            ("ne_pas".to_string(), vec!["ne", "pas"]),
            ("ne_que".to_string(), vec!["ne", "que"]),
        ];
        let matcher = DiscontinuousLemmaMatcher::new(&patterns, None);

        // Only ne...pas present, not ne...que
        let tokenization = make_tokenization(&["je", "ne", "sais", "pas"]);
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_label, "ne_pas");
    }

    #[test]
    fn test_max_gap_restarts_on_later_anchor() {
        let patterns = vec![("a_b".to_string(), vec!["a", "b"])];
        let matcher = DiscontinuousLemmaMatcher::new(&patterns, Some(1));

        // First "a" has gap too large to "b", but second "a" is close enough
        let tokenization = make_tokenization(&["a", "x", "y", "z", "a", "b"]);
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].positions, vec![4, 5]);
    }

    #[test]
    fn test_gap_constraint_contains_pos() {
        // Pattern: ne ... que, where the gap must contain a verb
        let patterns = vec![(
            "ne_que".to_string(),
            vec![
                ("ne", None),
                ("que", Some(GapConstraint::ContainsPos(PartOfSpeech::Verb))),
            ],
        )];
        let matcher = DiscontinuousLemmaMatcher::with_constraints(&patterns, None);

        // Gap contains a verb — should match
        let tokenization = Tokenization {
            tokens: vec![
                make_token("ne"),
                make_token_with_pos("soit", PartOfSpeech::Verb),
                make_token("que"),
            ],
        };
        let matches = matcher.find_all(&tokenization);
        assert_eq!(matches.len(), 1);

        // Gap contains only a noun — should not match
        let tokenization = Tokenization {
            tokens: vec![
                make_token("ne"),
                make_token_with_pos("chat", PartOfSpeech::Noun),
                make_token("que"),
            ],
        };
        let matches = matcher.find_all(&tokenization);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_gap_constraint_skips_bad_match_tries_next() {
        // Pattern: a ... b, where gap must contain a verb
        let patterns = vec![(
            "a_b".to_string(),
            vec![
                ("a", None),
                ("b", Some(GapConstraint::ContainsPos(PartOfSpeech::Verb))),
            ],
        )];
        let matcher = DiscontinuousLemmaMatcher::with_constraints(&patterns, None);

        // First "b" has no verb in gap, but second "b" does
        let tokenization = Tokenization {
            tokens: vec![
                make_token("a"),
                make_token("b"), // gap is empty — no verb
                make_token_with_pos("run", PartOfSpeech::Verb),
                make_token("b"), // gap has a verb
            ],
        };
        let matches = matcher.find_all(&tokenization);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].positions, vec![0, 3]);
    }
}

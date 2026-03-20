use crate::matching::aho_corasick::{AhoCorasick, Match};
use crate::pos::PartOfSpeech;
use crate::{LemmaPos, Tokenization};

/// A pattern matcher that operates on lemmatized tokens with POS awareness.
///
/// Searches for sequences of (lemma, POS) pairs within a tokenization,
/// making it useful for finding semantic patterns regardless of inflection
/// while distinguishing different grammatical roles (e.g., "être" as VERB vs AUX).
///
/// # Example
///
/// ```ignore
/// use lexide::matching::LemmaMatcher;
/// use lexide::pos::PartOfSpeech;
///
/// let patterns = vec![
///     ("be_happy".to_string(), vec![("be", PartOfSpeech::Verb), ("happy", PartOfSpeech::Adj)]),
/// ];
/// let matcher = LemmaMatcher::new(&patterns);
///
/// let matches = matcher.find_all(&tokenization);
/// ```
/// The type parameter `K` is the key/label type for patterns (defaults to `String`).
pub struct LemmaMatcher<K = String>
where
    K: Clone,
{
    automaton: AhoCorasick<LemmaPos, K>,
}

impl<K: Clone> LemmaMatcher<K> {
    /// Creates a new lemma matcher from labeled patterns with POS tags.
    ///
    /// # Arguments
    ///
    /// * `patterns` - A slice of (label, pattern) tuples, where each pattern is a slice of (lemma, POS) pairs
    pub fn new(patterns: &[(K, Vec<(&str, PartOfSpeech)>)]) -> Self {
        let lemma_pos_patterns: Vec<(K, Vec<LemmaPos>)> = patterns
            .iter()
            .map(|(label, pattern)| {
                (
                    label.clone(),
                    pattern
                        .iter()
                        .map(|&(s, pos)| LemmaPos {
                            lemma: s.to_string(),
                            pos,
                        })
                        .collect(),
                )
            })
            .collect();

        Self {
            automaton: AhoCorasick::new(&lemma_pos_patterns),
        }
    }

    /// Finds all occurrences of any pattern in the tokenization.
    ///
    /// # Arguments
    ///
    /// * `tokenization` - The tokenization to search within
    ///
    /// # Returns
    ///
    /// A vector of matches, where each match contains the pattern index and token positions.
    pub fn find_all<'a>(
        &'a self,
        tokenization: &'a Tokenization,
    ) -> Vec<Match<'a, LemmaPos, K>> {
        let lemma_pos_sequence: Vec<LemmaPos> = tokenization
            .tokens
            .iter()
            .map(|token| LemmaPos {
                lemma: token.lemma.lemma.clone(),
                pos: token.pos,
            })
            .collect();

        self.automaton.find_all(&lemma_pos_sequence).collect()
    }

    /// Checks if any pattern exists in the tokenization.
    ///
    /// This is more efficient than `find_all` when you only need to know
    /// whether a match exists.
    ///
    /// # Arguments
    ///
    /// * `tokenization` - The tokenization to search within
    ///
    /// # Returns
    ///
    /// `true` if any pattern is found, `false` otherwise.
    pub fn contains(&self, tokenization: &Tokenization) -> bool {
        let lemma_pos_sequence: Vec<LemmaPos> = tokenization
            .tokens
            .iter()
            .map(|token| LemmaPos {
                lemma: token.lemma.lemma.clone(),
                pos: token.pos,
            })
            .collect();

        self.automaton.contains(&lemma_pos_sequence)
    }

    /// Returns the number of patterns in this matcher.
    pub fn pattern_count(&self) -> usize {
        self.automaton.pattern_count()
    }
}

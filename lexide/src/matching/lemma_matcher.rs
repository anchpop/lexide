use crate::{Lemma, Tokenization};
use crate::matching::aho_corasick::{AhoCorasick, Match};

/// A pattern matcher that operates on lemmatized tokens.
///
/// Searches for sequences of lemmas within a tokenization,
/// making it useful for finding semantic patterns regardless of inflection.
///
/// # Example
///
/// ```ignore
/// use lexide::matching::LemmaMatcher;
///
/// let patterns = vec![
///     vec!["run", "quickly"],  // Will match "ran quickly", "running quickly", etc.
///     vec!["cat", "sleep"],    // Will match "cats sleeping", "cat slept", etc.
/// ];
/// let matcher = LemmaMatcher::new(&patterns);
///
/// let matches = matcher.find_all(&tokenization);
/// ```
pub struct LemmaMatcher {
    automaton: AhoCorasick<Lemma>,
}

impl LemmaMatcher {
    /// Creates a new lemma matcher from labeled string patterns.
    ///
    /// # Arguments
    ///
    /// * `patterns` - A slice of (label, pattern) tuples, where each pattern is a slice of lemma string references
    ///
    /// # Example
    ///
    /// ```ignore
    /// let patterns = vec![
    ///     ("happy_phrase".to_string(), vec!["be", "happy"]),
    ///     ("running".to_string(), vec!["run", "fast"]),
    /// ];
    /// let matcher = LemmaMatcher::new(&patterns);
    /// ```
    pub fn new(patterns: &[(String, Vec<&str>)]) -> Self {
        let lemma_patterns: Vec<(String, Vec<Lemma>)> = patterns
            .iter()
            .map(|(label, pattern)| {
                (
                    label.clone(),
                    pattern
                        .iter()
                        .map(|&s| Lemma {
                            lemma: s.to_string(),
                        })
                        .collect()
                )
            })
            .collect();

        Self {
            automaton: AhoCorasick::new(&lemma_patterns),
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
    pub fn find_all<'a>(&'a self, tokenization: &'a Tokenization) -> Vec<Match<'a, Lemma>> {
        let lemma_sequence: Vec<Lemma> = tokenization
            .tokens
            .iter()
            .map(|token| token.lemma.clone())
            .collect();

        self.automaton.find_all(&lemma_sequence).collect()
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
        let lemma_sequence: Vec<Lemma> = tokenization
            .tokens
            .iter()
            .map(|token| token.lemma.clone())
            .collect();

        self.automaton.contains(&lemma_sequence)
    }

    /// Returns the number of patterns in this matcher.
    pub fn pattern_count(&self) -> usize {
        self.automaton.pattern_count()
    }
}


use crate::{Text, Tokenization};
use crate::matching::aho_corasick::{AhoCorasick, Match};

/// A pattern matcher that operates on raw text tokens.
///
/// Searches for sequences of text tokens within a tokenization,
/// ignoring whitespace and other linguistic annotations.
///
/// # Example
///
/// ```ignore
/// use lexide::matching::TextMatcher;
///
/// let patterns = vec![
///     vec!["the", "cat"],
///     vec!["a", "dog"],
/// ];
/// let matcher = TextMatcher::new(&patterns);
///
/// let matches = matcher.find_all(&tokenization);
/// ```
pub struct TextMatcher {
    automaton: AhoCorasick<Text>,
}

impl TextMatcher {
    /// Creates a new text matcher from labeled string patterns.
    ///
    /// # Arguments
    ///
    /// * `patterns` - A slice of (label, pattern) tuples, where each pattern is a slice of string references
    ///
    /// # Example
    ///
    /// ```ignore
    /// let patterns = vec![
    ///     ("animal_phrase".to_string(), vec!["the", "quick", "brown"]),
    ///     ("lazy_phrase".to_string(), vec!["lazy", "dog"]),
    /// ];
    /// let matcher = TextMatcher::new(&patterns);
    /// ```
    pub fn new(patterns: &[(String, Vec<&str>)]) -> Self {
        let text_patterns: Vec<(String, Vec<Text>)> = patterns
            .iter()
            .map(|(label, pattern)| {
                (
                    label.clone(),
                    pattern
                        .iter()
                        .map(|&s| Text {
                            text: s.to_string(),
                        })
                        .collect()
                )
            })
            .collect();

        Self {
            automaton: AhoCorasick::new(&text_patterns),
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
    pub fn find_all<'a>(&'a self, tokenization: &'a Tokenization) -> Vec<Match<'a, Text>> {
        let text_sequence: Vec<Text> = tokenization
            .tokens
            .iter()
            .map(|token| token.text.clone())
            .collect();

        self.automaton.find_all(&text_sequence).collect()
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
        let text_sequence: Vec<Text> = tokenization
            .tokens
            .iter()
            .map(|token| token.text.clone())
            .collect();

        self.automaton.contains(&text_sequence)
    }

    /// Returns the number of patterns in this matcher.
    pub fn pattern_count(&self) -> usize {
        self.automaton.pattern_count()
    }
}

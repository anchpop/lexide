//! Pattern matching on tokenizations.
//!
//! This module provides specialized matchers for finding patterns in tokenized text,
//! using either raw text or lemmatized forms.

mod aho_corasick;
mod lemma_matcher;
mod text_matcher;
mod dependency_matcher;

pub use lemma_matcher::LemmaMatcher;
pub use text_matcher::TextMatcher;
pub use dependency_matcher::{DependencyMatcher, DependencyMatch, TreeNode};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dep::DependencyRelation;
    use crate::pos::PartOfSpeech;
    use crate::{Lemma, Text, Token, Tokenization};

    fn create_test_tokenization() -> Tokenization {
        Tokenization {
            tokens: vec![
                Token {
                    text: Text {
                        text: "The".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Det,
                    lemma: Lemma {
                        lemma: "the".to_string(),
                    },
                    dep: DependencyRelation::Det,
                    head: 1,
                },
                Token {
                    text: Text {
                        text: "cats".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Noun,
                    lemma: Lemma {
                        lemma: "cat".to_string(),
                    },
                    dep: DependencyRelation::Nsubj,
                    head: 2,
                },
                Token {
                    text: Text {
                        text: "are".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Aux,
                    lemma: Lemma {
                        lemma: "be".to_string(),
                    },
                    dep: DependencyRelation::Aux,
                    head: 3,
                },
                Token {
                    text: Text {
                        text: "sleeping".to_string(),
                    },
                    whitespace: ".".to_string(),
                    pos: PartOfSpeech::Verb,
                    lemma: Lemma {
                        lemma: "sleep".to_string(),
                    },
                    dep: DependencyRelation::Root,
                    head: 3,
                },
            ],
        }
    }

    #[test]
    fn test_text_matcher_finds_pattern() {
        let patterns = vec![("cats_are".to_string(), vec!["cats", "are"])];
        let matcher = TextMatcher::new(&patterns);

        let tokenization = create_test_tokenization();
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_index, 0);
        assert_eq!(matches[0].start, 1);
        assert_eq!(matches[0].end, 3);
        assert_eq!(matches[0].matched_label, "cats_are");
    }

    #[test]
    fn test_text_matcher_no_match() {
        let patterns = vec![("dogs_run".to_string(), vec!["dogs", "run"])];
        let matcher = TextMatcher::new(&patterns);

        let tokenization = create_test_tokenization();
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_text_matcher_contains() {
        let patterns = vec![("the_cats".to_string(), vec!["The", "cats"])];
        let matcher = TextMatcher::new(&patterns);

        let tokenization = create_test_tokenization();
        assert!(matcher.contains(&tokenization));
    }

    #[test]
    fn test_lemma_matcher_finds_pattern() {
        // Match using lemmas - "cat" lemma will match "cats" text
        let patterns = vec![("cat_be".to_string(), vec!["cat", "be"])];
        let matcher = LemmaMatcher::new(&patterns);

        let tokenization = create_test_tokenization();
        let matches = matcher.find_all(&tokenization);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_index, 0);
        assert_eq!(matches[0].start, 1);
        assert_eq!(matches[0].end, 3);
        assert_eq!(matches[0].matched_label, "cat_be");
    }

    #[test]
    fn test_lemma_matcher_multiple_patterns() {
        let patterns = vec![
            ("the_cat".to_string(), vec!["the", "cat"]),
            ("be_sleep".to_string(), vec!["be", "sleep"])
        ];
        let matcher = LemmaMatcher::new(&patterns);

        let tokenization = create_test_tokenization();
        let matches = matcher.find_all(&tokenization);

        // Should find both patterns
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_lemma_matcher_contains() {
        let patterns = vec![("cat_be_sleep".to_string(), vec!["cat", "be", "sleep"])];
        let matcher = LemmaMatcher::new(&patterns);

        let tokenization = create_test_tokenization();
        assert!(matcher.contains(&tokenization));
    }

    #[test]
    fn test_pattern_count() {
        let patterns = vec![
            ("a".to_string(), vec!["a"]),
            ("b".to_string(), vec!["b"]),
            ("c".to_string(), vec!["c"])
        ];
        let matcher = TextMatcher::new(&patterns);

        assert_eq!(matcher.pattern_count(), 3);
    }
}

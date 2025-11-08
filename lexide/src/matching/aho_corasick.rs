//! Generalized Aho-Corasick automaton for multi-pattern matching on arbitrary sequences.
//!
//! This implementation works with any sequence type and element comparison function,
//! making it suitable for matching on text, tokens, lemmas, or any other sequential data.

use std::collections::{HashMap, VecDeque};

/// A node in the Aho-Corasick trie.
#[derive(Debug, Clone)]
struct TrieNode<T> {
    /// Children of this node, indexed by element
    children: HashMap<T, usize>,
    /// Failure link - where to go if no matching child is found
    failure_link: Option<usize>,
    /// Output patterns - indices of patterns that end at this node
    output: Vec<usize>,
}

impl<T> TrieNode<T>
where
    T: Eq + std::hash::Hash,
{
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            failure_link: None,
            output: Vec::new(),
        }
    }
}

/// An Aho-Corasick automaton for multi-pattern matching.
///
/// This structure can match multiple patterns simultaneously against a sequence,
/// finding all occurrences of any pattern in linear time.
pub struct AhoCorasick<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    /// The trie nodes
    nodes: Vec<TrieNode<T>>,
    /// The patterns being matched (stored for reference)
    patterns: Vec<(String, Vec<T>)>,
}

/// A match found by the Aho-Corasick automaton.
#[derive(Clone, PartialEq, Eq)]
pub struct Match<'a, T> {
    /// Index of the pattern that matched
    pub pattern_index: usize,
    /// End position of the match in the searched sequence (exclusive)
    pub end: usize,
    /// Start position of the match in the searched sequence (inclusive)
    pub start: usize,
    /// Reference to the pattern that matched
    pub pattern: &'a [T],
    /// Matched label
    pub matched_label: String,
}

impl<T: std::fmt::Display> std::fmt::Debug for Match<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Match {{ ")?;
        for (i, item) in self.pattern.iter().enumerate() {
            write!(f, "{}", item)?;
            if i < self.pattern.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, " }}")?;
        Ok(())
    }
}

impl<T> AhoCorasick<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    /// Creates a new Aho-Corasick automaton from a collection of labeled patterns.
    ///
    /// # Arguments
    ///
    /// * `patterns` - A slice of (label, pattern) tuples
    ///
    /// # Example
    ///
    /// ```ignore
    /// let patterns = vec![
    ///     ("greeting".to_string(), vec!['h', 'e']),
    ///     ("pronoun".to_string(), vec!['s', 'h', 'e']),
    ///     ("possessive".to_string(), vec!['h', 'i', 's']),
    /// ];
    /// let ac = AhoCorasick::new(&patterns);
    /// ```
    pub fn new(patterns: &[(String, Vec<T>)]) -> Self {
        let mut ac = AhoCorasick {
            nodes: vec![TrieNode::new()],
            patterns: patterns.to_vec(),
        };

        // Build the trie
        ac.build_trie(patterns);

        // Build failure links
        ac.build_failure_links();

        ac
    }

    /// Builds the trie structure from the patterns.
    fn build_trie(&mut self, patterns: &[(String, Vec<T>)]) {
        for (pattern_idx, (_label, pattern)) in patterns.iter().enumerate() {
            let mut current_node = 0;

            for element in pattern.iter() {
                let next_node = if let Some(&child) = self.nodes[current_node].children.get(element)
                {
                    child
                } else {
                    let new_node_idx = self.nodes.len();
                    self.nodes.push(TrieNode::new());
                    self.nodes[current_node]
                        .children
                        .insert(element.clone(), new_node_idx);
                    new_node_idx
                };

                current_node = next_node;
            }

            // Mark this node as the end of a pattern
            self.nodes[current_node].output.push(pattern_idx);
        }
    }

    /// Builds the failure links using BFS.
    fn build_failure_links(&mut self) {
        let mut queue = VecDeque::new();

        // All depth-1 nodes have failure link to root
        let depth_1_children: Vec<usize> = self.nodes[0].children.values().copied().collect();
        for child_idx in depth_1_children {
            self.nodes[child_idx].failure_link = Some(0);
            queue.push_back(child_idx);
        }

        // BFS to compute failure links
        while let Some(current_idx) = queue.pop_front() {
            let children: Vec<(T, usize)> = self.nodes[current_idx]
                .children
                .iter()
                .map(|(k, &v)| (k.clone(), v))
                .collect();

            for (element, child_idx) in children {
                queue.push_back(child_idx);

                // Find the failure link for this child
                let mut failure_node = self.nodes[current_idx].failure_link;

                loop {
                    if let Some(fail_idx) = failure_node {
                        if let Some(&next_node) = self.nodes[fail_idx].children.get(&element) {
                            self.nodes[child_idx].failure_link = Some(next_node);

                            // Add outputs from failure link node
                            let outputs_to_add = self.nodes[next_node].output.clone();
                            self.nodes[child_idx].output.extend(outputs_to_add);
                            break;
                        } else {
                            failure_node = self.nodes[fail_idx].failure_link;
                        }
                    } else {
                        // Failed to root
                        self.nodes[child_idx].failure_link = Some(0);
                        break;
                    }
                }
            }
        }
    }

    /// Finds all occurrences of any pattern in the given sequence.
    ///
    /// Returns an iterator over all matches found.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to search within
    ///
    /// # Example
    ///
    /// ```ignore
    /// let text = vec!['u', 's', 'h', 'e', 'r', 's'];
    /// let matches: Vec<Match> = ac.find_all(&text).collect();
    /// ```
    pub fn find_all<'a, 'b>(&'a self, sequence: &'b [T]) -> impl Iterator<Item = Match<'a, T>> + 'a {
        let mut current_node = 0;
        let mut matches = Vec::new();

        for (pos, element) in sequence.iter().enumerate() {
            // Follow failure links until we find a matching edge or reach root
            loop {
                if let Some(&next_node) = self.nodes[current_node].children.get(element) {
                    current_node = next_node;
                    break;
                } else if current_node == 0 {
                    // At root and no match
                    break;
                } else {
                    // Follow failure link
                    current_node = self.nodes[current_node].failure_link.unwrap_or(0);
                }
            }

            // Report all patterns that end at this position
            for &pattern_idx in &self.nodes[current_node].output {
                let (label, pattern) = &self.patterns[pattern_idx];
                let pattern_len = pattern.len();
                matches.push(Match {
                    pattern_index: pattern_idx,
                    end: pos + 1,
                    start: pos + 1 - pattern_len,
                    pattern,
                    matched_label: label.clone(),
                });
            }
        }

        matches.into_iter()
    }

    /// Checks if any pattern exists in the given sequence.
    ///
    /// This is more efficient than `find_all` when you only need to know
    /// whether a match exists, not where it is.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The sequence to search within
    ///
    /// # Returns
    ///
    /// `true` if any pattern is found in the sequence, `false` otherwise.
    pub fn contains(&self, sequence: &[T]) -> bool {
        let mut current_node = 0;

        for element in sequence.iter() {
            // Follow failure links until we find a matching edge or reach root
            loop {
                if let Some(&next_node) = self.nodes[current_node].children.get(element) {
                    current_node = next_node;
                    break;
                } else if current_node == 0 {
                    // At root and no match
                    break;
                } else {
                    // Follow failure link
                    current_node = self.nodes[current_node].failure_link.unwrap_or(0);
                }
            }

            // Check if any pattern ends at this position
            if !self.nodes[current_node].output.is_empty() {
                return true;
            }
        }

        false
    }

    /// Returns the number of patterns in this automaton.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_char_matching() {
        let patterns = vec![
            ("he".to_string(), vec!['h', 'e']),
            ("she".to_string(), vec!['s', 'h', 'e']),
            ("his".to_string(), vec!['h', 'i', 's']),
            ("hers".to_string(), vec!['h', 'e', 'r', 's']),
        ];
        let ac = AhoCorasick::new(&patterns);

        let text = vec!['u', 's', 'h', 'e', 'r', 's'];
        let matches: Vec<Match<char>> = ac.find_all(&text).collect();

        assert_eq!(matches.len(), 3);

        // "she" at position 1-4
        assert!(matches.iter().any(|m| m.pattern_index == 1 && m.start == 1 && m.end == 4 && m.matched_label == "she"));
        // "he" at position 2-4
        assert!(matches.iter().any(|m| m.pattern_index == 0 && m.start == 2 && m.end == 4 && m.matched_label == "he"));
        // "hers" at position 2-6
        assert!(matches.iter().any(|m| m.pattern_index == 3 && m.start == 2 && m.end == 6 && m.matched_label == "hers"));
    }

    #[test]
    fn test_contains() {
        let patterns = vec![
            ("he".to_string(), vec!['h', 'e']),
            ("hello".to_string(), vec!['l', 'l', 'o'])
        ];
        let ac = AhoCorasick::new(&patterns);

        assert!(ac.contains(&['h', 'e', 'l', 'l', 'o']));
        assert!(ac.contains(&['h', 'e']));
        assert!(!ac.contains(&['w', 'o', 'r', 'l', 'd']));
    }

    #[test]
    fn test_integer_sequences() {
        let patterns = vec![
            ("seq1".to_string(), vec![1, 2, 3]),
            ("seq2".to_string(), vec![2, 3, 4]),
            ("seq3".to_string(), vec![3, 4])
        ];
        let ac = AhoCorasick::new(&patterns);

        let sequence = vec![0, 1, 2, 3, 4, 5];
        let matches: Vec<Match<i32>> = ac.find_all(&sequence).collect();

        assert_eq!(matches.len(), 3);
        assert!(matches.iter().any(|m| m.pattern_index == 0)); // [1,2,3]
        assert!(matches.iter().any(|m| m.pattern_index == 1)); // [2,3,4]
        assert!(matches.iter().any(|m| m.pattern_index == 2)); // [3,4]
    }

    #[test]
    fn test_overlapping_patterns() {
        let patterns = vec![
            ("aa".to_string(), vec!['a', 'a']),
            ("aaa".to_string(), vec!['a', 'a', 'a'])
        ];
        let ac = AhoCorasick::new(&patterns);

        let text = vec!['a', 'a', 'a'];
        let matches: Vec<Match<char>> = ac.find_all(&text).collect();

        // Should find "aa" at positions 0-2 and 1-3, and "aaa" at position 0-3
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_empty_patterns() {
        let patterns: Vec<(String, Vec<char>)> = vec![];
        let ac = AhoCorasick::new(&patterns);

        let text = vec!['a', 'b', 'c'];
        let matches: Vec<Match<char>> = ac.find_all(&text).collect();

        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_no_matches() {
        let patterns = vec![("xyz".to_string(), vec!['x', 'y', 'z'])];
        let ac = AhoCorasick::new(&patterns);

        let text = vec!['a', 'b', 'c'];
        let matches: Vec<Match<char>> = ac.find_all(&text).collect();

        assert_eq!(matches.len(), 0);
        assert!(!ac.contains(&text));
    }
}

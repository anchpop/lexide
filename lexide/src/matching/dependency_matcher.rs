use crate::{dep::DependencyRelation, Token, Tokenization};
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::convert::TryFrom;

/// A tree node representing a token and its dependency children.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TreeNode {
    pub token: Token,
    pub children: Vec<(DependencyRelation, TreeNode)>,
}

/// A match found by the dependency matcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyMatch<'a> {
    /// Index of the pattern that matched
    pub pattern_index: usize,
    /// Reference to the tree node that matched
    pub matched_node: &'a TreeNode,
    /// Matched label
    pub matched_label: String,
}

/// A pattern matcher that operates on dependency tree structures.
///
/// Searches for tree patterns within a tokenization's dependency tree,
/// making it useful for finding syntactic patterns based on grammatical structure.
///
/// # Example
///
/// ```ignore
/// use lexide::matching::{DependencyMatcher, TreeNode};
///
/// // Create a dependency tree from tokenization
/// let tree: TreeNode = tokenization.try_into().unwrap();
///
/// // Create pattern trees (e.g., "love" with an object child)
/// let love_pattern = TreeNode {
///     token: Token { lemma: Lemma { lemma: "love".to_string() }, ... },
///     children: vec![
///         (DependencyRelation::Obj, TreeNode { ... })
///     ],
/// };
///
/// let matcher = DependencyMatcher::new(&[love_pattern]);
/// let matches = matcher.find_all(&tree);
///
/// for match_result in matches {
///     println!("Pattern {} matched at node: {}",
///              match_result.pattern_index,
///              match_result.matched_node.token.lemma.lemma);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct DependencyMatcher {
    patterns: Vec<(String, TreeNode)>,
    root_index: HashMap<String, Vec<usize>>,
}

impl TryFrom<Tokenization> for TreeNode {
    type Error = &'static str;

    fn try_from(value: Tokenization) -> Result<Self, Self::Error> {
        // Find the root token (where dep == "root" or head points to itself in 1-indexed terms)
        let root_idx = value
            .tokens
            .iter()
            .enumerate()
            .position(|(idx, t)| t.dep == DependencyRelation::Root || t.head as usize == idx + 1)
            .ok_or("No root token found in tokenization")?;

        Ok(build_tree_node(&value.tokens, root_idx))
    }
}

impl DependencyMatcher {
    /// Creates a new dependency matcher from tree patterns.
    ///
    /// # Arguments
    ///
    /// * `patterns` - A slice of TreeNode patterns to match
    ///
    /// # Example
    ///
    /// ```ignore
    /// let patterns = vec![love_pattern, run_pattern];
    /// let matcher = DependencyMatcher::new(&patterns);
    /// ```
    pub fn new(patterns: &[(String, TreeNode)]) -> Self {
        // Build index: lemma -> pattern indices that start with this lemma
        let mut root_index: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, pattern) in patterns.iter().enumerate() {
            root_index
                .entry(pattern.1.token.lemma.lemma.clone())
                .or_default()
                .push(idx);
        }

        Self {
            patterns: patterns.to_vec(),
            root_index,
        }
    }

    /// Finds all occurrences of any pattern in the dependency tree.
    ///
    /// # Arguments
    ///
    /// * `tree` - The dependency tree to search within
    ///
    /// # Returns
    ///
    /// A vector of matches, where each match contains the pattern index and matched node.
    pub fn find_all<'a>(&'a self, tree: &'a TreeNode) -> Vec<DependencyMatch<'a>> {
        let mut matches = Vec::new();
        self.traverse_and_match(tree, &mut matches);
        matches
    }

    /// Checks if any pattern exists in the dependency tree.
    ///
    /// This is more efficient than `find_all` when you only need to know
    /// whether a match exists.
    ///
    /// # Arguments
    ///
    /// * `tree` - The dependency tree to search within
    ///
    /// # Returns
    ///
    /// `true` if any pattern is found, `false` otherwise.
    pub fn contains(&self, tree: &TreeNode) -> bool {
        self.has_match(tree)
    }

    /// Returns the number of patterns in this matcher.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    fn traverse_and_match<'a>(
        &'a self,
        node: &'a TreeNode,
        matches: &mut Vec<DependencyMatch<'a>>,
    ) {
        let mut visited = HashSet::new();
        self.traverse_and_match_impl(node, matches, &mut visited);
    }

    fn traverse_and_match_impl<'a>(
        &'a self,
        node: &'a TreeNode,
        matches: &mut Vec<DependencyMatch<'a>>,
        visited: &mut HashSet<*const TreeNode>,
    ) {
        // Use pointer address to detect cycles
        let node_ptr = node as *const TreeNode;
        if visited.contains(&node_ptr) {
            return; // Already visited this node, avoid infinite loop
        }
        visited.insert(node_ptr);

        // Check if this node starts any patterns
        if let Some(pattern_indices) = self.root_index.get(&node.token.lemma.lemma) {
            for &pattern_idx in pattern_indices {
                let (label, pattern) = &self.patterns[pattern_idx];

                // Check if this node matches the pattern (with children)
                if node.matches_pattern(pattern) {
                    matches.push(DependencyMatch {
                        pattern_index: pattern_idx,
                        matched_node: node,
                        matched_label: label.clone(),
                    });
                }
            }
        }

        // Recursively traverse children
        for (_, child) in &node.children {
            self.traverse_and_match_impl(child, matches, visited);
        }
    }

    fn has_match(&self, node: &TreeNode) -> bool {
        let mut visited = HashSet::new();
        self.has_match_impl(node, &mut visited)
    }

    fn has_match_impl(&self, node: &TreeNode, visited: &mut HashSet<*const TreeNode>) -> bool {
        // Use pointer address to detect cycles
        let node_ptr = node as *const TreeNode;
        if visited.contains(&node_ptr) {
            return false; // Already visited this node, avoid infinite loop
        }
        visited.insert(node_ptr);

        // Check if this node starts any patterns
        if let Some(pattern_indices) = self.root_index.get(&node.token.lemma.lemma) {
            for &pattern_idx in pattern_indices {
                let (_, pattern) = &self.patterns[pattern_idx];

                // Check if this node matches the pattern (with children)
                if node.matches_pattern(pattern) {
                    return true;
                }
            }
        }

        // Recursively check children
        for (_, child) in &node.children {
            if self.has_match_impl(child, visited) {
                return true;
            }
        }

        false
    }
}

impl TreeNode {
    /// Find multiple patterns in a single pass through the tree (for testing).
    /// Returns a map from pattern index to all locations where that pattern was found.
    #[cfg(test)]
    fn find_multiple<'a>(&'a self, patterns: &[&TreeNode]) -> HashMap<usize, Vec<&'a TreeNode>> {
        // Build index: lemma -> pattern indices that start with this lemma
        let mut root_index: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, pattern) in patterns.iter().enumerate() {
            root_index
                .entry(pattern.token.lemma.lemma.clone())
                .or_default()
                .push(idx);
        }

        let mut results = HashMap::new();
        self.traverse_with_patterns(patterns, &root_index, &mut results);
        results
    }

    #[cfg(test)]
    fn traverse_with_patterns<'a>(
        &'a self,
        patterns: &[&TreeNode],
        root_index: &HashMap<String, Vec<usize>>,
        results: &mut HashMap<usize, Vec<&'a TreeNode>>,
    ) {
        let mut visited = HashSet::new();
        self.traverse_with_patterns_impl(patterns, root_index, results, &mut visited);
    }

    #[cfg(test)]
    fn traverse_with_patterns_impl<'a>(
        &'a self,
        patterns: &[&TreeNode],
        root_index: &HashMap<String, Vec<usize>>,
        results: &mut HashMap<usize, Vec<&'a TreeNode>>,
        visited: &mut HashSet<*const TreeNode>,
    ) {
        // Use pointer address to detect cycles
        let node_ptr = self as *const TreeNode;
        if visited.contains(&node_ptr) {
            return; // Already visited this node, avoid infinite loop
        }
        visited.insert(node_ptr);

        // Check if this node starts any patterns
        if let Some(pattern_indices) = root_index.get(&self.token.lemma.lemma) {
            for &pattern_idx in pattern_indices {
                let pattern = patterns[pattern_idx];

                // Check if this node matches the pattern (with children)
                if self.matches_pattern(pattern) {
                    results.entry(pattern_idx).or_default().push(self);
                }
            }
        }

        // Recursively traverse children
        for (_, child) in &self.children {
            child.traverse_with_patterns_impl(patterns, root_index, results, visited);
        }
    }

    fn matches_pattern(&self, pattern: &TreeNode) -> bool {
        let mut visited = HashSet::new();
        self.matches_pattern_impl(pattern, &mut visited)
    }

    fn matches_pattern_impl(
        &self,
        pattern: &TreeNode,
        visited: &mut HashSet<*const TreeNode>,
    ) -> bool {
        // Lemmas must match (already checked by caller, but defensive)
        if self.token.lemma != pattern.token.lemma {
            return false;
        }

        // Use pointer address to detect cycles in the tree being searched
        let node_ptr = self as *const TreeNode;
        if visited.contains(&node_ptr) {
            // If we've seen this node before, treat it as having no children
            return pattern.children.is_empty();
        }
        visited.insert(node_ptr);

        // Build map of what children we need from the pattern
        let mut needed = pattern
            .children
            .iter()
            .map(|(dep, tree)| (*dep, tree))
            .collect::<HashMap<_, _>>();

        // Check each of our direct children only
        for (child_dep, child_tree) in &self.children {
            if let Entry::Occupied(entry) = needed.entry(*child_dep) {
                let pattern_child = entry.get();
                // Only check if the direct child's lemma matches - don't recurse deeply
                if child_tree.token.lemma == pattern_child.token.lemma {
                    // Now recursively check if this child's children match the pattern's children
                    if child_tree.matches_pattern_impl(pattern_child, visited) {
                        entry.remove();
                    }
                }
            }
        }

        // Pattern matches if all needed children were found
        needed.is_empty()
    }
}

fn build_tree_node(tokens: &[Token], token_idx: usize) -> TreeNode {
    let mut visited = HashSet::new();
    build_tree_node_impl(tokens, token_idx, &mut visited)
}

fn build_tree_node_impl(
    tokens: &[Token],
    token_idx: usize,
    visited: &mut HashSet<usize>,
) -> TreeNode {
    let token = tokens[token_idx].clone();

    // Check for cycles: if we've already visited this token in the current path, stop here
    if visited.contains(&token_idx) {
        // Break the cycle by returning a node with no children
        return TreeNode {
            token,
            children: Vec::new(),
        };
    }

    // Mark this token as visited
    visited.insert(token_idx);

    // Find all children of this token
    // Note: head indices are 1-indexed, so we compare with token_idx + 1
    let mut children = Vec::new();
    for (idx, child_token) in tokens.iter().enumerate() {
        if child_token.head as usize == token_idx + 1 && idx != token_idx {
            // Recursively build the child node
            let child_node = build_tree_node_impl(tokens, idx, visited);
            children.push((child_token.dep.clone(), child_node));
        }
    }

    // Unmark this token as we backtrack (allows the token to appear in other branches)
    visited.remove(&token_idx);

    TreeNode { token, children }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pos::PartOfSpeech, Lemma, Text};

    #[test]
    fn test_tree_node_from_tokenization() {
        // Test with: "I love programming."
        let tokenization = Tokenization {
            tokens: vec![
                Token {
                    text: Text {
                        text: "I".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Pron,
                    lemma: Lemma {
                        lemma: "I".to_string(),
                    },
                    dep: DependencyRelation::Nsubj,
                    head: 2,
                },
                Token {
                    text: Text {
                        text: "love".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Verb,
                    lemma: Lemma {
                        lemma: "love".to_string(),
                    },
                    dep: DependencyRelation::Root,
                    head: 0,
                },
                Token {
                    text: Text {
                        text: "programming".to_string(),
                    },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Noun,
                    lemma: Lemma {
                        lemma: "programming".to_string(),
                    },
                    dep: DependencyRelation::Obj,
                    head: 2,
                },
                Token {
                    text: Text {
                        text: ".".to_string(),
                    },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Punct,
                    lemma: Lemma {
                        lemma: ".".to_string(),
                    },
                    dep: DependencyRelation::Punct,
                    head: 2,
                },
            ],
        };

        let tree: TreeNode = tokenization.try_into().unwrap();

        // Root should be "love"
        assert_eq!(tree.token.text.text, "love");
        assert_eq!(tree.token.dep, DependencyRelation::Root);

        // Should have 3 children: I (Nsubj), programming (Obj), . (Punct)
        assert_eq!(tree.children.len(), 3);

        // Check children
        let child_texts: Vec<String> = tree
            .children
            .iter()
            .map(|(_, node)| node.token.text.text.clone())
            .collect();
        assert!(child_texts.contains(&"I".to_string()));
        assert!(child_texts.contains(&"programming".to_string()));
        assert!(child_texts.contains(&".".to_string()));

        // Check dependency relations
        for (dep_rel, child_node) in &tree.children {
            match child_node.token.text.text.as_str() {
                "I" => assert_eq!(*dep_rel, DependencyRelation::Nsubj),
                "programming" => assert_eq!(*dep_rel, DependencyRelation::Obj),
                "." => assert_eq!(*dep_rel, DependencyRelation::Punct),
                _ => panic!("Unexpected child token"),
            }
        }

        // Children should have no children of their own
        for (_, child) in &tree.children {
            assert_eq!(child.children.len(), 0);
        }
    }

    #[test]
    fn test_find_multiple_basic() {
        // Big tree: "I love programming"
        let big_tree = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Nsubj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "I".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "I".to_string(),
                            },
                            dep: DependencyRelation::Nsubj,
                            head: 2,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Obj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "programming".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Noun,
                            lemma: Lemma {
                                lemma: "programming".to_string(),
                            },
                            dep: DependencyRelation::Obj,
                            head: 2,
                        },
                        children: vec![],
                    },
                ),
            ],
        };

        // Pattern 1: just "love"
        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        // Pattern 2: "programming"
        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "programming".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Noun,
                lemma: Lemma {
                    lemma: "programming".to_string(),
                },
                dep: DependencyRelation::Obj,
                head: 2,
            },
            children: vec![],
        };

        // Pattern 3: "I"
        let pattern3 = TreeNode {
            token: Token {
                text: Text {
                    text: "I".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Pron,
                lemma: Lemma {
                    lemma: "I".to_string(),
                },
                dep: DependencyRelation::Nsubj,
                head: 2,
            },
            children: vec![],
        };

        let patterns = vec![&pattern1, &pattern2, &pattern3];
        let results = big_tree.find_multiple(&patterns);

        // All patterns should be found
        assert_eq!(results.len(), 3);
        assert_eq!(results[&0].len(), 1); // "love" found once
        assert_eq!(results[&1].len(), 1); // "programming" found once
        assert_eq!(results[&2].len(), 1); // "I" found once
    }

    #[test]
    fn test_find_multiple_with_structure() {
        // Big tree: "I love programming"
        let big_tree = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Nsubj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "I".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "I".to_string(),
                            },
                            dep: DependencyRelation::Nsubj,
                            head: 2,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Obj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "programming".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Noun,
                            lemma: Lemma {
                                lemma: "programming".to_string(),
                            },
                            dep: DependencyRelation::Obj,
                            head: 2,
                        },
                        children: vec![],
                    },
                ),
            ],
        };

        // Pattern 1: "love" with Obj child
        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Obj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "programming".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Noun,
                        lemma: Lemma {
                            lemma: "programming".to_string(),
                        },
                        dep: DependencyRelation::Obj,
                        head: 2,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern 2: "love" with Nsubj child (also should match)
        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Nsubj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "I".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Pron,
                        lemma: Lemma {
                            lemma: "I".to_string(),
                        },
                        dep: DependencyRelation::Nsubj,
                        head: 2,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern 3: "love" with wrong child (should NOT match)
        let pattern3 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Obj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "coding".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Noun,
                        lemma: Lemma {
                            lemma: "coding".to_string(),
                        },
                        dep: DependencyRelation::Obj,
                        head: 2,
                    },
                    children: vec![],
                },
            )],
        };

        let patterns = vec![&pattern1, &pattern2, &pattern3];
        let results = big_tree.find_multiple(&patterns);

        // Pattern 1 and 2 should match, pattern 3 should not
        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&0));
        assert!(results.contains_key(&1));
        assert!(!results.contains_key(&2));
    }

    #[test]
    fn test_find_multiple_empty_patterns() {
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let patterns: Vec<&TreeNode> = vec![];
        let results = tree.find_multiple(&patterns);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_multiple_single_node_tree() {
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "walk".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "walk".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let patterns = vec![&pattern1, &pattern2];
        let results = tree.find_multiple(&patterns);

        assert_eq!(results.len(), 1);
        assert!(results.contains_key(&0));
        assert!(!results.contains_key(&1));
    }

    #[test]
    fn test_find_multiple_all_patterns_fail() {
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "walk".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "walk".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "jump".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "jump".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let patterns = vec![&pattern1, &pattern2];
        let results = tree.find_multiple(&patterns);

        assert!(results.is_empty());
    }

    #[test]
    fn test_find_multiple_same_lemma_different_structures() {
        // Tree: "I love programming and love coding"
        // Two "love" nodes with different children
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Nsubj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "I".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "I".to_string(),
                            },
                            dep: DependencyRelation::Nsubj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Obj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "programming".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Noun,
                            lemma: Lemma {
                                lemma: "programming".to_string(),
                            },
                            dep: DependencyRelation::Obj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Conj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "love".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Verb,
                            lemma: Lemma {
                                lemma: "love".to_string(),
                            },
                            dep: DependencyRelation::Conj,
                            head: 1,
                        },
                        children: vec![(
                            DependencyRelation::Obj,
                            TreeNode {
                                token: Token {
                                    text: Text {
                                        text: "coding".to_string(),
                                    },
                                    whitespace: "".to_string(),
                                    pos: PartOfSpeech::Noun,
                                    lemma: Lemma {
                                        lemma: "coding".to_string(),
                                    },
                                    dep: DependencyRelation::Obj,
                                    head: 4,
                                },
                                children: vec![],
                            },
                        )],
                    },
                ),
            ],
        };

        // Pattern: "love" with "programming" as object
        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Obj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "programming".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Noun,
                        lemma: Lemma {
                            lemma: "programming".to_string(),
                        },
                        dep: DependencyRelation::Obj,
                        head: 1,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern: "love" with "coding" as object
        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Obj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "coding".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Noun,
                        lemma: Lemma {
                            lemma: "coding".to_string(),
                        },
                        dep: DependencyRelation::Obj,
                        head: 4,
                    },
                    children: vec![],
                },
            )],
        };

        let patterns = vec![&pattern1, &pattern2];
        let results = tree.find_multiple(&patterns);

        // Both patterns should match
        assert_eq!(results.len(), 2);
        assert_eq!(results[&0].len(), 1); // First "love" matches pattern1
        assert_eq!(results[&1].len(), 1); // Second "love" matches pattern2
    }

    #[test]
    fn test_find_multiple_duplicate_patterns() {
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let pattern = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        // Same pattern twice
        let patterns = vec![&pattern, &pattern];
        let results = tree.find_multiple(&patterns);

        // Both indices should have results (even though it's the same pattern)
        assert_eq!(results.len(), 2);
        assert_eq!(results[&0].len(), 1);
        assert_eq!(results[&1].len(), 1);
    }

    #[test]
    fn test_find_multiple_deep_nesting() {
        // Create a deeply nested tree: "I think you said he loves programming"
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "think".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "think".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Nsubj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "I".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "I".to_string(),
                            },
                            dep: DependencyRelation::Nsubj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Ccomp,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "said".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Verb,
                            lemma: Lemma {
                                lemma: "say".to_string(),
                            },
                            dep: DependencyRelation::Ccomp,
                            head: 1,
                        },
                        children: vec![
                            (
                                DependencyRelation::Nsubj,
                                TreeNode {
                                    token: Token {
                                        text: Text {
                                            text: "you".to_string(),
                                        },
                                        whitespace: " ".to_string(),
                                        pos: PartOfSpeech::Pron,
                                        lemma: Lemma {
                                            lemma: "you".to_string(),
                                        },
                                        dep: DependencyRelation::Nsubj,
                                        head: 2,
                                    },
                                    children: vec![],
                                },
                            ),
                            (
                                DependencyRelation::Ccomp,
                                TreeNode {
                                    token: Token {
                                        text: Text {
                                            text: "loves".to_string(),
                                        },
                                        whitespace: " ".to_string(),
                                        pos: PartOfSpeech::Verb,
                                        lemma: Lemma {
                                            lemma: "love".to_string(),
                                        },
                                        dep: DependencyRelation::Ccomp,
                                        head: 2,
                                    },
                                    children: vec![
                                        (
                                            DependencyRelation::Nsubj,
                                            TreeNode {
                                                token: Token {
                                                    text: Text {
                                                        text: "he".to_string(),
                                                    },
                                                    whitespace: " ".to_string(),
                                                    pos: PartOfSpeech::Pron,
                                                    lemma: Lemma {
                                                        lemma: "he".to_string(),
                                                    },
                                                    dep: DependencyRelation::Nsubj,
                                                    head: 3,
                                                },
                                                children: vec![],
                                            },
                                        ),
                                        (
                                            DependencyRelation::Obj,
                                            TreeNode {
                                                token: Token {
                                                    text: Text {
                                                        text: "programming".to_string(),
                                                    },
                                                    whitespace: "".to_string(),
                                                    pos: PartOfSpeech::Noun,
                                                    lemma: Lemma {
                                                        lemma: "programming".to_string(),
                                                    },
                                                    dep: DependencyRelation::Obj,
                                                    head: 3,
                                                },
                                                children: vec![],
                                            },
                                        ),
                                    ],
                                },
                            ),
                        ],
                    },
                ),
            ],
        };

        // Pattern for "programming" at depth 4
        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "programming".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Noun,
                lemma: Lemma {
                    lemma: "programming".to_string(),
                },
                dep: DependencyRelation::Obj,
                head: 3,
            },
            children: vec![],
        };

        // Pattern for "love" with children at depth 3
        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Nsubj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "he".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Pron,
                        lemma: Lemma {
                            lemma: "he".to_string(),
                        },
                        dep: DependencyRelation::Nsubj,
                        head: 3,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern for "say" at depth 2
        let pattern3 = TreeNode {
            token: Token {
                text: Text {
                    text: "say".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "say".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let patterns = vec![&pattern1, &pattern2, &pattern3];
        let results = tree.find_multiple(&patterns);

        // All three patterns should be found
        assert_eq!(results.len(), 3);
        assert_eq!(results[&0].len(), 1); // "programming"
        assert_eq!(results[&1].len(), 1); // "love" with "he"
        assert_eq!(results[&2].len(), 1); // "say"
    }

    #[test]
    fn test_find_multiple_pattern_subset_relationship() {
        // Tree with nested structure
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Nsubj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "I".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "I".to_string(),
                            },
                            dep: DependencyRelation::Nsubj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Obj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "marathon".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Noun,
                            lemma: Lemma {
                                lemma: "marathon".to_string(),
                            },
                            dep: DependencyRelation::Obj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
            ],
        };

        // Pattern 1: just "run" (no children required)
        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        // Pattern 2: "run" with Nsubj
        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Nsubj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "I".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Pron,
                        lemma: Lemma {
                            lemma: "I".to_string(),
                        },
                        dep: DependencyRelation::Nsubj,
                        head: 1,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern 3: "run" with both Nsubj and Obj
        let pattern3 = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Nsubj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "I".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "I".to_string(),
                            },
                            dep: DependencyRelation::Nsubj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Obj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "marathon".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Noun,
                            lemma: Lemma {
                                lemma: "marathon".to_string(),
                            },
                            dep: DependencyRelation::Obj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
            ],
        };

        let patterns = vec![&pattern1, &pattern2, &pattern3];
        let results = tree.find_multiple(&patterns);

        // All three should match (pattern1 is subset of pattern2, pattern2 is subset of pattern3)
        assert_eq!(results.len(), 3);
        assert_eq!(results[&0].len(), 1);
        assert_eq!(results[&1].len(), 1);
        assert_eq!(results[&2].len(), 1);
    }

    #[test]
    fn test_find_multiple_pattern_requires_missing_child() {
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Nsubj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "I".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Pron,
                        lemma: Lemma {
                            lemma: "I".to_string(),
                        },
                        dep: DependencyRelation::Nsubj,
                        head: 1,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern requires an Obj child that doesn't exist
        let pattern = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Obj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "marathon".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Noun,
                        lemma: Lemma {
                            lemma: "marathon".to_string(),
                        },
                        dep: DependencyRelation::Obj,
                        head: 1,
                    },
                    children: vec![],
                },
            )],
        };

        let patterns = vec![&pattern];
        let results = tree.find_multiple(&patterns);

        // Should not match
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_multiple_multiple_occurrences_same_pattern() {
        // Tree with multiple "I" nodes
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "think".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "think".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Nsubj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "I".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "I".to_string(),
                            },
                            dep: DependencyRelation::Nsubj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Ccomp,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "know".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Verb,
                            lemma: Lemma {
                                lemma: "know".to_string(),
                            },
                            dep: DependencyRelation::Ccomp,
                            head: 1,
                        },
                        children: vec![
                            (
                                DependencyRelation::Nsubj,
                                TreeNode {
                                    token: Token {
                                        text: Text {
                                            text: "I".to_string(),
                                        },
                                        whitespace: " ".to_string(),
                                        pos: PartOfSpeech::Pron,
                                        lemma: Lemma {
                                            lemma: "I".to_string(),
                                        },
                                        dep: DependencyRelation::Nsubj,
                                        head: 2,
                                    },
                                    children: vec![],
                                },
                            ),
                            (
                                DependencyRelation::Obj,
                                TreeNode {
                                    token: Token {
                                        text: Text {
                                            text: "you".to_string(),
                                        },
                                        whitespace: "".to_string(),
                                        pos: PartOfSpeech::Pron,
                                        lemma: Lemma {
                                            lemma: "you".to_string(),
                                        },
                                        dep: DependencyRelation::Obj,
                                        head: 2,
                                    },
                                    children: vec![],
                                },
                            ),
                        ],
                    },
                ),
            ],
        };

        // Pattern for "I"
        let pattern = TreeNode {
            token: Token {
                text: Text {
                    text: "I".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Pron,
                lemma: Lemma {
                    lemma: "I".to_string(),
                },
                dep: DependencyRelation::Nsubj,
                head: 0,
            },
            children: vec![],
        };

        let patterns = vec![&pattern];
        let results = tree.find_multiple(&patterns);

        // Should find both "I" nodes
        assert_eq!(results.len(), 1);
        assert_eq!(results[&0].len(), 2);
    }

    // Tests for DependencyMatcher interface
    #[test]
    fn test_dependency_matcher_basic() {
        let tokenization = Tokenization {
            tokens: vec![
                Token {
                    text: Text {
                        text: "I".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Pron,
                    lemma: Lemma {
                        lemma: "I".to_string(),
                    },
                    dep: DependencyRelation::Nsubj,
                    head: 2,
                },
                Token {
                    text: Text {
                        text: "love".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Verb,
                    lemma: Lemma {
                        lemma: "love".to_string(),
                    },
                    dep: DependencyRelation::Root,
                    head: 0,
                },
                Token {
                    text: Text {
                        text: "programming".to_string(),
                    },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Noun,
                    lemma: Lemma {
                        lemma: "programming".to_string(),
                    },
                    dep: DependencyRelation::Obj,
                    head: 2,
                },
            ],
        };

        // Pattern: "love"
        let pattern = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let matcher = DependencyMatcher::new(&[("love_pattern".to_string(), pattern)]);
        let tree: TreeNode = tokenization.try_into().unwrap();
        let matches = matcher.find_all(&tree);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_index, 0);
        assert_eq!(matches[0].matched_node.token.lemma.lemma, "love");
        assert_eq!(matches[0].matched_label, "love_pattern");
    }

    #[test]
    fn test_dependency_matcher_multiple_patterns() {
        let tokenization = Tokenization {
            tokens: vec![
                Token {
                    text: Text {
                        text: "I".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Pron,
                    lemma: Lemma {
                        lemma: "I".to_string(),
                    },
                    dep: DependencyRelation::Nsubj,
                    head: 2,
                },
                Token {
                    text: Text {
                        text: "love".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Verb,
                    lemma: Lemma {
                        lemma: "love".to_string(),
                    },
                    dep: DependencyRelation::Root,
                    head: 0,
                },
                Token {
                    text: Text {
                        text: "programming".to_string(),
                    },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Noun,
                    lemma: Lemma {
                        lemma: "programming".to_string(),
                    },
                    dep: DependencyRelation::Obj,
                    head: 2,
                },
            ],
        };

        // Pattern 1: "love"
        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        // Pattern 2: "programming"
        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "programming".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Noun,
                lemma: Lemma {
                    lemma: "programming".to_string(),
                },
                dep: DependencyRelation::Obj,
                head: 2,
            },
            children: vec![],
        };

        let matcher = DependencyMatcher::new(&[
            ("love_pattern".to_string(), pattern1),
            ("programming_pattern".to_string(), pattern2),
        ]);
        let tree: TreeNode = tokenization.try_into().unwrap();
        let matches = matcher.find_all(&tree);

        assert_eq!(matches.len(), 2);
        assert!(matches
            .iter()
            .any(|m| m.pattern_index == 0 && m.matched_label == "love_pattern"));
        assert!(matches
            .iter()
            .any(|m| m.pattern_index == 1 && m.matched_label == "programming_pattern"));
    }

    #[test]
    fn test_dependency_matcher_contains() {
        let tokenization = Tokenization {
            tokens: vec![
                Token {
                    text: Text {
                        text: "I".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Pron,
                    lemma: Lemma {
                        lemma: "I".to_string(),
                    },
                    dep: DependencyRelation::Nsubj,
                    head: 2,
                },
                Token {
                    text: Text {
                        text: "love".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Verb,
                    lemma: Lemma {
                        lemma: "love".to_string(),
                    },
                    dep: DependencyRelation::Root,
                    head: 0,
                },
                Token {
                    text: Text {
                        text: "programming".to_string(),
                    },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Noun,
                    lemma: Lemma {
                        lemma: "programming".to_string(),
                    },
                    dep: DependencyRelation::Obj,
                    head: 2,
                },
            ],
        };

        // Pattern that exists
        let pattern_exists = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        // Pattern that doesn't exist
        let pattern_missing = TreeNode {
            token: Token {
                text: Text {
                    text: "hate".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "hate".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let matcher_exists = DependencyMatcher::new(&[("love_exists".to_string(), pattern_exists)]);
        let matcher_missing =
            DependencyMatcher::new(&[("hate_missing".to_string(), pattern_missing)]);

        let tree: TreeNode = tokenization.try_into().unwrap();

        assert!(matcher_exists.contains(&tree));
        assert!(!matcher_missing.contains(&tree));
    }

    #[test]
    fn test_dependency_matcher_pattern_count() {
        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "run".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "run".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let pattern3 = TreeNode {
            token: Token {
                text: Text {
                    text: "walk".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "walk".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![],
        };

        let matcher = DependencyMatcher::new(&[
            ("love".to_string(), pattern1),
            ("run".to_string(), pattern2),
            ("walk".to_string(), pattern3),
        ]);
        assert_eq!(matcher.pattern_count(), 3);
    }

    #[test]
    fn test_no_match_with_intermediate_nodes() {
        // Tree: E -> B -> C -> D -> A
        // Pattern: E -> A (where A is a DIRECT child of E with Nsubj relation)
        // This should NOT match because A is not a direct child of E

        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "E".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "E".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Ccomp,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "B".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Verb,
                        lemma: Lemma {
                            lemma: "B".to_string(),
                        },
                        dep: DependencyRelation::Ccomp,
                        head: 1,
                    },
                    children: vec![(
                        DependencyRelation::Ccomp,
                        TreeNode {
                            token: Token {
                                text: Text {
                                    text: "C".to_string(),
                                },
                                whitespace: "".to_string(),
                                pos: PartOfSpeech::Verb,
                                lemma: Lemma {
                                    lemma: "C".to_string(),
                                },
                                dep: DependencyRelation::Ccomp,
                                head: 2,
                            },
                            children: vec![(
                                DependencyRelation::Ccomp,
                                TreeNode {
                                    token: Token {
                                        text: Text {
                                            text: "D".to_string(),
                                        },
                                        whitespace: "".to_string(),
                                        pos: PartOfSpeech::Verb,
                                        lemma: Lemma {
                                            lemma: "D".to_string(),
                                        },
                                        dep: DependencyRelation::Ccomp,
                                        head: 3,
                                    },
                                    children: vec![(
                                        DependencyRelation::Nsubj,
                                        TreeNode {
                                            token: Token {
                                                text: Text {
                                                    text: "A".to_string(),
                                                },
                                                whitespace: "".to_string(),
                                                pos: PartOfSpeech::Pron,
                                                lemma: Lemma {
                                                    lemma: "A".to_string(),
                                                },
                                                dep: DependencyRelation::Nsubj,
                                                head: 4,
                                            },
                                            children: vec![],
                                        },
                                    )],
                                },
                            )],
                        },
                    )],
                },
            )],
        };

        // Pattern: E with A as direct Nsubj child
        let pattern = TreeNode {
            token: Token {
                text: Text {
                    text: "E".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "E".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Nsubj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "A".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Pron,
                        lemma: Lemma {
                            lemma: "A".to_string(),
                        },
                        dep: DependencyRelation::Nsubj,
                        head: 1,
                    },
                    children: vec![],
                },
            )],
        };

        let matcher = DependencyMatcher::new(&[("E_with_A".to_string(), pattern)]);
        let matches = matcher.find_all(&tree);

        // Should NOT match because A is not a direct child of E
        assert_eq!(
            matches.len(),
            0,
            "Pattern should not match when child is not direct"
        );
    }

    #[test]
    fn test_match_with_direct_children_only() {
        // Tree: E -> A (A is direct child)
        //       E -> B (B is direct child)
        // Pattern: E -> A
        // This SHOULD match because A is a direct child of E

        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "E".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "E".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Nsubj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "A".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "A".to_string(),
                            },
                            dep: DependencyRelation::Nsubj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Obj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "B".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Noun,
                            lemma: Lemma {
                                lemma: "B".to_string(),
                            },
                            dep: DependencyRelation::Obj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
            ],
        };

        // Pattern: E with A as direct Nsubj child
        let pattern = TreeNode {
            token: Token {
                text: Text {
                    text: "E".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "E".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Nsubj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "A".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Pron,
                        lemma: Lemma {
                            lemma: "A".to_string(),
                        },
                        dep: DependencyRelation::Nsubj,
                        head: 1,
                    },
                    children: vec![],
                },
            )],
        };

        let matcher = DependencyMatcher::new(&[("E_direct_A".to_string(), pattern)]);
        let matches = matcher.find_all(&tree);

        // SHOULD match because A is a direct child of E
        assert_eq!(
            matches.len(),
            1,
            "Pattern should match when child is direct"
        );
        assert_eq!(matches[0].matched_node.token.lemma.lemma, "E");
        assert_eq!(matches[0].matched_label, "E_direct_A");
    }

    #[test]
    fn test_real_world_french_sentence_no_match() {
        // Real-world test case from user
        // Sentence: "Contrôlez-vous." (Control yourself)
        // Pattern: "logiciel rançonneur" (ransomware software)
        // These should NOT match at all

        // Tree for "Contrôlez-vous."
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "Contrôlez".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "contrôler".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Punct,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "-".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Punct,
                            lemma: Lemma {
                                lemma: "-".to_string(),
                            },
                            dep: DependencyRelation::Punct,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Obj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "vous".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "vous".to_string(),
                            },
                            dep: DependencyRelation::Obj,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Punct,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: ".".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Punct,
                            lemma: Lemma {
                                lemma: ".".to_string(),
                            },
                            dep: DependencyRelation::Punct,
                            head: 1,
                        },
                        children: vec![],
                    },
                ),
            ],
        };

        // Pattern for "logiciel rançonneur"
        let pattern = TreeNode {
            token: Token {
                text: Text {
                    text: "logiciel".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Noun,
                lemma: Lemma {
                    lemma: "logiciel".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Amod,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "rançonneur".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Adj,
                        lemma: Lemma {
                            lemma: "rançonneur".to_string(),
                        },
                        dep: DependencyRelation::Amod,
                        head: 1,
                    },
                    children: vec![],
                },
            )],
        };

        let matcher = DependencyMatcher::new(&[("logiciel_ranconneur".to_string(), pattern)]);
        let matches = matcher.find_all(&tree);

        // Should NOT match - completely different sentences
        assert_eq!(
            matches.len(),
            0,
            "French sentences with different lemmas should not match"
        );
    }

    #[test]
    fn test_tree_node_from_questce_tokenization() {
        // Test with: "qu'est-ce qu'il ne faut pas entendre"
        // This is a regression test for a bug where the wrong token was selected as root
        // Token at index 2 ("-") has head:2, which incorrectly matched idx==2 in 0-indexed
        let tokenization = Tokenization {
            tokens: vec![
                Token {
                    text: Text { text: "qu'".to_string() },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Pron,
                    lemma: Lemma { lemma: "que".to_string() },
                    dep: DependencyRelation::Obj,
                    head: 8,
                },
                Token {
                    text: Text { text: "est".to_string() },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Aux,
                    lemma: Lemma { lemma: "être".to_string() },
                    dep: DependencyRelation::Aux,
                    head: 8,
                },
                Token {
                    text: Text { text: "-".to_string() },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Punct,
                    lemma: Lemma { lemma: "-".to_string() },
                    dep: DependencyRelation::Punct,
                    head: 2,
                },
                Token {
                    text: Text { text: "ce".to_string() },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Pron,
                    lemma: Lemma { lemma: "ce".to_string() },
                    dep: DependencyRelation::Expl,
                    head: 2,
                },
                Token {
                    text: Text { text: "qu'".to_string() },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Sconj,
                    lemma: Lemma { lemma: "que".to_string() },
                    dep: DependencyRelation::Mark,
                    head: 8,
                },
                Token {
                    text: Text { text: "il".to_string() },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Pron,
                    lemma: Lemma { lemma: "il".to_string() },
                    dep: DependencyRelation::ExplImpers,
                    head: 8,
                },
                Token {
                    text: Text { text: "ne".to_string() },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Part,
                    lemma: Lemma { lemma: "ne".to_string() },
                    dep: DependencyRelation::Advmod,
                    head: 8,
                },
                Token {
                    text: Text { text: "faut".to_string() },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Aux,
                    lemma: Lemma { lemma: "falloir".to_string() },
                    dep: DependencyRelation::Root,
                    head: 0,
                },
                Token {
                    text: Text { text: "pas".to_string() },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Adv,
                    lemma: Lemma { lemma: "pas".to_string() },
                    dep: DependencyRelation::Advmod,
                    head: 8,
                },
                Token {
                    text: Text { text: "entendre".to_string() },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Verb,
                    lemma: Lemma { lemma: "entendre".to_string() },
                    dep: DependencyRelation::Xcomp,
                    head: 8,
                },
            ],
        };

        let tree: TreeNode = tokenization.try_into().unwrap();

        // Root should be "faut" (falloir), NOT the hyphen!
        assert_eq!(tree.token.text.text, "faut", "Root token should be 'faut', not '-'");
        assert_eq!(tree.token.lemma.lemma, "falloir");
        assert_eq!(tree.token.dep, DependencyRelation::Root);

        // The root should have multiple children (all tokens with head:8)
        // Children should include: qu', est, qu', il, ne, pas, entendre
        assert!(tree.children.len() >= 7, "Root should have at least 7 children, got {}", tree.children.len());

        // Verify that the hyphen is NOT the root (it should be a child of "est")
        let has_hyphen_as_root = tree.token.text.text == "-";
        assert!(!has_hyphen_as_root, "Hyphen should not be the root token");
    }

    #[test]
    fn test_french_sentence_questce_quil_ne_faut_pas_entendre() {
        // Real-world test case: "qu'est-ce qu'il ne faut pas entendre"
        // (What should one not hear / What things we shouldn't have to hear)
        // Pattern: "logiciel rançonneur" (ransomware)
        // Should NOT match - completely different sentence structure and lemmas

        // Tree for "qu'est-ce qu'il ne faut pas entendre"
        // Root is "faut" (falloir - to be necessary)
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "faut".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Aux,
                lemma: Lemma {
                    lemma: "falloir".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Obj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "qu'".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "que".to_string(),
                            },
                            dep: DependencyRelation::Obj,
                            head: 8,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Aux,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "est".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Aux,
                            lemma: Lemma {
                                lemma: "être".to_string(),
                            },
                            dep: DependencyRelation::Aux,
                            head: 8,
                        },
                        children: vec![
                            (
                                DependencyRelation::Punct,
                                TreeNode {
                                    token: Token {
                                        text: Text {
                                            text: "-".to_string(),
                                        },
                                        whitespace: "".to_string(),
                                        pos: PartOfSpeech::Punct,
                                        lemma: Lemma {
                                            lemma: "-".to_string(),
                                        },
                                        dep: DependencyRelation::Punct,
                                        head: 2,
                                    },
                                    children: vec![],
                                },
                            ),
                            (
                                DependencyRelation::Expl,
                                TreeNode {
                                    token: Token {
                                        text: Text {
                                            text: "ce".to_string(),
                                        },
                                        whitespace: " ".to_string(),
                                        pos: PartOfSpeech::Pron,
                                        lemma: Lemma {
                                            lemma: "ce".to_string(),
                                        },
                                        dep: DependencyRelation::Expl,
                                        head: 2,
                                    },
                                    children: vec![],
                                },
                            ),
                        ],
                    },
                ),
                (
                    DependencyRelation::Mark,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "qu'".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Sconj,
                            lemma: Lemma {
                                lemma: "que".to_string(),
                            },
                            dep: DependencyRelation::Mark,
                            head: 8,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::ExplImpers,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "il".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "il".to_string(),
                            },
                            dep: DependencyRelation::ExplImpers,
                            head: 8,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Advmod,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "ne".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Part,
                            lemma: Lemma {
                                lemma: "ne".to_string(),
                            },
                            dep: DependencyRelation::Advmod,
                            head: 8,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Advmod,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "pas".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Adv,
                            lemma: Lemma {
                                lemma: "pas".to_string(),
                            },
                            dep: DependencyRelation::Advmod,
                            head: 8,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Xcomp,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "entendre".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Verb,
                            lemma: Lemma {
                                lemma: "entendre".to_string(),
                            },
                            dep: DependencyRelation::Xcomp,
                            head: 8,
                        },
                        children: vec![],
                    },
                ),
            ],
        };

        // Pattern for "logiciel rançonneur" (ransomware)
        let pattern = TreeNode {
            token: Token {
                text: Text {
                    text: "logiciel".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Noun,
                lemma: Lemma {
                    lemma: "logiciel".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Amod,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "rançonneur".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Adj,
                        lemma: Lemma {
                            lemma: "rançonneur".to_string(),
                        },
                        dep: DependencyRelation::Amod,
                        head: 1,
                    },
                    children: vec![],
                },
            )],
        };

        let matcher = DependencyMatcher::new(&[("logiciel_ranconneur".to_string(), pattern)]);
        let matches = matcher.find_all(&tree);

        // Should NOT match - root is "falloir" not "logiciel"
        assert_eq!(
            matches.len(),
            0,
            "Pattern 'logiciel rançonneur' should not match 'qu'est-ce qu'il ne faut pas entendre'"
        );
    }

    #[test]
    fn test_multiple_patterns_no_false_positives() {
        // Test that when matching many patterns simultaneously, we don't get false positives
        // Tree: "I love programming" with nested structure
        let tree = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: " ".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![
                (
                    DependencyRelation::Nsubj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "I".to_string(),
                            },
                            whitespace: " ".to_string(),
                            pos: PartOfSpeech::Pron,
                            lemma: Lemma {
                                lemma: "I".to_string(),
                            },
                            dep: DependencyRelation::Nsubj,
                            head: 2,
                        },
                        children: vec![],
                    },
                ),
                (
                    DependencyRelation::Obj,
                    TreeNode {
                        token: Token {
                            text: Text {
                                text: "programming".to_string(),
                            },
                            whitespace: "".to_string(),
                            pos: PartOfSpeech::Noun,
                            lemma: Lemma {
                                lemma: "programming".to_string(),
                            },
                            dep: DependencyRelation::Obj,
                            head: 2,
                        },
                        children: vec![],
                    },
                ),
            ],
        };

        // Pattern 1: "love" with "programming" as obj (SHOULD match)
        let pattern1 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Obj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "programming".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Noun,
                        lemma: Lemma {
                            lemma: "programming".to_string(),
                        },
                        dep: DependencyRelation::Obj,
                        head: 2,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern 2: "hate" with "programming" as obj (should NOT match - wrong verb)
        let pattern2 = TreeNode {
            token: Token {
                text: Text {
                    text: "hate".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "hate".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Obj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "programming".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Noun,
                        lemma: Lemma {
                            lemma: "programming".to_string(),
                        },
                        dep: DependencyRelation::Obj,
                        head: 2,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern 3: "love" with "coding" as obj (should NOT match - wrong object)
        let pattern3 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Obj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "coding".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Noun,
                        lemma: Lemma {
                            lemma: "coding".to_string(),
                        },
                        dep: DependencyRelation::Obj,
                        head: 2,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern 4: "you" as nsubj of "love" (should NOT match - wrong subject)
        let pattern4 = TreeNode {
            token: Token {
                text: Text {
                    text: "love".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Verb,
                lemma: Lemma {
                    lemma: "love".to_string(),
                },
                dep: DependencyRelation::Root,
                head: 0,
            },
            children: vec![(
                DependencyRelation::Nsubj,
                TreeNode {
                    token: Token {
                        text: Text {
                            text: "you".to_string(),
                        },
                        whitespace: "".to_string(),
                        pos: PartOfSpeech::Pron,
                        lemma: Lemma {
                            lemma: "you".to_string(),
                        },
                        dep: DependencyRelation::Nsubj,
                        head: 2,
                    },
                    children: vec![],
                },
            )],
        };

        // Pattern 5: Just "I" (SHOULD match)
        let pattern5 = TreeNode {
            token: Token {
                text: Text {
                    text: "I".to_string(),
                },
                whitespace: "".to_string(),
                pos: PartOfSpeech::Pron,
                lemma: Lemma {
                    lemma: "I".to_string(),
                },
                dep: DependencyRelation::Nsubj,
                head: 2,
            },
            children: vec![],
        };

        // Create matcher with all 5 patterns
        let matcher = DependencyMatcher::new(&[
            ("love_programming".to_string(), pattern1.clone()),
            ("hate_programming".to_string(), pattern2),
            ("love_coding".to_string(), pattern3),
            ("you_love".to_string(), pattern4),
            ("just_I".to_string(), pattern5.clone()),
        ]);

        let matches = matcher.find_all(&tree);

        // Should only match patterns 0 (pattern1) and 4 (pattern5)
        assert_eq!(matches.len(), 2, "Should only match 2 out of 5 patterns");

        // Check that the correct patterns matched
        let pattern_indices: Vec<usize> = matches.iter().map(|m| m.pattern_index).collect();
        assert!(
            pattern_indices.contains(&0),
            "Pattern 1 (love + programming) should match"
        );
        assert!(
            pattern_indices.contains(&4),
            "Pattern 5 (just I) should match"
        );
        assert!(
            !pattern_indices.contains(&1),
            "Pattern 2 (hate) should NOT match"
        );
        assert!(
            !pattern_indices.contains(&2),
            "Pattern 3 (love + coding) should NOT match"
        );
        assert!(
            !pattern_indices.contains(&3),
            "Pattern 4 (you as subject) should NOT match"
        );

        // Verify the matched nodes and labels are correct
        for m in &matches {
            match m.pattern_index {
                0 => {
                    assert_eq!(m.matched_node.token.lemma.lemma, "love");
                    assert_eq!(m.matched_label, "love_programming");
                }
                4 => {
                    assert_eq!(m.matched_node.token.lemma.lemma, "I");
                    assert_eq!(m.matched_label, "just_I");
                }
                _ => panic!("Unexpected pattern index matched: {}", m.pattern_index),
            }
        }
    }
}

use crate::pos::PartOfSpeech;
use crate::{dep::DependencyRelation, Token, Tokenization};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::convert::TryFrom;

/// A tree node representing a token and its dependency children.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TreeNode {
    /// Index of this token in the tokenization the tree was built from.
    /// Hand-built pattern trees may leave this 0; matching never reads it on
    /// patterns, only on subject trees (to report matched token indices).
    pub index: usize,
    pub token: Token,
    pub children: Vec<(DependencyRelation, TreeNode)>,
}

/// A match found by the dependency matcher.
///
/// The type parameter `K` is the key/label type for patterns (defaults to `String`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyMatch<'a, K = String> {
    /// Index of the pattern that matched
    pub pattern_index: usize,
    /// Reference to the tree node that matched
    pub matched_node: &'a TreeNode,
    /// Matched label
    pub matched_label: K,
    /// Indices (into the tokenization the subject tree was built from) of the
    /// tokens bound by the pattern's nodes, sorted. Lets callers report which
    /// words of the sentence realized the pattern.
    pub matched_token_indices: Vec<usize>,
}

/// How a single pattern node decides whether it matches a token.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum NodeMatcher {
    /// Exact lemma match — the classic behavior for citation-form patterns.
    Lemma(String),
    /// Lemma must be one of `lemmas`; if `pos` is non-empty, the token's POS
    /// must also be in `pos`. Used for pronoun/determiner sets that can
    /// realize an argument slot (e.g. French dative clitics me/te/lui/...).
    LemmaSet {
        lemmas: BTreeSet<String>,
        pos: BTreeSet<PartOfSpeech>,
    },
    /// Wildcard: any token whose POS is in the set. Used for open argument
    /// slots ("quelqu'un"/"someone" filled by an arbitrary noun phrase).
    AnyPos(BTreeSet<PartOfSpeech>),
}

impl NodeMatcher {
    fn matches(&self, token: &Token) -> bool {
        match self {
            NodeMatcher::Lemma(lemma) => token.lemma.lemma == *lemma,
            NodeMatcher::LemmaSet { lemmas, pos } => {
                lemmas.contains(&token.lemma.lemma) && (pos.is_empty() || pos.contains(&token.pos))
            }
            NodeMatcher::AnyPos(pos) => pos.contains(&token.pos),
        }
    }

    /// The lemmas a matching token can have, or `None` if any lemma can match.
    /// Used to index patterns by their root for fast candidate lookup.
    fn anchor_lemmas(&self) -> Option<Vec<&str>> {
        match self {
            NodeMatcher::Lemma(lemma) => Some(vec![lemma.as_str()]),
            NodeMatcher::LemmaSet { lemmas, .. } => {
                Some(lemmas.iter().map(|l| l.as_str()).collect())
            }
            NodeMatcher::AnyPos(_) => None,
        }
    }
}

/// A pattern over dependency trees. Unlike [`TreeNode`] (which is a concrete
/// parse), pattern nodes can be wildcards or lemma sets, and each child edge
/// accepts a set of dependency relations (e.g. `{iobj, obj}` for a clitic
/// realization of a case-marked argument slot).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct PatternNode {
    pub matcher: NodeMatcher,
    pub children: Vec<(BTreeSet<DependencyRelation>, PatternNode)>,
}

impl From<&TreeNode> for PatternNode {
    /// A concrete parse tree used as a pattern matches by exact lemma at every
    /// node, with each child edge requiring the parse's dependency relation —
    /// the classic citation-form behavior.
    fn from(tree: &TreeNode) -> Self {
        PatternNode {
            matcher: NodeMatcher::Lemma(tree.token.lemma.lemma.clone()),
            children: tree
                .children
                .iter()
                .map(|(dep, child)| (BTreeSet::from([*dep]), PatternNode::from(child)))
                .collect(),
        }
    }
}

impl From<&PatternNode> for PatternNode {
    fn from(pattern: &PatternNode) -> Self {
        pattern.clone()
    }
}

impl PatternNode {
    /// If this pattern matches at `node`, returns the indices (into the
    /// tokenization the subject tree was built from) of the tokens bound by
    /// the pattern's nodes, sorted and deduplicated.
    fn matches(&self, node: &TreeNode) -> Option<Vec<usize>> {
        let mut indices = Vec::new();
        if self.matches_node(node, &mut indices) {
            indices.sort_unstable();
            indices.dedup();
            Some(indices)
        } else {
            None
        }
    }

    /// Whether this pattern matches at `node`, appending the bound token
    /// indices on success and leaving `indices` untouched on failure.
    ///
    /// A subject tree built by [`TreeNode::try_from`] is an owned, acyclic
    /// tree, so matching needs no cycle guard — walking it always terminates.
    fn matches_node(&self, node: &TreeNode, indices: &mut Vec<usize>) -> bool {
        if !self.matcher.matches(&node.token) {
            return false;
        }

        let checkpoint = indices.len();
        indices.push(node.index);

        let mut claimed = vec![false; node.children.len()];
        if self.assign_children(node, 0, &mut claimed, indices) {
            true
        } else {
            indices.truncate(checkpoint);
            false
        }
    }

    /// Assign this pattern's children to *distinct* children of `node`,
    /// backtracking when a choice paints a later requirement into a corner.
    ///
    /// Distinctness is what makes a pattern that requires, say, two `conj`
    /// children actually require two of them — matching both requirements
    /// against the same subject child would report a phrase that isn't there.
    fn assign_children(
        &self,
        node: &TreeNode,
        requirement: usize,
        claimed: &mut Vec<bool>,
        indices: &mut Vec<usize>,
    ) -> bool {
        let Some((deps, child_pattern)) = self.children.get(requirement) else {
            return true; // every requirement satisfied
        };

        for (i, (child_dep, child_node)) in node.children.iter().enumerate() {
            if claimed[i] || !deps.contains(child_dep) {
                continue;
            }
            let checkpoint = indices.len();
            claimed[i] = true;
            if child_pattern.matches_node(child_node, indices)
                && self.assign_children(node, requirement + 1, claimed, indices)
            {
                return true;
            }
            claimed[i] = false;
            indices.truncate(checkpoint);
        }
        false
    }
}

/// A pattern matcher that operates on dependency tree structures.
///
/// Searches for tree patterns within a tokenization's dependency tree,
/// making it useful for finding syntactic patterns based on grammatical structure.
///
/// Patterns are [`PatternNode`]s; anything convertible to one is accepted —
/// in particular a concrete [`TreeNode`] (e.g. the parse of a dictionary
/// citation form), which matches by exact lemmas.
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
/// let matcher = DependencyMatcher::new(&[("love_obj".to_string(), love_pattern)]);
/// let matches = matcher.find_all(&tree);
///
/// for match_result in matches {
///     println!("Pattern {} matched at node: {}",
///              match_result.pattern_index,
///              match_result.matched_node.token.lemma.lemma);
/// }
/// ```
/// The type parameter `K` is the key/label type for patterns (defaults to `String`).
#[derive(Debug, Clone)]
pub struct DependencyMatcher<K = String>
where
    K: Clone,
{
    patterns: Vec<(K, PatternNode)>,
    root_index: HashMap<String, Vec<usize>>,
    /// Patterns whose root has no anchor lemma (wildcard roots); checked at
    /// every node.
    unanchored: Vec<usize>,
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

        let tree = build_tree_node(&value.tokens, root_idx);

        // Validate that all tokens ended up in the tree. Tokens with out-of-bounds
        // head indices become orphaned and produce overly broad patterns that match
        // far too many sentences (e.g. a bare "être" root matching every sentence
        // containing "être").
        let node_count = count_nodes(&tree);
        if node_count != value.tokens.len() {
            return Err("Tree has orphaned tokens (likely due to invalid head indices)");
        }

        Ok(tree)
    }
}

impl<K: Clone> DependencyMatcher<K> {
    /// Creates a new dependency matcher from patterns.
    ///
    /// # Arguments
    ///
    /// * `patterns` - labeled patterns: anything convertible to a
    ///   [`PatternNode`], including concrete [`TreeNode`]s (exact-lemma
    ///   matching) and hand-built `PatternNode`s (wildcards / lemma sets).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let patterns = vec![("love".to_string(), love_pattern)];
    /// let matcher = DependencyMatcher::new(&patterns);
    /// ```
    pub fn new<P>(patterns: &[(K, P)]) -> Self
    where
        for<'a> PatternNode: From<&'a P>,
    {
        let patterns: Vec<(K, PatternNode)> = patterns
            .iter()
            .map(|(k, p)| (k.clone(), PatternNode::from(p)))
            .collect();

        // Build index: lemma -> pattern indices whose root can have this lemma
        let mut root_index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut unanchored = Vec::new();
        for (idx, (_, pattern)) in patterns.iter().enumerate() {
            match pattern.matcher.anchor_lemmas() {
                Some(lemmas) => {
                    for lemma in lemmas {
                        root_index.entry(lemma.to_string()).or_default().push(idx);
                    }
                }
                None => unanchored.push(idx),
            }
        }

        Self {
            patterns,
            root_index,
            unanchored,
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
    pub fn find_all<'t>(&self, tree: &'t TreeNode) -> Vec<DependencyMatch<'t, K>> {
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

    /// Pattern indices that could match at a node with this root lemma.
    fn candidate_indices<'s>(&'s self, lemma: &str) -> impl Iterator<Item = usize> + 's {
        self.root_index
            .get(lemma)
            .into_iter()
            .flatten()
            .copied()
            .chain(self.unanchored.iter().copied())
    }

    fn traverse_and_match<'t>(
        &self,
        node: &'t TreeNode,
        matches: &mut Vec<DependencyMatch<'t, K>>,
    ) {
        let mut visited = HashSet::new();
        self.traverse_and_match_impl(node, matches, &mut visited);
    }

    fn traverse_and_match_impl<'t>(
        &self,
        node: &'t TreeNode,
        matches: &mut Vec<DependencyMatch<'t, K>>,
        visited: &mut HashSet<*const TreeNode>,
    ) {
        // Use pointer address to detect cycles
        let node_ptr = node as *const TreeNode;
        if visited.contains(&node_ptr) {
            return; // Already visited this node, avoid infinite loop
        }
        visited.insert(node_ptr);

        // Check if this node starts any patterns
        for pattern_idx in self.candidate_indices(&node.token.lemma.lemma) {
            let (label, pattern) = &self.patterns[pattern_idx];

            // Check if this node matches the pattern (with children)
            if let Some(matched_token_indices) = pattern.matches(node) {
                matches.push(DependencyMatch {
                    pattern_index: pattern_idx,
                    matched_node: node,
                    matched_label: label.clone(),
                    matched_token_indices,
                });
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
        for pattern_idx in self.candidate_indices(&node.token.lemma.lemma) {
            let (_, pattern) = &self.patterns[pattern_idx];

            // Check if this node matches the pattern (with children)
            if pattern.matches(node).is_some() {
                return true;
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
        let labeled: Vec<(usize, PatternNode)> = patterns
            .iter()
            .enumerate()
            .map(|(idx, tree)| (idx, PatternNode::from(*tree)))
            .collect();
        let matcher = DependencyMatcher::new(&labeled);

        let mut results: HashMap<usize, Vec<&TreeNode>> = HashMap::new();
        for m in matcher.find_all(self) {
            results
                .entry(m.pattern_index)
                .or_default()
                .push(m.matched_node);
        }
        results
    }
}

fn count_nodes(node: &TreeNode) -> usize {
    1 + node
        .children
        .iter()
        .map(|(_, child)| count_nodes(child))
        .sum::<usize>()
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
            index: token_idx,
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
            children.push((child_token.dep, child_node));
        }
    }

    // Unmark this token as we backtrack (allows the token to appear in other branches)
    visited.remove(&token_idx);

    TreeNode {
        index: token_idx,
        token,
        children,
    }
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
                        index: 0,
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
                                index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
                                    index: 0,
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
                                    index: 0,
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
                                                index: 0,
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
                                                index: 0,
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
            index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
            index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
                                    index: 0,
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
                                    index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
            index: 0,
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
                    index: 0,
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
                            index: 0,
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
                                    index: 0,
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
                                            index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
                        index: 0,
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
            index: 0,
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
                    index: 0,
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
                Token {
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
                Token {
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
                Token {
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
                Token {
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
                Token {
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
                Token {
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
                Token {
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
                Token {
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
                Token {
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
            ],
        };

        let tree: TreeNode = tokenization.try_into().unwrap();

        // Root should be "faut" (falloir), NOT the hyphen!
        assert_eq!(
            tree.token.text.text, "faut",
            "Root token should be 'faut', not '-'"
        );
        assert_eq!(tree.token.lemma.lemma, "falloir");
        assert_eq!(tree.token.dep, DependencyRelation::Root);

        // The root should have multiple children (all tokens with head:8)
        // Children should include: qu', est, qu', il, ne, pas, entendre
        assert!(
            tree.children.len() >= 7,
            "Root should have at least 7 children, got {}",
            tree.children.len()
        );

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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
                                    index: 0,
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
                                    index: 0,
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
                        index: 0,
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
                        index: 0,
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
                        index: 0,
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
                        index: 0,
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
                        index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                        index: 0,
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
                        index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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
                    index: 0,
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
            index: 0,
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

    #[test]
    fn test_tree_node_rejects_orphaned_tokens() {
        // Simulates the "être sur son 31" bug: tokens 1-3 have head=5 (out of bounds),
        // so they become orphaned and the tree only contains the root "être".
        let tokenization = Tokenization {
            tokens: vec![
                Token {
                    text: Text {
                        text: "être".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Aux,
                    lemma: Lemma {
                        lemma: "être".to_string(),
                    },
                    dep: DependencyRelation::Root,
                    head: 0,
                },
                Token {
                    text: Text {
                        text: "sur".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Adp,
                    lemma: Lemma {
                        lemma: "sur".to_string(),
                    },
                    dep: DependencyRelation::Case,
                    head: 5, // Out of bounds! Only 4 tokens.
                },
                Token {
                    text: Text {
                        text: "son".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Det,
                    lemma: Lemma {
                        lemma: "son".to_string(),
                    },
                    dep: DependencyRelation::Det,
                    head: 5, // Out of bounds!
                },
                Token {
                    text: Text {
                        text: "31".to_string(),
                    },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Num,
                    lemma: Lemma {
                        lemma: "31".to_string(),
                    },
                    dep: DependencyRelation::Nummod,
                    head: 5, // Out of bounds!
                },
            ],
        };

        let result = TreeNode::try_from(tokenization);
        assert!(
            result.is_err(),
            "Should reject tokenization with orphaned tokens"
        );
    }

    #[test]
    fn test_tree_node_accepts_valid_tokenization() {
        // A valid tokenization where all tokens are reachable
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
            ],
        };

        let result = TreeNode::try_from(tokenization);
        assert!(result.is_ok(), "Should accept valid tokenization");
        let tree = result.unwrap();
        assert_eq!(tree.token.lemma.lemma, "run");
        assert_eq!(tree.children.len(), 1);
    }

    /// Build a token quickly for slot-pattern tests.
    fn tok(
        text: &str,
        lemma: &str,
        pos: PartOfSpeech,
        dep: DependencyRelation,
        head: i32,
    ) -> Token {
        Token {
            text: Text {
                text: text.to_string(),
            },
            whitespace: " ".to_string(),
            pos,
            lemma: Lemma {
                lemma: lemma.to_string(),
            },
            dep,
            head,
        }
    }

    /// "Ce qui leur est arrive" style tree: arriver(root) with an iobj PRON child.
    fn clitic_sentence_tree() -> TreeNode {
        // qui(nsubj) leur(iobj) est(aux) arriver(root)
        let tokenization = Tokenization {
            tokens: vec![
                tok(
                    "qui",
                    "qui",
                    PartOfSpeech::Pron,
                    DependencyRelation::Nsubj,
                    4,
                ),
                tok(
                    "leur",
                    "leur",
                    PartOfSpeech::Pron,
                    DependencyRelation::Iobj,
                    4,
                ),
                tok("est", "etre", PartOfSpeech::Aux, DependencyRelation::Aux, 4),
                tok(
                    "arrive",
                    "arriver",
                    PartOfSpeech::Verb,
                    DependencyRelation::Root,
                    0,
                ),
            ],
        };
        tokenization.try_into().unwrap()
    }

    /// "arrive a Jean" style tree: arriver(root) -> obl(Jean) -> case(a).
    fn filled_sentence_tree() -> TreeNode {
        let tokenization = Tokenization {
            tokens: vec![
                tok(
                    "arrive",
                    "arriver",
                    PartOfSpeech::Verb,
                    DependencyRelation::Root,
                    0,
                ),
                tok("à", "à", PartOfSpeech::Adp, DependencyRelation::Case, 3),
                tok(
                    "Jean",
                    "Jean",
                    PartOfSpeech::Propn,
                    DependencyRelation::Obl,
                    1,
                ),
            ],
        };
        tokenization.try_into().unwrap()
    }

    /// Clitic realization pattern: arriver with an {iobj, obj} child drawn
    /// from a clitic pronoun lemma set.
    fn clitic_pattern() -> PatternNode {
        PatternNode {
            matcher: NodeMatcher::Lemma("arriver".to_string()),
            children: vec![(
                BTreeSet::from([DependencyRelation::Iobj, DependencyRelation::Obj]),
                PatternNode {
                    matcher: NodeMatcher::LemmaSet {
                        lemmas: ["me", "te", "lui", "nous", "vous", "leur", "se"]
                            .into_iter()
                            .map(String::from)
                            .collect(),
                        pos: BTreeSet::from([PartOfSpeech::Pron]),
                    },
                    children: vec![],
                },
            )],
        }
    }

    /// Filled-slot realization pattern: arriver with an obl nominal wildcard
    /// that carries the case marker "a".
    fn filled_pattern() -> PatternNode {
        PatternNode {
            matcher: NodeMatcher::Lemma("arriver".to_string()),
            children: vec![(
                BTreeSet::from([DependencyRelation::Obl]),
                PatternNode {
                    matcher: NodeMatcher::AnyPos(BTreeSet::from([
                        PartOfSpeech::Noun,
                        PartOfSpeech::Propn,
                        PartOfSpeech::Pron,
                        PartOfSpeech::Num,
                    ])),
                    children: vec![(
                        BTreeSet::from([DependencyRelation::Case]),
                        PatternNode {
                            matcher: NodeMatcher::Lemma("à".to_string()),
                            children: vec![],
                        },
                    )],
                },
            )],
        }
    }

    #[test]
    fn test_slot_pattern_clitic_realization_matches() {
        let matcher = DependencyMatcher::new(&[("arriver_a_qqn".to_string(), clitic_pattern())]);
        let tree = clitic_sentence_tree();
        let matches = matcher.find_all(&tree);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_label, "arriver_a_qqn");
        assert_eq!(matches[0].matched_node.token.lemma.lemma, "arriver");
        // the pattern bound "leur" (index 1) and "arrive" (index 3)
        assert_eq!(matches[0].matched_token_indices, vec![1, 3]);
    }

    #[test]
    fn test_slot_pattern_clitic_rejects_non_clitic_pronoun() {
        let matcher = DependencyMatcher::new(&[("arriver_a_qqn".to_string(), clitic_pattern())]);

        // An iobj PRON whose lemma is outside the clitic set must not match —
        // this is what pins the LemmaSet lemma filtering itself.
        let wrong_lemma = Tokenization {
            tokens: vec![
                tok(
                    "qui",
                    "qui",
                    PartOfSpeech::Pron,
                    DependencyRelation::Nsubj,
                    4,
                ),
                tok(
                    "moi",
                    "moi",
                    PartOfSpeech::Pron,
                    DependencyRelation::Iobj,
                    4,
                ),
                tok("est", "etre", PartOfSpeech::Aux, DependencyRelation::Aux, 4),
                tok(
                    "arrive",
                    "arriver",
                    PartOfSpeech::Verb,
                    DependencyRelation::Root,
                    0,
                ),
            ],
        };
        let wrong_lemma: TreeNode = wrong_lemma.try_into().unwrap();
        assert!(matcher.find_all(&wrong_lemma).is_empty());

        // A clitic-set lemma on the wrong relation (nsubj rather than
        // iobj/obj) must not match either — this pins the dependency-set
        // filtering.
        let wrong_relation = Tokenization {
            tokens: vec![
                tok(
                    "leur",
                    "leur",
                    PartOfSpeech::Pron,
                    DependencyRelation::Nsubj,
                    3,
                ),
                tok("est", "etre", PartOfSpeech::Aux, DependencyRelation::Aux, 3),
                tok(
                    "arrive",
                    "arriver",
                    PartOfSpeech::Verb,
                    DependencyRelation::Root,
                    0,
                ),
            ],
        };
        let wrong_relation: TreeNode = wrong_relation.try_into().unwrap();
        assert!(matcher.find_all(&wrong_relation).is_empty());
    }

    #[test]
    fn test_slot_pattern_filled_wildcard_matches() {
        let matcher = DependencyMatcher::new(&[("arriver_a_qqn".to_string(), filled_pattern())]);
        let tree = filled_sentence_tree();
        assert_eq!(matcher.find_all(&tree).len(), 1);
    }

    #[test]
    fn test_slot_pattern_filled_requires_case_marker() {
        // arriver -> obl(Jean) but with case "de" instead of "a"
        let tokenization = Tokenization {
            tokens: vec![
                tok(
                    "arrive",
                    "arriver",
                    PartOfSpeech::Verb,
                    DependencyRelation::Root,
                    0,
                ),
                tok("de", "de", PartOfSpeech::Adp, DependencyRelation::Case, 3),
                tok(
                    "Jean",
                    "Jean",
                    PartOfSpeech::Propn,
                    DependencyRelation::Obl,
                    1,
                ),
            ],
        };
        let tree: TreeNode = tokenization.try_into().unwrap();
        let matcher = DependencyMatcher::new(&[("arriver_a_qqn".to_string(), filled_pattern())]);
        assert!(matcher.find_all(&tree).is_empty());
    }

    #[test]
    fn test_slot_pattern_wildcard_root_is_unanchored() {
        // A pattern with an AnyPos root should still match (checked at every node)
        let pattern = PatternNode {
            matcher: NodeMatcher::AnyPos(BTreeSet::from([PartOfSpeech::Propn])),
            children: vec![],
        };
        let matcher = DependencyMatcher::new(&[("any_propn".to_string(), pattern)]);
        let tree = filled_sentence_tree();
        let matches = matcher.find_all(&tree);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_node.token.lemma.lemma, "Jean");
    }

    #[test]
    fn test_two_requirements_need_two_distinct_children() {
        // A pattern requiring two Conj children must not be satisfied by one
        // subject child standing in for both.
        let leaf = |lemma: &str| PatternNode {
            matcher: NodeMatcher::Lemma(lemma.to_string()),
            children: vec![],
        };
        let pattern = PatternNode {
            matcher: NodeMatcher::Lemma("aimer".to_string()),
            children: vec![
                (BTreeSet::from([DependencyRelation::Conj]), leaf("pomme")),
                (BTreeSet::from([DependencyRelation::Conj]), leaf("pomme")),
            ],
        };
        let matcher = DependencyMatcher::new(&[("two_conj".to_string(), pattern)]);

        // One "pomme" conj child: must NOT match.
        let one = Tokenization {
            tokens: vec![
                tok(
                    "aime",
                    "aimer",
                    PartOfSpeech::Verb,
                    DependencyRelation::Root,
                    0,
                ),
                tok(
                    "pomme",
                    "pomme",
                    PartOfSpeech::Noun,
                    DependencyRelation::Conj,
                    1,
                ),
            ],
        };
        let one: TreeNode = one.try_into().unwrap();
        assert!(matcher.find_all(&one).is_empty());

        // Two "pomme" conj children: matches, binding both.
        let two = Tokenization {
            tokens: vec![
                tok(
                    "aime",
                    "aimer",
                    PartOfSpeech::Verb,
                    DependencyRelation::Root,
                    0,
                ),
                tok(
                    "pomme",
                    "pomme",
                    PartOfSpeech::Noun,
                    DependencyRelation::Conj,
                    1,
                ),
                tok(
                    "pomme",
                    "pomme",
                    PartOfSpeech::Noun,
                    DependencyRelation::Conj,
                    1,
                ),
            ],
        };
        let two: TreeNode = two.try_into().unwrap();
        let matches = matcher.find_all(&two);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_token_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_backtracks_when_greedy_choice_blocks_a_later_requirement() {
        // Requirement A accepts {Obj, Iobj}; requirement B accepts only Iobj.
        // Taking the Iobj child for A first would strand B, so the matcher has
        // to back out and give A the Obj child instead.
        let pattern = PatternNode {
            matcher: NodeMatcher::Lemma("donner".to_string()),
            children: vec![
                (
                    BTreeSet::from([DependencyRelation::Obj, DependencyRelation::Iobj]),
                    PatternNode {
                        matcher: NodeMatcher::AnyPos(BTreeSet::from([PartOfSpeech::Pron])),
                        children: vec![],
                    },
                ),
                (
                    BTreeSet::from([DependencyRelation::Iobj]),
                    PatternNode {
                        matcher: NodeMatcher::Lemma("lui".to_string()),
                        children: vec![],
                    },
                ),
            ],
        };
        let matcher = DependencyMatcher::new(&[("donner".to_string(), pattern)]);

        let sentence = Tokenization {
            tokens: vec![
                tok(
                    "donne",
                    "donner",
                    PartOfSpeech::Verb,
                    DependencyRelation::Root,
                    0,
                ),
                // Iobj comes first, so the wildcard requirement greedily claims
                // it and has to give it back once the "lui" requirement finds
                // nothing left. Ordering these the other way round would let a
                // matcher with no backtracking pass.
                tok(
                    "lui",
                    "lui",
                    PartOfSpeech::Pron,
                    DependencyRelation::Iobj,
                    1,
                ),
                tok("le", "le", PartOfSpeech::Pron, DependencyRelation::Obj, 1),
            ],
        };
        let tree: TreeNode = sentence.try_into().unwrap();
        let matches = matcher.find_all(&tree);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_token_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_failed_branch_leaves_no_stray_indices() {
        // The Obl requirement fails, so nothing (not even the root, nor the
        // successfully-matched Nsubj) should be reported.
        let pattern = PatternNode {
            matcher: NodeMatcher::Lemma("arriver".to_string()),
            children: vec![
                (
                    BTreeSet::from([DependencyRelation::Nsubj]),
                    PatternNode {
                        matcher: NodeMatcher::AnyPos(BTreeSet::from([PartOfSpeech::Pron])),
                        children: vec![],
                    },
                ),
                (
                    BTreeSet::from([DependencyRelation::Obl]),
                    PatternNode {
                        matcher: NodeMatcher::Lemma("Paris".to_string()),
                        children: vec![],
                    },
                ),
            ],
        };
        let matcher = DependencyMatcher::new(&[("arriver_paris".to_string(), pattern)]);
        let tree = clitic_sentence_tree();
        assert!(matcher.find_all(&tree).is_empty());
    }

    #[test]
    fn test_tree_node_pattern_still_matches_exactly() {
        // The From<&TreeNode> conversion preserves the classic exact-lemma behavior
        let tree = clitic_sentence_tree();
        let pattern_tree = clitic_sentence_tree();
        let matcher = DependencyMatcher::new(&[("exact".to_string(), pattern_tree)]);
        assert_eq!(matcher.find_all(&tree).len(), 1);
    }
}

use anyhow::Result;
use lexide::matching::{DependencyMatcher, LemmaMatcher, TextMatcher, TreeNode};
use lexide::{Language, Lexide};

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("HF_TOKEN").is_err() {
        eprintln!("Error: HF_TOKEN environment variable required");
        std::process::exit(1);
    }

    #[cfg(feature = "remote")]
    let lexide =
        Lexide::from_server("https://anchpop--lexide-gemma-3-27b-vllm-serve.modal.run")?;

    #[cfg(feature = "local")]
    let lexide = Lexide::from_pretrained(lexide::LexideConfig::default()).await?;

    #[cfg(not(any(feature = "remote", feature = "local")))]
    panic!("Either `remote` or `local` feature must be enabled!");

    println!("=== Text and Lemma Matching ===\n");

    let tokenization = lexide
        .analyze("The cats are sleeping.", Language::English)
        .await?;

    // TextMatcher - exact text
    let text_matcher = TextMatcher::new(&[
        ("cats_are".to_string(), vec!["cats", "are"]),
        ("the_cats".to_string(), vec!["The", "cats"])
    ]);
    let text_matches = text_matcher.find_all(&tokenization);
    println!("Text matches: {:?}", text_matches);

    // LemmaMatcher - matches lemmas (ignores inflection)
    let lemma_matcher = LemmaMatcher::new(&[
        ("cat_be".to_string(), vec!["cat", "be"]),
        ("cat_sleep".to_string(), vec!["cat", "sleep"])
    ]);
    let lemma_matches = lemma_matcher.find_all(&tokenization);
    println!("Lemma matches: {:?}", lemma_matches);

    // Quick existence check
    let contains_cat = LemmaMatcher::new(&[("cat".to_string(), vec!["cat"])]).contains(&tokenization);
    println!("Contains 'cat': {}", contains_cat);

    println!("\n=== Dependency Tree Matching ===\n");

    // DependencyMatcher - matches syntactic tree patterns
    // Let's create patterns from simple phrases and see if they match more complex ones

    // Pattern 1: "I love you" - basic structure
    println!("Creating pattern from: 'I love you'");
    let simple_phrase = lexide.analyze("I love you", Language::English).await?;
    let simple_tree = TreeNode::try_from(simple_phrase).unwrap();

    // Text to search: "I really love you" - has extra adverb
    println!("Searching in: 'I really love you'");
    let complex_phrase = lexide
        .analyze("I really love you", Language::English)
        .await?;
    let complex_tree = TreeNode::try_from(complex_phrase).unwrap();

    // Use the simple pattern as a template
    let matcher = DependencyMatcher::new(&[("i_love_you".to_string(), simple_tree.clone())]);
    let matches = matcher.find_all(&complex_tree);

    println!("Match result: {} pattern(s) found", matches.len());
    if !matches.is_empty() {
        println!("✓ The pattern 'I love you' matches within 'I really love you'!");
        for m in &matches {
            println!(
                "  Pattern '{}' (index {}) matched at node: '{}' (lemma: '{}')",
                m.matched_label, m.pattern_index, m.matched_node.token.text.text, m.matched_node.token.lemma.lemma
            );
        }
    } else {
        println!("✗ No match found");
    }

    // Pattern 2: Test with different structure
    println!("\nCreating pattern from: 'cats sleep'");
    let cat_pattern = lexide.analyze("cats sleep", Language::English).await?;
    let cat_tree = TreeNode::try_from(cat_pattern).unwrap();

    println!("Searching in: 'The cats are sleeping peacefully'");
    let cat_search = lexide
        .analyze("The cats are sleeping peacefully.", Language::English)
        .await?;
    let cat_search_tree = TreeNode::try_from(cat_search).unwrap();

    let cat_matcher = DependencyMatcher::new(&[("cats_sleep".to_string(), cat_tree)]);
    let cat_matches = cat_matcher.find_all(&cat_search_tree);

    println!("Match result: {} pattern(s) found", cat_matches.len());
    if !cat_matches.is_empty() {
        println!("✓ Found structural match!");
        for m in &cat_matches {
            println!(
                "  Pattern '{}' (index {}) matched at node: '{}' (lemma: '{}')",
                m.matched_label, m.pattern_index, m.matched_node.token.text.text, m.matched_node.token.lemma.lemma
            );
        }
    } else {
        println!("✗ No structural match (dependency structures may differ)");
    }

    // Pattern 3: Example that WON'T match - different structure
    println!("\n=== Example of Non-Matching Pattern ===\n");
    println!("Creating pattern from: 'I love cats'");
    let love_cats_pattern = lexide.analyze("I love cats", Language::English).await?;
    let love_cats_tree = TreeNode::try_from(love_cats_pattern).unwrap();

    println!("Searching in: 'Cats love me'");
    let cats_love_me = lexide.analyze("Cats love me", Language::English).await?;
    let cats_love_me_tree = TreeNode::try_from(cats_love_me).unwrap();

    let no_match_matcher = DependencyMatcher::new(&[("i_love_cats".to_string(), love_cats_tree)]);
    let no_matches = no_match_matcher.find_all(&cats_love_me_tree);

    println!("Match result: {} pattern(s) found", no_matches.len());
    if !no_matches.is_empty() {
        println!("✓ Found match (unexpected!)");
        for m in &no_matches {
            println!(
                "  Pattern '{}' (index {}) matched at node: '{}' (lemma: '{}')",
                m.matched_label, m.pattern_index, m.matched_node.token.text.text, m.matched_node.token.lemma.lemma
            );
        }
    } else {
        println!("✗ No match found - as expected!");
        println!("   Reason: Different grammatical structure:");
        println!("   - Pattern has 'I' as subject and 'cats' as object");
        println!("   - Search has 'cats' as subject and 'me' as object");
    }

    Ok(())
}

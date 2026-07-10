use anyhow::Result;
use lexide::{Language, Lexide};
#[cfg(feature = "local")]
use lexide::LocalConfig;

#[tokio::main]
async fn main() -> Result<()> {
    // Remote by default. Set LEXIDE_ENDPOINT_URL to point elsewhere; for the parsley CPU
    // tagger use Lexide::from_parsley_server(&url) instead (JSON response format).
    #[cfg(feature = "remote")]
    let lexide = {
        let url = std::env::var("LEXIDE_ENDPOINT_URL").unwrap_or_else(|_| {
            "https://anchpop--lexide-gemma-4-31b-vllm-serve.modal.run".to_string()
        });
        Lexide::from_server(&url)?
    };

    // Local parsley inference: ONNX artifacts come from LEXIDE_MODEL_DIR (or ./data/onnx).
    #[cfg(feature = "local")]
    let lexide = Lexide::from_pretrained(LocalConfig::default()).await?;

    #[cfg(not(any(feature = "remote", feature = "local")))]
    panic!("Either `remote` or `local` feature must be enabled!");

    let result: lexide::Tokenization = {
        use std::time::Instant;
        // Analyze a sentence
        println!("Analyzing sentence...");
        let start = Instant::now();
        let result = lexide
            .analyze("I love programming.", Language::English)
            .await?;
        let duration = start.elapsed();
        println!("Analysis complete! (took {:.2?})", duration);
        result
    };

    println!("Tokenization: {:?}", result);

    // Print results
    println!("\nFound {} tokens:", result.tokens.len());
    if result.tokens.is_empty() {
        println!(
            "\nNo tokens parsed! This might mean the model output is not in the expected format."
        );
    } else {
        for token in &result.tokens {
            println!(
                "{} [{}] -> lemma: {}, dep: {}, head: {}",
                token.text, token.pos, token.lemma, token.dep, token.head
            );
        }
    }

    Ok(())
}

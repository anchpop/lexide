use anyhow::Result;
use lexide::{Language, Lexide, LexideConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Check for HF_TOKEN
    if std::env::var("HF_TOKEN").is_err() {
        eprintln!("Error: HF_TOKEN environment variable is required for accessing Gemma models.");
        eprintln!("Please set it to your HuggingFace token:");
        eprintln!("  export HF_TOKEN=your_token_here");
        eprintln!("Get your token from: https://huggingface.co/settings/tokens");
        std::process::exit(1);
    }

    #[cfg(feature = "remote")]
    let lexide =
        Lexide::from_server("https://anchpop--lexide-gemma-3-27b-vllm-serve.modal.run")?;

    #[cfg(feature = "local")]
    let lexide = Lexide::from_pretrained(lexide::LexideConfig::default()).await?;

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

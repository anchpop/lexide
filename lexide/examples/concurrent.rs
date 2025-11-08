use anyhow::Result;
use futures::StreamExt as _;
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

    // Load the model with LoRA adapter (automatically reads HF_TOKEN)
    let lexide = Lexide::from_pretrained(LexideConfig::default()).await?;

    let sentences = vec![
        "I love programming.",
        "The cat is sleeping.",
        "The dog is running.",
        "The cat is sleeping.",
        "The parrot is flying.",
        "The mongoose is chasing the snake.",
        "The snake is hiding under the rock.",
        "The rock is rolling down the hill.",
        "The mouse is hiding under the rock.",
        "The fish is swimming in the water.",
        "The bird is flying in the sky.",
        "The eagle is flying in the sky.",
        "The hawk is flying in the sky.",
        "The owl is flying in the sky.",
        "The parrot is flying in the sky.",
        "The mongoose is chasing the snake.",
        "The snake is hiding under the rock.",
        "The rock is rolling speedily down the hill.",
        "The vehicle is driving on the road.",
        "The car is driving on the road.",
        "The truck is driving on the road.",
        "The bus is driving on the road.",
        "The train is driving on the road.",
        "The motorcycle is driving on the road.",
        "The bicycle is driving on the road.",
        "The pedestrian is walking on the road.",
        "The animal is walking on the road.",
        "The plant is growing on the road.",
        "The tree is growing on the road.",
        "The flower is growing on the road.",
        "The grass is growing on the road.",
        "The rock is growing on the road.",
        "The stone is growing on the road.",
        "The rock is growing on the road.",
    ];

    println!("Analyzing sentence...");
    use std::time::Instant;
    let start = Instant::now();

    futures::stream::iter(&sentences)
        .map(async |sentence| {
            let result: lexide::Tokenization = {
                // Analyze a sentence
                lexide
                    .analyze(sentence, Language::English)
                    .await
                    .unwrap()
            };

            println!("Tokenization: {:?}", result);
        })
        .buffer_unordered(400)
        .collect::<Vec<_>>()
        .await;

    let duration = start.elapsed();
    println!("Analysis complete! (took {:.2?}) ({:.2} sentences/s)", duration, sentences.len() as f64 / duration.as_secs_f64());

    Ok(())
}

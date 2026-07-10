use anyhow::Result;
use futures::StreamExt as _;
use lexide::{Language, Lexide};
#[cfg(feature = "local")]
use lexide::LocalConfig;

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(feature = "remote")]
    let lexide = {
        let url = std::env::var("LEXIDE_ENDPOINT_URL").unwrap_or_else(|_| {
            "https://anchpop--lexide-gemma-4-31b-vllm-serve.modal.run".to_string()
        });
        Lexide::from_server(&url)?
    };

    #[cfg(feature = "local")]
    let lexide = Lexide::from_pretrained(LocalConfig::default()).await?;

    #[cfg(not(any(feature = "remote", feature = "local")))]
    panic!("Either `remote` or `local` feature must be enabled!");

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
                lexide.analyze(sentence, Language::English).await.unwrap()
            };

            println!("Tokenization: {:?}", result);
        })
        .buffer_unordered(400)
        .collect::<Vec<_>>()
        .await;

    let duration = start.elapsed();
    println!(
        "Analysis complete! (took {:.2?}) ({:.2} sentences/s)",
        duration,
        sentences.len() as f64 / duration.as_secs_f64()
    );

    Ok(())
}

//! Segment passages into sentences, then analyze each sentence.
//!
//!     cargo run --example segment --no-default-features --features local
//!
//! This is the "segment sentences out from a list" workflow: for each passage in a list,
//! split it into its sentences (dropping the gaps between them) and hand each sentence to
//! the tagger. Segmentation runs the in-process byte-minGRU segmenter (local backend).

use anyhow::Result;
use lexide::{Language, Lexide};
#[cfg(feature = "local")]
use lexide::LocalConfig;

#[tokio::main]
async fn main() -> Result<()> {
    // Local parsley: downloads the ONNX + segmenter artifacts from HF on first use
    // (cached afterwards); set LEXIDE_MODEL_DIR to use a local directory instead.
    #[cfg(feature = "local")]
    let lexide = Lexide::from_pretrained(LocalConfig::default()).await?;

    #[cfg(not(feature = "local"))]
    {
        eprintln!("This example needs the `local` feature (segmentation is local-only).");
        return Ok(());
    }

    #[cfg(feature = "local")]
    {
        // A list of documents, each with several sentences and varied gaps/quotes.
        let passages = [
            ("First things first. Then, second—if there's time. \"Finally!\" she said.", Language::English),
            ("Guten Morgen. Wie geht es dir?\n\nMir geht es gut, danke!", Language::German),
            ("« Bonjour ! » dit-il. Puis il partit.", Language::French),
        ];

        for (passage, lang) in passages {
            println!("\n=== passage ({lang}) ===\n{passage}");
            let sentences = lexide.segment_sentences(passage)?;
            println!("-> {} sentences:", sentences.len());
            for (i, sentence) in sentences.iter().enumerate() {
                // Each sentence can now be tagged individually.
                let tok = lexide.analyze(sentence, lang).await?;
                println!("  [{i}] {sentence:?}  ({} tokens)", tok.tokens.len());
            }
        }
    }

    Ok(())
}

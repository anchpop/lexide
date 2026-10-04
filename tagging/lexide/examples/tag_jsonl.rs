//! Score shipped local parsley from stdin JSONL; stdout is predictions, stderr progress.
use anyhow::{Context, Result};
use lexide::{Language, Lexide, LocalConfig};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, BufRead, Write};
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    lang: String,
    text: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = Lexide::from_pretrained(LocalConfig {
        threads: 4,
        ..Default::default()
    })
    .await?;
    let start = Instant::now();
    let mut count = 0;
    let mut failures = 0;
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let input: Input = serde_json::from_str(&line)?;
        let language = Language::from_code(&input.lang)
            .with_context(|| format!("Unsupported language {}", input.lang))?;
        let mut tokens = Vec::new();
        match model.analyze(&input.text, language).await {
            Ok(analysis) => {
                let mut cursor = 0;
                for token in analysis.tokens() {
                    let end = cursor + token.text.text.chars().count();
                    tokens.push(json!({
                        "start": cursor, "end": end,
                        "pos": token.pos.to_string(), "lemma": token.lemma.lemma,
                        "dep": token.dep.to_string(), "head": token.head,
                    }));
                    cursor = end + token.whitespace.as_str().chars().count();
                }
                anyhow::ensure!(
                    cursor == input.text.chars().count(),
                    "Reconstruction mismatch"
                );
            }
            Err(error) => {
                // Preserve the evaluation denominator: failed inference is an empty
                // prediction, not a dropped gold row. Report every failure on stderr.
                failures += 1;
                eprintln!(
                    "row {} lang {} inference failed: {error:#}",
                    count + 1,
                    input.lang
                );
            }
        }
        serde_json::to_writer(
            &mut output,
            &json!({"lang": input.lang, "text": input.text, "tokens": tokens}),
        )?;
        writeln!(output)?;
        count += 1;
        if count % 250 == 0 {
            eprintln!(
                "{count} sentences, {failures} failures, {:.2} sentences/s",
                count as f64 / start.elapsed().as_secs_f64()
            );
        }
    }
    output.flush()?;
    eprintln!(
        "done: {count} sentences, {failures} failures, {:.2} sentences/s",
        count as f64 / start.elapsed().as_secs_f64()
    );
    Ok(())
}

//! Stream corpus word candidates through the exact preprocessing g2p build.
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::io::{BufRead, Write};

fn main() -> Result<()> {
    let identity = g2p::identity();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let mut row: Value = serde_json::from_str(&line?)?;
        let lang = g2p::Language::from_code(row["lang"].as_str().context("lang")?)
            .context("unsupported language")?;
        match g2p::phonemize(lang, row["word"].as_str().context("word")?) {
            Ok(result) => row["phonemes"] = json!(result.phonemes),
            Err(g2p::Error::Unlabelable(reason)) => row["exclude_reason"] = json!(reason),
            Err(g2p::Error::UnknownPhoneme(phone)) => {
                row["exclude_reason"] = json!(format!("unknown phone {phone:?}"))
            }
            // Candidate selection may discard an unpronounceable corpus form;
            // record the concrete backend error rather than inventing phones.
            Err(error) => row["exclude_reason"] = json!(format!("g2p_error: {error}")),
        }
        row["g2p_identity"] = json!(identity);
        serde_json::to_writer(&mut out, &row)?;
        writeln!(out)?;
    }
    Ok(())
}

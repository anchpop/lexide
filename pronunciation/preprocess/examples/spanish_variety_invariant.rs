//! Certify Spanish texts whose phone AND stress labels do not depend on variety.
use anyhow::{Context, Result};
use g2p::Language;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{BufRead, Write};

fn invariant(sentence: &str) -> Result<bool> {
    let european = g2p::phonemize(Language::SpanishEuro, sentence)?;
    let latin = g2p::phonemize(Language::SpanishLatinAmerica, sentence)?;
    Ok(european.phonemes == latin.phonemes && european.stress == latin.stress)
}

fn main() -> Result<()> {
    let identity = g2p::identity();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let (mut total, mut kept) = (0, 0);
    for line in std::io::stdin().lock().lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let sentence = row["sentence"].as_str().context("missing sentence")?;
        let same = invariant(sentence)?;
        let result = json!({
            "file": row["file"],
            "expected_sha256": format!("{:x}", Sha256::digest(sentence.as_bytes())),
            "g2p_identity": identity,
            "variety_invariant": same,
        });
        serde_json::to_writer(&mut out, &result)?;
        writeln!(out)?;
        total += 1;
        kept += usize::from(same);
    }
    eprintln!("Spanish variety-invariant labels: {kept}/{total}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_actual_variety_labels() {
        assert!(invariant("Hola, amigo").unwrap());
        assert!(!invariant("zapato").unwrap());
    }
}

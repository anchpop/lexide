//! Reads `{"id", "language", "text"}` lines on stdin and writes each with
//! g2p's `Phonemized` under `"g2p"` (or `"error"`), plus `g2p::identity()`.
use std::io::{BufRead, Write};

fn main() {
    let identity = g2p::identity();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let mut row: serde_json::Value = serde_json::from_str(&line).unwrap();
        let result = g2p::phonemize_lang(row["language"].as_str().unwrap(), row["text"].as_str().unwrap());
        match result {
            Ok(p) => row["g2p"] = serde_json::to_value(&p).unwrap(),
            Err(e) => row["error"] = e.to_string().into(),
        }
        row["g2p_identity"] = identity.clone().into();
        serde_json::to_writer(&mut out, &row).unwrap();
        out.write_all(b"\n").unwrap();
    }
}

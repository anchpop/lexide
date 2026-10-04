//! CPU load, warm throughput and resident memory over the multilingual parity corpus.
use anyhow::Result;
use lexide::{Language, LocalConfig, LocalLexide};
use serde_json::json;
use std::time::Instant;

fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}

fn main() -> Result<()> {
    let threads = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "1".into())
        .parse()?;
    let start = Instant::now();
    let model = LocalLexide::load(LocalConfig {
        threads,
        ..Default::default()
    })?;
    let load_seconds = start.elapsed().as_secs_f64();
    let records: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../tests/fixtures/parsley_reference.json"))?;
    model.analyze("The cats were sleeping.", Language::English)?;
    let start = Instant::now();
    for r in &records {
        model.analyze(
            r["text"].as_str().unwrap(),
            Language::from_code(r["lang"].as_str().unwrap()).unwrap(),
        )?;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"threads":threads,"load_seconds":load_seconds,"sentences":records.len(),
                          "seconds":seconds,"sentences_per_second":records.len() as f64/seconds,"rss_kib":rss_kib()})
    );
    Ok(())
}

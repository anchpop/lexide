//! Build compact fst lemma tables from the parsed Wiktionary JSON tables.
//!
//! Converts every `wikt_{lang}.json` (`{pos: {form: [lemmas]}}`, built by
//! `tagger/parse_wiktextract.py`) into a `wikt_{lang}.fst` the local backend loads —
//! candidate selection is resolved at build time, so files shrink ~10x and load in ms.
//!
//!     cargo run --release --features local --bin build-lemma-fst -- \
//!         --in ../data/lemma_tables --out ../data/onnx/lemma_fst

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut in_dir = PathBuf::from("../data/lemma_tables");
    let mut out_dir = PathBuf::from("../data/onnx/lemma_fst");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--in" => in_dir = args.next().context("--in needs a directory")?.into(),
            "--out" => out_dir = args.next().context("--out needs a directory")?.into(),
            other => bail!("unknown argument {other} (expected --in <dir> --out <dir>)"),
        }
    }

    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("cannot create {}", out_dir.display()))?;
    let mut built = 0;
    let mut entries: Vec<_> = std::fs::read_dir(&in_dir)
        .with_context(|| format!("cannot read {}", in_dir.display()))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(lang) = name
            .strip_prefix("wikt_")
            .and_then(|n| n.strip_suffix(".json"))
        else {
            continue;
        };
        let json: serde_json::Value =
            serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(entry.path())?))
                .with_context(|| format!("failed to parse {name}"))?;
        let out_path = out_dir.join(format!("wikt_{lang}.fst"));
        let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
        let (n, bytes) = lexide::build_table(&json, &mut out)?;
        let in_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
        println!(
            "{lang}: {n} entries, {:.1} MB json -> {:.1} MB fst",
            in_bytes as f64 / 1e6,
            bytes as f64 / 1e6
        );
        built += 1;
    }
    if built == 0 {
        bail!("no wikt_*.json files found in {}", in_dir.display());
    }
    Ok(())
}

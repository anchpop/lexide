//! Flag Pimsleur/other clips whose transcript is NOT entirely the target
//! language (LLM call only, with tysm prompt-aware caching).
//!
//! Pimsleur courses teach other languages, so many "English" clips actually
//! mix in foreign example phrases or instructions (e.g. Korean/Cantonese text +
//! "Listen and repeat."). espeak then phonemizes the foreign part as if it were
//! the target language → silently wrong labels (no error; Whisper still tags the
//! dominant language). This catches them by asking GPT-5.4-nano whether each
//! transcript is entirely <Language>.
//!
//! Output: train/lang_exclusions.jsonl, in the same schema the training loader
//! already understands (`load_asr_audit_exclusions`): one row per flagged file
//! with `expected_sha256 = sha256(sentence)`, so a clip is excluded only while
//! its (contaminated) transcript is unchanged. Add the filename to the default
//! sidecar list in train_unified.py and training auto-excludes them.
//!
//! Run with OPENAI_API_KEY set (from yap/.env):
//!   cargo run --release -- [DATA_DIR] [--langs eng deu ...] [--limit N]

use anyhow::{Context, Result};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::LazyLock;
use tokio::fs;
use tysm::chat_completions::ChatClient;

const LANGS: &[(&str, &str)] = &[
    ("eng", "English"), ("deu", "German"), ("fra", "French"), ("ita", "Italian"),
    ("por", "Portuguese"), ("rus", "Russian"), ("spa", "Spanish"), ("ara", "Arabic"),
    ("ces", "Czech"), ("dan", "Danish"), ("fas", "Persian"),
];

#[derive(Debug, Clone, Deserialize)]
struct ManifestRecord {
    file: String,
    sentence: String,
}

#[derive(Debug, Clone, Serialize)]
struct Exclusion {
    lang: String,
    file: String,
    expected_sha256: String,
    per: f64,            // 1.0 so it passes any min_per gate in the loader
    ok: bool,            // true => the loader keeps (applies) the exclusion
    reason: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct LangCheck {
    /// true if the snippet is entirely the target language; false if any part
    /// of it is not.
    is_target_language: bool,
}

static CHAT_CLIENT: LazyLock<ChatClient> = LazyLock::new(|| {
    ChatClient::from_env("gpt-5.4-nano")
        .expect("OPENAI_API_KEY not set")
        .with_cache_directory("./.cache")
});

fn system_prompt(language: &str) -> String {
    format!(
        "The target language is {language}. You will be provided a short snippet. Please \
respond `{{\"is_target_language\": true}}` if that snippet is entirely {language}. If any \
part of the snippet is not {language}, respond with `{{\"is_target_language\": false}}`."
    )
}

fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

async fn check(language: &str, sentence: &str) -> Result<LangCheck> {
    let user = format!("snippet: {sentence:?}"); // {:?} => snippet: "..."
    Ok(CHAT_CLIENT
        .chat_with_system_prompt(system_prompt(language), user)
        .await?)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let mut data_dir = PathBuf::from("../../data/audio");
    let mut only_langs: Option<Vec<String>> = None;
    let mut limit: Option<usize> = None;
    let mut out_path = PathBuf::from("../lang_exclusions.jsonl"); // cwd=train/lang-filter -> train/
    while let Some(a) = args.next() {
        match a.as_str() {
            "--langs" => {
                let mut v = vec![];
                while let Some(p) = args.peek() {
                    if p.starts_with("--") { break; }
                    v.push(args.next().unwrap());
                }
                only_langs = Some(v);
            }
            "--limit" => limit = args.next().and_then(|s| s.parse().ok()),
            "--out" => if let Some(p) = args.next() { out_path = PathBuf::from(p); },
            other => data_dir = PathBuf::from(other),
        }
    }

    let mut all_exclusions: Vec<Exclusion> = vec![];
    let mut totals: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // lang -> (checked, flagged)

    for (code, language) in LANGS {
        if let Some(ref ls) = only_langs {
            if !ls.iter().any(|l| l == code) { continue; }
        }
        let manifest = data_dir.join(code).join("manifest.jsonl");
        if !fs::try_exists(&manifest).await.unwrap_or(false) { continue; }
        let text = fs::read_to_string(&manifest).await
            .with_context(|| format!("read {}", manifest.display()))?;
        let mut records: Vec<ManifestRecord> = text.lines().filter(|l| !l.is_empty())
            .map(serde_json::from_str).collect::<Result<_, _>>()?;
        if let Some(n) = limit { records.truncate(n); }

        // Dedup by sentence -> the files sharing it (one LLM call per unique text).
        let mut by_sentence: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for r in &records {
            by_sentence.entry(r.sentence.clone()).or_default().push(r.file.clone());
        }
        let uniq: Vec<(String, Vec<String>)> = by_sentence.into_iter().collect();
        println!("{code}: {} records, {} unique sentences", records.len(), uniq.len());

        let pb = ProgressBar::new(uniq.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] {msg}").unwrap());
        pb.set_message(code.to_string());

        let results: Vec<(String, Vec<String>, Result<LangCheck>)> =
            stream::iter(uniq.into_iter())
                .map(|(sentence, files)| {
                    let pb = pb.clone();
                    async move {
                        let r = check(language, &sentence).await;
                        pb.inc(1);
                        (sentence, files, r)
                    }
                })
                .buffer_unordered(30)
                .collect()
                .await;
        pb.finish();

        let mut checked = 0;
        let mut flagged = 0;
        for (sentence, files, res) in results {
            match res {
                Ok(c) => {
                    checked += files.len();
                    if !c.is_target_language {
                        let hash = sha256_hex(&sentence);
                        for file in files {
                            flagged += 1;
                            all_exclusions.push(Exclusion {
                                lang: code.to_string(),
                                file,
                                expected_sha256: hash.clone(),
                                per: 1.0,
                                ok: true,
                                reason: "non_target_language".to_string(),
                            });
                        }
                    }
                }
                Err(e) => eprintln!("{code}: error on {sentence:?}: {e:#}"),
            }
        }
        totals.insert(code.to_string(), (checked, flagged));
    }

    let mut out = String::new();
    for e in &all_exclusions {
        out.push_str(&serde_json::to_string(e)?);
        out.push('\n');
    }
    fs::write(&out_path, out).await?;

    println!("\n=== flagged (non-target-language) per lang ===");
    for (lang, (checked, flagged)) in &totals {
        let pct = if *checked > 0 { 100.0 * *flagged as f64 / *checked as f64 } else { 0.0 };
        println!("  {lang}: {flagged}/{checked} ({pct:.1}%)");
    }
    println!("Wrote {} exclusions to {}", all_exclusions.len(), out_path.display());
    Ok(())
}

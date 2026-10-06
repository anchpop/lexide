//! Flag Pimsleur/other clips whose transcript is NOT entirely the target
//! language (LLM call only, with tysm prompt-aware caching).
//!
//! Pimsleur courses teach other languages, so many "English" clips actually
//! mix in foreign example phrases or instructions (e.g. Korean/Cantonese text +
//! "Listen and repeat."). espeak then phonemizes the foreign part as if it were
//! the target language → silently wrong labels (no error; Whisper still tags the
//! dominant language). This catches them by asking the model in llm::MODEL whether each
//! transcript is entirely <Language>.
//!
//! Output: train/lang_exclusions.jsonl, in the same schema the training loader
//! already understands (`load_asr_audit_exclusions`): one row per flagged file
//! with `expected_sha256 = sha256(sentence)`, so a clip is excluded only while
//! its (contaminated) transcript is unchanged. Add the filename to the default
//! sidecar list in train_unified.py and training auto-excludes them.
//!
//! Run with OPENAI_API_KEY set (from yap/.env):
//!   cargo run --release --manifest-path preprocess/Cargo.toml -- filter --langs eng deu

use anyhow::{Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use tokio::fs;

const LANGS: &[(&str, &str)] = &[
    ("eng", "English"),
    ("deu", "German"),
    ("fra", "French"),
    ("ita", "Italian"),
    ("por", "Portuguese"),
    ("rus", "Russian"),
    ("spa", "Spanish"),
    ("ara", "Arabic"),
    ("ces", "Czech"),
    ("dan", "Danish"),
    ("fas", "Persian"),
    ("hin", "Hindi"),
    ("jpn", "Japanese"),
    ("kor", "Korean"),
    ("tha", "Thai"),
    ("zho-hans", "Mandarin Chinese"),
];

#[derive(Debug, Clone, Deserialize)]
struct ManifestRecord {
    file: String,
    sentence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Exclusion {
    lang: String,
    file: String,
    expected_sha256: String,
    per: f64, // 1.0 so it passes any min_per gate in the loader
    ok: bool, // true => the loader keeps (applies) the exclusion
    reason: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct LangCheck {
    /// true if the snippet is entirely the target language; false if any part
    /// of it is not.
    is_target_language: bool,
}

fn system_prompt(language: &str) -> String {
    format!(
        "The target language is {language}. You will be provided a short snippet. Please \
respond `{{\"is_target_language\": true}}` if that snippet is entirely {language}. If any \
part of the snippet is not {language}, respond with `{{\"is_target_language\": false}}`."
    )
}

pub async fn run(data_dir: &Path, train_dir: &Path, langs: &[String], dry: bool) -> Result<()> {
    if !LANGS
        .iter()
        .any(|(code, _)| langs.iter().any(|l| l == code))
    {
        return Ok(());
    }
    let cache = train_dir.join("lang-filter/.cache");
    let out_path = train_dir.join("lang_exclusions.jsonl");
    // The judge is not deterministic across reruns, and a false negative
    // puts wrongly-phonemized foreign text back into training. Exclusions are
    // therefore monotonic: a prior flag stays while its sentence is unchanged.
    let prior: Vec<Exclusion> = if out_path.exists() {
        crate::corpus::read(&out_path)?
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<_, _>>()?
    } else {
        vec![]
    };
    let mut all_exclusions: Vec<Exclusion> = vec![];
    let mut totals: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // lang -> (checked, flagged)

    for (code, language) in LANGS {
        if !langs.iter().any(|l| l == code) {
            continue;
        }
        let manifest = data_dir.join(code).join("manifest.jsonl");
        if !fs::try_exists(&manifest).await.unwrap_or(false) {
            continue;
        }
        let text = fs::read_to_string(&manifest)
            .await
            .with_context(|| format!("read {}", manifest.display()))?;
        let records: Vec<ManifestRecord> = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;

        // Dedup by sentence -> the files sharing it (one LLM call per unique text).
        let mut by_sentence: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for r in &records {
            by_sentence
                .entry(r.sentence.clone())
                .or_default()
                .push(r.file.clone());
        }
        let uniq: Vec<(String, Vec<String>)> = by_sentence.into_iter().collect();
        println!(
            "{code}: {} records, {} unique sentences",
            records.len(),
            uniq.len()
        );

        let prompts: Vec<_> = uniq
            .iter()
            .map(|(sentence, _)| (system_prompt(language), format!("snippet: {sentence:?}")))
            .collect();
        let Some(verdicts) = crate::llm::run::<LangCheck>(&cache, &prompts, dry).await? else {
            continue;
        };
        let mut checked = 0;
        let mut flagged = 0;
        for ((sentence, files), verdict) in uniq.into_iter().zip(verdicts) {
            checked += files.len();
            if !verdict.is_target_language {
                let hash = crate::corpus::hash(sentence.as_bytes());
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
        let flagged_now: std::collections::HashSet<&str> = all_exclusions
            .iter()
            .filter(|e| e.lang == *code)
            .map(|e| e.file.as_str())
            .collect();
        let current: BTreeMap<&str, String> = records
            .iter()
            .map(|r| (r.file.as_str(), crate::corpus::hash(r.sentence.as_bytes())))
            .collect();
        let kept: Vec<Exclusion> = prior
            .iter()
            .filter(|e| {
                e.lang == *code
                    && !flagged_now.contains(e.file.as_str())
                    && current.get(e.file.as_str()) == Some(&e.expected_sha256)
            })
            .cloned()
            .collect();
        if !kept.is_empty() {
            println!(
                "{code}: kept {} prior exclusion(s) the judge no longer flags",
                kept.len()
            );
        }
        flagged += kept.len();
        all_exclusions.extend(kept);
        totals.insert(code.to_string(), (checked, flagged));
    }

    if dry {
        return Ok(());
    }
    let rows = all_exclusions
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()?;
    let checked_langs: Vec<String> = totals.keys().cloned().collect();
    crate::corpus::write_scoped(&out_path, rows, &checked_langs)?;

    println!("\n=== flagged (non-target-language) per lang ===");
    for (lang, (checked, flagged)) in &totals {
        let pct = if *checked > 0 {
            100.0 * *flagged as f64 / *checked as f64
        } else {
            0.0
        };
        println!("  {lang}: {flagged}/{checked} ({pct:.1}%)");
    }
    println!(
        "Wrote {} exclusions to {}",
        all_exclusions.len(),
        out_path.display()
    );
    Ok(())
}

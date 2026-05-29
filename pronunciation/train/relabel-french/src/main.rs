//! Identify rhythmic-group-final words in each French sentence (LLM call only).
//!
//! French has no lexical stress; stress falls on the last syllable of each
//! rhythmic group (groupe rythmique). espeak marks stress on every word's
//! final syllable, which is systematically wrong for French.
//!
//! This binary:
//! 1. Reads pronunciation/data/audio/fra/manifest.jsonl
//! 2. Calls GPT-5.4-nano (with tysm's prompt-aware caching) to get the
//!    rhythmic-group-final words (verbatim) for each sentence
//! 3. Writes pronunciation/data/audio/fra/stress_overrides.jsonl
//!
//! `train/scripts/preprocess.py` consumes the sidecar: when phonemizing
//! French it phonemizes the whole sentence with espeak (preserving liaison),
//! tracks word boundaries in the IPA output, then marks the last vowel of
//! each LLM-flagged word as primary stress and zeroes everything else.

use anyhow::{Context, Result};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::LazyLock;
use tokio::fs;
use tysm::chat_completions::ChatClient;

#[derive(Debug, Clone, Deserialize)]
struct ManifestRecord {
    file: String,
    sentence: String,
}

#[derive(Debug, Clone, Serialize)]
struct StressOverride {
    file: String,
    stressed_words: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct RhythmicGroupResponse {
    /// The words (verbatim, as they appear in the sentence) that end a rhythmic
    /// group and therefore carry the stress on their final syllable.
    /// Include punctuation if it's attached to the word (e.g. "plage.").
    stressed_words: Vec<String>,
}

static CHAT_CLIENT: LazyLock<ChatClient> = LazyLock::new(|| {
    ChatClient::from_env("gpt-5.4-nano")
        .expect("OPENAI_API_KEY not set")
        .with_cache_directory("./.cache")
});

const SYSTEM_PROMPT: &str = r#"You are a French prosody expert. Given a French sentence, identify which words end a rhythmic group (groupe rythmique).

In spoken French, stress falls on the final syllable of each rhythmic group. Rhythmic groups correspond to syntactic units such as:
- Noun phrases with their determiners and modifiers
- Verb phrases
- Prepositional phrases
- Short clauses

A typical rhythmic group is 3-7 syllables long.

CRITICAL RULE: ONLY content words can end a rhythmic group. Content words are:
- Nouns (chat, maison, idée, ...)
- Main verbs (manger, parler, ...) — NOT auxiliaries like "a", "est", "ont", "sont"
- Adjectives (rouge, grand, beau, ...)
- Adverbs (vite, bien, ...)

NEVER mark these as group-final (they are function words that attach to a following content word):
- Articles: le, la, les, un, une, des, du
- Prepositions: à, de, en, pour, sur, sous, avec, sans, dans, chez, par, vers, entre
- Auxiliaries (when helping a main verb): a, est, sont, ont, était, avait, sera, etc.
- Subject pronouns: je, tu, il, elle, on, nous, vous, ils, elles
- Object pronouns: me, te, se, le, la, lui, nous, vous, les, leur, y, en
- Conjunctions: et, ou, mais, donc, car, ni, si, que, qui, quand

SHORT FRAGMENTS: The input may be a fragment, idiom, dictionary entry, or single phrase rather than a complete sentence. In that case, treat the whole input as one rhythmic group and return the final content word (usually the last word, unless it's a function word that should be skipped).

Return the actual words (verbatim from the sentence, including any attached punctuation) that end a rhythmic group. Do NOT return indices.

Examples:

Input: "Le petit chat mange une pomme rouge."
Output: stressed_words = ["chat", "rouge."]

Input: "Il est parti hier soir après le dîner."
Output: stressed_words = ["parti", "soir", "dîner."]

Input: "À Berlin, la police estime qu'il y avait environ 6 500 manifestants."
Output: stressed_words = ["Berlin,", "estime", "manifestants."]

Input: "Une enquête a été ouverte."
Output: stressed_words = ["ouverte."]  (NOT "a" which is the auxiliary)

Input: "Il est mort à Osaka mardi."
Output: stressed_words = ["mort", "Osaka", "mardi."]  (NOT "à" which is a preposition)

Input: "Cependant, le conducteur a été grièvement blessé à la tête."
Output: stressed_words = ["conducteur", "blessé", "tête."]  (NOT "le", "a", or "à")

Input: "Cette petite rue mène directement à la plage."
Output: stressed_words = ["rue", "mène", "directement", "plage."]  (NOT "la" which is an article)

Input: "passe-temps"
Output: stressed_words = ["passe-temps"]

Input: "dernier soupir"
Output: stressed_words = ["soupir"]

Input: "zizanie des marais"
Output: stressed_words = ["marais"]

Input: "peine de mort"
Output: stressed_words = ["mort"]"#;

async fn get_stressed_words(sentence: &str) -> Result<Vec<String>> {
    let response: RhythmicGroupResponse = CHAT_CLIENT
        .chat_with_system_prompt(SYSTEM_PROMPT.to_string(), sentence.to_string())
        .await?;
    Ok(response.stressed_words)
}

async fn process_record(record: ManifestRecord) -> Result<StressOverride> {
    let stressed_words = get_stressed_words(&record.sentence).await?;
    Ok(StressOverride {
        file: record.file,
        stressed_words,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let data_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../../data/audio"));

    let input = data_dir.join("fra/manifest.jsonl");
    let output = data_dir.join("fra/stress_overrides.jsonl");

    let text = fs::read_to_string(&input)
        .await
        .with_context(|| format!("Failed to read {}", input.display()))?;

    let records: Vec<ManifestRecord> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;

    println!("Loaded {} French manifest records", records.len());

    let pb = ProgressBar::new(records.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] {msg}")
            .unwrap(),
    );

    let results: Vec<Result<StressOverride>> = stream::iter(records.into_iter())
        .map(|rec| {
            let pb = pb.clone();
            async move {
                let r = process_record(rec).await;
                pb.inc(1);
                r
            }
        })
        .buffer_unordered(30)
        .collect()
        .await;
    pb.finish();

    let mut output_text = String::new();
    let mut ok = 0;
    let mut errs = 0;
    for r in results {
        match r {
            Ok(rec) => {
                output_text.push_str(&serde_json::to_string(&rec)?);
                output_text.push('\n');
                ok += 1;
            }
            Err(e) => {
                eprintln!("Error: {e:#}");
                errs += 1;
            }
        }
    }

    fs::write(&output, output_text).await?;
    println!("Wrote {ok} overrides ({errs} errors) to {}", output.display());
    Ok(())
}

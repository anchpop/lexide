//! Re-label French stress using an LLM to identify rhythmic group endings.
//!
//! French has no lexical stress; stress falls on the last syllable of each
//! rhythmic group (groupe rythmique). espeak marks stress on every word's
//! final syllable, which is systematically wrong for French.
//!
//! This binary:
//! 1. Reads pronunciation/data/audio/fra/phonemes.jsonl
//! 2. Calls GPT-5.4-nano (with tysm's prompt-aware caching) to get
//!    rhythmic-group-final word indices for each sentence
//! 3. Phonemizes each word with espeak-ng to determine phoneme boundaries
//! 4. Writes updated phonemes.jsonl where only the last vowel of each
//!    group-final word carries primary stress

use anyhow::{Context, Result};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::sync::LazyLock;
use tokio::fs;
use tysm::chat_completions::ChatClient;

const ESPEAK_VOWELS: &str = "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝᵻ";
const VOWEL_CONTINUATIONS: &str = "ːˑ̠̞̯̥̃̊̈";
const WORD_BOUNDARIES: &str = " \t\n|_-";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhonemeRecord {
    file: String,
    lang: String,
    sentence: String,
    phonemes: Vec<String>,
    stress: Vec<u8>,
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

/// Strip leading/trailing punctuation for matching purposes.
fn strip_punct(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-' && c != '’')
        .to_lowercase()
}

async fn get_stressed_indices(sentence: &str) -> Result<Vec<usize>> {
    let response: RhythmicGroupResponse = CHAT_CLIENT
        .chat_with_system_prompt(SYSTEM_PROMPT.to_string(), sentence.to_string())
        .await?;

    // Match each returned word string to its index in the sentence.
    let sentence_words: Vec<&str> = sentence.split_whitespace().collect();
    let normalized: Vec<String> = sentence_words.iter().map(|w| strip_punct(w)).collect();

    let mut indices = Vec::new();
    // Track which sentence positions we've already consumed, so repeated words
    // get matched to successive occurrences (left-to-right).
    let mut used = vec![false; sentence_words.len()];
    for word in &response.stressed_words {
        let target = strip_punct(word);
        if target.is_empty() {
            continue;
        }
        if let Some(i) = normalized
            .iter()
            .enumerate()
            .find(|(i, w)| !used[*i] && **w == target)
            .map(|(i, _)| i)
        {
            indices.push(i);
            used[i] = true;
        }
    }
    Ok(indices)
}

/// Phonemize a single token with espeak-ng, using the same vowel-only stress
/// rule as the Python preprocess script. Returns (phonemes, stress_per_phoneme).
fn phonemize_word(word: &str) -> Result<(Vec<String>, Vec<u8>)> {
    let output = Command::new("espeak-ng")
        .args(["-v", "fr-fr", "-q", "--ipa", "-x", word])
        .output()
        .context("espeak-ng failed")?;
    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();

    let mut phonemes = Vec::new();
    let mut stress = Vec::new();
    let mut pending_stress: Option<u8> = None;
    let mut current_stress: u8 = 0;
    let mut in_vowel = false;

    for ch in raw.chars() {
        if ch == 'ˈ' {
            pending_stress = Some(1);
            in_vowel = false;
        } else if ch == 'ˌ' {
            pending_stress = Some(2);
            in_vowel = false;
        } else if WORD_BOUNDARIES.contains(ch) {
            pending_stress = None;
            current_stress = 0;
            in_vowel = false;
        } else if ESPEAK_VOWELS.contains(ch) {
            if let Some(s) = pending_stress.take() {
                current_stress = s;
            } else if !in_vowel {
                current_stress = 0;
            }
            in_vowel = true;
            phonemes.push(ch.to_string());
            stress.push(current_stress);
        } else if VOWEL_CONTINUATIONS.contains(ch) {
            // Combining diacritic (nasalization, length, etc) — append to the
            // previous phoneme so the combined string matches the tokenizer's
            // precomposed vocab (e.g. "ɛ̃"). Mirror of the fix in preprocess.py.
            if let Some(last) = phonemes.last_mut() {
                last.push(ch);
            } else {
                phonemes.push(ch.to_string());
                stress.push(0);
            }
        } else {
            in_vowel = false;
            current_stress = 0;
            phonemes.push(ch.to_string());
            stress.push(0);
        }
    }

    Ok((phonemes, stress))
}

/// Given a sentence, phonemize word-by-word and return phonemes + word spans.
/// word_spans[i] = (start, end) in the phonemes vec (end exclusive).
fn phonemize_sentence_with_spans(
    sentence: &str,
) -> Result<(Vec<String>, Vec<(usize, usize)>)> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let mut all_phonemes = Vec::new();
    let mut spans = Vec::new();
    for word in &words {
        let (ph, _) = phonemize_word(word)?;
        let start = all_phonemes.len();
        all_phonemes.extend(ph);
        spans.push((start, all_phonemes.len()));
    }
    Ok((all_phonemes, spans))
}

fn apply_stress(
    phonemes: &[String],
    spans: &[(usize, usize)],
    stressed_word_indices: &[usize],
) -> Vec<u8> {
    let mut stress = vec![0u8; phonemes.len()];
    for &widx in stressed_word_indices {
        if widx >= spans.len() {
            continue;
        }
        let (start, end) = spans[widx];
        // Mark the LAST vowel in this word's phoneme span as primary stress
        for i in (start..end).rev() {
            let first = phonemes[i].chars().next();
            if let Some(c) = first {
                if ESPEAK_VOWELS.contains(c) {
                    stress[i] = 1;
                    break;
                }
            }
        }
    }
    stress
}

async fn process_record(record: PhonemeRecord) -> Result<PhonemeRecord> {
    let indices = get_stressed_indices(&record.sentence).await?;
    let (phonemes, spans) = phonemize_sentence_with_spans(&record.sentence)?;
    let stress = apply_stress(&phonemes, &spans, &indices);
    Ok(PhonemeRecord {
        file: record.file,
        lang: record.lang,
        sentence: record.sentence,
        phonemes,
        stress,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let data_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../../data/audio"));

    let input = data_dir.join("fra/phonemes.jsonl");
    let output = data_dir.join("fra/phonemes.jsonl.new");

    let text = fs::read_to_string(&input)
        .await
        .with_context(|| format!("Failed to read {}", input.display()))?;

    let records: Vec<PhonemeRecord> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;

    println!("Loaded {} French records", records.len());

    let pb = ProgressBar::new(records.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] {msg}")
            .unwrap(),
    );

    let results: Vec<Result<PhonemeRecord>> = stream::iter(records.into_iter())
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
    println!("Wrote {ok} records ({errs} errors) to {}", output.display());
    println!("To replace: mv {} {}", output.display(), input.display());
    Ok(())
}

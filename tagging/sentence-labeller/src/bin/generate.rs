use anyhow::{Context, Result, bail};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf, sync::Arc};
use tokio::{fs, sync::Mutex};
use tysm::chat_completions::ChatClient;

const LANGUAGES: &[(&str, &str)] = &[
    ("deu", "German"),
    ("eng", "English"),
    ("fra", "French"),
    ("hin", "Hindi"),
    ("ita", "Italian"),
    ("jpn", "Japanese"),
    ("kor", "Korean"),
    ("por", "Portuguese"),
    ("rus", "Russian"),
    ("spa", "Spanish"),
];
const PROMPTS: &[&str] = &[
    "A traveler arrives at a remote railway station during unusual weather.",
    "Neighbors organize a community event that does not go entirely as planned.",
    "A student discovers an unexpected object in a library book.",
    "Describe a busy market from opening time until noon.",
    "A family prepares a complicated meal for an important visitor.",
    "An engineer investigates why a small town keeps losing electricity.",
    "Explain how a public transportation system responds to a major delay.",
    "A doctor and patient discuss a puzzling but non-emergency symptom.",
    "A museum curator prepares a fragile artifact for an exhibition.",
    "Researchers debate the results of an environmental field study.",
    "A software team diagnoses a failure shortly before a product launch.",
    "Explain a scientific instrument and what can make its measurements unreliable.",
    "A journalist interviews several people about a controversial local decision.",
    "Two old friends meet by chance after many years apart.",
    "A courtroom witness gives an account that is repeatedly interrupted.",
    "Write a lively scene in a café where several conversations overlap.",
    "A teacher leads a class discussion in which students strongly disagree.",
    "A radio host takes calls about a strange event seen across the city.",
    "A theater rehearsal goes wrong, with stage directions, corrections, and arguments.",
    "A parent tells a child a story while the child keeps asking questions.",
];

#[derive(Debug, Clone, Copy)]
enum Style {
    Simple,
    Technical,
    Dialogue,
}
impl Style {
    const ALL: [Self; 3] = [Self::Simple, Self::Technical, Self::Dialogue];
    fn id(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Technical => "technical",
            Self::Dialogue => "dialogue",
        }
    }
    fn instruction(self) -> &'static str {
        match self {
            Self::Simple => {
                "Use clear, everyday prose with mostly straightforward sentence structures."
            }
            Self::Technical => {
                "Use a technical or expert register, with realistic terminology, abbreviations, numbers, parentheticals, lists, or headings where natural."
            }
            Self::Dialogue => {
                "Make the passage dialogue-heavy, with many quotations, interruptions, dialogue markers, and varied punctuation. Include narration too."
            }
        }
    }
}
#[derive(Debug, Deserialize, JsonSchema)]
struct GeneratedPassage {
    content: String,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
struct OutputRecord {
    id: String,
    lang: String,
    book: u8,
    sample: u8,
    source: String,
    text: String,
    prompt: String,
    style: String,
}
fn system_prompt(language: &str, style: Style) -> String {
    format!(
        "Write an original passage entirely in {language}. {}\n\nAim for roughly 700–1,500 words, but end naturally. The passage should contain multiple paragraphs and varied sentence lengths. Preserve natural punctuation and typography for {language}. Do not discuss these instructions, identify the prompt, translate it, or wrap the passage in Markdown code fences. Return only the passage in the structured `content` field.",
        style.instruction()
    )
}
async fn generate_one(
    client: &ChatClient,
    lang: &str,
    language: &str,
    prompt_index: usize,
    prompt: &str,
    style: Style,
) -> Result<OutputRecord> {
    let response: GeneratedPassage = client
        .chat_with_system_prompt(system_prompt(language, style), prompt.to_owned())
        .await
        .with_context(|| format!("generate {lang}-{}-{}", prompt_index + 1, style.id()))?;
    if response.content.trim().is_empty() {
        bail!("generated an empty passage")
    }
    let style_index = Style::ALL
        .iter()
        .position(|candidate| candidate.id() == style.id())
        .unwrap();
    Ok(OutputRecord {
        id: format!("synthetic-{lang}-p{:02}-{}", prompt_index + 1, style.id()),
        lang: lang.to_owned(),
        book: 0,
        sample: (prompt_index * 3 + style_index + 1) as u8,
        source: "gpt-5.6-sol".to_owned(),
        text: response.content,
        prompt: prompt.to_owned(),
        style: style.id().to_owned(),
    })
}
#[tokio::main]
async fn main() -> Result<()> {
    let output = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/synthetic-passages.jsonl".into()),
    );
    let concurrency: usize = std::env::var("CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let existing = fs::read_to_string(&output).await.unwrap_or_default();
    let completed: HashSet<String> = existing
        .lines()
        .filter_map(|line| serde_json::from_str::<OutputRecord>(line).ok())
        .map(|r| r.id)
        .collect();
    let mut jobs = Vec::new();
    for &(lang, language) in LANGUAGES {
        for (prompt_index, &prompt) in PROMPTS.iter().enumerate() {
            for style in Style::ALL {
                let id = format!("synthetic-{lang}-p{:02}-{}", prompt_index + 1, style.id());
                if !completed.contains(&id) {
                    jobs.push((lang, language, prompt_index, prompt, style));
                }
            }
        }
    }
    println!(
        "{} already generated; {} pending",
        completed.len(),
        jobs.len()
    );
    if jobs.is_empty() {
        return Ok(());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::create_dir_all("./.cache").await?;
    let file = Arc::new(Mutex::new(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output)
            .await?,
    ));
    let client = Arc::new(
        ChatClient::from_env("gpt-5.6-sol")
            .context("create tysm client (is OPENAI_API_KEY set?)")?
            .with_cache_directory("./.cache"),
    );
    let progress = ProgressBar::new(jobs.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] {msg}")?,
    );
    let failures = Arc::new(Mutex::new(Vec::new()));
    stream::iter(jobs)
        .for_each_concurrent(
            concurrency,
            |(lang, language, prompt_index, prompt, style)| {
                let client = client.clone();
                let file = file.clone();
                let failures = failures.clone();
                let progress = progress.clone();
                async move {
                    let id = format!("synthetic-{lang}-p{:02}-{}", prompt_index + 1, style.id());
                    match generate_one(&client, lang, language, prompt_index, prompt, style).await {
                        Ok(record) => {
                            let mut line = serde_json::to_vec(&record).expect("serialize passage");
                            line.push(b'\n');
                            use tokio::io::AsyncWriteExt;
                            if let Err(error) = file.lock().await.write_all(&line).await {
                                failures
                                    .lock()
                                    .await
                                    .push(format!("{id}: write: {error:#}"));
                            }
                        }
                        Err(error) => failures.lock().await.push(format!("{id}: {error:#}")),
                    }
                    progress.inc(1);
                }
            },
        )
        .await;
    progress.finish();
    let failures = failures.lock().await;
    if !failures.is_empty() {
        for failure in failures.iter() {
            eprintln!("{failure}");
        }
        bail!("{} generations failed", failures.len())
    }
    println!("All passages written to {}", output.display());
    Ok(())
}

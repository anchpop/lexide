use anyhow::{Context, Result, bail};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf, sync::Arc};
use tokio::{fs, sync::Mutex};
use tysm::chat_completions::ChatClient;

const SYSTEM_PROMPT: &str = r#"Split the provided passage into sentences and gaps.

Return the passage as an ordered list of sections:
- `sentence`: exactly one complete sentence, including all punctuation that frames it: opening and closing quotation marks, and dialogue dashes or other dialogue markers.
- `gap`: text that belongs to no sentence, especially whitespace between sentences, headings, and document separators. Do not put quotation marks or dialogue markers in a gap when they frame a sentence.

Preserve every character exactly. Do not correct, normalize, translate, add, or remove anything. Never return an empty section. Copy content directly from the input, paying special attention to spaces around quotation marks and dashes. Concatenating every section's `content` in order MUST reproduce the input exactly. Spaces within a sentence belong to that sentence; spaces and newlines between sentences are gaps. Quotations are ordinary sentence content and require no special treatment. Before responding, verify the concatenation character-for-character.

Example input:
He said, "Hi!"  Then he left.

Example output:
{"sections":[{"type":"sentence","content":"He said, \"Hi!\""},{"type":"gap","content":"  "},{"type":"sentence","content":"Then he left."}]}"#;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct InputRecord {
    id: String,
    lang: String,
    book: u8,
    sample: u8,
    source: String,
    text: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum SectionType {
    Sentence,
    Gap,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
struct Section {
    #[serde(rename = "type")]
    section_type: SectionType,
    content: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct Annotation {
    sections: Vec<Section>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct OutputRecord {
    id: String,
    lang: String,
    book: u8,
    sample: u8,
    source: String,
    text: String,
    sections: Vec<Section>,
}

fn validate(text: &str, sections: &[Section]) -> Result<()> {
    if sections.is_empty() {
        bail!("response has no sections")
    }
    if sections.iter().any(|s| s.content.is_empty()) {
        bail!("response has an empty section")
    }
    let reconstructed: String = sections.iter().map(|s| s.content.as_str()).collect();
    if reconstructed != text {
        bail!(
            "sections do not reconstruct input (input {} bytes, output {} bytes)",
            text.len(),
            reconstructed.len()
        )
    }
    Ok(())
}

async fn annotate(client: &ChatClient, record: InputRecord) -> Result<OutputRecord> {
    let mut response: Annotation = client
        .chat_with_system_prompt(SYSTEM_PROMPT.to_owned(), record.text.clone())
        .await
        .with_context(|| format!("label {}", record.id))?;
    response
        .sections
        .retain(|section| !section.content.is_empty());
    validate(&record.text, &response.sections)
        .with_context(|| format!("validate {}", record.id))?;
    Ok(OutputRecord {
        id: record.id,
        lang: record.lang,
        book: record.book,
        sample: record.sample,
        source: record.source,
        text: record.text,
        sections: response.sections,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().unwrap_or_else(|| "data/samples.jsonl".into()));
    let output = PathBuf::from(args.next().unwrap_or_else(|| "data/labelled.jsonl".into()));
    let concurrency: usize = std::env::var("CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let input_text = fs::read_to_string(&input)
        .await
        .with_context(|| format!("read {}", input.display()))?;
    let records: Vec<InputRecord> = input_text
        .lines()
        .filter(|l| !l.is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;

    let existing = fs::read_to_string(&output).await.unwrap_or_default();
    let completed: HashSet<String> = existing
        .lines()
        .filter_map(|l| serde_json::from_str::<OutputRecord>(l).ok())
        .map(|r| r.id)
        .collect();
    let pending: Vec<_> = records
        .into_iter()
        .filter(|r| !completed.contains(&r.id))
        .collect();
    println!(
        "{} already labelled; {} pending",
        completed.len(),
        pending.len()
    );
    if pending.is_empty() {
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
        ChatClient::from_env("gpt-5.6-luna")
            .context("create tysm client (is OPENAI_API_KEY set?)")?
            .with_cache_directory("./.cache"),
    );
    let pb = ProgressBar::new(pending.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] {msg}")?,
    );

    let failures = Arc::new(Mutex::new(Vec::new()));
    stream::iter(pending)
        .for_each_concurrent(concurrency, |record| {
            let client = client.clone();
            let file = file.clone();
            let pb = pb.clone();
            let failures = failures.clone();
            async move {
                let id = record.id.clone();
                match annotate(&client, record).await {
                    Ok(labelled) => {
                        let mut line = serde_json::to_vec(&labelled).expect("serialize annotation");
                        line.push(b'\n');
                        use tokio::io::AsyncWriteExt;
                        if let Err(e) = file.lock().await.write_all(&line).await {
                            failures.lock().await.push(format!("{id}: write: {e:#}"));
                        }
                    }
                    Err(e) => failures.lock().await.push(format!("{id}: {e:#}")),
                }
                pb.inc(1);
            }
        })
        .await;
    pb.finish();

    let failures = failures.lock().await;
    if !failures.is_empty() {
        eprintln!(
            "{} failures (rerun to retry; successful records are retained):",
            failures.len()
        );
        for failure in failures.iter() {
            eprintln!("  {failure}");
        }
        bail!("labeling incomplete")
    }
    println!(
        "All annotations validated and written to {}",
        output.display()
    );
    Ok(())
}

use anyhow::{Context, Result, bail};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf, sync::Arc};
use tokio::{fs, sync::Mutex};
use tysm::chat_completions::ChatClient;

// The prompt is assembled per-record: HEAD + (ELLIPSIS_BULLET if the text has one) + TAIL.
// The bullet is conditional so that records without an ellipsis keep the exact prompt
// string of previous runs and stay hits in the tysm response cache.
const PROMPT_HEAD: &str = r#"Split the provided passage into sentences and gaps.

Return the passage as an ordered list of sections:
- `sentence`: exactly one complete sentence, including all punctuation that frames it: opening and closing quotation marks, and dialogue dashes or other dialogue markers.
- `gap`: text that belongs to no sentence, especially whitespace between sentences, headings, and document separators. Do not put quotation marks or dialogue markers in a gap when they frame a sentence.

Quotation policy — apply it the same way every time:
- A quote containing several sentences is split at its internal terminators; the opening mark goes with the first sentence and the closing mark with the last: `「おったまげた。` then `どうやって助かった？」`
- Latin-script dialogue: a quote plus its attribution clause is ONE sentence: `"Is this the place?" she asked.` Dash dialogue too: `—Ya voy —dijo Valdés—.` is one sentence; a multi-sentence dash turn splits at internal terminators, the dash staying with the first: `—No.` then `También contiene sales.`
- Japanese and Korean: the quote and what follows form ONE sentence only when a quotative binder attaches them (と, って, 라고, 하고): `「一人分しかないね」とハリーが言った。` With no binder the quote is its own sentence, even when what follows is an attribution: `「ピーブズ」` then `ハリーは声を殺した。` — and `“조용히 해.”` then `해리가 말했다.` Do not judge by the verb; check only for the binder.
- In scripts, a speaker label frames the sentence it introduces (`TOMÁS.— Atención, cabina.` is one sentence; a multi-sentence speech still splits after the first terminator). A bracketed stage direction is its own sentence, brackets included."#;

const ELLIPSIS_BULLET: &str = r#"
- Ellipsis: `…`/`...` followed by a new clause starting with a capital (or a new dialogue turn) is a sentence boundary: `—Ya voy. Espere…` then `Hay un sonido.` Followed by a lowercase continuation of the same clause it is mid-sentence: `esperó… y nada pasó.` In Japanese and Korean an ellipsis is mid-sentence unless a sentence terminator or a new turn follows it."#;

const PRE_QUOTE_BULLET: &str = r#"
- A clause BEFORE a quote binds to it only when it leads in with a comma or colon (`He said, "Hi!"` / `sagte sie: „Ja.“`). A preceding clause that ends with its own terminator is a separate sentence: `Ben erstarrte.` then `„Ich möchte festhalten, dass ich recht hatte.“`"#;

const DASH_TURN_BULLET: &str = r#"
- Dash turns in every language split at internal terminators: `— Любоваться будем позже.` then `Время отключения?` Speech resumed after an inline attribution starts a new sentence at the resuming dash: `— Да, — ответил Денис.` then `— Чайники, плиты.`"#;

const HINDI_BULLET: &str = r#"
- Hindi: like Japanese and Korean, a quote and what follows are ONE sentence only when bound (e.g. by कि or a comma); a bare attribution after the closing quote is a separate sentence: `“क्या आप कुछ ढूँढ़ रहे हैं?”` then `बूढ़े ने पूछा।`"#;

const PROMPT_TAIL: &str = r#"

Preserve every character exactly. Do not correct, normalize, translate, add, or remove anything. Never return an empty section. Copy content directly from the input, paying special attention to spaces around quotation marks and dashes. Concatenating every section's `content` in order MUST reproduce the input exactly. Spaces within a sentence belong to that sentence; spaces and newlines between sentences are gaps. Before responding, verify the concatenation character-for-character.

Example input:
He said, "Hi! Come in."  Then he left.

Example output:
{"sections":[{"type":"sentence","content":"He said, \"Hi!"},{"type":"gap","content":" "},{"type":"sentence","content":"Come in.\""},{"type":"gap","content":"  "},{"type":"sentence","content":"Then he left."}]}"#;

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
    // Conditional bullets: each is appended only when the passage can trigger it, so
    // records outside its scope keep a byte-identical prompt (and their cache entry).
    let mut bullets = String::new();
    if record.text.chars().any(|c| "\"“”„«»「『“".contains(c)) {
        bullets.push_str(PRE_QUOTE_BULLET);
    }
    if record.text.contains('—') || record.text.contains('–') {
        bullets.push_str(DASH_TURN_BULLET);
    }
    if record.text.contains('…') || record.text.contains("...") {
        bullets.push_str(ELLIPSIS_BULLET);
    }
    if record.lang == "hin" {
        bullets.push_str(HINDI_BULLET);
    }
    let mut response: Annotation = client
        .chat_with_system_prompt(format!("{PROMPT_HEAD}{bullets}{PROMPT_TAIL}"), record.text.clone())
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

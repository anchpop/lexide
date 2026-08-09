//! Label sentences with the Gemma teacher's tokenization — silver-data generation for
//! the byte char tokenizer (see `tagger/augment_tokenization_sentences.py`).
//!
//!     cargo run --release --no-default-features --features remote \
//!         --bin label-tokenization -- <lang> <sentences.txt> <out.jsonl> [concurrency]
//!
//! Output schema matches yap's generate-data / `data/big/<lang>/…tokenization.jsonl`:
//! one `{"sentence": ..., "tokens": [lexide::Token]}` per line. Incremental: sentences
//! already in the output file are skipped, so reruns only fill gaps/failures.

use std::collections::BTreeSet;
use std::io::{BufRead, Write};

use anyhow::{bail, Context, Result};
use futures::StreamExt;
use lexide::{Language, Lexide};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct TokenizedSentence {
    sentence: String,
    tokens: Vec<lexide::Token>,
}

fn language_from_code(code: &str) -> Result<Language> {
    let lang = match code {
        "deu" => Language::German,
        "eng" => Language::English,
        "fra" => Language::French,
        "hin" => Language::Hindi,
        "ita" => Language::Italian,
        "jpn" => Language::Japanese,
        "kor" => Language::Korean,
        "por" => Language::Portuguese,
        "rus" => Language::Russian,
        "spa" => Language::Spanish,
        "tha" => Language::Thai,
        "zho-hans" => Language::ChineseSimplified,
        other => bail!("unknown language code {other}"),
    };
    Ok(lang)
}

const GEMMA_URL: &str = "https://anchpop--lexide-gemma-4-31b-vllm-serve.modal.run";

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [lang_code, input, output, rest @ ..] = args.as_slice() else {
        bail!("usage: label-tokenization <lang> <sentences.txt> <out.jsonl> [concurrency]");
    };
    let concurrency: usize = rest.first().map(|s| s.parse()).transpose()?.unwrap_or(300);
    let language = language_from_code(lang_code)?;

    let sentences: Vec<String> = std::fs::read_to_string(input)
        .with_context(|| format!("read {input}"))?
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();

    // Incremental: skip sentences already labelled in the output file.
    let mut done = BTreeSet::new();
    if std::path::Path::new(output).exists() {
        let file = std::fs::File::open(output)?;
        for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
            if let Ok(rec) = serde_json::from_str::<TokenizedSentence>(&line) {
                done.insert(rec.sentence);
            }
        }
    }
    let todo: Vec<String> = sentences
        .into_iter()
        .filter(|s| !done.contains(s))
        .collect();
    eprintln!(
        "[{lang_code}] {} to label ({} already done), concurrency {concurrency}",
        todo.len(),
        done.len()
    );
    if todo.is_empty() {
        return Ok(());
    }

    let lexide = Lexide::from_server(GEMMA_URL)?;
    let out_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output)?;
    let mut writer = std::io::BufWriter::new(out_file);

    let total = todo.len();
    let mut ok = 0usize;
    let mut failed = 0usize;
    let mut results = futures::stream::iter(todo)
        .map(|sentence| {
            let lexide = &lexide;
            async move {
                let res = lexide.analyze(&sentence, language).await;
                (sentence, res)
            }
        })
        .buffer_unordered(concurrency);

    while let Some((sentence, res)) = results.next().await {
        match res {
            Ok(tokenization) => {
                let rec = TokenizedSentence {
                    sentence,
                    tokens: tokenization.tokens,
                };
                writeln!(writer, "{}", serde_json::to_string(&rec)?)?;
                ok += 1;
                if ok % 500 == 0 {
                    writer.flush()?;
                    eprintln!("[{lang_code}] {ok}/{total} labelled ({failed} failed)");
                }
            }
            Err(e) => {
                failed += 1;
                if failed <= 5 {
                    eprintln!("[{lang_code}] failed {sentence:?}: {e:#}");
                }
            }
        }
    }
    writer.flush()?;
    eprintln!("[{lang_code}] done: {ok} labelled, {failed} failed (rerun to retry failures)");
    Ok(())
}

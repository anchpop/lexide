//! Cache paid ASR responses by audio/request; recompute local scores on each run.
use crate::corpus;
use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use futures::{StreamExt, stream};
use reqwest::{Client, multipart};
use serde_json::{Value, json};
use std::{collections::HashMap, fs, path::Path, time::Duration};
use unicode_casefold::UnicodeCaseFold;
use unicode_general_category::get_general_category;
use unicode_normalization::UnicodeNormalization;

#[derive(Args)]
pub struct Options {
    /// Manifest sources to audit (defaults match the former upload script).
    #[arg(long, num_args = 1.., default_values = ["fleurs", "tatoeba"], value_parser = ["fleurs", "tatoeba", "tts", "kathbath"])]
    pub sources: Vec<String>,
    #[arg(long, default_value = "whisper-large-v3-turbo")]
    pub asr_model: String,
    #[arg(long, default_value = "8", value_parser = clap::value_parser!(u16).range(1..))]
    pub asr_workers: u16,
    #[arg(long, default_value_t = 4)]
    pub retries: u8,
    #[arg(long, default_value_t = 120)]
    pub timeout: u64,
    #[arg(long)]
    pub detect_language: bool,
    #[arg(long)]
    pub text_only: bool,
    /// Re-score saved audit transcripts, including older files; never calls Groq.
    #[arg(long)]
    pub rescore: bool,
    #[arg(
        long,
        hide = true,
        default_value = "https://api.groq.com/openai/v1/audio/transcriptions"
    )]
    pub asr_url: String,
}

fn normalize(s: &str) -> String {
    s.nfkc()
        .case_fold()
        .map(|c| {
            if c.is_ascii_punctuation() || get_general_category(c).abbreviation().starts_with('P') {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn distance<T: PartialEq>(a: &[T], b: &[T]) -> f64 {
    if a.is_empty() {
        return if b.is_empty() { 0.0 } else { 1.0 };
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut current = vec![i + 1];
        for (j, y) in b.iter().enumerate() {
            current.push(
                (previous[j + 1] + 1)
                    .min(current[j] + 1)
                    .min(previous[j] + usize::from(x != y)),
            );
        }
        previous = current;
    }
    previous[b.len()] as f64 / a.len() as f64
}

pub fn score(mut row: Value, text_only: bool) -> Result<Value> {
    let expected = corpus::text(&row, "expected")?.to_owned();
    let actual = corpus::text(&row, "whisper_text")?.to_owned();
    let e = normalize(&expected);
    let a = normalize(&actual);
    row["cer"] = json!(distance(
        &e.chars().filter(|c| *c != ' ').collect::<Vec<_>>(),
        &a.chars().filter(|c| *c != ' ').collect::<Vec<_>>()
    ));
    row["wer"] = json!(distance(
        &e.split_whitespace().collect::<Vec<_>>(),
        &a.split_whitespace().collect::<Vec<_>>()
    ));
    row["text_metric_version"] = json!(2);
    row["expected_sha256"] = json!(corpus::hash(expected.as_bytes()));
    for key in [
        "per",
        "expected_phonemes",
        "actual_phonemes",
        "g2p_reference_exclude_reason",
    ] {
        row.as_object_mut().unwrap().remove(key);
    }
    if !text_only {
        let language = corpus::language(&row, corpus::text(&row, "lang")?)?;
        row["g2p_language"] = serde_json::to_value(language)?;
        row["g2p_identity"] = json!(g2p::identity());
        match g2p::phonemize(language, &expected) {
            Ok(expected) => {
                let actual = match g2p::phonemize(language, &actual) {
                    Ok(p) => p.phonemes,
                    Err(g2p::Error::Unlabelable(_)) => vec![],
                    Err(error) => return Err(error.into()),
                };
                row["per"] = json!(distance(&expected.phonemes, &actual));
                row["expected_phonemes"] = json!(expected.phonemes);
                row["actual_phonemes"] = json!(actual);
            }
            Err(g2p::Error::Unlabelable(reason)) => {
                row["g2p_reference_exclude_reason"] = json!(reason)
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(row)
}

fn iso(code: &str) -> Option<&'static str> {
    Some(match code {
        "deu" => "de",
        "eng" => "en",
        "fra" => "fr",
        "ita" => "it",
        "por" => "pt",
        "rus" => "ru",
        "spa" => "es",
        "tha" => "th",
        "zho-hans" => "zh",
        "hin" => "hi",
        "jpn" => "ja",
        "kor" => "ko",
        "ara" => "ar",
        "ces" => "cs",
        "dan" => "da",
        _ => return None,
    })
}

async fn transcript(
    client: &Client,
    options: &Options,
    cache: &Path,
    path: &Path,
    lang: &str,
) -> Result<Value> {
    let bytes = tokio::fs::read(path).await?;
    let language = if options.detect_language {
        None
    } else {
        iso(lang)
    };
    let key = corpus::hash(&serde_json::to_vec(&json!({
        "audio": corpus::hash(&bytes), "model": options.asr_model, "language": language,
        "temperature": 0, "format": "verbose_json", "url": options.asr_url,
    }))?);
    let cached = cache.join(format!("{key}.json"));
    if cached.exists() {
        return Ok(serde_json::from_slice(&fs::read(cached)?)?);
    }
    let key = std::env::var("GROQ_API_KEY").context(
        "GROQ_API_KEY is required for uncached audio; use audit --rescore for saved transcripts",
    )?;
    let mut last_error = String::new();
    for attempt in 0..=options.retries {
        let part = multipart::Part::bytes(bytes.clone())
            .file_name(path.file_name().unwrap().to_string_lossy().into_owned())
            .mime_str("audio/wav")?;
        let mut form = multipart::Form::new()
            .text("model", options.asr_model.clone())
            .text("temperature", "0")
            .text("response_format", "verbose_json")
            .part("file", part);
        if let Some(language) = language {
            form = form.text("language", language);
        }
        let mut delay = Duration::from_secs((1u64 << attempt.min(5)).min(30));
        match client
            .post(&options.asr_url)
            .bearer_auth(&key)
            .multipart(form)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let payload: Value = response.json().await.context("decoding Groq transcript")?;
                corpus::text(&payload, "text")?;
                fs::create_dir_all(cache)?;
                let mut temp = tempfile::NamedTempFile::new_in(cache)?;
                serde_json::to_writer(&mut temp, &payload)?;
                temp.persist(&cached)?;
                return Ok(payload);
            }
            Ok(response) => {
                let status = response.status();
                if let Some(seconds) = response
                    .headers()
                    .get("retry-after")
                    .and_then(|s| s.to_str().ok())
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|s| s.is_finite() && *s >= 0.0)
                {
                    delay = Duration::from_secs_f64(seconds.min(3600.0));
                }
                last_error = format!("Groq HTTP {status}");
                if status.as_u16() != 429 && !status.is_server_error() {
                    bail!("{last_error}");
                }
            }
            Err(error) => last_error = error.to_string(),
        }
        if attempt < options.retries {
            tokio::time::sleep(delay).await;
        }
    }
    bail!("{last_error}")
}

fn with_payload(mut row: Value, payload: Value) -> Result<Value> {
    row["whisper_text"] = json!(corpus::text(&payload, "text")?);
    row["whisper_language"] = payload["language"].clone();
    let segments = payload["segments"].as_array().cloned().unwrap_or_default();
    for (source, target, mean) in [
        ("avg_logprob", "whisper_avg_logprob", true),
        ("no_speech_prob", "whisper_no_speech_prob", false),
        ("compression_ratio", "whisper_compression_ratio", false),
    ] {
        let values: Vec<_> = segments.iter().filter_map(|s| s[source].as_f64()).collect();
        row[target] = if values.is_empty() {
            Value::Null
        } else if mean {
            json!(values.iter().sum::<f64>() / values.len() as f64)
        } else {
            json!(values.into_iter().reduce(f64::max))
        };
    }
    row["whisper"] = payload;
    row["ok"] = json!(true);
    Ok(row)
}

pub async fn run(
    data_dir: &Path,
    train_dir: &Path,
    langs: &[String],
    options: &Options,
) -> Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(options.timeout))
        .build()?;
    let cache = data_dir.join(".cache/asr");
    for source in &options.sources {
        let output = train_dir.join(format!("{source}_asr_exclusions.jsonl"));
        let old = if output.exists() {
            corpus::read(&output)?
        } else {
            vec![]
        };
        let saved: HashMap<_, _> = old
            .iter()
            .filter(|r| r["ok"] == true)
            .map(|r| {
                (
                    (r["lang"].clone().to_string(), r["file"].clone().to_string()),
                    r,
                )
            })
            .collect();
        let mut records = Vec::new();
        for lang in langs {
            for rec in corpus::read(&data_dir.join(lang).join("manifest.jsonl"))? {
                if rec["source"] != *source {
                    continue;
                }
                let file = corpus::text(&rec, "file")?;
                records.push(json!({
                    "file": file, "path": data_dir.join(lang).join(file), "lang": lang,
                    "source": source, "voice": rec["voice"], "g2p_language": corpus::language(&rec, lang)?,
                    "expected": corpus::text(&rec, "sentence")?, "model": options.asr_model,
                }));
            }
        }
        eprintln!(
            "{source}: auditing {} recordings{}",
            records.len(),
            if options.rescore {
                " from saved transcripts"
            } else {
                " (cached audio is not resent)"
            }
        );
        let results = stream::iter(records)
            .map(|mut row| {
                let (client, cache, saved) = (&client, &cache, &saved);
                async move {
                    let result: Result<Value> = async {
                        if options.rescore {
                            let previous = saved
                                .get(&(row["lang"].to_string(), row["file"].to_string()))
                                .context("no saved successful transcript for offline rescoring")?;
                            for field in ["duration_sec", "rms", "model"] {
                                row[field] = previous[field].clone();
                            }
                            let payload = if previous["whisper"].is_object() {
                                previous["whisper"].clone()
                            } else {
                                json!({"text": corpus::text(previous, "whisper_text")?})
                            };
                            score(with_payload(row.clone(), payload)?, options.text_only)
                        } else {
                            let path = Path::new(corpus::text(&row, "path")?);
                            let (duration, rms) = crate::audio::stats(path)?;
                            let payload = transcript(
                                client,
                                options,
                                cache,
                                path,
                                corpus::text(&row, "lang")?,
                            )
                            .await?;
                            row["duration_sec"] = json!(duration);
                            row["rms"] = json!(rms);
                            score(with_payload(row.clone(), payload)?, options.text_only)
                        }
                    }
                    .await;
                    result.with_context(|| format!("{}/{}", row["lang"], row["file"]))
                }
            })
            .buffer_unordered(options.asr_workers.into())
            .collect::<Vec<_>>()
            .await;
        let mut rows = Vec::new();
        let mut failures = 0;
        for result in results {
            match result {
                Ok(row) => rows.push(row),
                Err(error) => {
                    failures += 1;
                    eprintln!("ASR audit failed: {error:#}");
                }
            }
        }
        // Successful requests are already cached, even if another clip failed.
        // Keep the previous exclusions intact until this scope is complete.
        ensure!(
            failures == 0,
            "{source}: {failures} audit failures; exclusions were not replaced"
        );
        rows.sort_by_key(|r| (r["lang"].to_string(), r["file"].to_string()));
        corpus::write_scoped(&output, rows, langs)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_and_refusal_dispositions() {
        assert_eq!(normalize("ＳＴＲＡＳＳＥ Straße。"), "strasse strasse");
        let row =
            |expected, actual| json!({"lang": "hin", "expected": expected, "whisper_text": actual});
        let refused = score(row("१२३", "नमस्ते"), false).unwrap();
        assert!(refused.get("per").is_none());
        assert!(refused.get("cer").is_some());
        let mismatch = score(row("नमस्ते", "१२३"), false).unwrap();
        assert_eq!(mismatch["per"], 1.0);
        // An engine error must not become a rejection of the recording.
        assert!(
            score(
                json!({"lang": "eng", "expected": "hello", "whisper_text": "a\u{0}b"}),
                false
            )
            .is_err()
        );
    }
}

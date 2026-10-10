//! Cache paid ASR responses by audio/request; recompute local scores on each run.
use crate::corpus;
use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use clap::{Args, ValueEnum};
use futures::{StreamExt, stream};
use reqwest::{Client, multipart};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};
use unicode_casefold::UnicodeCaseFold;
use unicode_general_category::get_general_category;
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Groq,
    Cloudflare,
}

#[derive(Args)]
pub struct Options {
    /// Manifest sources to audit (defaults match the former upload script).
    #[arg(long, num_args = 1.., default_values = ["fleurs", "tatoeba"], value_parser = ["fleurs", "tatoeba", "tts", "tts_word", "kathbath", "mls", "cv", "aishell1", "aishell3"])]
    pub sources: Vec<String>,
    /// ASR provider. Groq remains the default to preserve existing requests/caches.
    #[arg(long, value_enum, default_value = "groq")]
    pub asr_backend: Backend,
    /// Override the provider's Whisper Turbo model identifier.
    #[arg(long)]
    pub asr_model: Option<String>,
    /// Maximum selected clips TOTAL across all sources/languages, including cache hits.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub limit: Option<u32>,
    #[arg(long, default_value = "8", value_parser = clap::value_parser!(u16).range(1..))]
    pub asr_workers: u16,
    #[arg(long, default_value_t = 4)]
    pub retries: u8,
    #[arg(long, default_value_t = 120)]
    pub timeout: u64,
    #[arg(long)]
    pub detect_language: bool,
    /// Skip phoneme metrics for legacy sources; strict sources always compare phones.
    #[arg(long)]
    pub text_only: bool,
    /// Re-score saved audit transcripts, including older files; never calls an ASR API.
    #[arg(long)]
    pub rescore: bool,
    #[arg(long, hide = true)]
    pub asr_url: Option<String>,
}

impl Options {
    fn model(&self) -> &str {
        self.asr_model.as_deref().unwrap_or(match self.asr_backend {
            Backend::Groq => "whisper-large-v3-turbo",
            Backend::Cloudflare => "@cf/openai/whisper-large-v3-turbo",
        })
    }

    fn url(&self) -> Result<String> {
        if let Some(url) = &self.asr_url {
            return Ok(url.clone());
        }
        Ok(match self.asr_backend {
            Backend::Groq => "https://api.groq.com/openai/v1/audio/transcriptions".to_owned(),
            Backend::Cloudflare => {
                let account = std::env::var("CLOUDFLARE_ACCOUNT_ID")
                    .context("CLOUDFLARE_ACCOUNT_ID is required for Cloudflare ASR")?;
                format!(
                    "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{}",
                    self.model()
                )
            }
        })
    }
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

// Deliberately not NFKC/casefold: accents, ß and word boundaries are evidence.
fn words(s: &str) -> Vec<String> {
    s.nfc()
        .collect::<String>()
        .to_lowercase()
        .nfc()
        .map(|c| {
            if c.is_whitespace()
                || c.is_ascii_punctuation()
                || get_general_category(c).abbreviation().starts_with('P')
            {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
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

fn strict_source(row: &Value) -> bool {
    matches!(
        row["source"].as_str(),
        Some("mls" | "cv" | "aishell1" | "aishell3" | "tts_word")
    )
}

fn strict_phones(
    row: &mut Value,
    language: g2p::Language,
    text: &str,
    side: &str,
) -> Result<Option<Vec<g2p::Phoneme>>> {
    let phones = match g2p::phonemize(language, text) {
        Ok(target) => target.phonemes,
        Err(g2p::Error::Unlabelable(reason)) => {
            row[format!("g2p_{side}_exclude_reason")] = json!(reason);
            return Ok(None);
        }
        // Label generation treats an unrepresentable target as an explicit refusal too.
        Err(g2p::Error::UnknownPhoneme(unknown)) => {
            row[format!("g2p_{side}_exclude_reason")] =
                json!(format!("unknown_phoneme:{}", unknown.0));
            return Ok(None);
        }
        Err(error) => return Err(error.into()),
    };
    Ok(Some(phones))
}

pub fn score(mut row: Value, text_only: bool) -> Result<Value> {
    let expected = corpus::text(&row, "expected")?.to_owned();
    let actual = corpus::text(&row, "whisper_text")?.to_owned();
    row["word_match"] = json!(words(&expected) == words(&actual));
    row["word_match_version"] = json!(1);
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
        "g2p_actual_exclude_reason",
        "phone_match",
        "phone_match_version",
        "phone_match_reason",
    ] {
        row.as_object_mut().unwrap().remove(key);
    }
    if strict_source(&row) {
        // --text-only cannot weaken strict-source admission. Both calls use the
        // same dialect resolution as labels; only phones, not factor heads, count.
        let language = corpus::language(&row, corpus::text(&row, "lang")?)?;
        row["g2p_language"] = serde_json::to_value(language)?;
        row["g2p_identity"] = json!(g2p::identity());
        let expected = strict_phones(&mut row, language, &expected, "reference")?;
        let actual = strict_phones(&mut row, language, &actual, "actual")?;
        row["phone_match_version"] = json!(1);
        match (expected, actual) {
            (Some(expected), Some(actual)) => {
                let matched = expected == actual;
                row["phone_match"] = json!(matched);
                row["phone_match_reason"] = json!(if matched {
                    "identical_phones"
                } else {
                    "phone_disagreement"
                });
                row["per"] = json!(distance(&expected, &actual));
                row["expected_phonemes"] = json!(expected);
                row["actual_phonemes"] = json!(actual);
            }
            _ => {
                row["phone_match"] = json!(false);
                row["phone_match_reason"] = json!("g2p_refusal");
            }
        }
    } else if !text_only {
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
        "fas" => "fa",
        _ => return None,
    })
}

fn cloudflare_body(bytes: &[u8], language: Option<&str>) -> Value {
    let mut body = json!({"audio": STANDARD.encode(bytes), "task": "transcribe"});
    if let Some(language) = language {
        body["language"] = json!(language);
    }
    body
}

fn cache_key(options: &Options, bytes: &[u8], language: Option<&str>, url: &str) -> Result<String> {
    let request = match options.asr_backend {
        // Retain the original Groq key byte-for-byte; no paid cache invalidation.
        Backend::Groq => json!({
            "audio": corpus::hash(bytes), "model": options.model(), "language": language,
            "temperature": 0, "format": "verbose_json", "url": url,
        }),
        Backend::Cloudflare => json!({
            "provider": "cloudflare", "audio": corpus::hash(bytes), "model": options.model(),
            "language": language, "task": "transcribe", "encoding": "base64", "url": url,
        }),
    };
    Ok(corpus::hash(&serde_json::to_vec(&request)?))
}

fn decode_payload(backend: Backend, mut response: Value) -> Result<Value> {
    if backend == Backend::Cloudflare {
        ensure!(
            response["success"] == true,
            "Cloudflare ASR failed: {}",
            response["errors"]
        );
        response = response["result"].take();
    }
    corpus::text(&response, "text")?;
    Ok(response)
}

async fn transcript(
    client: &Client,
    options: &Options,
    cache: &Path,
    path: &Path,
    lang: &str,
    rate_limit: &Option<Mutex<Instant>>,
) -> Result<Value> {
    let bytes = tokio::fs::read(path).await?;
    let language = if options.detect_language {
        None
    } else {
        iso(lang)
    };
    let url = options.url()?;
    let key = cache_key(options, &bytes, language, &url)?;
    let cached = cache.join(format!("{key}.json"));
    if cached.exists() {
        return Ok(serde_json::from_slice(&fs::read(cached)?)?);
    }
    let credential = match options.asr_backend {
        Backend::Groq => "GROQ_API_KEY",
        Backend::Cloudflare => "CLOUDFLARE_API_TOKEN",
    };
    let key = std::env::var(credential).with_context(|| {
        format!(
            "{credential} is required for uncached audio; use audit --rescore for saved transcripts"
        )
    })?;
    let mut last_error = String::new();
    for attempt in 0..=options.retries {
        let request = client.post(&url).bearer_auth(&key);
        let request = match options.asr_backend {
            Backend::Groq => {
                let part = multipart::Part::bytes(bytes.clone())
                    .file_name(path.file_name().unwrap().to_string_lossy().into_owned())
                    .mime_str("audio/wav")?;
                let mut form = multipart::Form::new()
                    .text("model", options.model().to_owned())
                    .text("temperature", "0")
                    .text("response_format", "verbose_json")
                    .part("file", part);
                if let Some(language) = language {
                    form = form.text("language", language);
                }
                request.multipart(form)
            }
            Backend::Cloudflare => request.json(&cloudflare_body(&bytes, language)),
        };
        let mut delay = Duration::from_secs((1u64 << attempt.min(5)).min(30));
        if let Some(rate_limit) = rate_limit {
            let wait = {
                let mut next = rate_limit.lock().unwrap();
                let now = Instant::now();
                let start = (*next).max(now);
                *next = start + Duration::from_millis(100); // 600/min, below Workers AI's 720/min ASR limit.
                start.duration_since(now)
            };
            tokio::time::sleep(wait).await;
        }
        match request.send().await {
            Ok(response) if response.status().is_success() => {
                let payload = decode_payload(
                    options.asr_backend,
                    response.json().await.context("decoding ASR transcript")?,
                )?;
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
                last_error = format!("{:?} ASR HTTP {status}", options.asr_backend);
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
    row["whisper_language"] = payload
        .get("language")
        .or_else(|| {
            payload
                .get("transcription_info")
                .and_then(|info| info.get("language"))
        })
        .cloned()
        .unwrap_or(Value::Null);
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
    let rate_limit =
        (options.asr_backend == Backend::Cloudflare).then(|| Mutex::new(Instant::now()));
    let mut remaining = options.limit.map(|n| n as usize).unwrap_or(usize::MAX);
    for source in &options.sources {
        if remaining == 0 {
            break;
        }
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
                    "g2p_selection": {"g2p_language":rec["g2p_language"], "variety":rec["variety"], "espeak_voice":rec["espeak_voice"]},
                    "expected": corpus::text(&rec, "sentence")?, "model": options.model(),
                    "asr_backend": options.asr_backend,
                }));
            }
        }
        records.truncate(remaining);
        remaining -= records.len();
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
                let (client, cache, saved, rate_limit) = (&client, &cache, &saved, &rate_limit);
                async move {
                    let result: Result<Value> = async {
                        if options.rescore {
                            let previous = saved
                                .get(&(row["lang"].to_string(), row["file"].to_string()))
                                .context("no saved successful transcript for offline rescoring")?;
                            for field in ["duration_sec", "rms", "model"] {
                                row[field] = previous[field].clone();
                            }
                            row["asr_backend"] = previous
                                .get("asr_backend")
                                .cloned()
                                .unwrap_or(json!("groq"));
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
                                rate_limit,
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
        if options.limit.is_some() {
            let selected: std::collections::HashSet<_> = rows
                .iter()
                .map(|r| (r["lang"].to_string(), r["file"].to_string()))
                .collect();
            rows.extend(
                old.into_iter().filter(|r| {
                    !selected.contains(&(r["lang"].to_string(), r["file"].to_string()))
                }),
            );
            corpus::write(&output, &rows)?;
        } else {
            corpus::write_scoped(&output, rows, langs)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mls_is_an_explicit_audit_source() {
        use clap::Parser;
        let args = crate::Args::try_parse_from(["lexide-preprocess", "audit", "--sources", "mls"])
            .unwrap();
        assert_eq!(args.audit.sources, ["mls"]);
        let word =
            crate::Args::try_parse_from(["lexide-preprocess", "audit", "--sources", "tts_word"])
                .unwrap();
        assert_eq!(word.audit.sources, ["tts_word"]);
        assert!(strict_source(&json!({"source":"tts_word"})));
        let defaults = crate::Args::try_parse_from(["lexide-preprocess", "audit"]).unwrap();
        assert_eq!(defaults.audit.sources, ["fleurs", "tatoeba"]);
    }

    #[test]
    fn strict_word_matching_preserves_spelling_and_boundaries() {
        for (expected, actual, matched) in [
            ("Hello, WORLD!", "hello world", true),
            ("café", "cafe\u{301}", true),
            ("their", "there", false),
            ("sentence", "sen tence", false),
            ("café", "cafe", false),
            ("Straße", "strasse", false),
            ("12", "twelve", false),
        ] {
            let row = score(json!({"expected":expected,"whisper_text":actual}), true).unwrap();
            assert_eq!(row["word_match"], matched);
        }
    }

    #[test]
    fn strict_admission_compares_only_phones_even_in_text_only_mode() {
        for (lang, expected, actual) in [
            ("eng", "their", "there"),
            ("eng", "12", "twelve"),
            ("deu", "Straße", "Strasse"),
            ("fra", "café", "cafe"),
            ("zho-hans", "妈", "马"),
        ] {
            let row = score(
                json!({"source":"mls","lang":lang,"expected":expected,"whisper_text":actual}),
                true,
            )
            .unwrap();
            assert_eq!(row["phone_match_version"], 1);
            if row["phone_match_reason"] == "g2p_refusal" {
                assert_eq!(row["phone_match"], false);
            } else {
                assert_eq!(
                    row["phone_match"],
                    row["expected_phonemes"] == row["actual_phonemes"]
                );
            }
            if matches!(lang, "eng" if expected == "their") || lang == "zho-hans" {
                assert_eq!(row["word_match"], false);
                assert_eq!(row["phone_match"], true, "{row}");
            }
        }
        let row = score(json!({"source":"cv","lang":"spa","g2p_language":"spa-419","expected":"llama","whisper_text":"llama"}), true).unwrap();
        assert_eq!(row["g2p_language"], "spa-419");
        assert_eq!(row["phone_match"], true);
        assert!(
            score(
                json!({"source":"mls","lang":"eng","expected":"hello","whisper_text":"a\u{0}b"}),
                true
            )
            .is_err()
        );
    }

    #[test]
    fn strict_g2p_refusal_on_either_side_rejects() {
        for (expected, actual, field) in [
            ("१२३", "नमस्ते", "g2p_reference_exclude_reason"),
            ("नमस्ते", "१२३", "g2p_actual_exclude_reason"),
        ] {
            let row = score(
                json!({"source":"cv","lang":"hin","expected":expected,"whisper_text":actual}),
                true,
            )
            .unwrap();
            assert_eq!(row["phone_match"], false);
            assert_eq!(row["phone_match_reason"], "g2p_refusal");
            assert!(row[field].is_string());
        }
    }

    #[test]
    fn cloudflare_request_envelope_and_cache_isolation() {
        use clap::Parser;
        let groq = crate::Args::try_parse_from(["preprocess", "audit"])
            .unwrap()
            .audit;
        let cf =
            crate::Args::try_parse_from(["preprocess", "audit", "--asr-backend", "cloudflare"])
                .unwrap()
                .audit;
        assert_eq!(groq.asr_backend, Backend::Groq);
        assert_eq!(cf.model(), "@cf/openai/whisper-large-v3-turbo");
        assert_eq!(
            cloudflare_body(b"wav", Some("zh")),
            json!({"audio":"d2F2","language":"zh","task":"transcribe"})
        );
        assert!(cloudflare_body(b"wav", None).get("language").is_none());
        let url = groq.url().unwrap();
        let legacy = json!({"audio":corpus::hash(b"wav"),"model":"whisper-large-v3-turbo","language":"en","temperature":0,"format":"verbose_json","url":url});
        assert_eq!(
            cache_key(&groq, b"wav", Some("en"), &url).unwrap(),
            corpus::hash(&serde_json::to_vec(&legacy).unwrap())
        );
        assert_ne!(
            cache_key(&groq, b"wav", Some("en"), "same-url").unwrap(),
            cache_key(&cf, b"wav", Some("en"), "same-url").unwrap()
        );
        assert_ne!(
            cache_key(&cf, b"wav", Some("en"), "same-url").unwrap(),
            cache_key(&cf, b"wav", None, "same-url").unwrap()
        );
        let payload = decode_payload(Backend::Cloudflare, json!({"success":true,"result":{"text":"hello","transcription_info":{"language":"en"}}})).unwrap();
        let row = with_payload(json!({}), payload).unwrap();
        assert_eq!(row["whisper_text"], "hello");
        assert_eq!(row["whisper_language"], "en");
        assert!(row["whisper_avg_logprob"].is_null());
        assert!(
            decode_payload(
                Backend::Cloudflare,
                json!({"success":false,"errors":[{"message":"bad audio"}]})
            )
            .is_err()
        );
        assert!(decode_payload(Backend::Cloudflare, json!({"success":true,"result":{}})).is_err());
    }

    #[test]
    fn audit_limit_is_global_and_retains_unselected_rows_offline() {
        use clap::Parser;
        let args = crate::Args::try_parse_from([
            "preprocess",
            "audit",
            "--sources",
            "mls",
            "cv",
            "--asr-backend",
            "cloudflare",
            "--rescore",
            "--text-only",
            "--limit",
            "3",
        ])
        .unwrap();
        assert!(crate::Args::try_parse_from(["preprocess", "audit", "--limit", "0"]).is_err());
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("audio");
        let train = tmp.path().join("train");
        let mut saved = vec![];
        for lang in ["eng", "deu"] {
            let records: Vec<_> = ["a.wav", "b.wav"]
                .iter()
                .map(|file| json!({"file":file,"sentence":"hello","source":"mls"}))
                .collect();
            corpus::write(&data.join(lang).join("manifest.jsonl"), &records).unwrap();
            for file in ["a.wav", "b.wav"] {
                saved.push(json!({"lang":lang,"file":file,"ok":true,"expected":"hello","whisper_text":"hello","model":"old-groq","marker":"retain"}));
            }
        }
        corpus::write(&train.join("mls_asr_exclusions.jsonl"), &saved).unwrap();
        corpus::write(
            &train.join("cv_asr_exclusions.jsonl"),
            &[json!({"untouched":true})],
        )
        .unwrap();
        let cv_before = fs::read(train.join("cv_asr_exclusions.jsonl")).unwrap();
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(run(
                &data,
                &train,
                &["eng".into(), "deu".into()],
                &args.audit,
            ))
            .unwrap();
        let result = corpus::read(&train.join("mls_asr_exclusions.jsonl")).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.iter().filter(|r| r["word_match"] == true).count(), 3);
        assert!(
            result
                .iter()
                .filter(|r| r["word_match"] == true)
                .all(|r| r["asr_backend"] == "groq")
        );
        assert!(result.contains(saved.last().unwrap()));
        assert_eq!(
            fs::read(train.join("cv_asr_exclusions.jsonl")).unwrap(),
            cv_before
        );
    }

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

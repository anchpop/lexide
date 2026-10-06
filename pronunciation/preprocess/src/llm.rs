//! Shared, restartable OpenAI Batch transport; old tysm caches remain readable.
use anyhow::{Context, Result, bail, ensure};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};
use tysm::chat_completions::{
    ChatClient, ChatError, ChatMessage, ChatRequest, JsonSchemaFormat, ResponseFormat,
};

pub const MODEL: &str = "gpt-6-luna";

fn request<T: JsonSchema>(system: &str, user: &str) -> Result<Value> {
    Ok(serde_json::to_value(ChatRequest {
        model: MODEL.into(),
        messages: vec![ChatMessage::system(system), ChatMessage::user(user)],
        response_format: ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat::new::<T>(),
        },
        service_tier: None,
        reasoning_effort: None,
        extra_body: None,
    })?)
}

fn content<T: DeserializeOwned>(body: &Value) -> Result<T> {
    let choice = &body["choices"][0];
    ensure!(
        choice["finish_reason"] == "stop",
        "incomplete/refused batch response: {body}"
    );
    Ok(serde_json::from_str(
        choice["message"]["content"]
            .as_str()
            .context("missing batch response content")?,
    )?)
}

fn save(path: &Path, value: &Value) -> Result<()> {
    let mut temp = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
    serde_json::to_writer(&mut temp, value)?;
    temp.as_file().sync_all()?;
    temp.persist(path)?;
    Ok(())
}

async fn response(response: reqwest::Response) -> Result<Value> {
    let status = response.status();
    let text = response.text().await?;
    ensure!(status.is_success(), "OpenAI Batch HTTP {status}: {text}");
    Ok(serde_json::from_str(&text)?)
}

/// Requests retain exactly the old system/user/schema shape. Probe tysm in
/// cached-only mode (including its legacy-key migration), then the Batch cache.
/// Dry mode writes JSONL without credentials or any network activity.
pub async fn run<T: DeserializeOwned + JsonSchema>(
    cache: &Path,
    prompts: &[(String, String)],
    dry: bool,
) -> Result<Option<Vec<T>>> {
    fs::create_dir_all(cache)?;
    let dir = cache.join("batch");
    fs::create_dir_all(&dir)?;
    // Finish any earlier submission before splitting the now-smaller cache-miss
    // set. Otherwise a crash during ingestion could submit remaining rows twice.
    if !dry {
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if let Some(key) = path
                .file_name()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_suffix(".state.json"))
            {
                let state: Value = serde_json::from_slice(&fs::read(&path)?)?;
                if state["ingested"] != true {
                    let input = dir.join(format!("{key}.input.jsonl"));
                    execute::<T>(&dir, key, &input, &crate::corpus::read(&input)?).await?;
                }
            }
        }
    }
    let legacy = ChatClient::new("unused-cached-only", MODEL)
        .with_cache_directory(cache)
        .with_cached_only();
    let mut values = Vec::new();
    let mut pending = BTreeMap::new();
    let mut ids = Vec::new();
    for (system, user) in prompts {
        let body = request::<T>(system, user)?;
        let id = crate::corpus::hash(&serde_json::to_vec(&body)?);
        let path = dir.join(format!("{id}.response.json"));
        let value = if path.exists() {
            Some(content::<T>(&serde_json::from_slice(&fs::read(path)?)?)?)
        } else {
            match legacy.chat_with_system_prompt::<T>(system, user).await {
                Ok(value) => Some(value),
                Err(ChatError::CacheMiss) => None,
                Err(error) => return Err(error.into()),
            }
        };
        if value.is_none() {
            pending.insert(id.clone(), body);
        }
        values.push(value);
        ids.push(id);
    }
    // Stay below both official limits: 50,000 requests and 200 MB per file.
    let mut chunks: Vec<Vec<Value>> = vec![vec![]];
    let mut bytes = 0;
    for (id, body) in pending {
        let row = json!({"custom_id":id,"method":"POST","url":"/v1/chat/completions","body":body});
        let size = serde_json::to_vec(&row)?.len() + 1;
        ensure!(size < 180_000_000, "single batch request too large");
        if chunks.last().unwrap().len() == 40_000 || bytes + size > 180_000_000 {
            chunks.push(vec![]);
            bytes = 0;
        }
        bytes += size;
        chunks.last_mut().unwrap().push(row);
    }
    for rows in chunks.into_iter().filter(|c| !c.is_empty()) {
        let key = crate::corpus::hash(&serde_json::to_vec(&rows)?);
        let input = dir.join(format!("{key}.input.jsonl"));
        crate::corpus::write(&input, &rows)?;
        eprintln!("Batch: {} requests → {}", rows.len(), input.display());
        if !dry {
            execute::<T>(&dir, &key, &input, &rows).await?;
        }
    }
    if dry {
        return Ok(None);
    }
    for (value, id) in values.iter_mut().zip(ids) {
        if value.is_none() {
            *value = Some(content::<T>(&serde_json::from_slice(&fs::read(
                dir.join(format!("{id}.response.json")),
            )?)?)?);
        }
    }
    Ok(Some(values.into_iter().map(Option::unwrap).collect()))
}

async fn execute<T: DeserializeOwned>(
    dir: &Path,
    key: &str,
    input: &Path,
    rows: &[Value],
) -> Result<()> {
    let token = std::env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY required to submit/poll Batch; use --batch-dry-run offline")?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;
    let state_path = dir.join(format!("{key}.state.json"));
    let mut state: Value = if state_path.exists() {
        serde_json::from_slice(&fs::read(&state_path)?)?
    } else {
        json!({})
    };
    if state["batch_id"].is_null() {
        ensure!(
            state["submitting"] != true,
            "ambiguous Batch submission: inspect OpenAI batches and set batch_id in {}; do not blindly resubmit",
            state_path.display()
        );
        let form = reqwest::multipart::Form::new()
            .text("purpose", "batch")
            .part(
                "file",
                reqwest::multipart::Part::bytes(fs::read(input)?).file_name("requests.jsonl"),
            );
        let file = response(
            client
                .post("https://api.openai.com/v1/files")
                .bearer_auth(&token)
                .multipart(form)
                .send()
                .await?,
        )
        .await?;
        state = json!({"input_file_id":file["id"], "submitting":true});
        save(&state_path, &state)?;
        let batch = response(client.post("https://api.openai.com/v1/batches").bearer_auth(&token)
            .json(&json!({"input_file_id":file["id"],"endpoint":"/v1/chat/completions","completion_window":"24h","metadata":{"lexide_input_hash":key}})).send().await?).await?;
        state["batch_id"] = batch["id"].clone();
        save(&state_path, &state)?;
    }
    let id = state["batch_id"].as_str().context("missing batch ID")?;
    let batch = loop {
        let batch = response(
            client
                .get(format!("https://api.openai.com/v1/batches/{id}"))
                .bearer_auth(&token)
                .send()
                .await?,
        )
        .await?;
        match batch["status"].as_str().context("missing batch status")? {
            "validating" | "in_progress" | "finalizing" => {
                tokio::time::sleep(Duration::from_secs(60)).await
            }
            _ => break batch,
        }
    };
    save(&dir.join(format!("{key}.result.json")), &batch)?;
    for field in ["output_file_id", "error_file_id"] {
        if let Some(file) = batch[field].as_str() {
            let resp = client
                .get(format!("https://api.openai.com/v1/files/{file}/content"))
                .bearer_auth(&token)
                .send()
                .await?
                .error_for_status()?;
            fs::write(
                dir.join(format!("{key}.{field}.jsonl")),
                resp.bytes().await?,
            )?;
        }
    }
    let output = dir.join(format!("{key}.output_file_id.jsonl"));
    let mut expected: std::collections::HashSet<_> = rows
        .iter()
        .map(|r| r["custom_id"].as_str().unwrap())
        .collect();
    if output.exists() {
        for row in crate::corpus::read(&output)? {
            let id = row["custom_id"].as_str().context("missing custom_id")?;
            ensure!(
                expected.remove(id),
                "unknown or duplicate Batch custom_id {id}"
            );
            ensure!(
                row["error"].is_null() && row["response"]["status_code"] == 200,
                "Batch request failed: {row}"
            );
            let body = &row["response"]["body"];
            content::<T>(body)?;
            save(&dir.join(format!("{id}.response.json")), body)?;
        }
    }
    if batch["status"] != "completed" || !expected.is_empty() {
        bail!(
            "Batch {id} ended {} with {} missing results; inspect {}",
            batch["status"],
            expected.len(),
            dir.display()
        );
    }
    state["ingested"] = json!(true);
    save(&state_path, &state)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(serde::Deserialize, JsonSchema)]
    struct Verdict {
        accepted: bool,
    }
    #[test]
    fn offline_jsonl_and_response_validation() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
        let dir = tempfile::tempdir().unwrap();
        assert!(run::<Verdict>(dir.path(), &[("system".into(), "user".into())], true).await.unwrap().is_none());
        let input = fs::read_dir(dir.path().join("batch")).unwrap().next().unwrap().unwrap().path();
        let rows = crate::corpus::read(&input).unwrap();
        assert_eq!(rows[0]["body"]["model"], MODEL);
        assert_eq!(rows[0]["url"], "/v1/chat/completions");
        assert!(content::<Verdict>(&json!({"choices":[{"finish_reason":"stop","message":{"content":"{\"accepted\":true}"}}]})).unwrap().accepted);
        assert!(content::<Verdict>(&json!({"choices":[{"finish_reason":"length","message":{"content":"{}"}}]})).is_err());
        });
    }
}

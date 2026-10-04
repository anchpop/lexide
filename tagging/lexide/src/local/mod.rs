//! Local parsley: bge-m3 + character BiLSTM + joint word heads, CPU fp32 ONNX.
mod mst;
mod script;
mod tagger;

use crate::{raw::tokens_from_raw, Language, Tokenization};
use anyhow::{Context, Result};
use std::path::PathBuf;

/// Immutable artifact revision; v1's `onnx/` directory remains untouched.
pub const MODEL_REVISION: &str = "a4acccb7a921fa79493d02fd4f7ee6f22ab77a74";

/// Configuration for local parsley inference.
#[derive(Debug, Clone)]
pub struct LocalConfig {
    /// Directory containing encoder.onnx, encoder.onnx.data, heads.onnx,
    /// tokenizer.json, vocab.json and config.json. Defaults to LEXIDE_MODEL_DIR.
    pub model_dir: Option<PathBuf>,
    /// Hugging Face repository; downloads joint/* at MODEL_REVISION.
    pub hf_repo: String,
    /// ONNX intra-op threads (0 lets the runtime decide).
    pub threads: usize,
}
impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            model_dir: std::env::var("LEXIDE_MODEL_DIR").map(PathBuf::from).ok(),
            hf_repo: "anchpop/lexide-parsley".into(),
            threads: 0,
        }
    }
}

fn fetch_from_hub(repo_id: &str) -> Result<PathBuf> {
    let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        repo_id.into(),
        hf_hub::RepoType::Model,
        MODEL_REVISION.into(),
    ));
    for name in [
        "encoder.onnx.data",
        "heads.onnx",
        "tokenizer.json",
        "vocab.json",
        "config.json",
    ] {
        repo.get(&format!("joint/{name}"))
            .with_context(|| format!("downloading joint/{name}"))?;
    }
    Ok(repo
        .get("joint/encoder.onnx")?
        .parent()
        .expect("cached model parent")
        .to_path_buf())
}

/// One joint model, no dictionary or boundary-prior dependencies.
pub struct LocalLexide {
    tagger: tagger::OnnxTagger,
}
impl LocalLexide {
    /// Download (~2.4 GB, cached) and load away from the async executor.
    pub async fn from_pretrained(config: LocalConfig) -> Result<Self> {
        tokio::task::spawn_blocking(move || Self::load(config))
            .await
            .context("model loading task panicked")?
    }
    pub fn load(config: LocalConfig) -> Result<Self> {
        let dir = match config.model_dir {
            Some(dir) => dir,
            None => fetch_from_hub(&config.hf_repo)?,
        };
        Ok(Self {
            tagger: tagger::OnnxTagger::load(&dir, config.threads)?,
        })
    }
    pub fn analyze(&self, sentence: &str, language: Language) -> Result<Tokenization> {
        Ok(tokens_from_raw(
            &self.tagger.tag(sentence, language.code())?,
            sentence,
        )?)
    }
}

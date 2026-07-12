//! Local parsley inference: the ONNX multi-task tagger + byte-minGRU segmenter +
//! Wiktionary lemma tables, all running in-process on CPU. This replaces the old
//! mistralrs Gemma backend (which was unusably slow) with the same pipeline the
//! parsley Modal serve runs, minus the network.
//!
//! By default the artifacts are downloaded from HF `anchpop/lexide-parsley/onnx/` into
//! the standard HuggingFace cache on first load; set `LocalConfig::model_dir` (or
//! `LEXIDE_MODEL_DIR`) to use a local directory instead. Expected `model_dir` layout
//! (published by `tagging/release.sh`):
//!   tagger.onnx                  encoder + heads, exported ONNX graph
//!   tokenizer.json               XLM-R fast tokenizer
//!   vocab.json                   POS / dep / lemma edit-script vocabularies
//!   char_tokenizer.safetensors   byte-minGRU token boundary tagger weights
//!   sentence_segmenter.safetensors  byte-minGRU sentence segmenter weights (optional)
//!   lemma_fst/wikt_{lang}.fst    optional per-language lemma tables (build-lemma-fst)

mod byte_bio;
mod chartok;
mod lemma;
mod script;
mod sentence;
mod tagger;

pub use sentence::Sentence;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::{bail, Context, Result};

use crate::raw::{tokens_from_raw, RawToken};
use crate::{Language, Tokenization};

pub use lemma::{build_table, LemmaTable};

/// Configuration for local parsley inference.
#[derive(Debug, Clone)]
pub struct LocalConfig {
    /// Directory with tagger.onnx, tokenizer.json, vocab.json, char_tokenizer.safetensors.
    /// `None` (the default when `LEXIDE_MODEL_DIR` is unset) downloads the artifacts from
    /// `hf_repo` into the standard HuggingFace cache on first use.
    pub model_dir: Option<PathBuf>,
    /// HuggingFace repo to fetch from when `model_dir` is None; artifacts live under its
    /// `onnx/` folder. The repo is public, so no token is needed (`HF_TOKEN` is honored
    /// if set, e.g. for a private fork).
    pub hf_repo: String,
    /// Directory with per-language `wikt_{lang}.fst` lemma tables.
    /// Defaults to `{model_dir}/lemma_fst`; missing tables just mean model-only lemmas,
    /// matching the parsley server's behavior for languages without a table.
    pub lemma_tables_dir: Option<PathBuf>,
    /// Intra-op threads for ONNX Runtime (0 = let the runtime decide).
    pub threads: usize,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            model_dir: std::env::var("LEXIDE_MODEL_DIR").map(PathBuf::from).ok(),
            hf_repo: "anchpop/lexide-parsley".to_string(),
            lemma_tables_dir: None,
            threads: 0,
        }
    }
}

/// The languages with published lemma tables (jpn isn't served; see OVERVIEW.md).
const TABLE_LANGS: [&str; 9] = ["deu", "eng", "fra", "hin", "ita", "kor", "por", "rus", "spa"];

/// Fetch the model artifacts from the hub into the HF cache (no-op when already cached)
/// and return the cache directory that mirrors a local model_dir layout.
fn fetch_from_hub(repo_id: &str) -> Result<PathBuf> {
    let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build().context("failed to build HF hub client")?;
    let repo = api.model(repo_id.to_string());

    let mut tagger_path = None;
    for f in [
        "tagger.onnx",
        "tokenizer.json",
        "vocab.json",
        "char_tokenizer.safetensors",
    ] {
        let p = repo
            .get(&format!("onnx/{f}"))
            .with_context(|| format!("failed to download onnx/{f} from {repo_id}"))?;
        if f == "tagger.onnx" {
            tagger_path = Some(p);
        }
    }
    // Optional: the sentence segmenter. Older repo snapshots may not have it; a miss just
    // means `segment_sentences` is unavailable, not a failed load.
    if let Err(e) = repo.get("onnx/sentence_segmenter.safetensors") {
        eprintln!("lexide: no sentence segmenter on {repo_id} (segment_sentences disabled): {e}");
    }
    for lang in TABLE_LANGS {
        // Optional: a table missing on the hub just means model-only lemmas for that language.
        if let Err(e) = repo.get(&format!("onnx/lemma_fst/wikt_{lang}.fst")) {
            eprintln!("lexide: no lemma table for {lang} on {repo_id}: {e}");
        }
    }
    // The snapshot layout mirrors the repo, so tagger.onnx's parent is a valid model_dir
    // (with lemma_fst/ beneath it).
    Ok(tagger_path
        .expect("tagger.onnx was just downloaded")
        .parent()
        .expect("cached file has a parent directory")
        .to_path_buf())
}

/// Local inference pipeline: segment (byte minGRU) -> tag (ONNX) -> lemma floor (fst).
pub struct LocalLexide {
    chartok: chartok::CharTokenizer,
    tagger: tagger::OnnxTagger,
    // Optional: absent when the artifact isn't published/downloaded (older snapshots).
    segmenter: Option<sentence::SentenceSegmenter>,
    lemma_dir: PathBuf,
    // Tables load lazily per language (a table is a few MB; most callers use one language).
    tables: RwLock<HashMap<&'static str, Option<Arc<LemmaTable>>>>,
}

impl LocalLexide {
    /// Load the model. With the default config this downloads the artifacts from the hub
    /// into the HF cache on first use (~1.2 GB; subsequent loads reuse the cache), so the
    /// download runs off the async executor.
    pub async fn from_pretrained(config: LocalConfig) -> Result<Self> {
        tokio::task::spawn_blocking(move || Self::load(config))
            .await
            .context("model loading task panicked")?
    }

    pub fn load(config: LocalConfig) -> Result<Self> {
        let dir = match &config.model_dir {
            Some(dir) => {
                if !dir.join("tagger.onnx").exists() {
                    bail!(
                        "no tagger.onnx in {} — point LocalConfig.model_dir (or \
                         LEXIDE_MODEL_DIR) at the parsley ONNX artifacts, or leave it unset \
                         to download them from HF {}",
                        dir.display(),
                        config.hf_repo
                    );
                }
                dir.clone()
            }
            None => fetch_from_hub(&config.hf_repo)?,
        };
        let dir = &dir;
        let chartok = chartok::CharTokenizer::load(&dir.join("char_tokenizer.safetensors"))
            .context("failed to load the char tokenizer")?;
        let tagger = tagger::OnnxTagger::load(dir, config.threads)
            .context("failed to load the ONNX tagger")?;
        // Optional: present only if the artifact was published/downloaded. A present-but-
        // unreadable file is worth surfacing; an absent one just disables segmentation.
        let seg_path = dir.join("sentence_segmenter.safetensors");
        let segmenter = if seg_path.exists() {
            Some(
                sentence::SentenceSegmenter::load(&seg_path)
                    .context("failed to load the sentence segmenter")?,
            )
        } else {
            None
        };
        let lemma_dir = config
            .lemma_tables_dir
            .unwrap_or_else(|| dir.join("lemma_fst"));
        Ok(Self {
            chartok,
            tagger,
            segmenter,
            lemma_dir,
            tables: RwLock::new(HashMap::new()),
        })
    }

    fn table(&self, language: Language) -> Option<Arc<LemmaTable>> {
        let code = language.code();
        if let Some(cached) = self.tables.read().expect("lemma tables lock").get(code) {
            return cached.clone();
        }
        let path = self.lemma_dir.join(format!("wikt_{code}.fst"));
        let table = match LemmaTable::load(&path) {
            Ok(t) => Some(Arc::new(t)),
            Err(_) if !path.exists() => None, // no table built for this language
            Err(e) => {
                // A present-but-unreadable table is worth a warning, not a hard failure:
                // lemmas degrade to model-only, same as serving without tables.
                eprintln!("lexide: ignoring lemma table {}: {e:#}", path.display());
                None
            }
        };
        self.tables
            .write()
            .expect("lemma tables lock")
            .insert(code, table.clone());
        table
    }

    /// Analyze a sentence: segment into tokens, tag POS/lemma/dependencies, and apply the
    /// language's Wiktionary lemma floor. Mirrors the parsley server token-for-token.
    pub fn analyze(&self, sentence: &str, language: Language) -> Result<Tokenization> {
        let spans = self.chartok.segment(sentence);
        let tagged = self.tagger.tag(sentence, &spans)?;
        let table = self.table(language);
        let rtoks: Vec<RawToken> = tagged
            .into_iter()
            .map(|t| {
                let lemma = match &table {
                    Some(tb) => tb.resolve(&t.text, &t.pos, &t.lemma),
                    None => t.lemma,
                };
                RawToken {
                    text: t.text,
                    start: t.start,
                    end: t.end,
                    pos: t.pos,
                    lemma,
                    dep: t.dep,
                    head: t.head,
                }
            })
            .collect();
        Ok(tokens_from_raw(&rtoks, sentence))
    }

    /// Split a passage into its sentences (gaps between sentences dropped), each with its
    /// char span. Errors if the segmenter artifact wasn't available at load time.
    pub fn segment_sentences_detailed(&self, text: &str) -> Result<Vec<Sentence>> {
        let seg = self.segmenter.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "sentence segmenter not loaded (sentence_segmenter.safetensors missing from \
                 the model dir / HF repo)"
            )
        })?;
        Ok(seg.segment(text))
    }

    /// Split a passage into its sentence strings, in order.
    pub fn segment_sentences(&self, text: &str) -> Result<Vec<String>> {
        let seg = self.segmenter.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "sentence segmenter not loaded (sentence_segmenter.safetensors missing from \
                 the model dir / HF repo)"
            )
        })?;
        Ok(seg.sentences(text))
    }
}

/// Test helper: locate a model artifact, honoring LEXIDE_MODEL_DIR and falling back to
/// the repo-relative `../data/onnx` (where `modal volume get lexide-onnx` drops them).
#[cfg(test)]
pub(crate) mod test_support {
    use std::path::PathBuf;

    pub fn model_dir() -> PathBuf {
        std::env::var("LEXIDE_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/onnx"))
    }

    pub fn model_file(name: &str) -> Option<PathBuf> {
        let p = model_dir().join(name);
        p.exists().then_some(p)
    }
}

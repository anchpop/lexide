//! Local parsley inference: the ONNX multi-task tagger + byte-minGRU segmenter +
//! Wiktionary lemma tables, all running in-process on CPU. This replaces the old
//! mistralrs Gemma backend (which was unusably slow) with the same pipeline the
//! parsley Modal serve runs, minus the network.
//!
//! Expected `model_dir` layout (artifacts live on the `lexide-onnx` Modal volume;
//! see `tagger/export_onnx.py` + `tagger/export_char_modal.py`):
//!   tagger.onnx                  encoder + heads, exported ONNX graph
//!   tokenizer.json               XLM-R fast tokenizer
//!   vocab.json                   POS / dep / lemma edit-script vocabularies
//!   char_tokenizer.safetensors   byte-minGRU segmenter weights
//!   lemma_fst/wikt_{lang}.fst    optional per-language lemma tables (build-lemma-fst)

mod chartok;
mod lemma;
mod script;
mod tagger;

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
    pub model_dir: PathBuf,
    /// Directory with per-language `wikt_{lang}.fst` lemma tables.
    /// Defaults to `{model_dir}/lemma_fst`; missing tables just mean model-only lemmas,
    /// matching the parsley server's behavior for languages without a table.
    pub lemma_tables_dir: Option<PathBuf>,
    /// Intra-op threads for ONNX Runtime (0 = let the runtime decide).
    pub threads: usize,
}

impl Default for LocalConfig {
    fn default() -> Self {
        let model_dir = std::env::var("LEXIDE_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/onnx"));
        Self {
            model_dir,
            lemma_tables_dir: None,
            threads: 0,
        }
    }
}

/// Local inference pipeline: segment (byte minGRU) -> tag (ONNX) -> lemma floor (fst).
pub struct LocalLexide {
    chartok: chartok::CharTokenizer,
    tagger: tagger::OnnxTagger,
    lemma_dir: PathBuf,
    // Tables load lazily per language (a table is a few MB; most callers use one language).
    tables: RwLock<HashMap<&'static str, Option<Arc<LemmaTable>>>>,
}

impl LocalLexide {
    /// Load all model artifacts from `config.model_dir`. Async only for API compatibility
    /// with the old backend — loading is synchronous and takes ~a second.
    pub async fn from_pretrained(config: LocalConfig) -> Result<Self> {
        Self::load(config)
    }

    pub fn load(config: LocalConfig) -> Result<Self> {
        let dir = &config.model_dir;
        if !dir.join("tagger.onnx").exists() {
            bail!(
                "no tagger.onnx in {} — set LocalConfig.model_dir (or LEXIDE_MODEL_DIR) to a \
                 directory with the parsley ONNX artifacts (see `modal volume get lexide-onnx`)",
                dir.display()
            );
        }
        let chartok = chartok::CharTokenizer::load(&dir.join("char_tokenizer.safetensors"))
            .context("failed to load the char tokenizer")?;
        let tagger = tagger::OnnxTagger::load(dir, config.threads)
            .context("failed to load the ONNX tagger")?;
        let lemma_dir = config
            .lemma_tables_dir
            .unwrap_or_else(|| dir.join("lemma_fst"));
        Ok(Self {
            chartok,
            tagger,
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

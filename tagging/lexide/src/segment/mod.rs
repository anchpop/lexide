//! Standalone sentence segmentation: the 1M-param byte-minGRU and nothing else.
//!
//! This is the lightest way to use parsley — enabled by the `segment` cargo feature
//! (which `local` builds on), it needs only `safetensors` + `hf-hub`, and
//! [`Segmenter::from_pretrained`] downloads a single ~4 MB artifact into the standard
//! HF cache. For the full pipeline (tokenize / POS / lemma / dependencies) use
//! [`Lexide`](crate::Lexide) with the `local` feature instead.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! let parsley = lexide::Segmenter::from_pretrained()?; // ~4 MB, cached after first run
//! assert_eq!(
//!     parsley.segment_in("Dr. Smith arrived at 3 p.m. — he wasn't late. \"Is this the place?\" she asked.", lexide::Language::English),
//!     vec!["Dr. Smith arrived at 3 p.m. — he wasn't late.", "\"Is this the place?\" she asked."],
//! );
//! # Ok(()) }
//! ```

pub(crate) mod byte_bio;
pub(crate) mod sentence;

use std::path::Path;

use anyhow::{Context, Result};

use crate::Language;
pub use sentence::Sentence;

/// A self-contained sentence segmenter (byte-minGRU over UTF-8, ~1M params).
///
/// Gaps between sentences (whitespace, headings, separators) are dropped; punctuation
/// that frames a sentence (its quotes, a leading dialogue dash) stays attached to it.
pub struct Segmenter {
    inner: sentence::SentenceSegmenter,
}

impl Segmenter {
    /// Download the segmenter weights from HF `anchpop/lexide-parsley` (~4 MB, reused
    /// from the standard HF cache after the first call) and load them.
    ///
    /// Honors `LEXIDE_MODEL_DIR`: when set and it contains
    /// `sentence_segmenter.safetensors`, that file is used and nothing is downloaded.
    pub fn from_pretrained() -> Result<Self> {
        if let Ok(dir) = std::env::var("LEXIDE_MODEL_DIR") {
            let p = Path::new(&dir).join("sentence_segmenter.safetensors");
            if p.exists() {
                return Self::from_file(&p);
            }
        }
        let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
        if let Ok(token) = std::env::var("HF_TOKEN") {
            builder = builder.with_token(Some(token));
        }
        let api = builder.build().context("failed to build HF hub client")?;
        let path = api
            .model("anchpop/lexide-parsley".to_string())
            .get("onnx/sentence_segmenter.safetensors")
            .context("failed to download the segmenter from HF anchpop/lexide-parsley")?;
        Self::from_file(&path)
    }

    /// Load from a local `sentence_segmenter.safetensors`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            inner: sentence::SentenceSegmenter::load(path.as_ref())?,
        })
    }

    /// Split a passage into its sentence strings without a language hint.
    pub fn segment(&self, text: &str) -> Vec<String> {
        self.inner.sentences(text, None)
    }

    /// Like [`segment`](Self::segment) with a language hint, which improves ambiguous
    /// boundaries (abbreviations like "Mr.", quote attributions).
    pub fn segment_in(&self, text: &str, language: Language) -> Vec<String> {
        self.inner.sentences(text, Some(language.code()))
    }

    /// Each sentence with its `[start, end)` char (code point) span in the passage.
    pub fn segment_detailed(&self, text: &str, language: Option<Language>) -> Vec<Sentence> {
        self.inner.segment(text, language.map(|l| l.code()))
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

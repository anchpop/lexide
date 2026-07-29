//! WASM bindings for the byte-minGRU models: the char tokenizer and the sentence
//! segmenter, running fully in-browser. Reuses the lexide crate's pure-Rust
//! `byte_bio` module (single source of truth — included by path, not copied).
//!
//! Spans are char (code point) `[start, end)` indices into the input, matching the
//! Rust/Python pipelines — JS callers must index by code point (`Array.from(text)`),
//! not UTF-16 unit.

#[path = "../../lexide/src/segment/byte_bio.rs"]
#[allow(dead_code)]
mod byte_bio;
// byte_bio reads its prior symbols from these; included by path for the same reason —
// one source of truth. unidic is pulled in because prior refers to it, but the browser
// never has the 87MB artifact to hand it, so no dictionary is ever loaded here.
#[path = "../../lexide/src/segment/prior.rs"]
#[allow(dead_code)]
mod prior;
#[path = "../../lexide/src/segment/unidic.rs"]
#[allow(dead_code)]
mod unidic;

use byte_bio::ByteBioModel;
use prior::PriorSet;
use wasm_bindgen::prelude::*;

fn js_err(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

/// The proposal is built per call and never truncated — the model does not truncate its
/// input either, so a shorter prior would misalign against a long paste.
const NO_PRIOR_LIMIT: usize = usize::MAX;

#[wasm_bindgen]
pub struct Parsley {
    tokenizer: ByteBioModel,
    segmenter: ByteBioModel,
    /// Empty: the browser has no room for the 87MB Japanese dictionary. Spaced languages
    /// still get their exact whitespace proposal for free, and Japanese gets an all-NONE
    /// one — "no information" rather than whitespace's false claim that the sentence is a
    /// single word. Checkpoints trained with `--prior-dropout` handle that gracefully; a
    /// checkpoint trained without it will segment Japanese poorly here, which is the
    /// documented cost of running the demo without the dictionary.
    priors: PriorSet,
}

#[wasm_bindgen]
impl Parsley {
    /// Build from the two safetensors artifacts (fetched by the page).
    #[wasm_bindgen(constructor)]
    pub fn new(tokenizer_weights: &[u8], segmenter_weights: &[u8]) -> Result<Parsley, JsError> {
        Ok(Parsley {
            tokenizer: ByteBioModel::from_bytes(tokenizer_weights).map_err(js_err)?,
            segmenter: ByteBioModel::from_bytes(segmenter_weights).map_err(js_err)?,
            priors: PriorSet::default(),
        })
    }

    /// Token `[start, end)` char spans as a JSON array of pairs. `lang` is an optional
    /// three-letter code (deu/eng/fra/hin/ita/jpn/kor/por/rus/spa); null = language-free.
    pub fn token_spans(&self, text: &str, lang: Option<String>) -> String {
        let lang = lang.as_deref();
        let p = self
            .tokenizer
            .wants_prior()
            .then(|| self.priors.ids(text, lang, NO_PRIOR_LIMIT));
        serde_json::to_string(&self.tokenizer.segment_with_prior(text, lang, p.as_deref()))
            .expect("span serialization")
    }

    /// Sentence `[start, end)` char spans as a JSON array of pairs.
    pub fn sentence_spans(&self, text: &str, lang: Option<String>) -> String {
        serde_json::to_string(&self.segmenter.segment(text, lang.as_deref()))
            .expect("span serialization")
    }
}

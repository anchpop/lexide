//! WASM bindings for the byte-minGRU models: the char tokenizer and the sentence
//! segmenter, running fully in-browser. Reuses the lexide crate's pure-Rust
//! `byte_bio` module (single source of truth — included by path, not copied).
//!
//! Spans are char (code point) `[start, end)` indices into the input, matching the
//! Rust/Python pipelines — JS callers must index by code point (`Array.from(text)`),
//! not UTF-16 unit.

#[path = "../../lexide/src/local/byte_bio.rs"]
#[allow(dead_code)]
mod byte_bio;

use byte_bio::ByteBioModel;
use wasm_bindgen::prelude::*;

fn js_err(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

#[wasm_bindgen]
pub struct Parsley {
    tokenizer: ByteBioModel,
    segmenter: ByteBioModel,
}

#[wasm_bindgen]
impl Parsley {
    /// Build from the two safetensors artifacts (fetched by the page).
    #[wasm_bindgen(constructor)]
    pub fn new(tokenizer_weights: &[u8], segmenter_weights: &[u8]) -> Result<Parsley, JsError> {
        Ok(Parsley {
            tokenizer: ByteBioModel::from_bytes(tokenizer_weights).map_err(js_err)?,
            segmenter: ByteBioModel::from_bytes(segmenter_weights).map_err(js_err)?,
        })
    }

    /// Token `[start, end)` char spans as a JSON array of pairs. `lang` is an optional
    /// three-letter code (deu/eng/fra/hin/ita/jpn/kor/por/rus/spa); null = language-free.
    pub fn token_spans(&self, text: &str, lang: Option<String>) -> String {
        serde_json::to_string(&self.tokenizer.segment(text, lang.as_deref()))
            .expect("span serialization")
    }

    /// Sentence `[start, end)` char spans as a JSON array of pairs.
    pub fn sentence_spans(&self, text: &str, lang: Option<String>) -> String {
        serde_json::to_string(&self.segmenter.segment(text, lang.as_deref()))
            .expect("span serialization")
    }
}

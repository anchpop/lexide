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
use unidic::UniDic;
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
    /// Starts empty and can be given the Japanese dictionary via
    /// [`Parsley::load_japanese_dictionary`]. Spaced languages need nothing loaded — their
    /// proposal is whitespace, which is exact and free, and identical to what the server
    /// pipeline uses. Only Japanese needs the 87MB artifact, which is why fetching it is
    /// the caller's choice.
    ///
    /// Until it is loaded, Japanese gets an all-NONE proposal rather than the whitespace
    /// one. Measured on a curriculum-trained checkpoint: 80.4 F1 with NONE against 33.5
    /// with whitespace, because whitespace on a language with no spaces does not say "no
    /// information", it asserts the sentence is a single token. With the dictionary
    /// loaded it is 92.7.
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

    /// Install the Japanese boundary dictionary (`onnx/jpn-unidic.bin`, ~87MB) fetched by
    /// the page. Optional and only affects Japanese; everything else is already exact.
    pub fn load_japanese_dictionary(&mut self, bytes: Vec<u8>) -> Result<(), JsError> {
        self.priors.set_unidic(UniDic::from_bytes(bytes).map_err(js_err)?);
        Ok(())
    }

    /// Whether the Japanese dictionary has been loaded.
    pub fn has_japanese_dictionary(&self) -> bool {
        self.priors.has_japanese()
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

//! wasm-bindgen surface for the lexide-speech worker. Grams and options cross as JSON, samples as a Float32Array.

use wasm_bindgen::prelude::*;

use crate::{Gram, Options, Voice};

#[wasm_bindgen(js_name = Voice)]
pub struct WebVoice(Voice);

fn options(json: &str) -> Result<Options, JsError> {
    Ok(serde_json::from_str(json)?)
}

impl WebVoice {
    fn parser(&self) -> Result<&crate::Espeak, JsError> {
        self.0
            .espeak()
            .ok_or_else(|| JsError::new("this voice reads yap grams"))
    }
}

#[wasm_bindgen(js_class = Voice)]
impl WebVoice {
    #[wasm_bindgen(constructor)]
    pub fn new(bundle: &[u8]) -> Result<WebVoice, JsError> {
        Ok(WebVoice(Voice::load(bundle).map_err(|e| JsError::new(&e))?))
    }

    #[wasm_bindgen(getter)]
    pub fn speakers(&self) -> Vec<String> {
        self.0.speakers.clone()
    }

    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.0.sample_rate
    }

    /// The espeak voice name for distilled voices, which speak text.
    #[wasm_bindgen(getter)]
    pub fn espeak(&self) -> Option<String> {
        self.0.espeak().map(|e| e.spec.voice.clone())
    }

    pub fn speak(&self, grams: &str, options_json: &str) -> Result<Vec<f32>, JsError> {
        let grams: Vec<Gram> = serde_json::from_str(grams)?;
        Ok(self
            .0
            .speak(&grams, &options(options_json)?)
            .map_err(|e| JsError::new(&e))?
            .samples)
    }

    /// The stretches of `text` to phonemize for `speak_text`.
    pub fn chunks(&self, text: &str) -> Result<Vec<String>, JsError> {
        Ok(self.parser()?.chunks(text))
    }

    /// Speaks text given espeak's output for each of its `chunks`.
    #[wasm_bindgen(js_name = speakText)]
    pub fn speak_text(
        &self,
        text: &str,
        ipa: Vec<String>,
        options_json: &str,
    ) -> Result<Vec<f32>, JsError> {
        let grams = self
            .parser()?
            .grams(text, &ipa)
            .map_err(|e| JsError::new(&e))?;
        Ok(self
            .0
            .speak(&grams, &options(options_json)?)
            .map_err(|e| JsError::new(&e))?
            .samples)
    }

    pub fn warmup(&self) -> Result<(), JsError> {
        self.0.warmup().map_err(|e| JsError::new(&e))
    }
}

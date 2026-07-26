//! Byte-level token boundary tagger — the tokenizer. A thin wrapper over the shared
//! [`ByteBioModel`](super::byte_bio) whose O/B/I spans are *tokens*: B = a token begins
//! here, I = token continues, O = not part of a token. Token spans are recovered as each
//! B and its trailing I run. This is the piece that replaces the LLM's implicit tokenizer;
//! see `tagger/model.py::CharBoundaryTagger`.

use std::path::Path;

use anyhow::Result;

use crate::segment::byte_bio::ByteBioModel;

pub struct CharTokenizer {
    model: ByteBioModel,
}

impl CharTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            model: ByteBioModel::load(path)?,
        })
    }

    /// Per-position O/B/I logits for `[LANG or BOS] + utf8(text) + [EOS]`.
    #[allow(dead_code)] // used by the parity test; not on any production path
    pub fn logits(&self, text: &str, lang: Option<&str>) -> Vec<[f32; 3]> {
        self.model.logits(text, lang)
    }

    /// Raw text -> token (start, end) char spans. The optional language hint improves
    /// ambiguous boundaries on lang-token checkpoints; harmless no-op on older ones.
    pub fn segment(&self, text: &str, lang: Option<&str>) -> Vec<(usize, usize)> {
        self.model.segment(text, lang)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::byte_bio::{argmax3, spans_from_byte_labels};
    use crate::segment::test_support::model_file;

    #[test]
    fn segment_uses_shared_span_recovery() {
        // "ab cd": B I O B I -> [(0,2),(3,5)] (guards the wrapper wiring, not the model).
        let labels = [0u8, 1, 2, 0, 1, 2, 0];
        assert_eq!(spans_from_byte_labels("ab cd", &labels), vec![(0, 2), (3, 5)]);
    }

    /// Bit-for-bit parity with the Python CharBoundaryTagger on multilingual fixtures
    /// (labels + spans exact, first-row logits numerically close). Skips when the model
    /// artifacts aren't present locally (they live on the lexide-onnx Modal volume).
    #[test]
    fn matches_python_reference_fixtures() {
        let (Some(weights), Some(fixtures)) = (
            model_file("char_tokenizer.safetensors"),
            model_file("char_tokenizer_fixtures.json"),
        ) else {
            eprintln!("skipping: char tokenizer artifacts not found (set LEXIDE_MODEL_DIR)");
            return;
        };
        let tok = CharTokenizer::load(&weights).unwrap();
        let fixtures: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixtures).unwrap()).unwrap();
        for fx in fixtures.as_array().unwrap() {
            let text = fx["text"].as_str().unwrap();
            // Older fixture files predate language conditioning and carry no lang key.
            let lang = fx.get("lang").and_then(|v| v.as_str());
            let want_labels: Vec<u8> = fx["byte_labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            let want_spans: Vec<(usize, usize)> = fx["spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize))
                .collect();

            let logits = tok.logits(text, lang);
            let labels: Vec<u8> = logits.iter().map(argmax3).collect();
            assert_eq!(labels, want_labels, "byte labels diverge for {text:?}");
            assert_eq!(tok.segment(text, lang), want_spans, "spans diverge for {text:?}");

            let want_first: Vec<f32> = fx["first_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            for (got, want) in logits[1].iter().zip(&want_first) {
                assert!(
                    (got - want).abs() < 1e-3,
                    "logits diverge for {text:?}: {:?} vs {want_first:?}",
                    logits[1]
                );
            }
        }
    }
}

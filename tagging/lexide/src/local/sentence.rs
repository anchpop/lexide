//! Byte-level sentence segmenter. A thin wrapper over the shared [`ByteBioModel`]
//! (super::byte_bio) whose O/B/I spans are *sentences*: B = a sentence begins here,
//! I = inside a sentence, O = a gap (whitespace, headings, separators between sentences).
//!
//! Trained by `sentence-labeller/train_segmenter.py` on LLM/mechanically labelled
//! passages; the same architecture, weight layout, and export path as the token
//! [`CharTokenizer`](super::chartok). Given a passage it recovers each sentence's char
//! span, so a raw text (or a list of texts) can be split into its sentences with the
//! gaps between them dropped — the whitespace/markers a sentence *frames* (its own quotes,
//! leading dashes) stay attached because they were labelled as part of the sentence.

use std::path::Path;

use anyhow::Result;

use super::byte_bio::ByteBioModel;

/// One segmented sentence: its text plus the `[start, end)` char (code-point) span it
/// occupies in the original passage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

pub struct SentenceSegmenter {
    model: ByteBioModel,
}

impl SentenceSegmenter {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            model: ByteBioModel::load(path)?,
        })
    }

    /// Per-position O/B/I logits for `[BOS] + utf8(text) + [EOS]` (exposed for parity tests).
    #[allow(dead_code)] // used by the parity test; not on any production path
    pub fn logits(&self, text: &str) -> Vec<[f32; 3]> {
        self.model.logits(text)
    }

    /// Sentence `[start, end)` char spans within `text`.
    pub fn spans(&self, text: &str) -> Vec<(usize, usize)> {
        self.model.segment(text)
    }

    /// Split a passage into its sentences, dropping the gaps between them. Char indexing
    /// is by code point, matching the span recovery.
    pub fn segment(&self, text: &str) -> Vec<Sentence> {
        let chars: Vec<char> = text.chars().collect();
        self.spans(text)
            .into_iter()
            .map(|(start, end)| Sentence {
                text: chars[start..end].iter().collect(),
                start,
                end,
            })
            .collect()
    }

    /// Convenience: just the sentence strings, in order.
    pub fn sentences(&self, text: &str) -> Vec<String> {
        self.segment(text).into_iter().map(|s| s.text).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::byte_bio::argmax3;
    use crate::local::test_support::model_file;

    /// Bit-for-bit parity with the Python segmenter on multilingual multi-sentence
    /// fixtures. Skips when the segmenter artifacts aren't present locally.
    #[test]
    fn matches_python_reference_fixtures() {
        let (Some(weights), Some(fixtures)) = (
            model_file("sentence_segmenter.safetensors"),
            model_file("sentence_segmenter_fixtures.json"),
        ) else {
            eprintln!("skipping: sentence segmenter artifacts not found (set LEXIDE_MODEL_DIR)");
            return;
        };
        let seg = SentenceSegmenter::load(&weights).unwrap();
        let fixtures: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixtures).unwrap()).unwrap();
        for fx in fixtures.as_array().unwrap() {
            let text = fx["text"].as_str().unwrap();
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
            let want_sentences: Vec<String> = fx["sentences"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();

            let logits = seg.logits(text);
            let labels: Vec<u8> = logits.iter().map(argmax3).collect();
            assert_eq!(labels, want_labels, "byte labels diverge for {text:?}");
            assert_eq!(seg.spans(text), want_spans, "spans diverge for {text:?}");
            assert_eq!(seg.sentences(text), want_sentences, "sentences diverge for {text:?}");

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

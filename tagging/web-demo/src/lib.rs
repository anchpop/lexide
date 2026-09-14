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

// Shared wire validation and nonblank-first CTC decoder from the lexide library.
#[path = "../../lexide/src/pronunciation/mod.rs"]
#[allow(dead_code)]
mod pronunciation;

/// Own the unpacked matrix in WASM rather than round-tripping float arrays through JS.
#[wasm_bindgen]
pub struct PronunciationMatrix {
    matrix: pronunciation::FrameMatrix,
}

#[wasm_bindgen]
impl PronunciationMatrix {
    #[wasm_bindgen(constructor)]
    pub fn new(payload: &str) -> Result<PronunciationMatrix, JsError> {
        let decode = || -> anyhow::Result<_> {
            let payload: pronunciation::FrameMatrixPayload = serde_json::from_str(payload)?;
            anyhow::ensure!(
                payload.shape.len() == 2
                    && payload.shape[0] > 0
                    && payload.shape[0] <= 2000
                    && payload.shape[1] >= 2
                    && payload.shape[1] <= 4096,
                "matrix dimensions exceed demo limits"
            );
            Ok(Self {
                matrix: pronunciation::FrameMatrix::decode(&payload)?,
            })
        };
        decode().map_err(|e| {
            JsError::new(&format!(
                "The model returned an invalid frame matrix: {e:#}"
            ))
        })
    }

    /// Stress stays on its original frame; only phone IDs determine CTC runs.
    pub fn decode(&self, frames: &str) -> Result<String, JsError> {
        self.decode_ui(frames).map_err(js_err)
    }
}

impl PronunciationMatrix {
    fn decode_ui(&self, frames: &str) -> anyhow::Result<String> {
        use serde_json::json;
        let mut frames: Vec<serde_json::Value> = serde_json::from_str(frames)
            .map_err(|_| anyhow::anyhow!("The model's phoneme and stress frames do not align."))?;
        anyhow::ensure!(
            frames.len() == self.matrix.frames
                && frames.iter().enumerate().all(|(i, frame)| {
                    frame.is_object()
                        && frame["frame"].as_u64() == Some(i as u64)
                        && matches!(frame["stress"].as_u64(), Some(0..=2))
                }),
            "The model's phoneme and stress frames do not align."
        );
        let decoded = self.matrix.decode_path()?;
        let columns = self.matrix.vocab.len();
        let values = self.matrix.log_probs();
        for (frame, &id) in decoded.path.iter().enumerate() {
            let blank = id == self.matrix.blank_id;
            let record = frames[frame]
                .as_object_mut()
                .expect("validated frame object");
            record.insert("id".into(), json!(id));
            record.insert(
                "phoneme".into(),
                json!(if blank {
                    "CTC blank"
                } else {
                    &self.matrix.vocab[id]
                }),
            );
            record.insert("blank".into(), json!(blank));
            record.insert(
                "probability".into(),
                json!(values[frame * columns + id].exp()),
            );
        }
        let phone_ids: Vec<_> = self
            .matrix
            .vocab
            .iter()
            .enumerate()
            .filter(|(id, token)| {
                *id != self.matrix.blank_id && pronunciation::is_phone_token(token)
            })
            .map(|(id, _)| id)
            .collect();
        let mut phones = Vec::with_capacity(decoded.runs.len());
        for run in decoded.runs {
            let mut sums = vec![0.0_f64; columns];
            for frame in run.start_frame..run.end_frame {
                let row = &values[frame * columns..(frame + 1) * columns];
                // Conditional phone probabilities, excluding blank/specials. Subtract
                // the maximum so valid but very negative phone logits never underflow.
                let max = phone_ids
                    .iter()
                    .map(|&id| row[id])
                    .fold(f32::NEG_INFINITY, f32::max) as f64;
                let mass: f64 = phone_ids
                    .iter()
                    .map(|&id| (row[id] as f64 - max).exp())
                    .sum();
                for &id in &phone_ids {
                    sums[id] += (row[id] as f64 - max).exp() / mass;
                }
            }
            let count = (run.end_frame - run.start_frame) as f64;
            let mut ranked: Vec<_> = phone_ids
                .iter()
                .copied()
                .filter(|&id| sums[id] > 0.0)
                .collect();
            ranked.sort_by(|&a, &b| sums[b].total_cmp(&sums[a]).then(a.cmp(&b)));
            let alternatives: Vec<_> = ranked
                .iter()
                .take(3)
                .map(|&id| {
                    json!({
                        "id": id, "phoneme": self.matrix.vocab[id], "probability": sums[id] / count,
                    })
                })
                .collect();
            phones.push(json!({"id": run.id, "phoneme": self.matrix.vocab[run.id],
                "startFrame": run.start_frame, "endFrame": run.end_frame,
                "confidence": sums[run.id] / count, "top_k": alternatives}));
        }
        Ok(serde_json::to_string(
            &json!({"path": frames, "phones": phones}),
        )?)
    }
}

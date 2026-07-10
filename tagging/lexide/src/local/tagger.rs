//! ONNX inference for the multi-task tagger — the Rust port of `predict.Pipeline.tag`.
//!
//! The graph (exported by `tagger/export_onnx.py`) is the XLM-R encoder + first-subword
//! word pooling + POS/lemma/biaffine heads in one forward:
//! (input_ids, attention_mask, word_first_sub, word_mask) ->
//! (pos_logits [1,W,P], lemma_logits [1,W,L], arc_scores [1,W,W+1], rel_scores [1,W,W+1,R]).
//! We run batch=1 with no padding, so the arc padding mask the export intentionally left
//! out of the graph is a no-op here.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;

use super::script::apply_script;

/// Matches `predict.Pipeline.tag`'s `max_length=192` subword truncation.
const MAX_SUBWORDS: usize = 192;

#[derive(Debug, Deserialize)]
pub struct Vocab {
    pub pos: Vec<String>,
    pub dep: Vec<String>,
    pub lemma_scripts: Vec<String>,
}

/// A tagged token with char offsets, before lemma-table resolution.
#[derive(Debug)]
pub struct TaggedToken {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub pos: String,
    pub lemma: String,
    pub dep: String,
    /// 0 = ROOT, else 1-indexed token position (parsley convention).
    pub head: i32,
}

pub struct OnnxTagger {
    // ort's Session::run takes &mut self (it binds outputs); analyze() takes &self.
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    vocab: Vocab,
}

impl OnnxTagger {
    pub fn load(model_dir: &Path, threads: usize) -> Result<Self> {
        let onnx = model_dir.join("tagger.onnx");
        // ort's builder errors carry the (non-Send) builder back, so map them eagerly.
        let mut builder = Session::builder().map_err(|e| anyhow!("ort session builder: {e}"))?;
        if threads > 0 {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|e| anyhow!("ort intra threads: {e}"))?;
        }
        let session = builder
            .commit_from_file(&onnx)
            .with_context(|| format!("failed to load ONNX tagger at {}", onnx.display()))?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("failed to load tokenizer.json: {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SUBWORDS,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("failed to set truncation: {e}"))?;

        let vocab: Vocab = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("vocab.json"))
                .context("failed to read vocab.json")?,
        )
        .context("failed to parse vocab.json")?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            vocab,
        })
    }

    /// Given token char spans (from the char tokenizer), produce per-token labels.
    /// Model-only lemmas — the caller applies the Wiktionary floor.
    pub fn tag(&self, text: &str, spans: &[(usize, usize)]) -> Result<Vec<TaggedToken>> {
        // Subword-encode with char offsets, exactly like the Python fast tokenizer call.
        let enc = self
            .tokenizer
            .encode_char_offsets(text, true)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?;
        let offsets = enc.get_offsets();

        // char index -> first subword covering it (special tokens report (0,0) and are skipped)
        let mut char_to_sub: HashMap<usize, usize> = HashMap::new();
        for (si, &(a, b)) in offsets.iter().enumerate() {
            if a == 0 && b == 0 {
                continue;
            }
            for c in a..b {
                char_to_sub.entry(c).or_insert(si);
            }
        }

        // Each of our tokens is represented by its first subword; tokens the encoder
        // truncated away (or that map to no subword) are dropped, as in Python.
        let mut word_first: Vec<i64> = Vec::with_capacity(spans.len());
        let mut keep: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
        for &(s, e) in spans {
            let sub = char_to_sub
                .get(&s)
                .or_else(|| char_to_sub.get(&e.wrapping_sub(1)));
            if let Some(&sub) = sub {
                word_first.push(sub as i64);
                keep.push((s, e));
            }
        }
        if word_first.is_empty() {
            return Ok(Vec::new());
        }

        let w = keep.len();
        let s_len = enc.get_ids().len();
        let input_ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
        let attention_mask: Vec<i64> = enc.get_attention_mask().iter().map(|&v| v as i64).collect();

        let mut session = self.session.lock().expect("tagger session poisoned");
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array((vec![1i64, s_len as i64], input_ids))?,
            "attention_mask" => Tensor::from_array((vec![1i64, s_len as i64], attention_mask))?,
            "word_first_sub" => Tensor::from_array((vec![1i64, w as i64], word_first))?,
            "word_mask" => Tensor::from_array((vec![1i64, w as i64], vec![1i64; w]))?,
        ])?;

        let (_, pos_logits) = outputs["pos_logits"].try_extract_tensor::<f32>()?;
        let (_, lemma_logits) = outputs["lemma_logits"].try_extract_tensor::<f32>()?;
        let (_, arc_scores) = outputs["arc_scores"].try_extract_tensor::<f32>()?;
        let (_, rel_scores) = outputs["rel_scores"].try_extract_tensor::<f32>()?;

        let n_pos = self.vocab.pos.len();
        let n_dep = self.vocab.dep.len();
        let n_lemma = self.vocab.lemma_scripts.len();
        if pos_logits.len() != w * n_pos
            || lemma_logits.len() != w * n_lemma
            || arc_scores.len() != w * (w + 1)
            || rel_scores.len() != w * (w + 1) * n_dep
        {
            bail!("ONNX output shapes don't match vocab sizes — model/vocab.json mismatch?");
        }

        let chars: Vec<char> = text.chars().collect();
        let mut result = Vec::with_capacity(w);
        for i in 0..w {
            let (s, e) = keep[i];
            let form: String = chars[s..e].iter().collect();
            let pos_idx = argmax(&pos_logits[i * n_pos..(i + 1) * n_pos]);
            let lemma_idx = argmax(&lemma_logits[i * n_lemma..(i + 1) * n_lemma]);
            // head over [ROOT, w1..wW]: 0 = ROOT, else 1-indexed token
            let head = argmax(&arc_scores[i * (w + 1)..(i + 1) * (w + 1)]);
            // relation scored at the predicted head
            let rel_base = (i * (w + 1) + head) * n_dep;
            let rel_idx = argmax(&rel_scores[rel_base..rel_base + n_dep]);

            let lemma = apply_script(&form, &self.vocab.lemma_scripts[lemma_idx]);
            result.push(TaggedToken {
                start: s,
                end: e,
                text: form,
                pos: self.vocab.pos[pos_idx].clone(),
                lemma,
                dep: self.vocab.dep[rel_idx].clone(),
                head: head as i32,
            });
        }
        Ok(result)
    }
}

/// First maximal index — matches torch.argmax's tie-breaking.
fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > xs[best] {
            best = i;
        }
    }
    best
}

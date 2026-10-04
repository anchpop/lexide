//! Single-sentence fp32 joint inference, matching predict_joint.py.
use super::{mst::single_root_mst, script::apply_script};
use crate::raw::RawToken;
use anyhow::{anyhow, ensure, Context, Result};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use std::path::Path;
use std::sync::Mutex;

#[derive(Deserialize)]
struct Vocab {
    pos: Vec<String>,
    dep: Vec<String>,
    lemma_scripts: Vec<String>,
    langs: Vec<String>,
}
#[derive(Deserialize)]
struct Config {
    max_subwords: usize,
    char_buckets: u32,
}

pub struct OnnxTagger {
    sessions: Mutex<(Session, Session)>,
    tokenizer: tokenizers::Tokenizer,
    vocab: Vocab,
    config: Config,
}

fn session(path: &Path, threads: usize) -> Result<Session> {
    let mut builder = Session::builder().map_err(|e| anyhow!("ort session builder: {e}"))?;
    if threads > 0 {
        builder = builder
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("ort threads: {e}"))?;
    }
    builder
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))
}

impl OnnxTagger {
    pub fn load(dir: &Path, threads: usize) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!("tokenizer truncation: {e}"))?;
        tokenizer.with_padding(None);
        Ok(Self {
            sessions: Mutex::new((
                session(&dir.join("encoder.onnx"), threads)?,
                session(&dir.join("heads.onnx"), threads)?,
            )),
            tokenizer,
            vocab: serde_json::from_slice(&std::fs::read(dir.join("vocab.json"))?)?,
            config: serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?,
        })
    }

    pub fn tag(&self, text: &str, lang: &str) -> Result<Vec<RawToken>> {
        let characters: Vec<char> = text.chars().collect();
        let c = characters.len();
        if characters.iter().all(|&ch| is_space(ch)) {
            return Ok(Vec::new());
        }
        let enc = self
            .tokenizer
            .encode_char_offsets(text, true)
            .map_err(|e| anyhow!("tokenization: {e}"))?;
        let s = enc.len();
        ensure!(
            s <= self.config.max_subwords,
            "Sentence has {s} subwords; maximum is {}. Split passages into sentences first.",
            self.config.max_subwords
        );
        let mut mapping = vec![-1i64; c];
        let mut first = vec![false; c];
        for (i, &(a, b)) in enc.get_offsets().iter().enumerate() {
            if a < b {
                if a < c {
                    first[a] = true;
                }
                for value in &mut mapping[a.min(c)..b.min(c)] {
                    if *value < 0 {
                        *value = i as i64;
                    }
                }
            }
        }
        let features: Vec<i64> = characters
            .iter()
            .enumerate()
            .map(|(i, &ch)| {
                i64::from(is_space(ch)) + 2 * i64::from(first[i]) + 4 * i64::from(mapping[i] < 0)
            })
            .collect();
        let char_ids: Vec<i64> = characters
            .iter()
            .map(|&ch| i64::from(ch as u32 % self.config.char_buckets + 1))
            .collect();
        let lang = self
            .vocab
            .langs
            .iter()
            .position(|l| l == lang)
            .map_or(0, |i| i as i64 + 1);
        let mut sessions = self.sessions.lock().expect("joint sessions poisoned");
        let (encoder, heads) = &mut *sessions;
        let encoded = encoder.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1, s], enc.get_ids().iter().map(|&v| i64::from(v)).collect::<Vec<_>>()))?,
            "attention_mask" => Tensor::from_array(([1, s], vec![1i64; s]))?,
            "char_to_sub" => Tensor::from_array(([1, c], mapping))?,
            "char_ids" => Tensor::from_array(([1, c], char_ids))?,
            "char_features" => Tensor::from_array(([1, c], features))?,
            "lang_id" => Tensor::from_array(([1], vec![lang]))?,
        ])?;
        let (_, logits) = encoded["boundary_logits"].try_extract_tensor::<f32>()?;
        let labels: Vec<_> = logits.chunks_exact(3).map(argmax).collect();
        let spans = spans_from_char_labels(&characters, &labels);
        let w = spans.len();
        let starts: Vec<i64> = spans.iter().map(|s| s.0 as i64).collect();
        let ends: Vec<i64> = spans.iter().map(|s| s.1 as i64).collect();
        let output = heads.run(ort::inputs![
            "chars" => &encoded["chars"],
            "sub_at_char" => &encoded["sub_at_char"],
            "starts" => Tensor::from_array(([1, w], starts))?,
            "ends" => Tensor::from_array(([1, w], ends))?,
        ])?;
        let (_, pos) = output["pos_logits"].try_extract_tensor::<f32>()?;
        let (_, lemma) = output["lemma_logits"].try_extract_tensor::<f32>()?;
        let (_, arcs) = output["arc_scores"].try_extract_tensor::<f32>()?;
        let (_, rels) = output["rel_scores"].try_extract_tensor::<f32>()?;
        let p = self.vocab.pos.len();
        let l = self.vocab.lemma_scripts.len();
        let r = self.vocab.dep.len();
        ensure!(
            pos.len() == w * p
                && lemma.len() == w * l
                && arcs.len() == w * (w + 1)
                && rels.len() == w * (w + 1) * r,
            "ONNX output shapes do not match vocab"
        );
        let parents = single_root_mst(arcs, w)?;
        Ok(spans
            .into_iter()
            .enumerate()
            .map(|(i, (start, end))| {
                let text: String = characters[start..end].iter().collect();
                let rel = (i * (w + 1) + parents[i]) * r;
                RawToken {
                    lemma: apply_script(
                        &text,
                        &self.vocab.lemma_scripts[argmax(&lemma[i * l..(i + 1) * l])],
                    ),
                    text,
                    start,
                    end,
                    pos: self.vocab.pos[argmax(&pos[i * p..(i + 1) * p])].clone(),
                    dep: self.vocab.dep[argmax(&rels[rel..rel + r])].clone(),
                    head: parents[i] as i32,
                }
            })
            .collect())
    }
}

// Python str.isspace additionally includes these four information separators.
fn is_space(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

fn spans_from_char_labels(text: &[char], labels: &[usize]) -> Vec<(usize, usize)> {
    let label = |i: usize| labels.get(i).copied().unwrap_or(0);
    let mut spans = Vec::new();
    let mut start = None;
    for (i, &ch) in text.iter().enumerate() {
        if is_space(ch) {
            let Some(s) = start else {
                continue;
            };
            let mut j = i;
            while j < text.len() && is_space(text[j]) {
                j += 1;
            }
            if !(label(i) == 2 && j < text.len() && label(j) == 2) {
                spans.push((s, i));
                start = None;
            }
        } else if start.is_none() || label(i) == 1 {
            if let Some(s) = start {
                spans.push((s, i));
            }
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_dropped_text() {
        for (text, labels, expected) in [
            (
                "你好吗 x",
                vec![2, 1, 2, 0, 1],
                vec![(0, 1), (1, 3), (4, 5)],
            ),
            ("ab cd", vec![1, 2, 0, 0, 0], vec![(0, 2), (3, 5)]),
            ("New York", vec![1, 2, 2, 2, 2, 2, 2, 2], vec![(0, 8)]),
            ("ab cd", vec![1, 2, 2, 1, 2], vec![(0, 2), (3, 5)]),
            ("你好", vec![1], vec![(0, 2)]),
            ("", vec![], vec![]),
        ] {
            assert_eq!(
                spans_from_char_labels(&text.chars().collect::<Vec<_>>(), &labels),
                expected
            );
        }
    }
}

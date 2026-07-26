//! Shared byte-level bidirectional minGRU — a pure-Rust reimplementation of the
//! `CharBoundaryTagger` in `tagger/model.py` (~0.31M params, so no ML runtime needed;
//! the sequential-scan recurrence is also what keeps it out of the ONNX graph).
//!
//! Two models share this exact architecture and weight layout, differing only in what
//! their O/B/I spans mean:
//!   * [`super::chartok::CharTokenizer`] — B/I/O over *tokens* (the tokenizer).
//!   * [`super::sentence::SentenceSegmenter`] — B/I/O over *sentences* (the segmenter).
//!
//! Input is `[BOS] + utf8(text) + [EOS]` byte ids; output is a per-byte O/B/I label read
//! at each character's first byte to recover char spans (see [`spans_from_byte_labels`]).

use std::path::Path;

use anyhow::{bail, Context, Result};

const BOS_BYTE: usize = 257;
const EOS_BYTE: usize = 258;
const VOCAB: usize = 259; // 256 byte values + PAD(256) + BOS + EOS
const LN_EPS: f32 = 1e-5; // torch LayerNorm default

/// Language-conditioned BOS rows appended after the base vocab, in this fixed order
/// (matches `tagger/dataset.py::LANG_ORDER`). A checkpoint with 259 embedding rows is
/// language-blind; one with 259+10 accepts a language hint via its first token.
pub const LANG_ORDER: [&str; 10] = [
    "deu", "eng", "fra", "hin", "ita", "jpn", "kor", "por", "rus", "spa",
];

fn lang_index(code: &str) -> Option<usize> {
    LANG_ORDER.iter().position(|&l| l == code)
}

struct Linear {
    w: Vec<f32>,  // [out, in] row-major, as torch stores it
    wt: Vec<f32>, // [in, out] transpose — for the vectorizable axpy path in apply_all
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        for (o, out_o) in out.iter_mut().enumerate() {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            let mut acc = self.b[o];
            for (wi, xi) in row.iter().zip(x) {
                acc += wi * xi;
            }
            *out_o = acc;
        }
    }

    /// [len, in] -> [len, out] for a whole sequence. Axpy formulation over the transposed
    /// weights: `out[t] += x[t][i] * wt[i]` — contiguous writes with no reduction, so it
    /// vectorizes, and tiling over timesteps keeps each weight column's traffic to once
    /// per tile (at ~1M params the weights no longer fit in cache; the naive per-timestep
    /// order was memory-bound). Per-element accumulation order (bias + ascending i) is
    /// identical to `apply`, so results are bit-for-bit the same.
    fn apply_all(&self, xs: &[f32], len: usize) -> Vec<f32> {
        const TILE: usize = 32;
        let (in_d, out_d) = (self.in_dim, self.out_dim);
        let mut out = Vec::with_capacity(len * out_d);
        for _ in 0..len {
            out.extend_from_slice(&self.b);
        }
        for t0 in (0..len).step_by(TILE) {
            let t1 = (t0 + TILE).min(len);
            for i in 0..in_d {
                let col = &self.wt[i * out_d..(i + 1) * out_d];
                for t in t0..t1 {
                    let xi = xs[t * in_d + i];
                    let row = &mut out[t * out_d..(t + 1) * out_d];
                    for (oj, cj) in row.iter_mut().zip(col) {
                        *oj += xi * cj;
                    }
                }
            }
        }
        out
    }
}

/// minGRU (Feng et al.): z = sigmoid(Wz x), h_cand = Wh x, h = (1-z)*h + z*h_cand.
/// The candidate doesn't depend on the hidden state, so the scan is a cheap loop.
struct MinGru {
    to_z: Linear,
    to_h: Linear,
}

impl MinGru {
    fn hidden(&self) -> usize {
        self.to_z.out_dim
    }

    /// xs: [L, in_dim] flattened. Writes each step's hidden state into
    /// out[t * stride + offset ..][..hidden] (so fwd/bwd can interleave into one buffer).
    /// Projections are batched over the whole sequence (they don't depend on the hidden
    /// state); only the cheap elementwise recurrence runs sequentially.
    fn scan(&self, xs: &[f32], len: usize, reverse: bool, out: &mut [f32], stride: usize, offset: usize) {
        let h_dim = self.hidden();
        let z_all = self.to_z.apply_all(xs, len);
        let cand_all = self.to_h.apply_all(xs, len);
        let mut h = vec![0.0f32; h_dim];
        for i in 0..len {
            let t = if reverse { len - 1 - i } else { i };
            let z = &z_all[t * h_dim..(t + 1) * h_dim];
            let cand = &cand_all[t * h_dim..(t + 1) * h_dim];
            for j in 0..h_dim {
                let zj = 1.0 / (1.0 + (-z[j]).exp());
                h[j] = (1.0 - zj) * h[j] + zj * cand[j];
            }
            out[t * stride + offset..t * stride + offset + h_dim].copy_from_slice(&h);
        }
    }
}

struct BiMinGru {
    fwd: MinGru,
    bwd: MinGru,
}

impl BiMinGru {
    /// [L, in] -> [L, 2*hidden] (forward states then backward states, as torch.cat does)
    fn forward(&self, xs: &[f32], len: usize) -> Vec<f32> {
        let h = self.fwd.hidden();
        let mut out = vec![0.0f32; len * 2 * h];
        self.fwd.scan(xs, len, false, &mut out, 2 * h, 0);
        self.bwd.scan(xs, len, true, &mut out, 2 * h, h);
        out
    }
}

/// The full byte tagger: embedding, N BiMinGRU layers, LayerNorm, linear to O/B/I logits.
pub struct ByteBioModel {
    emb: Vec<f32>, // [VOCAB (+ n_langs), emb_dim]
    emb_dim: usize,
    n_langs: usize, // 0 = language-blind checkpoint
    layers: Vec<BiMinGru>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    out: Linear,
}

impl ByteBioModel {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("failed to read byte-minGRU weights at {}", path.display()))?;
        Self::from_bytes(&data)
    }

    /// Load from in-memory safetensors bytes (e.g. fetched over HTTP in the wasm demo).
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let st = safetensors::SafeTensors::deserialize(data)
            .context("failed to parse byte-minGRU safetensors")?;

        let tensor = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let view = st
                .tensor(name)
                .with_context(|| format!("byte-minGRU weights missing tensor {name}"))?;
            if view.dtype() != safetensors::Dtype::F32 {
                bail!("tensor {name} is {:?}, expected F32", view.dtype());
            }
            let vals = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok((view.shape().to_vec(), vals))
        };
        let linear = |prefix: &str| -> Result<Linear> {
            let (wshape, w) = tensor(&format!("{prefix}.weight"))?;
            let (_, b) = tensor(&format!("{prefix}.bias"))?;
            let (out_dim, in_dim) = (wshape[0], wshape[1]);
            let mut wt = vec![0.0f32; w.len()];
            for o in 0..out_dim {
                for i in 0..in_dim {
                    wt[i * out_dim + o] = w[o * in_dim + i];
                }
            }
            Ok(Linear {
                in_dim,
                out_dim,
                w,
                wt,
                b,
            })
        };

        let (emb_shape, emb) = tensor("emb.weight")?;
        if emb_shape[0] < VOCAB || emb_shape[0] > VOCAB + LANG_ORDER.len() {
            bail!(
                "unexpected byte vocab size {} (expected {VOCAB}..={} — base vocab plus \
                 optional language tokens)",
                emb_shape[0],
                VOCAB + LANG_ORDER.len()
            );
        }
        let n_langs = emb_shape[0] - VOCAB;
        let mut layers = Vec::new();
        for i in 0.. {
            if st.tensor(&format!("layers.{i}.fwd.to_z.weight")).is_err() {
                break;
            }
            layers.push(BiMinGru {
                fwd: MinGru {
                    to_z: linear(&format!("layers.{i}.fwd.to_z"))?,
                    to_h: linear(&format!("layers.{i}.fwd.to_h"))?,
                },
                bwd: MinGru {
                    to_z: linear(&format!("layers.{i}.bwd.to_z"))?,
                    to_h: linear(&format!("layers.{i}.bwd.to_h"))?,
                },
            });
        }
        if layers.is_empty() {
            bail!("byte-minGRU weights contain no BiMinGRU layers");
        }
        let (_, norm_w) = tensor("norm.weight")?;
        let (_, norm_b) = tensor("norm.bias")?;
        let out = linear("out")?;

        Ok(Self {
            emb_dim: emb_shape[1],
            emb,
            n_langs,
            layers,
            norm_w,
            norm_b,
            out,
        })
    }

    /// Per-position O/B/I logits for `[LANG or BOS] + utf8(text) + [EOS]`. `lang` is a
    /// three-letter code from [`LANG_ORDER`]; unknown codes — or any code on a
    /// language-blind checkpoint — fall back to the generic BOS.
    pub fn logits(&self, text: &str, lang: Option<&str>) -> Vec<[f32; 3]> {
        let first = lang
            .and_then(lang_index)
            .filter(|&i| i < self.n_langs)
            .map(|i| VOCAB + i)
            .unwrap_or(BOS_BYTE);
        let mut ids: Vec<usize> = Vec::with_capacity(text.len() + 2);
        ids.push(first);
        ids.extend(text.as_bytes().iter().map(|&b| b as usize));
        ids.push(EOS_BYTE);
        let len = ids.len();

        let mut h: Vec<f32> = Vec::with_capacity(len * self.emb_dim);
        for id in ids {
            h.extend_from_slice(&self.emb[id * self.emb_dim..(id + 1) * self.emb_dim]);
        }
        for layer in &self.layers {
            h = layer.forward(&h, len);
        }

        // LayerNorm over the feature dim, then the output projection
        let d = self.norm_w.len();
        let mut logits = Vec::with_capacity(len);
        let mut normed = vec![0.0f32; d];
        for t in 0..len {
            let x = &h[t * d..(t + 1) * d];
            let mean = x.iter().sum::<f32>() / d as f32;
            let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + LN_EPS).sqrt();
            for j in 0..d {
                normed[j] = (x[j] - mean) * inv * self.norm_w[j] + self.norm_b[j];
            }
            let mut row = [0.0f32; 3];
            self.out.apply(&normed, &mut row);
            logits.push(row);
        }
        logits
    }

    /// Raw text -> (start, end) char spans (each B and its trailing I run).
    pub fn segment(&self, text: &str, lang: Option<&str>) -> Vec<(usize, usize)> {
        let labels: Vec<u8> = self.logits(text, lang).iter().map(argmax3).collect();
        spans_from_byte_labels(text, &labels)
    }
}

pub fn argmax3(row: &[f32; 3]) -> u8 {
    let mut best = 0;
    for i in 1..3 {
        if row[i] > row[best] {
            best = i;
        }
    }
    best as u8
}

/// Map per-byte O/B/I labels back to character spans, mirroring
/// `predict.spans_from_byte_labels`: labels align to `[BOS] + utf8(text) + [EOS]` and each
/// character reads the label at its first byte. Spans are char (code point) indices.
pub fn spans_from_byte_labels(text: &str, labels: &[u8]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut pos = 1; // skip BOS
    let mut cur_start: Option<usize> = None;
    let mut n_chars = 0;
    for (ci, ch) in text.chars().enumerate() {
        n_chars = ci + 1;
        let lab = labels.get(pos).copied().unwrap_or(0);
        match lab {
            1 => {
                // B: close any open span, start a new one
                if let Some(s) = cur_start {
                    spans.push((s, ci));
                }
                cur_start = Some(ci);
            }
            0 => {
                // O: close any open span
                if let Some(s) = cur_start.take() {
                    spans.push((s, ci));
                }
            }
            _ => {} // I: continue
        }
        pos += ch.len_utf8();
    }
    if let Some(s) = cur_start {
        spans.push((s, n_chars));
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_basic() {
        // "ab cd": B I O B I -> [(0,2),(3,5)]
        let labels = [0, 1, 2, 0, 1, 2, 0]; // BOS a b ' ' c d EOS
        assert_eq!(spans_from_byte_labels("ab cd", &labels), vec![(0, 2), (3, 5)]);
    }

    #[test]
    fn spans_adjacent_and_trailing_open() {
        // B at "b" closes the open "a" span; trailing I run closes at end of text.
        let labels = [0, 1, 1, 2, 0]; // BOS a b b EOS -> "a" then "bb"
        assert_eq!(spans_from_byte_labels("abb", &labels), vec![(0, 1), (1, 3)]);
    }

    #[test]
    fn spans_multibyte_reads_first_byte_label() {
        // "яб": я is 2 bytes; its label is at byte pos 1, б's at pos 3.
        let labels = [0, 1, 0, 2, 0]; // BOS я(2b) б(2b) EOS -> one span covering both chars
        assert_eq!(spans_from_byte_labels("яб", &labels), vec![(0, 2)]);
    }
}

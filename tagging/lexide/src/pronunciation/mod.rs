//! Score a phoneme sequence against the model's per-frame distribution.
//!
//! Greedy decode + edit distance asks "what did the model think it heard,
//! and does that string match?" — it throws the distribution away and turns
//! every soft disagreement into a hard edit. CTC asks the question we care
//! about directly: the total probability, over every frame alignment, that
//! the audio spells *this* target. Mass the model put on the right phoneme
//! still counts when it lost the argmax, and no per-language equivalence
//! table is needed for the number to mean something.
//!
//! The endpoint returns the whole `(T, V)` log-prob matrix (fp16, zlib) so
//! the forward pass is paid once per clip and any sequence — a re-segmented
//! sentence, a corrected subtitle, a rebuilt espeak reference — is scored
//! here, locally, without a GPU. CTC likelihood and Viterbi alignment use
//! the original joint log-probabilities. Free decoding instead makes the
//! nonblank decision first, then chooses a phone: splitting speech mass
//! between several phones must not turn speech into a blank.
//!
//! Ported from yap's `phoneme-verify/src/ctc.rs`. This module performs no
//! inference; enable it with the `pronunciation` feature. Optional networking
//! lives in `pronunciation::remote` behind `pronunciation-remote`.

use anyhow::{bail, Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cache-key decoder revision. Any change to decoding must bump this version.
pub const DECODER_VERSION: &str = "nonblank_v1";
mod frame_matrix;
pub use frame_matrix::*;
mod training_labels;
pub use training_labels::{training_labels, TrainingLabels};

/// Identity reported by the serving container's `marker_only` probe.
/// Model fields are required: missing identity must never produce a cache key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelIdentity {
    pub model_id: String,
    pub model_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deploy_marker: Option<String>,
}

/// Model-specific cache partition, compatible with yap's pronunciation cache.
pub fn cache_version(identity: &ModelIdentity) -> String {
    format!(
        "{}@{}__{}",
        identity.model_id.replace('/', "_"),
        identity.model_revision.chars().take(12).collect::<String>(),
        DECODER_VERSION,
    )
}

/// One clip of mono audio for the hosted phonemizer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PredictRequest {
    /// Standard base64 of little-endian float32 samples, not an encoded audio file.
    pub audio_f32_b64: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_phonemes: Option<Vec<String>>,
    #[serde(default)]
    pub return_frame_matrix: bool,
    /// Include all checkpoint auxiliary heads in the matrix, regardless of language.
    #[serde(default, skip_serializing_if = "is_false")]
    pub return_all_heads: bool,
    #[serde(default)]
    pub return_frames: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn default_sample_rate() -> u32 {
    16000
}
fn default_top_k() -> usize {
    3
}

impl PredictRequest {
    /// Encode mono samples as little-endian float32 base64, with 16 kHz defaults.
    /// Set `sample_rate` if the source differs; the server validates the waveform.
    pub fn from_samples(samples: &[f32]) -> Self {
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        Self {
            audio_f32_b64: base64::engine::general_purpose::STANDARD.encode(bytes),
            ..Self::default()
        }
    }
}

impl Default for PredictRequest {
    fn default() -> Self {
        Self {
            audio_f32_b64: String::new(),
            sample_rate: default_sample_rate(),
            top_k: default_top_k(),
            language: None,
            target_phonemes: None,
            return_frame_matrix: false,
            return_all_heads: false,
            return_frames: false,
        }
    }
}

/// A bare-phone alternative; stress is reported separately on the emission.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhonemeAlternative {
    pub phoneme: String,
    pub probability: f64,
}

/// One CTC-collapsed phoneme, with stress embedded in `phoneme`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmittedPhoneme {
    pub phoneme: String,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub top_k: Vec<PhonemeAlternative>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stress: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tone: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_accent: Option<usize>,
}

/// Diagnostic phone-only top-k for a frame, including non-emitting frames.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PredictionFrame {
    pub frame: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stress: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p_nonblank: Option<f64>,
    #[serde(default)]
    pub top_k: Vec<PhonemeAlternative>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tone: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_accent: Option<usize>,
}

/// The endpoint cannot score a target with no in-vocabulary phones.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetScoreError {
    pub error: String,
    #[serde(default)]
    pub oov: Vec<String>,
}

/// Server target score, including its all-OOV error response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PredictTargetScore {
    Error(TargetScoreError),
    Score(TargetScore),
}

/// Successful prediction. `phonemes` is required so error objects cannot be
/// mistaken for successful silence; optional diagnostics may be absent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_version: Option<String>,
    pub phonemes: Vec<EmittedPhoneme>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_score: Option<PredictTargetScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_matrix: Option<FrameMatrixPayload>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<Vec<PredictionFrame>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deploy_marker: Option<String>,
}

/// A rejected clip within an otherwise successful batch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PredictionError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Batch entries retain errors at their original request index.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BatchResult {
    Error { error: PredictionError },
    Prediction(PredictResponse),
}

/// Ordered batch results. The marker is stamped on the envelope, not each item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_version: Option<String>,
    pub results: Vec<BatchResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deploy_marker: Option<String>,
}

#[cfg(feature = "pronunciation-remote")]
pub mod remote;

/// A contiguous nonblank run. Both indices are zero-based; end is exclusive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PhoneRun {
    pub id: usize,
    pub start_frame: usize,
    pub end_frame: usize,
}

/// Per-frame labels (including blanks) and CTC-collapsed nonblank runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodedPath {
    pub path: Vec<usize>,
    pub runs: Vec<PhoneRun>,
}

/// Whether a vocab label represents a phone rather than a special token.
/// Blank IDs must additionally be excluded, regardless of their label.
pub fn is_phone_token(token: &str) -> bool {
    !token.is_empty() && token != "|" && !(token.starts_with('<') && token.ends_with('>'))
}

fn matrix_len(frames: usize, width: usize, blank_id: usize) -> Result<usize> {
    if width == 0 || blank_id >= width {
        bail!("frame matrix blank id {blank_id} is outside vocab width {width}");
    }
    frames
        .checked_mul(width)
        .context("frame matrix shape overflows")
}

fn validate_probs(log_probs: &[f32], frames: usize, width: usize, blank_id: usize) -> Result<()> {
    let expected = matrix_len(frames, width, blank_id)?;
    if log_probs.len() != expected {
        bail!(
            "frame matrix holds {} values, expected {expected}",
            log_probs.len()
        );
    }
    // Do not enforce row sums: fp16 quantization changes them slightly.
    if log_probs.iter().any(|&p| p.is_nan() || p > 0.0) {
        bail!("frame matrix log-probabilities must be nonpositive and not NaN");
    }
    if log_probs
        .chunks_exact(width)
        .any(|row| row.iter().all(|p| *p == f32::NEG_INFINITY))
    {
        bail!("frame matrix row has no probability mass");
    }
    Ok(())
}

/// Decode row-major joint log-probabilities with a nonblank-first decision.
///
/// Emit blank when its log-probability is at least `ln(0.5)`. Otherwise choose
/// the highest-probability finite nonblank slot (ties choose the lowest ID).
/// This fixes the split-mass regression: P(nonblank)=0.6 and conditional
/// phone probabilities 0.4/0.3/0.3 give joint probabilities 0.24/0.18/0.18;
/// a joint argmax incorrectly chooses blank (0.4) rather than the first phone.
/// With P(nonblank)=0.4, or exactly 0.5, the result is blank.
///
/// Masked `-inf` slots cannot be phones. A speech frame without a finite phone
/// candidate is invalid. This helper has no vocab; use [`FrameMatrix::decode_path`]
/// to exclude special labels too. Blanks reset repeat collapsing. Zero frames
/// yield an empty path; invalid dimensions, NaN, positive log-probabilities,
/// and rows with no mass are errors.
pub fn decode_path(
    log_probs: &[f32],
    frames: usize,
    vocab_size: usize,
    blank_id: usize,
) -> Result<DecodedPath> {
    decode_path_with(log_probs, frames, vocab_size, blank_id, |_| true)
}

fn decode_path_with(
    log_probs: &[f32],
    frames: usize,
    width: usize,
    blank_id: usize,
    is_phone: impl Fn(usize) -> bool,
) -> Result<DecodedPath> {
    validate_probs(log_probs, frames, width, blank_id)?;
    let mut path = Vec::with_capacity(frames);
    let mut runs: Vec<PhoneRun> = Vec::new();
    for (t, row) in log_probs.chunks_exact(width).enumerate() {
        let mut best = blank_id;
        if row[blank_id] < 0.5f32.ln() {
            let mut best_lp = f32::NEG_INFINITY;
            for (id, &lp) in row.iter().enumerate() {
                if id != blank_id && is_phone(id) && lp > best_lp {
                    best = id;
                    best_lp = lp;
                }
            }
            if best == blank_id {
                bail!("speech frame {t} has no finite phone candidate");
            }
        }
        if best != blank_id {
            if path.last() == Some(&best) {
                runs.last_mut()
                    .expect("previous nonblank has a run")
                    .end_frame = t + 1;
            } else {
                runs.push(PhoneRun {
                    id: best,
                    start_frame: t,
                    end_frame: t + 1,
                });
            }
        }
        path.push(best);
    }
    Ok(DecodedPath { path, runs })
}

/// The frame matrix exactly as the endpoint ships it: compressed, so a cache
/// entry stays ~24 KB per audio-second.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyFrameMatrixPayload {
    /// `[T, V]`.
    pub shape: Vec<usize>,
    pub dtype: String,
    pub encoding: String,
    pub blank_id: usize,
    /// Row labels, from the tokenizer's own vocab (not `decode`, which
    /// renders 78 of 461 entries differently and would misindex rows).
    pub vocab: Vec<String>,
    /// zlib-compressed little-endian fp16, row-major.
    pub data: String,
}

/// Decoded per-frame log-probabilities.
#[derive(Debug, Clone)]
pub struct FrameMatrix {
    /// None for legacy artifacts; no provenance or timing is fabricated.
    pub schema_version: Option<u32>,
    pub producer: Option<FrameMatrixProducer>,
    pub trained_against_g2p: Option<String>,
    pub frame_rate_ms: Option<f64>,
    pub sample_rate: Option<u32>,
    /// Empty for legacy artifacts. Scoring still uses the original phone matrix.
    pub heads: HashMap<String, FrameHead>,
    pub frames: usize,
    pub vocab: Vec<String>,
    pub blank_id: usize,
    index: HashMap<String, usize>,
    /// Row-major `[frames × vocab.len()]`.
    log_probs: Vec<f32>,
}

impl FrameMatrix {
    fn decode_legacy(payload: &LegacyFrameMatrixPayload) -> Result<Self> {
        if payload.dtype != "float16" || payload.encoding != "zlib+base64" {
            bail!(
                "unsupported frame matrix format {}/{}",
                payload.dtype,
                payload.encoding
            );
        }
        let [frames, width] = payload.shape[..] else {
            bail!("frame matrix shape {:?} is not [T, V]", payload.shape);
        };
        if width != payload.vocab.len() {
            bail!(
                "frame matrix has {width} columns but {} vocab entries",
                payload.vocab.len()
            );
        }
        matrix_len(frames, width, payload.blank_id)?;
        let log_probs = frame_matrix::decode_values(
            &payload.shape,
            &payload.dtype,
            &payload.encoding,
            &payload.data,
        )?;
        let index: HashMap<String, usize> = payload
            .vocab
            .iter()
            .enumerate()
            .map(|(i, tok)| (tok.clone(), i))
            .collect();
        if index.len() != width {
            bail!("frame matrix vocab contains duplicate labels");
        }
        let matrix = Self {
            schema_version: None,
            producer: None,
            trained_against_g2p: None,
            frame_rate_ms: None,
            sample_rate: None,
            heads: HashMap::new(),
            frames,
            vocab: payload.vocab.clone(),
            blank_id: payload.blank_id,
            index,
            log_probs,
        };
        matrix.decode_path()?;
        Ok(matrix)
    }

    #[inline]
    fn lp(&self, t: usize, v: usize) -> f64 {
        f64::from(self.log_probs[t * self.vocab.len() + v])
    }

    /// Vocab id of a phoneme token, if the model knows it.
    pub fn id(&self, token: &str) -> Option<usize> {
        self.index.get(token).copied()
    }

    /// Original joint log-probabilities, row-major; never renormalized.
    pub fn log_probs(&self) -> &[f32] {
        &self.log_probs
    }

    /// Vocab-aware nonblank-first decoding; excludes special labels and `-inf`.
    /// See [`decode_path`] for threshold and collapse semantics.
    pub fn decode_path(&self) -> Result<DecodedPath> {
        decode_path_with(
            &self.log_probs,
            self.frames,
            self.vocab.len(),
            self.blank_id,
            |id| is_phone_token(&self.vocab[id]),
        )
    }

    /// The model's own reading, with blanks and repeats collapsed.
    ///
    /// Uses the nonblank-first rule of [`decode_path`], not joint argmax:
    /// P(nonblank)=0.6 with conditional phones 0.4/0.3/0.3 emits the first
    /// phone, even though blank (0.4) exceeds every joint phone probability.
    /// P(nonblank)=0.4 and exactly 0.5 emit blank, resetting repeat collapse.
    /// Special labels and masked `-inf` slots never become phones.
    ///
    /// The matrix must retain the valid metadata established by [`Self::decode`].
    pub fn greedy_ids(&self) -> Vec<usize> {
        self.decode_path()
            .expect("valid decoded frame matrix")
            .runs
            .into_iter()
            .map(|run| run.id)
            .collect()
    }

    fn valid_target(&self, target: &[usize]) -> bool {
        !target.is_empty()
            && target.len() <= self.frames
            && target.iter().all(|&id| {
                id != self.blank_id
                    && self
                        .vocab
                        .get(id)
                        .is_some_and(|token| is_phone_token(token))
            })
    }

    /// `log P(target | audio)` summed over all CTC alignments — the standard
    /// forward recursion in log space over the blank-interleaved target.
    /// `None` when the target is empty, includes blank/special/out-of-range
    /// IDs, or has no possible alignment within the available frames.
    pub fn log_likelihood(&self, target: &[usize]) -> Option<f64> {
        if !self.valid_target(target) {
            return None;
        }
        // Extended label sequence: blank, l1, blank, l2, …, blank.
        let ext_len = 2 * target.len() + 1;
        let label = |s: usize| -> usize {
            if s.is_multiple_of(2) {
                self.blank_id
            } else {
                target[s / 2]
            }
        };
        let mut alpha = vec![f64::NEG_INFINITY; ext_len];
        alpha[0] = self.lp(0, self.blank_id);
        if ext_len > 1 {
            alpha[1] = self.lp(0, label(1));
        }
        let mut next = vec![f64::NEG_INFINITY; ext_len];
        for t in 1..self.frames {
            for (s, slot) in next.iter_mut().enumerate() {
                let mut acc = alpha[s];
                if s >= 1 {
                    acc = log_add(acc, alpha[s - 1]);
                }
                // A skip over the blank is allowed between two different labels.
                if s >= 2 && s % 2 == 1 && label(s) != label(s - 2) {
                    acc = log_add(acc, alpha[s - 2]);
                }
                *slot = if acc == f64::NEG_INFINITY {
                    acc
                } else {
                    acc + self.lp(t, label(s))
                };
            }
            std::mem::swap(&mut alpha, &mut next);
        }
        let total = log_add(alpha[ext_len - 1], alpha[ext_len - 2]);
        (total != f64::NEG_INFINITY).then_some(total)
    }

    /// Fraction of frames in `[from, to)` the model considers speech —
    /// P(blank) below ½, i.e. some phoneme (any language) is carrying real
    /// probability mass. The model is a better voice detector on film audio
    /// than an energy VAD: ambience and score mostly stay blank, a voice in
    /// any language does not. `None` when the range holds no frames.
    pub fn speech_fraction(&self, from: usize, to: usize) -> Option<f64> {
        let to = to.min(self.frames);
        if from >= to {
            return None;
        }
        let half = f64::from(0.5f32.ln());
        let speech = (from..to)
            .filter(|&t| self.lp(t, self.blank_id) < half)
            .count();
        Some(speech as f64 / (to - from) as f64)
    }

    /// Best-path (Viterbi) CTC alignment of `target` to the frames: where in
    /// the audio each target phoneme was emitted, and how strongly.
    ///
    /// [`log_likelihood`](Self::log_likelihood) answers "is the whole target
    /// supported"; this answers *where* — a phoneme that is not in the audio
    /// at all still gets assigned frames (Viterbi must pass through every
    /// label), but its frames carry very low probability, so `logp_mean`
    /// exposes exactly the phonemes the clip is missing. `None` under the
    /// same conditions as `log_likelihood`.
    pub fn force_align(&self, target: &[usize]) -> Option<Vec<AlignedPhoneme>> {
        if !self.valid_target(target) {
            return None;
        }
        let ext_len = 2 * target.len() + 1;
        let label = |s: usize| -> usize {
            if s.is_multiple_of(2) {
                self.blank_id
            } else {
                target[s / 2]
            }
        };
        // delta[t][s]: best log-prob of any path through state s at frame t;
        // from[t][s]: which state it came from.
        let mut delta = vec![f64::NEG_INFINITY; ext_len];
        let mut from = vec![vec![0usize; ext_len]; self.frames];
        delta[0] = self.lp(0, self.blank_id);
        if ext_len > 1 {
            delta[1] = self.lp(0, label(1));
            from[0][1] = 1;
        }
        let mut next = vec![f64::NEG_INFINITY; ext_len];
        for (t, from_t) in from.iter_mut().enumerate().skip(1) {
            for (s, slot) in next.iter_mut().enumerate() {
                let (mut best, mut arg) = (delta[s], s);
                if s >= 1 && delta[s - 1] > best {
                    (best, arg) = (delta[s - 1], s - 1);
                }
                if s >= 2 && s % 2 == 1 && label(s) != label(s - 2) && delta[s - 2] > best {
                    (best, arg) = (delta[s - 2], s - 2);
                }
                from_t[s] = arg;
                *slot = if best == f64::NEG_INFINITY {
                    best
                } else {
                    best + self.lp(t, label(s))
                };
            }
            std::mem::swap(&mut delta, &mut next);
        }
        let mut state = if delta[ext_len - 1] >= delta[ext_len - 2] {
            ext_len - 1
        } else {
            ext_len - 2
        };
        if delta[state] == f64::NEG_INFINITY {
            return None;
        }
        // Walk the path back, collecting the frames each label state emitted.
        let mut spans = vec![
            AlignedPhoneme {
                start_frame: usize::MAX,
                end_frame: 0,
                frames: 0,
                logp_mean: 0.0,
            };
            target.len()
        ];
        for t in (0..self.frames).rev() {
            if state % 2 == 1 {
                let p = &mut spans[state / 2];
                p.start_frame = t;
                p.end_frame = p.end_frame.max(t);
                p.frames += 1;
                p.logp_mean += self.lp(t, label(state));
            }
            state = from[t][state];
        }
        for p in &mut spans {
            debug_assert!(p.frames > 0, "Viterbi must visit every label");
            p.logp_mean /= p.frames as f64;
        }
        Some(spans)
    }

    /// Score `target` using joint CTC likelihood and a nonblank-first free decode.
    /// Unknown and special tokens are reported in `oov` and omitted.
    pub fn score_target(&self, target: &[String]) -> TargetScore {
        let mut ids = Vec::with_capacity(target.len());
        let mut oov = Vec::new();
        for tok in target {
            match self.id(tok) {
                Some(id) if id != self.blank_id && is_phone_token(tok) => ids.push(id),
                _ => oov.push(tok.clone()),
            }
        }
        let free = self.greedy_ids();
        let logp_target = if ids.is_empty() {
            None
        } else {
            self.log_likelihood(&ids)
        };
        let logp_free = self.log_likelihood(&free);
        let ratio = match (logp_target, logp_free) {
            (Some(t), Some(f)) => Some((t - f) / ids.len() as f64),
            _ => None,
        };
        TargetScore {
            logp_target,
            logp_target_per_phoneme: logp_target.map(|t| t / ids.len() as f64),
            logp_free,
            ratio,
            target_len: ids.len(),
            free_len: free.len(),
            oov,
        }
    }
}

/// One target phoneme's place in the audio under the best CTC alignment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlignedPhoneme {
    /// First emitted frame, inclusive.
    pub start_frame: usize,
    /// Last emitted frame, inclusive (unlike [`PhoneRun::end_frame`]).
    pub end_frame: usize,
    /// Frames the best path spent emitting this label (≥ 1).
    pub frames: usize,
    /// Mean per-frame log-prob of the label over those frames — very low
    /// when the phoneme is not actually in the audio.
    pub logp_mean: f64,
}

/// How well the audio supports one specific phoneme sequence.
///
/// `ratio = (logP(target) − logP(free)) / target_len`: log-odds per phoneme
/// of the claimed sentence against the model's own preferred reading. 0
/// means the target *is* what the model would have said; more negative means
/// the audio increasingly fails to support it. The free decode is not
/// necessarily the maximum-likelihood sequence, so positive ratios are possible.
/// Scale-free across clip lengths.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetScore {
    pub logp_target: Option<f64>,
    pub logp_target_per_phoneme: Option<f64>,
    pub logp_free: Option<f64>,
    pub ratio: Option<f64>,
    pub target_len: usize,
    pub free_len: usize,
    /// Target phonemes outside the model's vocabulary, or special tokens — dropped from the
    /// scored sequence, so a non-empty list means the score is of a
    /// *shorter* target than asked for.
    pub oov: Vec<String>,
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(vocab: &[&str], blank_id: usize, rows: &[&[f32]]) -> FrameMatrix {
        let width = vocab.len();
        let mut log_probs = Vec::new();
        for row in rows {
            assert_eq!(row.len(), width);
            let z: f32 = row.iter().map(|p| p.exp()).sum::<f32>();
            assert!((z - 1.0).abs() < 1e-4, "rows must be log-probabilities");
            log_probs.extend_from_slice(row);
        }
        FrameMatrix {
            schema_version: None,
            producer: None,
            trained_against_g2p: None,
            frame_rate_ms: None,
            sample_rate: None,
            heads: HashMap::new(),
            frames: rows.len(),
            vocab: vocab.iter().map(|s| s.to_string()).collect(),
            blank_id,
            index: vocab
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_string(), i))
                .collect(),
            log_probs,
        }
    }

    fn lp(p: &[f32]) -> Vec<f32> {
        p.iter().map(|x| x.ln()).collect()
    }

    #[test]
    fn nonblank_first_split_mass_and_threshold() {
        let speech = lp(&[0.4, 0.24, 0.18, 0.18]);
        let silence = lp(&[0.6, 0.16, 0.12, 0.12]);
        let tie = lp(&[0.5, 0.2, 0.15, 0.15]);
        let m = matrix(&["<pad>", "a", "b", "c"], 0, &[&speech, &silence, &tie]);
        assert_eq!(m.decode_path().unwrap().path, [1, 0, 0]);
        assert_eq!(m.greedy_ids(), [1]);
        assert_eq!(m.speech_fraction(0, 3), Some(1.0 / 3.0));
        assert_eq!(m.speech_fraction(2, 100), Some(0.0));
        assert_eq!(m.speech_fraction(3, 100), None);
        assert_eq!(m.speech_fraction(2, 1), None);
        // Scoring and alignment must retain JOINT 0.24, not conditional 0.4.
        let m = matrix(&["<pad>", "a", "b", "c"], 0, &[&speech]);
        assert!((m.log_likelihood(&[1]).unwrap() - 0.24f64.ln()).abs() < 1e-6);
        assert!((m.force_align(&[1]).unwrap()[0].logp_mean - 0.24f64.ln()).abs() < 1e-6);
        assert_eq!(m.score_target(&["a".into()]).ratio, Some(0.0));
    }

    #[test]
    fn runs_are_exclusive_and_blank_resets_collapse() {
        let a = lp(&[0.8, 0.1, 0.1]);
        let b = lp(&[0.1, 0.8, 0.1]);
        let blank = lp(&[0.1, 0.1, 0.8]);
        // Nonzero blank id, leading/trailing blanks, adjacent different phones,
        // and the same phone separated by a blank.
        let rows = [&blank, &a, &a, &blank, &a, &b, &b, &blank];
        let values: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let got = decode_path(&values, rows.len(), 3, 2).unwrap();
        assert_eq!(got.path, [2, 0, 0, 2, 0, 1, 1, 2]);
        assert_eq!(
            got.runs,
            [
                PhoneRun {
                    id: 0,
                    start_frame: 1,
                    end_frame: 3
                },
                PhoneRun {
                    id: 0,
                    start_frame: 4,
                    end_frame: 5
                },
                PhoneRun {
                    id: 1,
                    start_frame: 5,
                    end_frame: 7
                },
            ]
        );
        assert_eq!(
            serde_json::from_str::<DecodedPath>(&serde_json::to_string(&got).unwrap()).unwrap(),
            got
        );
    }

    #[test]
    fn specials_masked_slots_and_tied_phones() {
        let row = lp(&[0.1, 0.5, 0.0, 0.2, 0.2]);
        let m = matrix(&["<pad>", "<unk>", "masked", "a", "b"], 0, &[&row]);
        assert_eq!(m.greedy_ids(), [3]);
        // The raw helper has no labels, so only masks -inf and blank.
        assert_eq!(decode_path(&row, 1, 5, 0).unwrap().path, [1]);
        assert_eq!(
            decode_path(&lp(&[0.4, 0.0, 0.3, 0.3]), 1, 4, 0)
                .unwrap()
                .path,
            [2]
        );
        for special in ["", "|", "<pad>", "<s>", "</s>", "<unk>", "<blank>"] {
            assert!(!is_phone_token(special));
        }
        assert!(is_phone_token("tʃ"));
        assert_eq!(m.id("<unk>"), Some(1)); // lookup preserves wire vocab
        assert_eq!(
            m.score_target(&["<unk>".into(), "<pad>".into()]).oov,
            ["<unk>", "<pad>"]
        );
        for target in [vec![], vec![0], vec![1], vec![5], vec![usize::MAX], vec![2]] {
            assert!(m.log_likelihood(&target).is_none());
            assert!(m.force_align(&target).is_none());
        }
    }

    #[test]
    fn rejects_invalid_raw_matrices() {
        assert!(decode_path(&[], 0, 0, 0).is_err());
        assert!(decode_path(&[], 0, 2, 2).is_err());
        assert!(decode_path(&[], usize::MAX, 2, 0).is_err());
        assert!(decode_path(&[-0.1], 1, 2, 0).is_err());
        for bad in [f32::NAN, f32::INFINITY, 0.1] {
            assert!(decode_path(&[-0.1, bad], 1, 2, 0).is_err());
        }
        assert!(decode_path(&[f32::NEG_INFINITY; 2], 1, 2, 0).is_err());
        assert!(decode_path(&[-1.0, f32::NEG_INFINITY], 1, 2, 0).is_err());
        assert_eq!(
            decode_path(&[0.0, f32::NEG_INFINITY], 1, 2, 0)
                .unwrap()
                .path,
            [0]
        );
        assert!(decode_path(&[], 0, 2, 0).unwrap().runs.is_empty());
        let empty = matrix(&["<pad>", "a"], 0, &[]);
        assert!(empty.greedy_ids().is_empty());
        assert_eq!(empty.speech_fraction(0, 1), None);
        assert!(empty.force_align(&[1]).is_none());
        assert!(empty.log_likelihood(&[1]).is_none());
        assert_eq!(empty.score_target(&[]).ratio, None);
    }

    fn payload(rows: &[f32], frames: usize, vocab: &[&str]) -> LegacyFrameMatrixPayload {
        use std::io::Write;
        let bytes: Vec<u8> = rows
            .iter()
            .flat_map(|&p| half::f16::from_f32(p).to_le_bytes())
            .collect();
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&bytes).unwrap();
        LegacyFrameMatrixPayload {
            shape: vec![frames, vocab.len()],
            dtype: "float16".into(),
            encoding: "zlib+base64".into(),
            blank_id: 0,
            vocab: vocab.iter().map(|s| (*s).into()).collect(),
            data: base64::engine::general_purpose::STANDARD.encode(enc.finish().unwrap()),
        }
    }

    #[test]
    fn validates_wire_payloads() {
        let good = payload(&lp(&[0.4, 0.6, 0.0]), 1, &["<pad>", "a", "<unk>"]);
        let decoded = FrameMatrix::decode(&FrameMatrixPayload::Legacy(good.clone())).unwrap();
        assert_eq!(decoded.greedy_ids(), [1]);
        assert_eq!(decoded.log_probs()[2], f32::NEG_INFINITY);
        let mut bad = good.clone();
        bad.dtype = "float32".into();
        assert!(FrameMatrix::decode(&FrameMatrixPayload::Legacy(bad)).is_err());
        let mut bad = good.clone();
        bad.encoding = "base64".into();
        assert!(FrameMatrix::decode(&FrameMatrixPayload::Legacy(bad)).is_err());
        for shape in [
            vec![],
            vec![3],
            vec![1, 3, 1],
            vec![1, 2],
            vec![2, 3],
            vec![usize::MAX, 3],
        ] {
            let mut bad = good.clone();
            bad.shape = shape;
            assert!(FrameMatrix::decode(&FrameMatrixPayload::Legacy(bad)).is_err());
        }
        let mut bad = good.clone();
        bad.blank_id = 3;
        assert!(FrameMatrix::decode(&FrameMatrixPayload::Legacy(bad)).is_err());
        let mut bad = good.clone();
        bad.vocab[2] = "a".into();
        assert!(FrameMatrix::decode(&FrameMatrixPayload::Legacy(bad)).is_err());
        let mut bad = good.clone();
        bad.data = "!invalid base64!".into();
        assert!(FrameMatrix::decode(&FrameMatrixPayload::Legacy(bad)).is_err());
        let mut bad = good;
        bad.data = base64::engine::general_purpose::STANDARD.encode(b"not zlib");
        assert!(FrameMatrix::decode(&FrameMatrixPayload::Legacy(bad)).is_err());
        for rows in [
            vec![f32::NAN, -1.0],
            vec![f32::INFINITY, -1.0],
            vec![0.1, -1.0],
            vec![-1.0, f32::NEG_INFINITY],
            vec![f32::NEG_INFINITY; 2],
            vec![-1.0; 3],
        ] {
            assert!(FrameMatrix::decode_legacy(&payload(&rows, 1, &["<pad>", "a"])).is_err());
        }
        assert!(
            FrameMatrix::decode_legacy(&payload(&[-1.0, -0.5], 1, &["<pad>", "<unk>"])).is_err()
        );
        assert!(
            FrameMatrix::decode_legacy(&payload(&[], 0, &["<pad>", "a"]))
                .unwrap()
                .greedy_ids()
                .is_empty()
        );
    }

    /// Two frames, vocab {blank, a}. P("a") = every path whose collapse is
    /// "a": (a,a), (a,-), (-,a) = 0.7·0.7 + 0.7·0.3 + 0.3·0.7 = 0.91.
    #[test]
    fn sums_over_all_alignments() {
        let r = lp(&[0.3, 0.7]);
        let m = matrix(&["<pad>", "a"], 0, &[&r, &r]);
        let got = m.log_likelihood(&[m.id("a").unwrap()]).unwrap();
        assert!((got - 0.91f64.ln()).abs() < 1e-5, "{got}");
        // "a a" needs a blank between repeats: impossible in two frames.
        assert!(m.log_likelihood(&[1, 1]).is_none());
    }

    /// Target longer than the frames can spell has no alignment.
    #[test]
    fn impossible_targets_are_none() {
        let r = lp(&[0.5, 0.5]);
        let m = matrix(&["<pad>", "a"], 0, &[&r]);
        assert!(m.log_likelihood(&[1, 1]).is_none());
        assert!(m.log_likelihood(&[]).is_none());
    }

    #[test]
    fn greedy_collapses_blanks_and_repeats() {
        let a = lp(&[0.1, 0.8, 0.1]);
        let b = lp(&[0.1, 0.1, 0.8]);
        let blank = lp(&[0.8, 0.1, 0.1]);
        let m = matrix(&["<pad>", "a", "b"], 0, &[&a, &a, &blank, &a, &b, &b]);
        assert_eq!(m.greedy_ids(), vec![1, 1, 2]);
        let s = m.score_target(&["a".into(), "a".into(), "b".into()]);
        assert_eq!(s.free_len, 3);
        assert_eq!(s.target_len, 3);
        // The greedy path is the model's preferred reading: ratio is ≈ 0.
        assert!(s.ratio.unwrap().abs() < 1e-9, "{s:?}");
        let worse = m.score_target(&["b".into(), "a".into()]);
        assert!(worse.ratio.unwrap() < s.ratio.unwrap());
        assert_eq!(
            m.score_target(&["zz".into(), "a".into()]).oov,
            vec!["zz".to_string()]
        );
    }

    /// Forced alignment localises each phoneme and exposes a missing one.
    #[test]
    fn force_align_localises_and_scores() {
        let a = lp(&[0.1, 0.8, 0.1]);
        let b = lp(&[0.1, 0.1, 0.8]);
        let blank = lp(&[0.8, 0.1, 0.1]);
        let m = matrix(&["<pad>", "a", "b"], 0, &[&a, &a, &blank, &b, &b, &blank]);
        let spans = m.force_align(&[1, 2]).unwrap();
        assert_eq!(spans.len(), 2);
        assert!(spans[0].end_frame < spans[1].start_frame);
        assert!((spans[0].logp_mean - 0.8f64.ln()).abs() < 1e-5);
        // "b a b": the audio spells "a b", so the leading "b" exists nowhere
        // — its best frames still carry only the 0.1 the model left it,
        // while "a" and the real "b" align to their frames at 0.8.
        let spans = m.force_align(&[2, 1, 2]).unwrap();
        assert!((spans[0].logp_mean - 0.1f64.ln()).abs() < 1e-5);
        assert!((spans[1].logp_mean - 0.8f64.ln()).abs() < 1e-5);
        assert!(spans[0].logp_mean < spans[2].logp_mean);
    }

    /// Round-trips the endpoint's wire format.
    #[test]
    fn decodes_the_wire_format() {
        use std::io::Write;
        let rows: Vec<f32> = vec![-0.1, -2.5, -3.0, -0.2, -1.0, -2.0];
        let mut bytes = Vec::new();
        for v in &rows {
            bytes.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&bytes).unwrap();
        let payload = LegacyFrameMatrixPayload {
            shape: vec![2, 3],
            dtype: "float16".into(),
            encoding: "zlib+base64".into(),
            blank_id: 0,
            vocab: vec!["<pad>".into(), "a".into(), "b".into()],
            data: base64::engine::general_purpose::STANDARD.encode(enc.finish().unwrap()),
        };
        let m = FrameMatrix::decode(&FrameMatrixPayload::Legacy(payload)).unwrap();
        assert_eq!(m.frames, 2);
        assert!((m.lp(1, 2) - -2.0).abs() < 1e-3);
        assert!(m.greedy_ids().is_empty()); // blank wins both frames
    }
}

//! Versioned, self-describing frame artifacts. Absent version means legacy;
//! a present but unsupported/malformed version never falls back to legacy.
use super::*;

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum FrameMatrixPayload {
    Legacy(LegacyFrameMatrixPayload),
    V1(FrameMatrixV1),
}

impl<'de> Deserialize<'de> for FrameMatrixPayload {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value.get("schema_version") {
            None => serde_json::from_value(value).map(Self::Legacy),
            Some(version) if version.as_u64() == Some(1) => {
                serde_json::from_value(value).map(Self::V1)
            }
            Some(version) => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported frame matrix schema_version {version}"
                )))
            }
        }
        .map_err(serde::de::Error::custom)
    }
}

/// Deliberately NOT `#[serde(default)]`: the key must be present, even as null.
/// A server that cannot establish which g2p its checkpoint was labelled against
/// has to say so explicitly; omitting the field would make "unknown" and "older
/// server that never heard of this field" indistinguishable.
fn deserialize_training_identity<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error> {
    Option::<String>::deserialize(deserializer)
}

/// Required serving identity, independent of the optional legacy HTTP envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FrameMatrixProducer {
    pub model_id: String,
    pub model_revision: String,
    pub deploy_marker: String,
    pub decoder_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FrameMatrixV1 {
    pub schema_version: u32,
    pub producer: FrameMatrixProducer,
    /// Exact g2p::identity() of source labels; null means unknown, not compatible.
    /// Does not identify post-G2P remapping, narrowing, or supervision masks.
    #[serde(deserialize_with = "deserialize_training_identity")]
    pub trained_against_g2p: Option<String>,
    pub frame_rate_ms: f64,
    pub sample_rate: u32,
    pub heads: HashMap<String, FrameHeadPayload>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FrameValueSemantics {
    JointLogProbability,
    LogProbability,
    Probability,
    SigmoidProbability,
}

/// Row-major little-endian float16 values. Labels index the final dimension;
/// the scalar nonblank head is [T] with the single label "nonblank".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FrameHeadPayload {
    pub shape: Vec<usize>,
    pub labels: Vec<String>,
    pub dtype: String,
    pub encoding: String,
    pub value_semantics: FrameValueSemantics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blank_id: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    pub data: String,
}

/// Retains the complete wire descriptor (including compressed data) and values.
#[derive(Debug, Clone)]
pub struct FrameHead {
    pub payload: FrameHeadPayload,
    pub values: Vec<f32>,
}

pub(super) fn decode_values(
    shape: &[usize],
    dtype: &str,
    encoding: &str,
    data: &str,
) -> Result<Vec<f32>> {
    if dtype != "float16" || encoding != "zlib+base64" {
        bail!("unsupported frame matrix format {dtype}/{encoding}");
    }
    let expected = shape
        .iter()
        .try_fold(2usize, |size, &dim| size.checked_mul(dim))
        .context("frame matrix byte length overflows")?;
    let limit = expected
        .checked_add(1)
        .context("frame matrix decompression limit overflows")?;
    let compressed = base64::engine::general_purpose::STANDARD
        .decode(data)
        .context("frame matrix base64")?;
    // Bound output without reserving untrusted dimensions. Require StreamEnd:
    // read-to-EOF alone accepts a truncated zlib trailer.
    let mut raw = Vec::new();
    let mut decoder = flate2::Decompress::new(true);
    loop {
        let mut chunk = [0u8; 8192];
        let available = chunk.len().min(limit - raw.len());
        let before_in = decoder.total_in();
        let before_out = decoder.total_out();
        let status = decoder
            .decompress(
                &compressed[before_in as usize..],
                &mut chunk[..available],
                flate2::FlushDecompress::None,
            )
            .context("frame matrix zlib")?;
        let written = (decoder.total_out() - before_out) as usize;
        raw.extend_from_slice(&chunk[..written]);
        if raw.len() > expected {
            bail!("frame matrix output exceeds shape");
        }
        if status == flate2::Status::StreamEnd {
            if raw.len() != expected || decoder.total_in() != compressed.len() as u64 {
                bail!("frame matrix byte length mismatch or trailing compressed data");
            }
            break;
        }
        if decoder.total_in() == before_in && decoder.total_out() == before_out {
            bail!("truncated frame matrix zlib stream");
        }
    }
    Ok(raw
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

impl FrameMatrix {
    pub fn decode(payload: &FrameMatrixPayload) -> Result<Self> {
        let payload = match payload {
            FrameMatrixPayload::Legacy(legacy) => return Self::decode_legacy(legacy),
            FrameMatrixPayload::V1(payload) => payload,
        };
        if payload.schema_version != 1 {
            bail!("unsupported frame matrix schema_version");
        }
        if payload.sample_rate == 0
            || !payload.frame_rate_ms.is_finite()
            || payload.frame_rate_ms <= 0.0
        {
            bail!("invalid frame matrix timebase");
        }
        let producer = &payload.producer;
        if [
            &producer.model_id,
            &producer.model_revision,
            &producer.deploy_marker,
            &producer.decoder_version,
        ]
        .iter()
        .any(|s| s.is_empty())
            || payload
                .trained_against_g2p
                .as_ref()
                .is_some_and(|s| s.is_empty())
        {
            bail!("empty frame matrix provenance");
        }
        let phone = payload.heads.get("phone").context("missing phone head")?;
        if phone.value_semantics != FrameValueSemantics::JointLogProbability {
            bail!("phone head must contain joint log probabilities");
        }
        let mut matrix = Self::decode_legacy(&LegacyFrameMatrixPayload {
            shape: phone.shape.clone(),
            dtype: phone.dtype.clone(),
            encoding: phone.encoding.clone(),
            blank_id: phone.blank_id.context("phone head needs blank_id")?,
            vocab: phone.labels.clone(),
            data: phone.data.clone(),
        })?;
        for (name, head) in &payload.heads {
            let width = match head.shape.as_slice() {
                [t] if name == "nonblank" && *t == matrix.frames => 1,
                [t, width] if *t == matrix.frames && *width > 0 => *width,
                _ => bail!("invalid head shape for {name}"),
            };
            if head.labels.len() != width
                || head.labels.iter().any(|label| label.is_empty())
                || head
                    .labels
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    != width
            {
                bail!("invalid labels for {name}");
            }
            if name != "phone" && head.blank_id.is_some() {
                bail!("only phone has blank_id");
            }
            if !["phone", "stress", "nonblank"].contains(&name.as_str())
                && (head.language.as_ref().is_none_or(|s| s.is_empty())
                    || head.target.as_ref().is_none_or(|s| s.is_empty()))
            {
                bail!("auxiliary head {name} needs language and target");
            }
            let values = if name == "phone" {
                matrix.log_probs.clone()
            } else {
                decode_values(&head.shape, &head.dtype, &head.encoding, &head.data)?
            };
            for &value in &values {
                let valid = match head.value_semantics {
                    FrameValueSemantics::JointLogProbability
                    | FrameValueSemantics::LogProbability => value <= 0.0,
                    FrameValueSemantics::Probability | FrameValueSemantics::SigmoidProbability => {
                        (0.0..=1.0).contains(&value)
                    }
                };
                if !valid {
                    bail!("invalid value in {name}");
                }
            }
            matrix.heads.insert(
                name.clone(),
                FrameHead {
                    payload: head.clone(),
                    values,
                },
            );
        }
        let nonblank = matrix
            .heads
            .get("nonblank")
            .context("missing nonblank head")?;
        if nonblank.payload.labels != ["nonblank"]
            || nonblank.payload.value_semantics != FrameValueSemantics::SigmoidProbability
        {
            bail!("invalid nonblank head declaration");
        }
        let stress = matrix.heads.get("stress").context("missing stress head")?;
        if stress.payload.labels != ["none", "primary", "secondary"]
            || stress.payload.value_semantics != FrameValueSemantics::Probability
        {
            bail!("invalid stress head declaration");
        }
        matrix.schema_version = Some(1);
        matrix.producer = Some(payload.producer.clone());
        matrix.trained_against_g2p = payload.trained_against_g2p.clone();
        matrix.frame_rate_ms = Some(payload.frame_rate_ms);
        matrix.sample_rate = Some(payload.sample_rate);
        Ok(matrix)
    }
}

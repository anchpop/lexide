//! Offline verification of artifacts written by pronunciation/modal/verify_frame_matrix.py.
//! cargo run --features pronunciation --example verify_frame_matrix -- artifact.json ...
use anyhow::{ensure, Context, Result};
use base64::Engine;
use lexide::pronunciation::{FrameMatrix, FrameMatrixPayload};
use serde_json::{json, Value};
use std::io::Read;

fn main() -> Result<()> {
    for path in std::env::args().skip(1) {
        let artifact: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let raw = &artifact["response"]["frame_matrix"];
        let payload: FrameMatrixPayload = serde_json::from_value(raw.clone())?;
        ensure!(serde_json::to_value(&payload)? == *raw, "wire fields lost");
        let matrix = FrameMatrix::decode(&payload)?;
        ensure!(json!(matrix.schema_version) == raw["schema_version"]);
        ensure!(serde_json::to_value(&matrix.producer)? == raw["producer"]);
        ensure!(json!(matrix.trained_against_g2p) == raw["trained_against_g2p"]);
        ensure!(json!(matrix.sample_rate) == raw["sample_rate"]);
        ensure!(json!(matrix.frame_rate_ms) == raw["frame_rate_ms"]);
        ensure!(matrix.heads.len() == raw["heads"].as_object().unwrap().len());
        for (name, head) in &matrix.heads {
            ensure!(
                serde_json::to_value(&head.payload)? == raw["heads"][name],
                "head metadata lost: {name}"
            );
            let compressed = base64::engine::general_purpose::STANDARD
                .decode(raw["heads"][name]["data"].as_str().unwrap())?;
            let mut bytes = Vec::new();
            flate2::read::ZlibDecoder::new(compressed.as_slice()).read_to_end(&mut bytes)?;
            let decoded_bytes: Vec<_> = head
                .values
                .iter()
                .flat_map(|&value| half::f16::from_f32(value).to_le_bytes())
                .collect();
            ensure!(bytes == decoded_bytes, "head values changed: {name}");
        }
        let phone = &raw["heads"]["phone"];
        let legacy: FrameMatrixPayload = serde_json::from_value(json!({"shape": phone["shape"],
            "vocab": phone["labels"], "blank_id":phone["blank_id"], "dtype":phone["dtype"],
            "encoding":phone["encoding"], "data":phone["data"]}))?;
        let old = FrameMatrix::decode(&legacy)?;
        ensure!(old.log_probs() == matrix.log_probs());
        ensure!(old.producer.is_none() && old.schema_version.is_none() && old.heads.is_empty());
        let duration_ms = artifact["duration_ms"].as_f64().context("duration")?;
        let frame_duration_ms = matrix.frames as f64 * matrix.frame_rate_ms.unwrap();
        ensure!(duration_ms >= frame_duration_ms && duration_ms - frame_duration_ms < 30.0);
        let target = lexide::pronunciation::Phonemized {
            phonemes: serde_json::from_value(artifact["target"].clone())?,
            ..Default::default()
        };
        let score = matrix.score_target(&target);
        ensure!(score == old.score_target(&target), "legacy score differs");
        let endpoint = &artifact["response"]["target_score"];
        if artifact["impossible"].as_bool() == Some(true) {
            ensure!(
                score.logp_target.is_none(),
                "impossible Rust alignment was accepted"
            );
            // Deliberately report, do not paper over, Python zero_infinity=True.
        } else {
            let local = serde_json::to_value(&score)?;
            for field in ["target_len", "free_len", "oov"] {
                ensure!(
                    local[field] == endpoint[field],
                    "scoring metadata differs: {field}"
                );
            }
            for field in [
                "logp_target",
                "logp_target_per_phoneme",
                "logp_free",
                "ratio",
            ] {
                let delta = local[field].as_f64().context("Rust score")?
                    - endpoint[field].as_f64().context("endpoint score")?;
                ensure!(
                    delta.abs() < 0.1,
                    "normal CTC parity exceeds fp16 tolerance for {field}: {delta}"
                );
            }
        }
        println!(
            "{}",
            json!({"artifact":path,"every_field_verified":true,"legacy_verified":true,
            "frames":matrix.frames,"duration_ms":duration_ms,"frame_duration_ms":frame_duration_ms,
            "endpoint_score":endpoint,"rust_score":score,
            "logp_delta": score.logp_target.zip(endpoint["logp_target"].as_f64()).map(|(a,b)|a-b)})
        );
    }
    Ok(())
}

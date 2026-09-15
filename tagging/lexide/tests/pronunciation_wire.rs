#![cfg(feature = "pronunciation")]

use lexide::pronunciation::*;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};

// Field names/nesting transcribed from wav2vec2_phoneme.py's predict,
// _aggregate_group, _frames_topk, _score_target, _frame_matrix and predict_batch
// dictionary constructors. Values are illustrative, not live GPU captures.
fn round_trip<T: DeserializeOwned + Serialize>(wire: Value) -> T {
    let parsed: T = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), wire);
    parsed
}

fn identity() -> Value {
    json!({
        "model_id": "anchpop/lexide-pronunciation",
        "model_revision": "edcbbbf43a7ff337f43d233a9d89566509715e63",
        "deploy_marker": "test-deploy"
    })
}

fn phoneme() -> Value {
    json!({"phoneme": "ˈa", "confidence": 0.75,
        "top_k": [{"phoneme": "a", "probability": 0.75}],
        "stress": 1, "tone": 2, "pitch_accent": 0})
}

fn frame() -> Value {
    json!({"frame": 0, "stress": 1, "p_nonblank": 0.9,
        "top_k": [{"phoneme": "a", "probability": 0.75}],
        "tone": 2, "pitch_accent": 0})
}

fn score() -> Value {
    json!({"logp_target": -1.0, "logp_target_per_phoneme": -1.0,
        "logp_free": null, "ratio": null, "target_len": 1, "free_len": 0, "oov": []})
}

fn matrix() -> Value {
    // zlib-compressed empty byte array, representing a zero-frame matrix.
    json!({"shape": [0, 2], "dtype": "float16", "encoding": "zlib+base64",
        "blank_id": 0, "vocab": ["<pad>", "a"], "data": "eJwDAAAAAAE="})
}

#[test]
fn identity_and_cache_version_match_yap() {
    let identity = round_trip::<ModelIdentity>(identity());
    assert_eq!(DECODER_VERSION, "nonblank_v1");
    assert_eq!(
        cache_version(&identity),
        "anchpop_lexide-pronunciation@edcbbbf43a7f__nonblank_v1"
    );
    let mut short = identity;
    short.model_revision = "é短".into();
    assert_eq!(
        cache_version(&short),
        "anchpop_lexide-pronunciation@é短__nonblank_v1"
    );
}

#[test]
fn request_round_trip_and_python_defaults() {
    round_trip::<PredictRequest>(
        json!({"audio_f32_b64": "AAAAAAAAAD8AAAC/", "sample_rate": 16000,
        "top_k": 3, "language": "eng", "target_phonemes": ["a"],
        "return_frame_matrix": true, "return_frames": true}),
    );
    let minimal: PredictRequest = serde_json::from_value(json!({"audio_f32_b64": ""})).unwrap();
    assert_eq!(minimal, PredictRequest::default());
    assert_eq!(PredictRequest::from_samples(&[]), minimal);
    assert!(serde_json::from_value::<PredictRequest>(json!({"audio": []})).is_err());
}

#[test]
fn request_audio_encoding_matches_known_little_endian_bytes() {
    // IEEE 754: +0 = 00000000, +0.5 = 3f000000, -0.5 = bf000000.
    // The '/' distinguishes standard base64 from the URL-safe alphabet.
    let request = PredictRequest::from_samples(&[0.0, 0.5, -0.5]);
    assert_eq!(request.audio_f32_b64, "AAAAAAAAAD8AAAC/");
    assert_eq!(
        PredictRequest::from_samples(&[1.0]).audio_f32_b64,
        "AACAPw=="
    );
    assert_eq!(
        serde_json::to_value(request).unwrap(),
        json!({"audio_f32_b64": "AAAAAAAAAD8AAAC/", "sample_rate": 16000,
            "top_k": 3, "return_frame_matrix": false, "return_frames": false})
    );
}

#[test]
fn request_audio_base64_round_trip_preserves_sample_bits() {
    use base64::{engine::general_purpose::STANDARD, Engine};

    let samples = [0.0, -0.0, 0.5, -0.5, f32::MIN_POSITIVE, f32::MAX];
    let request = PredictRequest::from_samples(&samples);
    let parsed = round_trip::<PredictRequest>(serde_json::to_value(&request).unwrap());
    assert_eq!(parsed, request);
    let bytes = STANDARD.decode(&parsed.audio_f32_b64).unwrap();
    let bits: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(bits, samples.map(f32::to_bits));
}

#[test]
fn nested_response_types_round_trip() {
    round_trip::<PhonemeAlternative>(json!({"phoneme": "a", "probability": 0.75}));
    round_trip::<EmittedPhoneme>(phoneme());
    round_trip::<PredictionFrame>(frame());
    round_trip::<TargetScore>(score());
    round_trip::<PredictTargetScore>(score());
    let error = json!({"error": "no target phoneme was in the model vocab", "oov": ["unknown"]});
    round_trip::<TargetScoreError>(error.clone());
    assert!(matches!(
        round_trip::<PredictTargetScore>(error),
        PredictTargetScore::Error(_)
    ));
    let payload = round_trip::<FrameMatrixPayload>(matrix());
    assert_eq!(FrameMatrix::decode(&payload).unwrap().frames, 0);
}

#[test]
fn prediction_round_trip() {
    round_trip::<PredictResponse>(json!({"phonemes": [phoneme()],
        "target_score": score(), "frame_matrix": matrix(), "frames": [frame()],
        "deploy_marker": "test-deploy"}));
    round_trip::<PredictResponse>(json!({"phonemes": [], "target_score": {
        "error": "no target phoneme was in the model vocab", "oov": ["unknown"]}}));
    let old = round_trip::<PredictResponse>(json!({"phonemes": []}));
    assert!(old.deploy_marker.is_none());
    assert!(old.frame_matrix.is_none());
    assert!(old.frames.is_none());
    assert!(old.target_score.is_none());
    assert!(serde_json::from_value::<PredictResponse>(
        json!({"error": {"type": "ValueError", "message": "invalid"}})
    )
    .is_err());
}

#[test]
fn batch_round_trip_preserves_item_errors() {
    let error = json!({"type": "ValueError", "message": "invalid audio"});
    round_trip::<PredictionError>(error.clone());
    round_trip::<BatchResult>(json!({"error": error.clone()}));
    round_trip::<BatchResult>(json!({"phonemes": [phoneme()]}));
    let batch = round_trip::<BatchResponse>(json!({"results": [
        {"phonemes": [phoneme()]}, {"error": error}], "deploy_marker": "test-deploy"}));
    assert!(matches!(&batch.results[0], BatchResult::Prediction(p) if p.deploy_marker.is_none()));
    assert!(
        matches!(&batch.results[1], BatchResult::Error { error } if error.error_type == "ValueError")
    );
    round_trip::<BatchResponse>(json!({"results": []}));
}

#[test]
fn new_identity_metadata_round_trips_on_every_envelope() {
    let mut metadata = identity();
    metadata["decoder_version"] = json!("nonblank_v1");
    round_trip::<ModelIdentity>(metadata.clone());
    let mut single = metadata.clone();
    single["phonemes"] = json!([]);
    let prediction = round_trip::<PredictResponse>(single);
    assert_eq!(
        prediction.model_id.as_deref(),
        Some("anchpop/lexide-pronunciation")
    );
    let mut batch = metadata;
    batch["results"] = json!([]);
    let batch = round_trip::<BatchResponse>(batch);
    assert_eq!(batch.decoder_version.as_deref(), Some("nonblank_v1"));
}

#[test]
fn optional_fields_and_unknown_server_extensions_are_compatible() {
    let old: EmittedPhoneme =
        serde_json::from_value(json!({"phoneme": "a", "future_head": 4})).unwrap();
    assert_eq!(old.confidence, 0.0);
    assert!(old.top_k.is_empty());
    assert!(old.stress.is_none());
    let mut probe = identity();
    probe["language_head_specs"] = json!({});
    assert!(serde_json::from_value::<ModelIdentity>(probe).is_ok());
    assert!(serde_json::from_value::<ModelIdentity>(
        json!({"deploy_marker": "broken", "load_error": "traceback"})
    )
    .is_err());
    let old: ModelIdentity =
        serde_json::from_value(json!({"model_id": "m", "model_revision": "r"})).unwrap();
    assert!(old.deploy_marker.is_none());
    assert!(old.decoder_version.is_none());
    let response: PredictResponse = serde_json::from_value(json!({"phonemes": [],
        "target_score": null, "frame_matrix": null, "frames": null,
        "deploy_marker": null, "future_diagnostic": {}}))
    .unwrap();
    assert!(response.target_score.is_none());
}

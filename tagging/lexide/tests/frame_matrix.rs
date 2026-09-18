#![cfg(feature = "pronunciation")]
use base64::Engine;
use lexide::pronunciation::{FrameMatrix, FrameMatrixPayload};
use serde_json::{json, Value};
use std::io::Write;

fn data(values: &[f32]) -> String {
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    for &value in values {
        enc.write_all(&half::f16::from_f32(value).to_le_bytes())
            .unwrap();
    }
    base64::engine::general_purpose::STANDARD.encode(enc.finish().unwrap())
}
fn fixture() -> Value {
    json!({"schema_version":1,
    "producer":{"model_id":"model", "model_revision":"revision", "deploy_marker":"deploy", "decoder_version":"nonblank_v1"},
    "trained_against_g2p":"g2p/0.4.0 espeak-ng/aa907af78d5665d8 thai/ad66331eca29d4ea korean/9e4bc6b854f6a903",
    "sample_rate":16000, "frame_rate_ms":20.0,
    "heads":{
        "phone":{"shape":[2,2],"labels":["<pad>","a"],"blank_id":0,"value_semantics":"joint_log_probability",
            "dtype":"float16","encoding":"zlib+base64","data":data(&[-1.0,-0.5,-0.2,-2.0])},
        "nonblank":{"shape":[2],"labels":["nonblank"],"value_semantics":"sigmoid_probability",
            "dtype":"float16","encoding":"zlib+base64","data":data(&[0.6,0.2])},
        "stress":{"shape":[2,3],"labels":["none","primary","secondary"],"value_semantics":"probability",
            "dtype":"float16","encoding":"zlib+base64","data":data(&[0.5,0.25,0.25,0.5,0.25,0.25])},
        "jpn_pitch_accent":{"shape":[2,3],"labels":["not_bearer","low","high"],"value_semantics":"probability",
            "language":"jpn","target":"pitch_accent","dtype":"float16","encoding":"zlib+base64",
            "data":data(&[0.5,0.25,0.25,0.25,0.5,0.25])}
    }})
}
fn decode(value: Value) -> anyhow::Result<FrameMatrix> {
    FrameMatrix::decode(&serde_json::from_value(value)?)
}

#[test]
fn retains_all_v1_metadata_and_values() {
    let raw = fixture();
    let payload: FrameMatrixPayload = serde_json::from_value(raw.clone()).unwrap();
    assert_eq!(serde_json::to_value(&payload).unwrap(), raw);
    let m = FrameMatrix::decode(&payload).unwrap();
    assert_eq!(m.schema_version, Some(1));
    assert_eq!(serde_json::to_value(&m.producer).unwrap(), raw["producer"]);
    assert_eq!(
        m.trained_against_g2p.as_deref(),
        raw["trained_against_g2p"].as_str()
    );
    assert_eq!(m.sample_rate, Some(16000));
    assert_eq!(m.frame_rate_ms, Some(20.0));
    for (name, head) in &m.heads {
        assert_eq!(
            serde_json::to_value(&head.payload).unwrap(),
            raw["heads"][name]
        );
        assert_eq!(
            head.values.len(),
            head.payload.shape.iter().product::<usize>()
        );
    }
    assert_eq!(
        m.heads["jpn_pitch_accent"].values,
        [0.5, 0.25, 0.25, 0.25, 0.5, 0.25]
    );
    assert_eq!(m.log_probs(), m.heads["phone"].values);
    assert_eq!(m.greedy_ids(), [1]);
    let mut scalar_column = raw;
    scalar_column["heads"]["nonblank"]["shape"] = json!([2, 1]);
    assert!(decode(scalar_column).is_ok());
}

#[test]
fn legacy_has_no_fabricated_provenance_or_timebase() {
    let raw = fixture();
    let phone = &raw["heads"]["phone"];
    let legacy = json!({"shape":phone["shape"],"vocab":phone["labels"],"blank_id":0,
        "dtype":phone["dtype"],"encoding":phone["encoding"],"data":phone["data"]});
    let m = decode(legacy.clone()).unwrap();
    assert!(m.schema_version.is_none() && m.producer.is_none() && m.trained_against_g2p.is_none());
    assert!(m.sample_rate.is_none() && m.frame_rate_ms.is_none() && m.heads.is_empty());
    assert_eq!(
        m.score_target(&lexide::pronunciation::Phonemized::from_ipa_tokens("a")),
        decode(raw)
            .unwrap()
            .score_target(&lexide::pronunciation::Phonemized::from_ipa_tokens("a"))
    );
    for version in [json!(2), json!(null), json!("1"), json!(true), json!(1.0)] {
        let mut invalid = legacy.clone();
        invalid["schema_version"] = version;
        assert!(decode(invalid).is_err());
    }
}

#[test]
fn rejects_malformed_v1_without_legacy_fallback() {
    for field in [
        "producer",
        "trained_against_g2p",
        "sample_rate",
        "frame_rate_ms",
        "heads",
    ] {
        let mut raw = fixture();
        raw.as_object_mut().unwrap().remove(field);
        assert!(decode(raw).is_err(), "{field}");
    }
    for field in [
        "model_id",
        "model_revision",
        "deploy_marker",
        "decoder_version",
    ] {
        let mut raw = fixture();
        raw["producer"].as_object_mut().unwrap().remove(field);
        assert!(decode(raw).is_err(), "{field}");
    }
    for (path, value) in [
        ("/schema_version", json!(9)),
        ("/sample_rate", json!(0)),
        ("/frame_rate_ms", json!(-1)),
        ("/producer/model_id", json!("")),
        ("/heads/phone/blank_id", json!(9)),
        ("/heads/phone/value_semantics", json!("probability")),
        ("/heads/stress/shape", json!([3, 3])),
        ("/heads/stress/labels", json!(["none", "none", "secondary"])),
        ("/heads/stress/dtype", json!("float32")),
        ("/heads/stress/encoding", json!("base64")),
        ("/heads/stress/data", json!("!")),
        ("/heads/stress/data", json!(data(&[1.0]))),
        ("/heads/stress/data", json!(data(&[f32::NAN; 6]))),
        ("/heads/stress/data", json!(data(&[1.1; 6]))),
        ("/heads/nonblank/shape", json!([2, 1, 1])),
    ] {
        let mut raw = fixture();
        *raw.pointer_mut(path).unwrap() = value;
        assert!(decode(raw).is_err(), "{path}");
    }
    for name in ["phone", "nonblank", "stress"] {
        let mut raw = fixture();
        raw["heads"].as_object_mut().unwrap().remove(name);
        assert!(decode(raw).is_err(), "{name}");
    }
    let mut unknown = fixture();
    unknown["trained_against_g2p"] = Value::Null;
    assert!(decode(unknown).unwrap().trained_against_g2p.is_none());
}

#[test]
fn rejects_truncated_and_trailing_zlib_streams() {
    let raw = fixture();
    let compressed = base64::engine::general_purpose::STANDARD
        .decode(raw["heads"]["stress"]["data"].as_str().unwrap())
        .unwrap();
    for removed in 1..=4 {
        let mut broken = raw.clone();
        broken["heads"]["stress"]["data"] = json!(base64::engine::general_purpose::STANDARD
            .encode(&compressed[..compressed.len() - removed]));
        assert!(decode(broken).is_err());
    }
    let mut broken = raw;
    let mut extra = compressed;
    extra.push(0);
    broken["heads"]["stress"]["data"] =
        json!(base64::engine::general_purpose::STANDARD.encode(extra));
    assert!(decode(broken).is_err());
}

#[test]
fn full_inventory_request_round_trips() {
    let request = lexide::pronunciation::PredictRequest {
        return_frame_matrix: true,
        return_all_heads: true,
        ..Default::default()
    };
    let raw = serde_json::to_value(&request).unwrap();
    assert_eq!(raw["return_all_heads"], true);
    assert_eq!(
        serde_json::from_value::<lexide::pronunciation::PredictRequest>(raw).unwrap(),
        request
    );
}

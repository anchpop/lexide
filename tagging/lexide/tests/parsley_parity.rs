//! Token-for-token parity with CPU fp32 predict_joint (not the bf16 GPU serve).
//! Set LEXIDE_MODEL_DIR to verified joint artifacts, or download the pinned release.

#![cfg(feature = "local")]

use std::path::PathBuf;

use lexide::pos::PartOfSpeech;
use lexide::{Language, LocalConfig, LocalLexide};

fn language(code: &str) -> Language {
    match code {
        "deu" => Language::German,
        "eng" => Language::English,
        "fra" => Language::French,
        "spa" => Language::SpanishEuro,
        "ita" => Language::Italian,
        "por" => Language::PortugueseBrazil,
        "rus" => Language::Russian,
        "kor" => Language::Korean,
        "hin" => Language::Hindi,
        "jpn" => Language::Japanese,
        "tha" => Language::Thai,
        "zho-hans" => Language::ChineseSimplified,
        other => panic!("unknown language code in fixtures: {other}"),
    }
}

#[test]
fn local_pipeline_matches_pytorch_fp32() {
    let lexide = LocalLexide::load(LocalConfig {
        threads: 4,
        ..Default::default()
    })
    .expect("failed to load local pipeline");
    let error = lexide
        .analyze(&"hello ".repeat(10_000), Language::English)
        .unwrap_err();
    assert!(error.to_string().contains("maximum is 8192"), "{error:#}");

    let fixtures: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parsley_reference.json"),
        )
        .unwrap(),
    )
    .unwrap();

    let mut sentences = 0;
    let mut tokens = 0;
    for fx in fixtures.as_array().unwrap() {
        let lang = language(fx["lang"].as_str().unwrap());
        let text = fx["text"].as_str().unwrap();
        let want = fx["tokens"].as_array().unwrap();

        let got = lexide
            .analyze(text, lang)
            .unwrap_or_else(|e| panic!("analyze failed for {text:?}: {e:#}"));

        assert_eq!(
            got.tokens().len(),
            want.len(),
            "token count diverges for {lang:?} {text:?}: got {:?}",
            got.tokens()
                .iter()
                .map(|t| &t.text.text)
                .collect::<Vec<_>>()
        );
        for (i, (g, w)) in got.tokens().iter().zip(want).enumerate() {
            let ctx = format!("{lang:?} {text:?} token {i} ({:?})", w["text"]);
            assert_eq!(g.text.text, w["text"].as_str().unwrap(), "text: {ctx}");
            assert_eq!(
                g.whitespace.as_str(),
                w["whitespace"].as_str().unwrap(),
                "whitespace: {ctx}"
            );
            let want_pos: PartOfSpeech = serde_plain::from_str(w["pos"].as_str().unwrap())
                .expect("PyTorch POS must be represented by the Rust API");
            assert_eq!(g.pos, want_pos, "pos: {ctx}");
            assert_eq!(g.lemma.lemma, w["lemma"].as_str().unwrap(), "lemma: {ctx}");
            let want_dep: lexide::DependencyRelation =
                serde_plain::from_str(w["dep"].as_str().unwrap())
                    .expect("PyTorch dependency relation must be represented by the Rust API");
            assert_eq!(g.dep, want_dep, "dep: {ctx}");
            assert_eq!(g.head, w["head"].as_i64().unwrap() as i32, "head: {ctx}");
            tokens += 1;
        }
        sentences += 1;
    }
    println!("parity OK: {sentences} sentences, {tokens} tokens match PyTorch fp32");
}

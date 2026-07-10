//! Token-for-token parity between the local ONNX pipeline and the parsley Modal serve.
//!
//! `tests/fixtures/parsley_reference.json` holds recorded responses from the live endpoint
//! (24 sentences across all 10 languages; refresh with
//! `scratchpad/fetch_parsley_fixtures.py`-style POSTs). The local pipeline — byte-minGRU
//! segmentation, ONNX tagging, fst lemma floor — must reproduce them exactly: same token
//! boundaries, POS, lemma, dependency relation, and head.
//!
//! Skips (with a note) when the model artifacts aren't present; fetch them with
//! `modal volume get lexide-onnx ...` into `tagging/data/onnx/` or point LEXIDE_MODEL_DIR
//! at them, and build the lemma tables with `build-lemma-fst`.

#![cfg(feature = "local")]

use std::path::PathBuf;

use lexide::pos::PartOfSpeech;
use lexide::{Language, LocalConfig, LocalLexide};

fn model_dir() -> PathBuf {
    std::env::var("LEXIDE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/onnx"))
}

fn language(code: &str) -> Language {
    match code {
        "deu" => Language::German,
        "eng" => Language::English,
        "fra" => Language::French,
        "spa" => Language::Spanish,
        "ita" => Language::Italian,
        "por" => Language::Portuguese,
        "rus" => Language::Russian,
        "kor" => Language::Korean,
        "hin" => Language::Hindi,
        "jpn" => Language::Japanese,
        other => panic!("unknown language code in fixtures: {other}"),
    }
}

#[test]
fn local_pipeline_matches_parsley_server() {
    let dir = model_dir();
    if !dir.join("tagger.onnx").exists() {
        eprintln!(
            "skipping: no ONNX artifacts at {} (set LEXIDE_MODEL_DIR)",
            dir.display()
        );
        return;
    }
    let have_tables = dir.join("lemma_fst").exists();
    assert!(
        have_tables,
        "lemma tables missing at {}/lemma_fst — run build-lemma-fst first \
         (the fixtures were recorded with the Wiktionary lemma floor active)",
        dir.display()
    );

    let lexide = LocalLexide::load(LocalConfig {
        model_dir: dir,
        lemma_tables_dir: None,
        threads: 0,
    })
    .expect("failed to load local pipeline");

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
            got.tokens.len(),
            want.len(),
            "token count diverges for {lang:?} {text:?}: got {:?}",
            got.tokens.iter().map(|t| &t.text.text).collect::<Vec<_>>()
        );
        for (i, (g, w)) in got.tokens.iter().zip(want).enumerate() {
            let ctx = format!("{lang:?} {text:?} token {i} ({:?})", w["text"]);
            assert_eq!(g.text.text, w["text"].as_str().unwrap(), "text: {ctx}");
            // Map the recorded strings through the same serde funnel the remote client
            // uses, so both sides degrade unknown tags identically.
            let want_pos: PartOfSpeech =
                serde_plain::from_str(w["pos"].as_str().unwrap()).unwrap_or(PartOfSpeech::X);
            assert_eq!(g.pos, want_pos, "pos: {ctx}");
            assert_eq!(g.lemma.lemma, w["lemma"].as_str().unwrap(), "lemma: {ctx}");
            let want_dep: lexide::DependencyRelation =
                serde_plain::from_str(w["dep"].as_str().unwrap())
                    .unwrap_or(lexide::DependencyRelation::Dep);
            assert_eq!(g.dep, want_dep, "dep: {ctx}");
            assert_eq!(g.head, w["head"].as_i64().unwrap() as i32, "head: {ctx}");
            tokens += 1;
        }
        sentences += 1;
    }
    println!("parity OK: {sentences} sentences, {tokens} tokens match the parsley serve");
}

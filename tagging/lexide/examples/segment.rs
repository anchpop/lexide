//! The whole sentence-segmentation API — no tagger, no ONNX, one ~4 MB download:
//!
//!     cargo run --example segment --no-default-features --features segment
//!
//! To go on and tag the sentences (POS/lemma/dependencies), load `Lexide` with the
//! `local` feature and call `analyze(&sentence, lang)` on each — see `simple.rs`.

fn main() -> anyhow::Result<()> {
    let parsley = lexide::Segmenter::from_pretrained()?; // ~4 MB download, cached
    let sentences = parsley.segment_in(
        "Dr. Smith arrived at 3 p.m. — he wasn't late. \"Is this the place?\" she asked.",
        lexide::Language::English,
    );
    assert_eq!(
        sentences,
        vec![
            "Dr. Smith arrived at 3 p.m. — he wasn't late.",
            "\"Is this the place?\" she asked.",
        ],
    );
    println!("{sentences:#?}");
    Ok(())
}

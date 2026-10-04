//! Exercise the unchanged parsley JSON client against a supplied serving URL.
#[cfg(feature = "remote")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use lexide::{Language, Lexide};
    let url = std::env::args().nth(1).expect("usage: parsley_remote URL");
    let client = Lexide::from_parsley_server(&url)?;
    for (text, language) in [
        ("The cats were sleeping.", Language::English),
        ("Eine Fundgrube.", Language::German),
        ("猫が寝ています。", Language::Japanese),
        ("", Language::English),
    ] {
        let result = client.analyze(text, language).await?;
        assert_eq!(result.reconstruct_text(), text);
        println!("{text:?}: {:?}", result.tokens());
    }
    Ok(())
}

#[cfg(not(feature = "remote"))]
fn main() {
    panic!("Run with --features remote");
}

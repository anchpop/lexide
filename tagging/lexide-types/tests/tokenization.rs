use lexide_types::{
    DependencyRelation, Lemma, PartOfSpeech, Text, Token, Tokenization, TokenizationError,
    Whitespace,
};
use serde_json::json;

fn token(text: &str, whitespace: Whitespace) -> Token {
    Token {
        text: Text { text: text.into() },
        whitespace,
        pos: PartOfSpeech::Noun,
        lemma: Lemma { lemma: text.into() },
        dep: DependencyRelation::Root,
        head: 0,
    }
}

#[test]
fn whitespace_is_strict_and_serializes_literal_strings() {
    for invalid in [json!({" ": null}), json!(null), json!(0)] {
        assert!(serde_json::from_value::<Whitespace>(invalid).is_err());
    }
    for (gap, expected) in [
        ("", Whitespace::None),
        (" ", Whitespace::Space),
        ("\u{a0}", Whitespace::Nbsp),
        ("\u{202f}", Whitespace::NarrowNbsp),
    ] {
        assert_eq!(gap.parse::<Whitespace>().unwrap(), expected);
        assert_eq!(expected.as_str(), gap);
        assert_eq!(expected.to_string(), gap);
        assert_eq!(serde_json::to_value(expected).unwrap(), json!(gap));
        assert_eq!(
            serde_json::from_value::<Whitespace>(json!(gap)).unwrap(),
            expected
        );
    }
    for gap in [
        ", ", "  ", "\t", "\n", "\u{2009}", "\u{200a}", "\u{200b}", "\u{3000}", "Space", "none",
        "_",
    ] {
        assert_eq!(gap.parse::<Whitespace>().unwrap_err().0, gap);
        assert!(serde_json::from_value::<Whitespace>(json!(gap)).is_err());
    }
}

#[test]
fn constructor_and_accessors_preserve_sentence_and_tokens() {
    let tokens = vec![
        token("Hello", Whitespace::Space),
        token("world", Whitespace::None),
    ];
    let parsed = Tokenization::new("Hello world", tokens.clone()).unwrap();
    assert_eq!(parsed.sentence(), "Hello world");
    assert_eq!(parsed.reconstruct_text(), parsed.sentence());
    assert_eq!(parsed.tokens(), tokens);
    assert_eq!(
        parsed.texts(),
        tokens.iter().map(|t| t.text.clone()).collect::<Vec<_>>()
    );
    assert_eq!(
        parsed.lemmas(),
        tokens.iter().map(|t| t.lemma.clone()).collect::<Vec<_>>()
    );
    assert_eq!(parsed.clone().into_tokens(), tokens);
    assert_eq!(parsed.into_parts(), ("Hello world".into(), tokens));
    assert!(Tokenization::new("", vec![]).is_ok());
    assert!(Tokenization::new("x", vec![]).is_err());
}

#[test]
fn constructor_rejects_empty_or_whitespace_containing_tokens() {
    for text in [
        "",
        "a b",
        "a\tb",
        "a\nb",
        "a\u{a0}b",
        "a\u{202f}b",
        "\u{3000}",
    ] {
        assert_eq!(
            Tokenization::new(
                format!("ok {text}"),
                vec![
                    token("ok", Whitespace::Space),
                    token(text, Whitespace::None)
                ]
            )
            .unwrap_err(),
            TokenizationError::InvalidTokenText {
                index: 1,
                text: text.into()
            }
        );
    }
}

#[test]
fn constructor_rejects_mismatch() {
    assert_eq!(
        Tokenization::new("Hello!", vec![token("Hello", Whitespace::None)]).unwrap_err(),
        TokenizationError::ReconstructionMismatch {
            sentence: "Hello!".into(),
            reconstructed: "Hello".into()
        }
    );
}

#[test]
fn serde_enforces_constructor_and_requires_sentence() {
    let valid = Tokenization::new("hello", vec![token("hello", Whitespace::None)]).unwrap();
    let wire = serde_json::to_value(&valid).unwrap();
    assert_eq!(wire["sentence"], "hello");
    assert_eq!(
        serde_json::from_value::<Tokenization>(wire.clone())
            .unwrap()
            .tokens(),
        valid.tokens()
    );
    for text in ["", "hello world"] {
        let mut invalid = wire.clone();
        invalid["tokens"][0]["text"]["text"] = json!(text);
        invalid["sentence"] = json!(text);
        assert!(serde_json::from_value::<Tokenization>(invalid).is_err());
    }
    let mut mismatch = wire.clone();
    mismatch["sentence"] = json!("other");
    assert!(serde_json::from_value::<Tokenization>(mismatch).is_err());
    let mut missing = wire;
    missing.as_object_mut().unwrap().remove("sentence");
    assert!(serde_json::from_value::<Tokenization>(missing).is_err());
}

#[test]
fn hindi_comma_must_be_an_explicit_token_not_a_gap() {
    let mut comma = token(",", Whitespace::Space);
    comma.pos = PartOfSpeech::Punct;
    comma.dep = DependencyRelation::Punct;
    let parsed = Tokenization::new(
        "हाँ, ठीक",
        vec![
            token("हाँ", Whitespace::None),
            comma,
            token("ठीक", Whitespace::None),
        ],
    )
    .unwrap();
    let mut wire = serde_json::to_value(&parsed).unwrap();
    assert!(serde_json::from_value::<Tokenization>(wire.clone()).is_ok());
    wire["tokens"].as_array_mut().unwrap().remove(1);
    wire["tokens"][0]["whitespace"] = json!(", ");
    assert!(serde_json::from_value::<Tokenization>(wire).is_err());
}

#[cfg(feature = "rkyv")]
#[test]
fn whitespace_archive_round_trip() {
    for gap in [
        Whitespace::None,
        Whitespace::Space,
        Whitespace::Nbsp,
        Whitespace::NarrowNbsp,
    ] {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&gap).unwrap();
        assert_eq!(
            rkyv::from_bytes::<Whitespace, rkyv::rancor::Error>(&bytes).unwrap(),
            gap
        );
    }
}

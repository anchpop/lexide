//! Shared conversion from raw tagger output (strings + char offsets) into `Tokenization`.
//! Used by both the parsley remote client and the local ONNX backend so whitespace
//! reconstruction and unknown-tag degradation behave identically.

use crate::{dep::DependencyRelation, pos::PartOfSpeech, Lemma, Text, Token, Tokenization};

/// One token as produced by the parsley tagger (server or local): surface form, char
/// offsets into the sentence, and string labels.
#[derive(Debug, Clone)]
pub(crate) struct RawToken {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub pos: String,
    pub lemma: String,
    pub dep: String,
    pub head: i32,
}

/// Build a `Tokenization` from raw tokens. Each token's whitespace is the character gap to
/// the next token's start offset (offsets are char indices, so we index `sentence` by
/// char). Unknown POS/dep strings degrade to `X`/`dep` rather than failing a user request.
pub(crate) fn tokens_from_raw(
    rtoks: &[RawToken],
    sentence: &str,
) -> Result<Tokenization, crate::TokenizationError> {
    let chars: Vec<char> = sentence.chars().collect();
    let mut tokens = Vec::with_capacity(rtoks.len());
    for (i, rt) in rtoks.iter().enumerate() {
        let next_start = rtoks.get(i + 1).map(|n| n.start).unwrap_or(chars.len());
        if !(rt.start <= rt.end && rt.end <= next_start && next_start <= chars.len()) {
            return Err(crate::TokenizationError::InvalidOffsets {
                index: i,
                text: rt.text.clone(),
                start: rt.start,
                end: rt.end,
                next_start,
                sentence_len: chars.len(),
            });
        }
        let gap: String = chars[rt.end..next_start].iter().collect();
        let whitespace = gap
            .parse()
            .map_err(|_| crate::TokenizationError::InvalidGap { index: i, gap })?;
        let pos: PartOfSpeech = serde_plain::from_str(&rt.pos).unwrap_or(PartOfSpeech::X);
        let dep: DependencyRelation =
            serde_plain::from_str(&rt.dep).unwrap_or(DependencyRelation::Dep);
        tokens.push(Token {
            text: Text {
                text: rt.text.clone(),
            },
            whitespace,
            pos,
            lemma: Lemma {
                lemma: rt.lemma.clone(),
            },
            dep,
            head: rt.head,
        });
    }
    Tokenization::new(sentence, tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dep::DependencyRelation, pos::PartOfSpeech};

    fn rt(
        text: &str,
        start: usize,
        end: usize,
        pos: &str,
        lemma: &str,
        dep: &str,
        head: i32,
    ) -> RawToken {
        RawToken {
            text: text.into(),
            start,
            end,
            pos: pos.into(),
            lemma: lemma.into(),
            dep: dep.into(),
            head,
        }
    }

    #[test]
    fn raw_tokens_reconstruct_and_map() {
        let sentence = "Eine Fundgrube.";
        let rtoks = vec![
            rt("Eine", 0, 4, "DET", "ein", "det", 2),
            rt("Fundgrube", 5, 14, "NOUN", "Fundgrube", "root", 0),
            rt(".", 14, 15, "PUNCT", ".", "punct", 2),
        ];
        let t = tokens_from_raw(&rtoks, sentence).unwrap();
        // whitespace derived from offsets reconstructs the sentence exactly
        assert_eq!(t.reconstruct_text(), sentence);
        assert_eq!(t.tokens()[0].whitespace, crate::Whitespace::Space);
        assert_eq!(t.tokens()[1].whitespace, crate::Whitespace::None);
        assert_eq!(t.tokens()[0].pos, PartOfSpeech::Det);
        assert_eq!(t.tokens()[1].lemma.lemma, "Fundgrube");
        assert_eq!(t.tokens()[0].head, 2);
        assert_eq!(t.tokens()[2].dep, DependencyRelation::Punct);
    }

    #[test]
    fn raw_offsets_are_char_indexed_not_byte() {
        // Cyrillic + a multibyte gap: offsets must index chars, not bytes.
        let sentence = "я им";
        let rtoks = vec![
            rt("я", 0, 1, "PRON", "я", "nsubj", 2),
            rt("им", 2, 4, "PRON", "они", "obl", 0),
        ];
        let t = tokens_from_raw(&rtoks, sentence).unwrap();
        assert_eq!(t.reconstruct_text(), sentence);
        assert_eq!(t.tokens()[0].whitespace, crate::Whitespace::Space);
        assert_eq!(t.tokens()[1].lemma.lemma, "они");
    }

    #[test]
    fn skipped_hindi_comma_is_not_whitespace() {
        let raw = vec![
            rt("हाँ", 0, 3, "INTJ", "हाँ", "root", 0),
            rt("ठीक", 5, 8, "ADJ", "ठीक", "dep", 1),
        ];
        let error = tokens_from_raw(&raw, "हाँ, ठीक").unwrap_err();
        assert_eq!(
            error,
            crate::TokenizationError::InvalidGap {
                index: 0,
                gap: ", ".into()
            }
        );
        let mut explicit = raw;
        explicit.insert(1, rt(",", 3, 4, "PUNCT", ",", "punct", 1));
        assert!(tokens_from_raw(&explicit, "हाँ, ठीक").is_ok());
    }

    #[test]
    fn invalid_offsets_and_shapes_are_rejected() {
        for (text, start, end, sentence) in [
            ("x", 1, 0, "x"),
            ("x", 0, 2, "x"),
            ("x", 0, 1, "x\t"),
            ("a ", 0, 2, "a "),
            (" a", 0, 2, " a"),
            ("x", 1, 2, " x"),
        ] {
            assert!(
                tokens_from_raw(&[rt(text, start, end, "X", text, "root", 0)], sentence).is_err()
            );
        }
    }

    #[test]
    fn unknown_tags_degrade_gracefully() {
        let rtoks = vec![rt("x", 0, 1, "WEIRD", "x", "nonsense:sub", 0)];
        let t = tokens_from_raw(&rtoks, "x").unwrap();
        assert_eq!(t.tokens()[0].pos, PartOfSpeech::X);
        assert_eq!(t.tokens()[0].dep, DependencyRelation::Dep);
    }
}

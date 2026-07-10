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
pub(crate) fn tokens_from_raw(rtoks: &[RawToken], sentence: &str) -> Tokenization {
    let chars: Vec<char> = sentence.chars().collect();
    let mut tokens = Vec::with_capacity(rtoks.len());
    for (i, rt) in rtoks.iter().enumerate() {
        let next_start = rtoks.get(i + 1).map(|n| n.start).unwrap_or(chars.len());
        let whitespace: String = if rt.end <= next_start && next_start <= chars.len() {
            chars[rt.end..next_start].iter().collect()
        } else {
            String::new()
        };
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
    Tokenization { tokens }
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
        let t = tokens_from_raw(&rtoks, sentence);
        // whitespace derived from offsets reconstructs the sentence exactly
        assert_eq!(t.reconstruct_text(), sentence);
        assert_eq!(t.tokens[0].whitespace, " ");
        assert_eq!(t.tokens[1].whitespace, "");
        assert_eq!(t.tokens[0].pos, PartOfSpeech::Det);
        assert_eq!(t.tokens[1].lemma.lemma, "Fundgrube");
        assert_eq!(t.tokens[0].head, 2);
        assert_eq!(t.tokens[2].dep, DependencyRelation::Punct);
    }

    #[test]
    fn raw_offsets_are_char_indexed_not_byte() {
        // Cyrillic + a multibyte gap: offsets must index chars, not bytes.
        let sentence = "я им";
        let rtoks = vec![
            rt("я", 0, 1, "PRON", "я", "nsubj", 2),
            rt("им", 2, 4, "PRON", "они", "obl", 0),
        ];
        let t = tokens_from_raw(&rtoks, sentence);
        assert_eq!(t.reconstruct_text(), sentence);
        assert_eq!(t.tokens[0].whitespace, " ");
        assert_eq!(t.tokens[1].lemma.lemma, "они");
    }

    #[test]
    fn unknown_tags_degrade_gracefully() {
        let rtoks = vec![rt("x", 0, 1, "WEIRD", "x", "nonsense:sub", 0)];
        let t = tokens_from_raw(&rtoks, "x");
        assert_eq!(t.tokens[0].pos, PartOfSpeech::X);
        assert_eq!(t.tokens[0].dep, DependencyRelation::Dep);
    }
}

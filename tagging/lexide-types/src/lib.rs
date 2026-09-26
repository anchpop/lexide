pub mod dep;
pub mod pos;
pub use dep::DependencyRelation;
pub use pos::PartOfSpeech;
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Text {
    pub text: String,
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Lemma {
    pub lemma: String,
}

impl fmt::Display for Lemma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.lemma)
    }
}

/// A lemma paired with its part of speech, for POS-aware matching.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct LemmaPos {
    pub lemma: String,
    pub pos: PartOfSpeech,
}

impl fmt::Display for LemmaPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.lemma, self.pos)
    }
}

/// Represents a single token with its linguistic annotations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Token {
    pub text: Text,
    pub whitespace: Whitespace,
    pub pos: PartOfSpeech,
    pub lemma: Lemma,
    pub dep: DependencyRelation,
    pub head: i32,
}

/// The only supported gaps after a token. Serialized as the literal gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub enum Whitespace {
    None,
    Space,
    Nbsp,
    NarrowNbsp,
}

impl Serialize for Whitespace {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Whitespace {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

impl Whitespace {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Space => " ",
            Self::Nbsp => "\u{00a0}",
            Self::NarrowNbsp => "\u{202f}",
        }
    }
}

impl fmt::Display for Whitespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidWhitespace(pub String);

impl fmt::Display for InvalidWhitespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unsupported whitespace gap {:?}", self.0)
    }
}
impl std::error::Error for InvalidWhitespace {}

impl FromStr for Whitespace {
    type Err = InvalidWhitespace;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "" => Ok(Self::None),
            " " => Ok(Self::Space),
            "\u{00a0}" => Ok(Self::Nbsp),
            "\u{202f}" => Ok(Self::NarrowNbsp),
            _ => Err(InvalidWhitespace(s.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizationError {
    InvalidOffsets {
        index: usize,
        text: String,
        start: usize,
        end: usize,
        next_start: usize,
        sentence_len: usize,
    },
    InvalidTokenText {
        index: usize,
        text: String,
    },
    InvalidGap {
        index: usize,
        gap: String,
    },
    ReconstructionMismatch {
        sentence: String,
        reconstructed: String,
    },
}

impl fmt::Display for TokenizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOffsets {
                index,
                text,
                start,
                end,
                next_start,
                sentence_len,
            } => write!(
                f,
                "token {index} ({text:?}) has invalid character offsets {start}..{end} (next start {next_start}, sentence length {sentence_len})"
            ),
            Self::InvalidTokenText { index, text } => write!(
                f,
                "token {index} has empty text or whitespace at its edge {text:?}"
            ),
            Self::InvalidGap { index, gap } => {
                write!(f, "token {index} has unsupported whitespace gap {gap:?}")
            }
            Self::ReconstructionMismatch {
                sentence,
                reconstructed,
            } => write!(
                f,
                "reconstructed text {reconstructed:?} does not match sentence {sentence:?}"
            ),
        }
    }
}
impl std::error::Error for TokenizationError {}

/// A sentence and tokens that reconstruct it exactly. Construction and deserialization
/// validate token text; accessors cannot mutate the validated representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "RawTokenization")]
pub struct Tokenization {
    sentence: String,
    tokens: Vec<Token>,
}

#[derive(Deserialize)]
struct RawTokenization {
    sentence: String,
    tokens: Vec<Token>,
}

impl TryFrom<RawTokenization> for Tokenization {
    type Error = TokenizationError;

    fn try_from(raw: RawTokenization) -> Result<Self, Self::Error> {
        Self::new(raw.sentence, raw.tokens)
    }
}

impl Tokenization {
    pub fn new(sentence: impl Into<String>, tokens: Vec<Token>) -> Result<Self, TokenizationError> {
        for (index, token) in tokens.iter().enumerate() {
            // Leading or trailing whitespace belongs in the gap, never in the
            // token. Internal whitespace is allowed: multiword names such as
            // "New York" are deliberately single tokens.
            let text = &token.text.text;
            if text.is_empty() || text.trim() != text {
                return Err(TokenizationError::InvalidTokenText {
                    index,
                    text: token.text.text.clone(),
                });
            }
        }
        let result = Self {
            sentence: sentence.into(),
            tokens,
        };
        let reconstructed = result.reconstruct_text();
        if reconstructed != result.sentence {
            return Err(TokenizationError::ReconstructionMismatch {
                sentence: result.sentence,
                reconstructed,
            });
        }
        Ok(result)
    }

    pub fn sentence(&self) -> &str {
        &self.sentence
    }
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }
    pub fn into_tokens(self) -> Vec<Token> {
        self.tokens
    }
    pub fn into_parts(self) -> (String, Vec<Token>) {
        (self.sentence, self.tokens)
    }

    pub fn reconstruct_text(&self) -> String {
        self.tokens
            .iter()
            .map(|token| format!("{}{}", token.text, token.whitespace))
            .collect()
    }
    pub fn texts(&self) -> Vec<Text> {
        self.tokens.iter().map(|token| token.text.clone()).collect()
    }
    pub fn lemmas(&self) -> Vec<Lemma> {
        self.tokens
            .iter()
            .map(|token| token.lemma.clone())
            .collect()
    }
}

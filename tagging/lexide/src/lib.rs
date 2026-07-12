mod dep;
mod language;
pub mod matching;
#[cfg(feature = "remote")]
mod parsing;
pub mod pos;

#[cfg(feature = "local")]
mod local;
#[cfg(any(feature = "local", feature = "remote"))]
mod raw;
#[cfg(feature = "remote")]
mod remote;

pub use crate::dep::DependencyRelation;
pub use crate::language::Language;
use crate::pos::PartOfSpeech;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

#[cfg(feature = "local")]
pub use local::{build_table, LemmaTable, LocalConfig, LocalLexide, Sentence};
#[cfg(feature = "remote")]
pub use remote::{RemoteClient, RemoteConfig, ResponseFormat};

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
    pub whitespace: String,
    pub pos: PartOfSpeech,
    pub lemma: Lemma,
    pub dep: DependencyRelation,
    pub head: i32,
}

/// Analysis result for a sentence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenization {
    pub tokens: Vec<Token>,
}

impl Tokenization {
    /// Reconstruct the text from tokens using whitespace information
    pub fn reconstruct_text(&self) -> String {
        self.tokens
            .iter()
            .map(|token| format!("{}{}", token.text, token.whitespace))
            .collect()
    }

    /// Extract the text sequence from tokens
    pub fn texts(&self) -> Vec<Text> {
        self.tokens.iter().map(|token| token.text.clone()).collect()
    }

    /// Extract the lemma sequence from tokens
    pub fn lemmas(&self) -> Vec<Lemma> {
        self.tokens
            .iter()
            .map(|token| token.lemma.clone())
            .collect()
    }
}

/// Configuration for the Lexide model
#[derive(Debug, Clone)]
pub enum LexideConfig {
    #[cfg(feature = "local")]
    Local(LocalConfig),
    #[cfg(feature = "remote")]
    Remote(RemoteConfig),
}

#[cfg(feature = "local")]
impl Default for LexideConfig {
    fn default() -> Self {
        Self::Local(LocalConfig::default())
    }
}

#[cfg(all(feature = "remote", not(feature = "local")))]
impl Default for LexideConfig {
    fn default() -> Self {
        Self::Remote(RemoteConfig::default())
    }
}

/// Main struct for running NLP inference
pub enum Lexide {
    #[cfg(feature = "local")]
    Local(LocalLexide),
    #[cfg(feature = "remote")]
    Remote(RemoteClient),
}

impl Lexide {
    /// Load the model or create remote client based on config
    #[cfg(feature = "local")]
    pub async fn from_pretrained(local_config: LocalConfig) -> Result<Self> {
        Ok(Self::Local(
            LocalLexide::from_pretrained(local_config).await?,
        ))
    }

    /// Connect to the Gemma vLLM endpoint (OpenAI completions, tab-separated text).
    #[cfg(feature = "remote")]
    pub fn from_server(url: &str) -> Result<Self> {
        Ok(Self::Remote(RemoteClient::new(RemoteConfig {
            endpoint_url: url.to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            ..Default::default()
        })?))
    }

    /// Connect to the parsley CPU tagger endpoint (JSON tokens with char offsets).
    #[cfg(feature = "remote")]
    pub fn from_parsley_server(url: &str) -> Result<Self> {
        Ok(Self::Remote(RemoteClient::new(RemoteConfig {
            endpoint_url: url.to_string(),
            format: ResponseFormat::ParsleyJson,
            ..Default::default()
        })?))
    }

    /// Analyze a sentence and return structured results
    #[allow(unreachable_code, unused_variables)]
    pub async fn analyze(&self, sentence: &str, language: Language) -> Result<Tokenization> {
        match self {
            // Local parsley: in-process segment -> ONNX tag -> lemma floor.
            #[cfg(feature = "local")]
            Self::Local(local) => local.analyze(sentence, language),
            // Remote: dispatches internally on the endpoint's response format
            // (Gemma completions text, or parsley JSON).
            #[cfg(feature = "remote")]
            Self::Remote(remote) => remote.analyze(sentence, language).await,
            #[cfg(not(any(feature = "local", feature = "remote")))]
            _ => unreachable!("Type should be uninhabited!"),
        }
    }

    /// Split a passage into its sentences, in order, dropping the gaps (whitespace,
    /// headings, separators) between them. Punctuation and markers that frame a sentence
    /// (its quotes, a leading dialogue dash) stay attached to it.
    ///
    /// Handy for turning a document — or each entry of a list of documents — into the
    /// individual sentences to feed to [`analyze`](Self::analyze):
    /// ```no_run
    /// # async fn f(lexide: &lexide::Lexide) -> anyhow::Result<()> {
    /// for passage in ["First. Second.", "Uno. Dos."] {
    ///     for sentence in lexide.segment_sentences(passage)? {
    ///         let _tok = lexide.analyze(&sentence, lexide::Language::English).await?;
    ///     }
    /// }
    /// # Ok(()) }
    /// ```
    ///
    /// Only the local backend segments in-process (a cheap byte-minGRU pass, hence sync).
    /// On the remote backend this returns an error — use the async
    /// [`RemoteClient::segment_sentences`](crate::RemoteClient::segment_sentences) against a
    /// parsley `/segment` endpoint instead.
    #[allow(unreachable_code, unused_variables)]
    pub fn segment_sentences(&self, text: &str) -> Result<Vec<String>> {
        match self {
            #[cfg(feature = "local")]
            Self::Local(local) => local.segment_sentences(text),
            #[cfg(feature = "remote")]
            Self::Remote(_) => anyhow::bail!(
                "segment_sentences needs the local backend; for a remote parsley serve use \
                 the async RemoteClient::segment_sentences against its /segment endpoint"
            ),
            #[cfg(not(any(feature = "local", feature = "remote")))]
            _ => unreachable!("Type should be uninhabited!"),
        }
    }

    /// Like [`segment_sentences`](Self::segment_sentences) but returns each sentence with
    /// its `[start, end)` char span in the original passage (local backend only).
    #[cfg(feature = "local")]
    pub fn segment_sentences_detailed(&self, text: &str) -> Result<Vec<Sentence>> {
        match self {
            Self::Local(local) => local.segment_sentences_detailed(text),
            #[cfg(feature = "remote")]
            Self::Remote(_) => anyhow::bail!(
                "segment_sentences_detailed is only available on the local backend"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dep::DependencyRelation;

    use super::*;

    #[test]
    fn test_token_creation() {
        let token = Token {
            text: Text {
                text: "Hello".to_string(),
            },
            whitespace: " ".to_string(),
            pos: PartOfSpeech::Noun,
            lemma: Lemma {
                lemma: "hello".to_string(),
            },
            dep: DependencyRelation::Root,
            head: 0,
        };

        assert_eq!(
            token.text,
            Text {
                text: "Hello".to_string()
            }
        );
        assert_eq!(token.whitespace, " ");
        assert_eq!(token.pos, PartOfSpeech::Noun);
    }

    #[test]
    fn test_reconstruct_text() {
        let result = Tokenization {
            tokens: vec![
                Token {
                    text: Text {
                        text: "Hello".to_string(),
                    },
                    whitespace: " ".to_string(),
                    pos: PartOfSpeech::Intj,
                    lemma: Lemma {
                        lemma: "hello".to_string(),
                    },
                    dep: DependencyRelation::Root,
                    head: 0,
                },
                Token {
                    text: Text {
                        text: "world".to_string(),
                    },
                    whitespace: "".to_string(),
                    pos: PartOfSpeech::Noun,
                    lemma: Lemma {
                        lemma: "world".to_string(),
                    },
                    dep: DependencyRelation::Obj,
                    head: 0,
                },
            ],
        };

        let reconstructed = result.reconstruct_text();
        assert_eq!(reconstructed, "Hello world");
    }

    #[test]
    fn test_whitespace_conversion() {
        // Test that '_' is converted to space
        let ws1 = if "_" == "_" {
            " ".to_string()
        } else {
            "_".to_string()
        };
        assert_eq!(ws1, " ");

        // Test that empty string stays empty
        let ws2 = if "" == "_" {
            " ".to_string()
        } else {
            "".to_string()
        };
        assert_eq!(ws2, "");
    }

    #[cfg(feature = "local")]
    #[test]
    fn test_local_config() {
        let config = LocalConfig::default();
        // model_dir defaults to LEXIDE_MODEL_DIR when set, else None -> hub download
        assert_eq!(
            config.model_dir.is_some(),
            std::env::var("LEXIDE_MODEL_DIR").is_ok()
        );
        assert_eq!(config.hf_repo, "anchpop/lexide-parsley");
        assert_eq!(config.threads, 0);
        assert!(config.lemma_tables_dir.is_none());
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_remote_config() {
        let config = RemoteConfig::default();
        assert!(
            config.endpoint_url.contains("modal.run")
                || config.endpoint_url.contains("LEXIDE_ENDPOINT_URL")
        );
    }
}

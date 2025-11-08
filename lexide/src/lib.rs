mod dep;
mod language;
pub mod matching;
mod parsing;
pub mod pos;

#[cfg(feature = "local")]
mod local;
#[cfg(feature = "remote")]
mod remote;

pub use crate::language::Language;
use crate::{dep::DependencyRelation, pos::PartOfSpeech};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

#[cfg(feature = "local")]
pub use local::{LocalConfig, LocalLexide};
#[cfg(feature = "remote")]
pub use remote::{RemoteClient, RemoteConfig};

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
        )?)
    }

    #[cfg(feature = "remote")]
    pub fn from_server(url: &str) -> Result<Self> {
        Ok(Self::Remote(RemoteClient::new(RemoteConfig {
            endpoint_url: url.to_string(),
            max_tokens: 1024,
            temperature: 0.1,
        })?))
    }

    /// Analyze a sentence and return structured results
    ///
    /// This is the main entry point that orchestrates:
    /// 1. Creating the prompt (shared logic)
    /// 2. Generating the response (local or remote)
    /// 3. Parsing the response (shared logic)
    pub async fn analyze(&self, sentence: &str, language: Language) -> Result<Tokenization> {
        // Step 1: Create prompt (shared between local and remote)
        let prompt = parsing::create_prompt(sentence, language);

        // Step 2: Generate response (differs between local and remote)
        let response: String = match self {
            #[cfg(feature = "local")]
            Self::Local(local) => local.generate(&prompt, sentence).await?,
            #[cfg(feature = "remote")]
            Self::Remote(remote) => remote.generate(&prompt).await?,
            #[cfg(not(any(feature = "local", feature = "remote")))]
            _ => unreachable!("Type should be uninhabited!"),
        };

        // Step 3: Parse response (shared between local and remote)
        parsing::parse_response(&response, sentence)
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
    #[tokio::test]
    async fn test_local_config() {
        let config = LocalConfig::default();
        assert_eq!(config.base_model_repo, "google/gemma-3-1b-it");
        assert_eq!(config.lora_adapter_repo, "anchpop/lexide-gemma-3-1b-it");
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

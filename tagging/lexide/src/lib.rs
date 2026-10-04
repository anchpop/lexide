pub use lexide_types::*;
pub mod matching;
#[cfg(feature = "remote")]
mod parsing;
#[cfg(feature = "pronunciation")]
pub mod pronunciation;

#[cfg(feature = "local")]
mod local;
#[cfg(any(feature = "local", feature = "remote"))]
mod raw;
#[cfg(feature = "remote")]
mod remote;
#[cfg(feature = "segment")]
pub mod segment;

use anyhow::Result;
pub use g2p_types::Language;

#[cfg(feature = "local")]
pub use local::{LocalConfig, LocalLexide, MODEL_REVISION};
#[cfg(feature = "remote")]
pub use remote::{RemoteClient, RemoteConfig, ResponseFormat};
#[cfg(feature = "segment")]
pub use segment::{Segmenter, Sentence};

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

    /// Connect to the parsley joint tagger endpoint (JSON tokens with char offsets).
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
            // Local parsley: joint encoder and word heads.
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
}

#[cfg(all(test, any(feature = "local", feature = "remote")))]
mod tests {
    use super::*;

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

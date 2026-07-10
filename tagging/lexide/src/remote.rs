/// Remote inference client for Modal endpoints (Gemma completions or the parsley JSON tagger)
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{dep::DependencyRelation, pos::PartOfSpeech, Language, Lemma, Text, Token, Tokenization};

/// Which response format the configured endpoint speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Gemma vLLM: OpenAI `/v1/completions` returning tab-separated text (the original serve).
    GemmaCompletions,
    /// parsley: `POST {sentences, lang}` -> JSON tokens with char offsets (the CPU tagger).
    ParsleyJson,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    temperature: f64,
    max_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ParsleyRequest {
    sentences: Vec<String>,
    lang: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ParsleyResponse {
    results: Vec<Vec<ParsleyToken>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ParsleyToken {
    text: String,
    start: usize,
    end: usize,
    pos: String,
    lemma: String,
    dep: String,
    head: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompletionResponse {
    choices: Vec<CompletionChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompletionChoice {
    text: String,
}

/// Configuration for remote inference
#[derive(Debug, Clone)]
pub struct RemoteConfig {
    /// Modal endpoint URL
    pub endpoint_url: String,
    /// Maximum tokens for generation
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f64,
    /// Maximum idle connections per host (default: 256)
    pub pool_max_idle_per_host: usize,
    /// Idle connection timeout in seconds (default: 300)
    pub pool_idle_timeout_secs: u64,
    /// Response format the endpoint speaks (default: Gemma completions)
    pub format: ResponseFormat,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            endpoint_url: std::env::var("LEXIDE_ENDPOINT_URL")
                .unwrap_or_else(|_| "https://anchpop--lexide-gemma-4-31b-vllm-serve.modal.run".to_string()),
            max_tokens: 512,
            temperature: 0.0,
            pool_max_idle_per_host: 256,
            pool_idle_timeout_secs: 300,
            format: ResponseFormat::GemmaCompletions,
        }
    }
}

/// Remote inference client
pub struct RemoteClient {
    config: RemoteConfig,
    client: reqwest::Client,
}

impl RemoteClient {
    /// Create a new remote client with connection pooling
    pub fn new(config: RemoteConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600)) // 10 minutes for cold starts
            .connect_timeout(std::time::Duration::from_secs(30))
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .pool_idle_timeout(std::time::Duration::from_secs(
                config.pool_idle_timeout_secs,
            ))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { config, client })
    }

    /// Generate a response from a prompt
    ///
    /// This is the only method that differs between local and remote implementations.
    /// It takes a pre-formatted prompt and returns the raw model response.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        // Use completions API with "1\t" prefix to prime the model for proper format
        let primed_prompt = format!("{}1\t", prompt);

        let request = CompletionRequest {
            model: "lexide-gemma-4-31b".to_string(),
            prompt: primed_prompt,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        // Use /v1/completions endpoint (not chat/completions)
        let url = format!(
            "{}/v1/completions",
            self.config.endpoint_url.trim_end_matches('/')
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request to remote endpoint")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Remote endpoint returned error {}: {}", status, error_text);
        }

        let completion_response: CompletionResponse = response
            .json()
            .await
            .context("Failed to parse response from remote endpoint")?;

        // Prepend "1\t" back to the response since we used it to prime the model
        let text = completion_response
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        Ok(format!("1\t{}", text))
    }

    /// Analyze a sentence, dispatching on the configured response format.
    pub async fn analyze(&self, sentence: &str, language: Language) -> Result<Tokenization> {
        match self.config.format {
            ResponseFormat::GemmaCompletions => {
                let prompt = crate::parsing::create_prompt(sentence, language);
                let response = self.generate(&prompt).await?;
                crate::parsing::parse_response(&response, sentence)
            }
            ResponseFormat::ParsleyJson => self.tokenize_parsley(sentence, language).await,
        }
    }

    /// parsley path: POST the raw sentence, deserialize JSON tokens, and rebuild each token's
    /// trailing whitespace from the char offsets (exact by construction — no reconstruction
    /// heuristics needed, unlike the Gemma text parser).
    async fn tokenize_parsley(&self, sentence: &str, language: Language) -> Result<Tokenization> {
        let request = ParsleyRequest {
            sentences: vec![sentence.to_string()],
            lang: language.code().to_string(),
        };
        let url = self.config.endpoint_url.trim_end_matches('/');

        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request to parsley endpoint")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("parsley endpoint returned error {}: {}", status, error_text);
        }

        let parsed: ParsleyResponse = response
            .json()
            .await
            .context("Failed to parse JSON from parsley endpoint")?;

        let toks = parsed
            .results
            .into_iter()
            .next()
            .context("parsley returned no results for the sentence")?;

        Ok(tokens_from_parsley(&toks, sentence))
    }
}

/// Build a `Tokenization` from parsley tokens. Each token's whitespace is the character gap to
/// the next token's start offset (offsets are Python char indices, so we index `sentence` by
/// char). Unknown POS/dep strings degrade to `X`/`dep` rather than failing a user request.
fn tokens_from_parsley(ptoks: &[ParsleyToken], sentence: &str) -> Tokenization {
    let chars: Vec<char> = sentence.chars().collect();
    let mut tokens = Vec::with_capacity(ptoks.len());
    for (i, pt) in ptoks.iter().enumerate() {
        let next_start = ptoks.get(i + 1).map(|n| n.start).unwrap_or(chars.len());
        let whitespace: String = if pt.end <= next_start && next_start <= chars.len() {
            chars[pt.end..next_start].iter().collect()
        } else {
            String::new()
        };
        let pos: PartOfSpeech = serde_plain::from_str(&pt.pos).unwrap_or(PartOfSpeech::X);
        let dep: DependencyRelation =
            serde_plain::from_str(&pt.dep).unwrap_or(DependencyRelation::Dep);
        tokens.push(Token {
            text: Text {
                text: pt.text.clone(),
            },
            whitespace,
            pos,
            lemma: Lemma {
                lemma: pt.lemma.clone(),
            },
            dep,
            head: pt.head,
        });
    }
    Tokenization { tokens }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dep::DependencyRelation, pos::PartOfSpeech};

    fn pt(text: &str, start: usize, end: usize, pos: &str, lemma: &str, dep: &str, head: i32) -> ParsleyToken {
        ParsleyToken {
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
    fn parsley_tokens_reconstruct_and_map() {
        let sentence = "Eine Fundgrube.";
        let ptoks = vec![
            pt("Eine", 0, 4, "DET", "ein", "det", 2),
            pt("Fundgrube", 5, 14, "NOUN", "Fundgrube", "root", 0),
            pt(".", 14, 15, "PUNCT", ".", "punct", 2),
        ];
        let t = tokens_from_parsley(&ptoks, sentence);
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
    fn parsley_offsets_are_char_indexed_not_byte() {
        // Cyrillic + a multibyte gap: offsets must index chars, not bytes.
        let sentence = "я им";
        let ptoks = vec![pt("я", 0, 1, "PRON", "я", "nsubj", 2), pt("им", 2, 4, "PRON", "они", "obl", 0)];
        let t = tokens_from_parsley(&ptoks, sentence);
        assert_eq!(t.reconstruct_text(), sentence);
        assert_eq!(t.tokens[0].whitespace, " ");
        assert_eq!(t.tokens[1].lemma.lemma, "они");
    }

    #[test]
    fn unknown_tags_degrade_gracefully() {
        let ptoks = vec![pt("x", 0, 1, "WEIRD", "x", "nonsense:sub", 0)];
        let t = tokens_from_parsley(&ptoks, "x");
        assert_eq!(t.tokens[0].pos, PartOfSpeech::X);
        assert_eq!(t.tokens[0].dep, DependencyRelation::Dep);
    }
}

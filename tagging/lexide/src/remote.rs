/// Remote inference client for Modal endpoints (OpenAI-compatible API)
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    temperature: f64,
    max_tokens: usize,
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
}

/// Remote inference client for Modal endpoints (OpenAI-compatible API)
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Choice {
    message: ChatMessage,
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
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            endpoint_url: std::env::var("LEXIDE_ENDPOINT_URL")
                .unwrap_or_else(|_| "https://your-modal-endpoint.modal.run/inference".to_string()),
            max_tokens: 512,
            temperature: 0.1,
        }
    }
}

/// Remote inference client
pub struct RemoteClient {
    config: RemoteConfig,
    client: reqwest::Client,
}

impl RemoteClient {
    /// Create a new remote client
    pub fn new(config: RemoteConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600)) // 10 minutes for cold starts
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { config, client })
    }

    /// Generate a response from a prompt
    ///
    /// This is the only method that differs between local and remote implementations.
    /// It takes a pre-formatted prompt and returns the raw model response.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        // Use OpenAI-compatible chat completions API
        let request = ChatCompletionRequest {
            model: "lexide-gemma-3-27b".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        // Append /v1/chat/completions to the endpoint URL
        let url = format!("{}/v1/chat/completions", self.config.endpoint_url.trim_end_matches('/'));

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

        let chat_response: ChatCompletionResponse = response
            .json()
            .await
            .context("Failed to parse response from remote endpoint")?;

        Ok(chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default())
    }
}

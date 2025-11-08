/// Local inference implementation using mistral.rs
use anyhow::{Context, Result};
use mistralrs::{
    Constraint, LoraModelBuilder, Model as MistralModel, RequestBuilder, TextMessageRole,
    TextMessages, TokenSource, VisionModelBuilder,
};

/// Configuration for local inference
#[derive(Debug, Clone)]
pub struct LocalConfig {
    /// Base model repository (e.g., "google/gemma-3-1b-it")
    pub base_model_repo: String,
    /// LoRA adapter repository (e.g., "anchpop/lexide-gemma-3-1b-it")
    pub lora_adapter_repo: String,
    /// HuggingFace API token (required for Gemma models)
    pub hf_token: Option<String>,
    /// Maximum sequence length for generation
    pub max_length: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f64,
    /// Top-p nucleus sampling parameter
    pub top_p: f64,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            base_model_repo: "google/gemma-3-1b-it".to_string(),
            lora_adapter_repo: "anchpop/lexide-gemma-3-1b-it".to_string(),
            hf_token: std::env::var("HF_TOKEN").ok(),
            max_length: 512,
            temperature: 0.1, // Lower temperature for more deterministic parsing
            top_p: 0.9,
        }
    }
}

/// Local inference implementation
pub struct LocalLexide {
    model: MistralModel,
}

impl LocalLexide {
    /// Load the model from HuggingFace Hub with LoRA adapter
    pub async fn from_pretrained(config: LocalConfig) -> Result<Self> {
        println!("Loading base model: {}", config.base_model_repo);
        println!("Loading LoRA adapter: {}", config.lora_adapter_repo);

        let mut builder = VisionModelBuilder::new(config.base_model_repo.clone()).with_logging();

        if let Some(ref token) = config.hf_token {
            builder = builder.with_token_source(TokenSource::Literal(token.clone()));
        }

        let model = LoraModelBuilder::from_vision_model_builder(
            builder,
            vec![config.lora_adapter_repo.clone()],
        )
        .build()
        .await?;

        println!("✓ Model loaded successfully!");

        Ok(Self { model })
    }

    /// Generate a response from a prompt
    ///
    /// This is the only method that differs between local and remote implementations.
    /// It takes a pre-formatted prompt and returns the raw model response.
    pub async fn generate(&self, prompt: &str, sentence: &str) -> Result<String> {
        // Conservative token limit: 10 + 8 * input length
        let max_tokens = 10 + 8 * sentence.len();
        let _constraint = create_constraint(sentence);

        let messages = TextMessages::new().add_message(TextMessageRole::User, prompt);
        let request = RequestBuilder::from(messages).set_sampler_max_len(max_tokens);
        // Not actually using .set_constraint(constraint); yet because it makes the model way worse

        let response = self
            .model
            .send_chat_request(request)
            .await
            .context("Did not receive a valid response from the model")?;

        Ok(response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.to_string())
            .unwrap_or_default())
    }
}

/// Create a regex constraint to ensure valid output format
/// (Currently not used as it makes the model worse, but kept for future use)
fn create_constraint(sentence: &str) -> Constraint {
    // Get unique characters from the input sentence and escape them for regex
    let chars: std::collections::HashSet<char> = sentence.chars().collect();
    let escaped_chars: Vec<String> = chars
        .iter()
        .map(|c| regex::escape(&c.to_string()))
        .collect();
    let char_class = escaped_chars.join("");

    // Valid whitespace encodings
    let ws_pattern = "(_|none|nbsp|thinsp|hairsp|zwsp|ideogrp)";

    // Valid POS tags (from your pos.rs)
    let pos_pattern = "(ADJ|ADP|ADV|AUX|CCONJ|DET|INTJ|NOUN|NUM|PART|PRON|PROPN|PUNCT|SCONJ|SYM|VERB|X)";

    // Valid dependency relations (from your dep.rs)
    let dep_pattern = "(acl|advcl|advmod|amod|appos|aux|case|cc|ccomp|clf|compound|conj|cop|csubj|dep|det|discourse|dislocated|expl|fixed|flat|goeswith|iobj|list|mark|nmod|nsubj|nummod|obj|obl|orphan|parataxis|punct|reparandum|root|vocative|xcomp)";

    // Build the regex pattern for a single line
    // Format: idx\ttoken\tws\tPOS\tlemma\tdep\thead
    let line_pattern = format!(
        r"\d+\t[{char_class}]+\t{ws_pattern}\t{pos_pattern}\t[{char_class}]+\t{dep_pattern}\t-?\d+"
    );

    // Pattern for a sentence group (one or more lines)
    let sentence_group = format!(r"({line_pattern}\n)*{line_pattern}");

    // Allow optional conversational prefix, multiple sentence groups separated by -----,
    // and end with </analysis>
    let full_pattern = format!(
        r"^(Here's the token analysis:\n\n)?{sentence_group}(\n-----\n{sentence_group})*\n\n</analysis>$"
    );

    Constraint::Regex(full_pattern)
}

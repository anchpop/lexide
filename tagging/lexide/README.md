# Lexide

A Rust library for multilingual NLP analysis using a LoRA-adapted Gemma model. Performs tokenization, POS tagging, lemmatization, and dependency parsing.

## Features

- **No Model Merging Required!** - Uses mistral.rs with native LoRA support
- Multilingual support (English, French, German, Spanish, Korean, and more)
- Token-level analysis:
  - Part-of-Speech (POS) tagging
  - Lemmatization
  - Dependency parsing
  - Dependency head identification
  - Whitespace preservation for accurate text reconstruction
- Built with **mistral.rs** for blazingly fast inference
- Uses LoRA-adapted Gemma 3 1B model (`anchpop/lexide-gemma-3-1b-it`)
- Async API powered by Tokio

## Installation

### 1. Add Dependency

Add this to your `Cargo.toml`:

```toml
[dependencies]
lexide = { path = "../lexide" }
```

Or if published to crates.io:

```toml
[dependencies]
lexide = "0.1.0"
```

### 2. Set HuggingFace Token

Gemma models require authentication. Set your HuggingFace token:

```bash
export HF_TOKEN=your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

**Note:** The library automatically reads `HF_TOKEN` from environment variables when using `LexideConfig::default()`.

## Usage

### Basic Example

```rust
use lexide::{Lexide, LexideConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load the model with LoRA adapter (no merging needed!)
    let config = LexideConfig::default();
    let lexide = Lexide::from_pretrained(config).await?;

    // Analyze a sentence
    let result = lexide.analyze("The cat sat on the mat.", "English").await?;

    // Print results
    for token in result.tokens {
        println!(
            "{}: {} [{}] -> lemma: {}, dep: {}, head: {}",
            token.index, token.text, token.pos,
            token.lemma, token.dep, token.head
        );
    }

    Ok(())
}
```

The model automatically downloads both:
- Base model: `google/gemma-3-1b-it`
- LoRA adapter: `anchpop/lexide-gemma-3-1b-it`

And applies the LoRA weights on-the-fly - **no manual merging required!**

### Custom Configuration

```rust
use lexide::{Lexide, LexideConfig};

let config = LexideConfig {
    base_model_repo: "google/gemma-3-1b-it".to_string(),
    lora_adapter_repo: "anchpop/lexide-gemma-3-1b-it".to_string(),
    max_length: 512,
    temperature: 0.1, // Lower for more deterministic parsing
    top_p: 0.9,
};

let lexide = Lexide::from_pretrained(config).await?;
```

### Supported Languages

- English
- French
- German
- Spanish
- Korean

## API Reference

### `LexideConfig`

Configuration options for the Lexide model.

**Fields:**
- `base_model_repo: String` - Base HuggingFace model (default: "google/gemma-3-1b-it")
- `lora_adapter_repo: String` - LoRA adapter repository (default: "anchpop/lexide-gemma-3-1b-it")
- `hf_token: Option<String>` - HuggingFace API token (default: reads from `HF_TOKEN` env var)
- `max_length: usize` - Maximum sequence length (default: 512)
- `temperature: f64` - Sampling temperature (default: 0.1 for deterministic parsing)
- `top_p: f64` - Nucleus sampling parameter (default: 0.9)

**Example with custom token:**
```rust
let config = LexideConfig {
    hf_token: Some("your_token_here".to_string()),
    ..Default::default()
};
```

### `Lexide`

Main interface for NLP analysis.

**Methods:**

#### `from_pretrained(config: LexideConfig) -> Result<Self>`

Load the model from HuggingFace Hub.

```rust
let config = LexideConfig::default();
let lexide = Lexide::from_pretrained(config)?;
```

#### `analyze(&mut self, sentence: &str, language: &str) -> Result<AnalysisResult>`

Analyze a sentence and return structured results.

```rust
let result = lexide.analyze("Hello world", "English")?;
```

### `Token`

Represents a single token with linguistic annotations.

**Fields:**
- `index: usize` - Token position in sentence
- `text: String` - Token text
- `whitespace: String` - Trailing whitespace (e.g., " " for space, "" for none)
- `pos: String` - Part-of-speech tag
- `lemma: String` - Base form of the word
- `dep: String` - Dependency relation
- `head: i32` - Index of the head token

### `AnalysisResult`

Complete analysis for a sentence.

**Fields:**
- `language: String` - Language of the sentence
- `sentence: String` - Original sentence
- `tokens: Vec<Token>` - Analyzed tokens

**Methods:**

#### `reconstruct_text(&self) -> String`

Reconstruct the original text from tokens using whitespace information.

```rust
let result = lexide.analyze("Hello, world!", "English")?;
let reconstructed = result.reconstruct_text();
assert_eq!(reconstructed, "Hello, world!");
```

## Examples

### Running the Simple Example

```bash
cargo run --example simple
```

This will analyze "I love programming." and display the linguistic annotations.

**Expected output:**
```
0: I [PRON] -> lemma: I, dep: nsubj, head: 1
1: love [VERB] -> lemma: love, dep: root, head: 0
2: programming [NOUN] -> lemma: programming, dep: obj, head: 1
3: . [PUNCT] -> lemma: ., dep: punct, head: 1
```

### Running the Comprehensive Example

```bash
cargo run --example basic_usage
```

This example demonstrates:
- Basic sentence analysis
- Text reconstruction from tokens
- Multilingual analysis (English, French, German, Spanish)
- JSON serialization
- Working with dependency trees

### Running the CLI Application

```bash
cargo run --release
```

The CLI provides both batch examples and an interactive mode:

```
Loading Lexide NLP model...
Model loaded successfully!

Analyzing: The cat sat on the mat. (English)
─────────────────────────────────────────
Tokens: 7

Idx	Token		POS	Lemma		Dep	Head
───────────────────────────────────────────────────────────
0	The       	DET	the       	det	1
1	cat       	NOUN	cat       	nsubj	2
2	sat       	VERB	sit       	ROOT	2
...

Enter sentences to analyze (Ctrl+C to quit):
Format: <language> | <sentence>
Example: English | I love programming
```

## Performance

The model runs efficiently on both CPU and GPU:
- CPU: ~1-2s per sentence
- GPU (CUDA): ~100-200ms per sentence

## Troubleshooting

### 401 Unauthorized Error

If you get a `RequestError(Status(401, Response[status: 401, status_text: Unauthorized]))` error:

1. Make sure you've set the `HF_TOKEN` environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. Verify your token is valid at https://huggingface.co/settings/tokens

3. Ensure you've accepted the Gemma license at https://huggingface.co/google/gemma-3-1b-it

### Model Not Found

If the model fails to download:
- Check your internet connection
- Verify the model repository exists: https://huggingface.co/anchpop/lexide-gemma-3-1b-it
- Try clearing the HuggingFace cache: `rm -rf ~/.cache/huggingface/`

## Model Details

This library uses a LoRA-adapted version of Gemma 3 1B IT, fine-tuned on multilingual Universal Dependencies data. The model is hosted at [anchpop/lexide-gemma-3-1b-it](https://huggingface.co/anchpop/lexide-gemma-3-1b-it).

## License

MIT OR Apache-2.0

## Contributing

Contributions welcome! Please open an issue or PR.

//! Shared handling for the GPT-5.4-nano sidecar stages.
use std::future::Future;
use std::time::Duration;

use anyhow::Result;

/// The judge behind the French stress and language-filter sidecars. Switching
/// it re-judges everything: tysm's cache is keyed by model.
pub const MODEL: &str = "gpt-6-luna";

/// Retry a transient OpenAI failure (5xx, rate limit, network) a few times
/// before failing the stage: one bad response must not kill a 100k-request run.
pub async fn retry<T, F, Fut>(mut call: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    const ATTEMPTS: u32 = 4;
    let mut delay = Duration::from_secs(2);
    let mut attempt = 1;
    loop {
        match call().await {
            Ok(value) => return Ok(value),
            Err(error) if attempt < ATTEMPTS => {
                eprintln!(
                    "LLM request failed (attempt {attempt}/{ATTEMPTS}), retrying in {}s: {error:#}",
                    delay.as_secs()
                );
                tokio::time::sleep(delay).await;
                delay *= 4;
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

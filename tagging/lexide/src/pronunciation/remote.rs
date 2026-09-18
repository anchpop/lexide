//! Async client for Modal's separate predict and predict-batch URLs.
//! Owns bounded audio loading, request batching, coalescing and retries.
//! Cache storage and policy are configured on the client; callers supply keys.

use anyhow::{bail, Context, Result};
use serde::{de::DeserializeOwned, Serialize};

mod activity;
mod audio;
mod batching;
mod cache;
pub use activity::{RequestActivity, RequestActivitySnapshot};
pub use audio::{decode_audio_bytes, min_samples, request_from_samples, AudioInput};
pub use batching::AudioClip;
pub use cache::{response_identity, CachePolicy};
use std::sync::Arc;
use tokio::sync::{mpsc, OnceCell, Semaphore};

use super::{BatchResponse, ModelIdentity, PredictRequest, PredictResponse, RawBatchResponse};

#[derive(Clone)]
pub struct PhonemizerClient {
    store: Option<osmo::Store>,
    cache_policy: CachePolicy,
    expected_identity: Option<ModelIdentity>,
    expected_deploy_marker: Option<String>,
    check_identity_on_miss: bool,
    live_identity: Arc<OnceCell<ModelIdentity>>,
    http: reqwest::Client,
    predict_url: String,
    batch_url: String,
    queue: Arc<OnceCell<mpsc::Sender<batching::QueuedClip>>>,
    requests: Arc<Semaphore>,
    activity: Option<Arc<RequestActivity>>,
}

impl PhonemizerClient {
    /// Derive Modal's `-predict-batch.modal.run` sibling from a predict URL.
    /// Custom endpoints must use [`Self::with_endpoints`].
    pub fn new(predict_url: impl Into<String>) -> Result<Self> {
        let predict_url = predict_url.into();
        let batch_url = batch_endpoint(&predict_url)?;
        Self::with_endpoints(reqwest::Client::builder().build()?, predict_url, batch_url)
    }

    /// Supply custom URLs and an HTTP client (e.g. with timeouts or auth).
    pub fn with_endpoints(
        http: reqwest::Client,
        predict_url: impl Into<String>,
        batch_url: impl Into<String>,
    ) -> Result<Self> {
        let predict_url = predict_url.into();
        let batch_url = batch_url.into();
        reqwest::Url::parse(&predict_url).context("invalid predict URL")?;
        reqwest::Url::parse(&batch_url).context("invalid batch URL")?;
        Ok(Self {
            store: None,
            cache_policy: CachePolicy::default(),
            expected_identity: None,
            expected_deploy_marker: None,
            check_identity_on_miss: false,
            live_identity: Arc::new(OnceCell::new()),
            http,
            predict_url,
            batch_url,
            queue: Arc::new(OnceCell::new()),
            requests: Arc::new(Semaphore::new(batching::REQUESTS)),
            activity: None,
        })
    }

    /// Replace the HTTP client while retaining the configured endpoint URLs.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.queue = Arc::new(OnceCell::new());
        self.http = http;
        self
    }

    /// Discover model identity without inference. The deployed Python endpoint
    /// has no GET health route: it accepts `POST {"marker_only": true}` on predict.
    /// Rejects `load_error` even if the response also contains model fields.
    pub async fn identity(&self) -> Result<ModelIdentity> {
        #[derive(Serialize)]
        struct Probe {
            marker_only: bool,
        }
        let response = self
            .http
            .post(&self.predict_url)
            .json(&Probe { marker_only: true })
            .send()
            .await?;
        let probe = checked(response)
            .await?
            .json()
            .await
            .context("invalid model identity")?;
        parse_identity(probe)
    }

    /// Probe and require an exact deploy marker, including rejecting its absence.
    /// This checks only the probe's container: callers must also check prediction
    /// and batch markers before caching responses from later requests.
    pub async fn check_identity(&self, expected_deploy_marker: &str) -> Result<ModelIdentity> {
        let identity = self.identity().await?;
        if identity.deploy_marker.as_deref() != Some(expected_deploy_marker) {
            bail!(
                "deploy-marker mismatch: endpoint reported {:?}, expected {:?}",
                identity.deploy_marker,
                expected_deploy_marker
            );
        }
        Ok(identity)
    }

    pub async fn predict(&self, request: &PredictRequest) -> Result<PredictResponse> {
        parse_response(self.predict_raw(request).await?)
            .await
            .context("invalid prediction response")
    }

    /// Return the exact response body, including unknown fields and whitespace.
    /// Checks HTTP status, not JSON or prediction validity. Callers must parse
    /// and validate the response and its identity before using or caching it.
    pub async fn predict_raw(&self, request: &PredictRequest) -> Result<Vec<u8>> {
        let response = self
            .http
            .post(&self.predict_url)
            .json(request)
            .send()
            .await?;
        Ok(checked(response)
            .await?
            .bytes()
            .await
            .context("invalid prediction response")?
            .to_vec())
    }

    /// Send 1–64 clips, preserving per-item errors and the envelope marker.
    pub async fn predict_batch(&self, requests: &[PredictRequest]) -> Result<BatchResponse> {
        let batch: BatchResponse = parse_response(self.batch_body(requests).await?)
            .await
            .context("invalid batch response")?;
        check_count(&batch, requests.len())?;
        Ok(batch)
    }

    /// Preserve all envelope and item fields for per-clip caching.
    /// Enforces request bounds, metadata types and result count. Individual
    /// results are not type-checked on success; callers must validate them and
    /// the envelope identity before use or caching.
    pub async fn predict_batch_raw(&self, requests: &[PredictRequest]) -> Result<RawBatchResponse> {
        let body = self.batch_body(requests).await?;
        let batch: reqwest::Result<BatchResponse<Box<serde_json::value::RawValue>>> =
            parse_response(body.clone()).await;
        if !batch
            .as_ref()
            .is_ok_and(|batch| batch.results.len() == requests.len())
        {
            // Full typed parsing wins over metadata/count failures, just as it
            // did before raw access. Avoid decoding matrices on the happy path.
            let typed: BatchResponse = parse_response(body.clone())
                .await
                .context("invalid batch response")?;
            check_count(&typed, requests.len())?;
        }
        let batch = batch.context("invalid batch response")?;
        let mut envelope: std::collections::BTreeMap<String, Box<serde_json::value::RawValue>> =
            parse_response(body)
                .await
                .context("invalid batch response")?;
        envelope.remove("results");
        Ok(RawBatchResponse { batch, envelope })
    }

    async fn batch_body(&self, requests: &[PredictRequest]) -> Result<Vec<u8>> {
        if !(1..=batching::BATCH_SIZE).contains(&requests.len()) {
            bail!(
                "requests must contain between 1 and {} items",
                batching::BATCH_SIZE
            );
        }
        #[derive(Serialize)]
        struct BatchRequest<'a> {
            requests: &'a [PredictRequest],
        }
        let response = self
            .http
            .post(&self.batch_url)
            .json(&BatchRequest { requests })
            .send()
            .await?;
        Ok(checked(response)
            .await?
            .bytes()
            .await
            .context("invalid batch response")?
            .to_vec())
    }
}

fn check_count<T>(batch: &BatchResponse<T>, expected: usize) -> Result<()> {
    if batch.results.len() != expected {
        bail!(
            "batch returned {} results for {} clips",
            batch.results.len(),
            expected
        );
    }
    Ok(())
}

// Keep reqwest's decode-error wrapper in the chain, just as Response::json did
// before raw access was exposed. This only parses memory; it performs no I/O.
async fn parse_response<T: DeserializeOwned>(body: Vec<u8>) -> reqwest::Result<T> {
    reqwest::Response::from(http::Response::new(body))
        .json()
        .await
}

fn parse_identity(probe: serde_json::Value) -> Result<ModelIdentity> {
    if let Some(error) = probe.get("load_error") {
        bail!("model identity probe reported load_error: {error}");
    }
    serde_json::from_value(probe).context("invalid model identity")
}

/// How much of an error response to quote. Modal's rejections are short JSON;
/// the limit only guards against an endpoint returning an HTML error page.
const BODY_EXCERPT_CHARS: usize = 2000;

/// Fail on an error status while keeping the endpoint's response body, which
/// is where the reason a clip was rejected lives (`error_for_status` alone
/// discards it). The `reqwest::Error` stays in the chain, so callers can still
/// recover the status to decide whether the failure is worth retrying.
async fn checked(response: reqwest::Response) -> Result<reqwest::Response> {
    let Some(error) = response.error_for_status_ref().err() else {
        return Ok(response);
    };
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let excerpt: String = body.trim().chars().take(BODY_EXCERPT_CHARS).collect();
    Err(anyhow::Error::new(error).context(if excerpt.is_empty() {
        format!("endpoint returned {status}")
    } else {
        format!("endpoint returned {status}: {excerpt}")
    }))
}

fn batch_endpoint(single: &str) -> Result<String> {
    match single.strip_suffix("-predict.modal.run") {
        Some(prefix) => Ok(format!("{prefix}-predict-batch.modal.run")),
        None => bail!("supply an explicit batch URL for custom predict endpoint {single}"),
    }
}

impl std::fmt::Debug for PhonemizerClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhonemizerClient")
            .field("predict_url", &self.predict_url)
            .field("batch_url", &self.batch_url)
            .field("cache_enabled", &self.store.is_some())
            .field("cache_policy", &self.cache_policy)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modal_batch_url_matches_yap() {
        assert_eq!(
            batch_endpoint("https://anchpop--wav2vec2-phoneme-wav2vec2phoneme-predict.modal.run")
                .unwrap(),
            "https://anchpop--wav2vec2-phoneme-wav2vec2phoneme-predict-batch.modal.run"
        );
        assert!(batch_endpoint("http://localhost/single").is_err());
    }
}

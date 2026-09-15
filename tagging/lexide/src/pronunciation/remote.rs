//! Thin async client for Modal's separate predict and predict-batch URLs.
//! No caching, retries, runtime creation, or implicit deploy-marker checks.

use anyhow::{bail, Context, Result};
use serde::Serialize;

use super::{BatchResponse, ModelIdentity, PredictRequest, PredictResponse};

#[derive(Debug, Clone)]
pub struct PhonemizerClient {
    http: reqwest::Client,
    predict_url: String,
    batch_url: String,
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
            http,
            predict_url,
            batch_url,
        })
    }

    /// Replace the HTTP client while retaining the configured endpoint URLs.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
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
        let probe = self
            .http
            .post(&self.predict_url)
            .json(&Probe { marker_only: true })
            .send()
            .await?
            .error_for_status()?
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
        self.http
            .post(&self.predict_url)
            .json(request)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
            .context("invalid prediction response")
    }

    /// Send 1–64 clips, preserving per-item errors and the envelope marker.
    pub async fn predict_batch(&self, requests: &[PredictRequest]) -> Result<BatchResponse> {
        if !(1..=64).contains(&requests.len()) {
            bail!("requests must contain between 1 and 64 items");
        }
        #[derive(Serialize)]
        struct BatchRequest<'a> {
            requests: &'a [PredictRequest],
        }
        let batch: BatchResponse = self
            .http
            .post(&self.batch_url)
            .json(&BatchRequest { requests })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
            .context("invalid batch response")?;
        if batch.results.len() != requests.len() {
            bail!(
                "batch returned {} results for {} clips",
                batch.results.len(),
                requests.len()
            );
        }
        Ok(batch)
    }
}

fn parse_identity(probe: serde_json::Value) -> Result<ModelIdentity> {
    if let Some(error) = probe.get("load_error") {
        bail!("model identity probe reported load_error: {error}");
    }
    serde_json::from_value(probe).context("invalid model identity")
}

fn batch_endpoint(single: &str) -> Result<String> {
    match single.strip_suffix("-predict.modal.run") {
        Some(prefix) => Ok(format!("{prefix}-predict-batch.modal.run")),
        None => bail!("supply an explicit batch URL for custom predict endpoint {single}"),
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

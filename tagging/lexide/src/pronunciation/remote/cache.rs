//! Client-owned raw response caching. Cache identity is supplied by the caller.
use super::PhonemizerClient;
use crate::pronunciation::{
    FrameMatrixPayload, ModelIdentity, PredictResponse as ModalResponse, RawPrediction,
    DECODER_VERSION,
};
use anyhow::{Context, Result};
use std::path::Path;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CachePolicy {
    /// Reuse valid entries; infer and save misses.
    #[default]
    ReadThrough,
    /// Never prepare audio, probe identity, or infer on a miss.
    CachedOnly,
    /// Explicitly replace entries using live inference.
    Refresh,
}

impl PhonemizerClient {
    pub fn with_cache_directory(self, directory: impl AsRef<Path>) -> Self {
        self.with_cache(osmo::Store::open(directory))
    }

    pub fn with_cache(mut self, store: osmo::Store) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_cache_policy(mut self, policy: CachePolicy) -> Self {
        self.cache_policy = policy;
        self
    }

    pub fn with_cached_only(self) -> Self {
        self.with_cache_policy(CachePolicy::CachedOnly)
    }

    pub fn with_expected_identity(mut self, identity: ModelIdentity) -> Self {
        self.expected_identity = Some(identity);
        self
    }

    pub fn with_expected_deploy_marker(mut self, marker: impl Into<String>) -> Self {
        self.expected_deploy_marker = Some(marker.into());
        self
    }

    /// Discover the live model once, on the first miss, and validate live results
    /// against it. Clones share discovery. Cache hits never contact the endpoint.
    pub fn with_identity_check(mut self) -> Self {
        self.check_identity_on_miss = true;
        self
    }

    /// Cache-only inspection. Preserves absent versus malformed entries and never
    /// checks a historical response against the currently deployed model.
    pub async fn cached(&self, key: &str) -> Option<Result<RawPrediction>> {
        if self.cache_policy == CachePolicy::Refresh {
            return None;
        }
        let bytes = self.store.as_ref()?.read(key).await?;
        Some((|| {
            let raw: RawPrediction = serde_json::from_slice(&bytes).context("cached response")?;
            raw.frames()?;
            Ok(raw)
        })())
    }

    pub(super) async fn live_client(&self) -> Result<Self> {
        let mut client = self.clone();
        if self.check_identity_on_miss && self.expected_identity.is_none() {
            let identity = self
                .live_identity
                .get_or_try_init(|| async {
                    let identity = self.identity().await?;
                    check_decoder(identity.decoder_version.as_deref())?;
                    check_marker(
                        self.expected_deploy_marker.as_deref(),
                        identity.deploy_marker.as_deref(),
                    )?;
                    Ok::<_, anyhow::Error>(identity)
                })
                .await?;
            client.expected_identity = Some(identity.clone());
        }
        Ok(client)
    }

    /// Validate a live response before persisting it. The caller owns the full
    /// key; no model/version/endpoint prefix is added implicitly.
    pub async fn cache_response(
        &self,
        key: Option<&str>,
        raw: RawPrediction,
    ) -> Result<RawPrediction> {
        self.validate_response(&raw.decode()?)?;
        if let (Some(store), Some(key)) = (&self.store, key) {
            raw.frames()?;
            store.write(key, &serde_json::to_vec(&raw)?).await?;
        }
        Ok(raw)
    }

    /// Per-response freshness check: the one-shot marker_only probe only proves
    /// the *first* request hit a fresh container. Verifying the marker on every
    /// response guarantees no later request was routed to a stale/contaminated
    /// warm container and silently cached under the wrong model's key.
    pub fn validate_response(&self, modal: &ModalResponse) -> Result<()> {
        // V1's matrix producer is authoritative, but must agree with any envelope
        // metadata too. Never label a matrix using the context's desired producer.
        let observed = response_identity(modal);
        if let Some(observed) = &observed {
            for (name, outer, inner) in [
                (
                    "model id",
                    modal.model_id.as_deref(),
                    Some(observed.model_id.as_str()),
                ),
                (
                    "model revision",
                    modal.model_revision.as_deref(),
                    Some(observed.model_revision.as_str()),
                ),
                (
                    "decoder",
                    modal.decoder_version.as_deref(),
                    observed.decoder_version.as_deref(),
                ),
                (
                    "deploy-marker",
                    modal.deploy_marker.as_deref(),
                    observed.deploy_marker.as_deref(),
                ),
            ] {
                anyhow::ensure!(
                    outer.is_none() || inner == outer,
                    "response {name} disagrees with matrix producer"
                );
            }
            check_decoder(observed.decoder_version.as_deref())?;
            check_marker(
                self.expected_deploy_marker.as_deref(),
                observed.deploy_marker.as_deref(),
            )?;
            if let Some(expected) = &self.expected_identity {
                anyhow::ensure!(
                    observed.model_id == expected.model_id
                        && observed.model_revision == expected.model_revision,
                    "matrix producer does not match expected model identity"
                );
            }
        }
        check_decoder(modal.decoder_version.as_deref())?;
        if let Some(identity) = &self.expected_identity {
            for (name, reported, expected) in [
                ("model id", modal.model_id.as_ref(), &identity.model_id),
                (
                    "model revision",
                    modal.model_revision.as_ref(),
                    &identity.model_revision,
                ),
            ] {
                if let Some(reported) = reported {
                    anyhow::ensure!(
                    reported == expected,
                    "{name} mismatch: endpoint reported {reported:?}, expected {expected:?} — refusing to cache prediction"
                );
                }
            }
        }
        check_marker(
            self.expected_deploy_marker.as_deref(),
            observed
                .as_ref()
                .and_then(|identity| identity.deploy_marker.as_deref())
                .or(modal.deploy_marker.as_deref()),
        )
    }
}

pub fn response_identity(response: &ModalResponse) -> Option<ModelIdentity> {
    if let Some(FrameMatrixPayload::V1(payload)) = &response.frame_matrix {
        let producer = &payload.producer;
        return Some(ModelIdentity {
            model_id: producer.model_id.clone(),
            model_revision: producer.model_revision.clone(),
            decoder_version: Some(producer.decoder_version.clone()),
            deploy_marker: Some(producer.deploy_marker.clone()),
        });
    }
    Some(ModelIdentity {
        model_id: response.model_id.clone()?,
        model_revision: response.model_revision.clone()?,
        decoder_version: response.decoder_version.clone(),
        deploy_marker: response.deploy_marker.clone(),
    })
}

pub(super) fn check_decoder(decoder: Option<&str>) -> Result<()> {
    if let Some(decoder) = decoder {
        anyhow::ensure!(
            decoder == DECODER_VERSION,
            "decoder mismatch: endpoint reported {decoder:?}, expected {DECODER_VERSION:?}"
        );
    }
    Ok(())
}

fn check_marker(expected: Option<&str>, reported: Option<&str>) -> Result<()> {
    if let Some(expected) = expected {
        anyhow::ensure!(
            reported == Some(expected),
            "deploy-marker mismatch: endpoint reported {reported:?}, expected {expected:?}"
        );
    }
    Ok(())
}

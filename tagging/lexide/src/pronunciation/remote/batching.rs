use super::{AudioInput, CachePolicy, PhonemizerClient, RequestActivity};
use crate::pronunciation::{PredictRequest, RawPrediction};
use anyhow::{Context, Result};
use futures::{Stream, StreamExt};
use std::{sync::Arc, time::Duration};
use tokio::sync::{mpsc, oneshot, OnceCell};

pub(super) const BATCH_SIZE: usize = 64;
const PREPARATIONS: usize = 8;
pub(super) const REQUESTS: usize = 2;
const MAX_ATTEMPTS: usize = 5;
const LINGER: Duration = Duration::from_millis(200);

/// The ID is returned unchanged, including on errors. Duration is a grouping
/// hint; paths are opened lazily and all response fields are retained.
pub struct AudioClip<I> {
    pub id: I,
    pub duration: Duration,
    pub audio: AudioInput,
    /// Optional extra identity (e.g. expected phonemes). Audio is always hashed.
    pub cache_context: Option<String>,
}

pub(super) struct QueuedClip {
    audio: AudioInput,
    reply: oneshot::Sender<Result<RawPrediction>>,
}

impl PhonemizerClient {
    pub fn with_activity(mut self, activity: Arc<RequestActivity>) -> Self {
        self.queue = Arc::new(OnceCell::new());
        self.activity = Some(activity);
        self
    }

    /// Any number of clips, with bounded preparation and HTTP concurrency.
    /// Results stream in completion order, keyed by the caller's ID. Local or
    /// per-item errors affect one clip; exhausted HTTP failures affect one batch.
    /// Dropping the stream stops scheduling further clips.
    pub fn predict_many<'a, I: 'a>(
        &'a self,
        clips: Vec<AudioClip<I>>,
    ) -> impl Stream<Item = (I, Result<RawPrediction>)> + 'a {
        async_stream::stream! {
            let mut misses = Vec::new();
            for clip in clips {
                let key = match self.input_cache_key(&clip.audio, clip.cache_context.as_deref()).await {
                    Ok(key) => key,
                    Err(error) => { yield (clip.id, Err(error)); continue; }
                };
                if let Some(key) = key.as_deref() {
                    if let Some(Ok(raw)) = self.cached(key).await {
                        yield (clip.id, Ok(raw));
                        continue;
                    }
                }
                if self.cache_policy == CachePolicy::CachedOnly {
                    yield (clip.id, Err(anyhow::anyhow!("response cache miss; cache-only mode is enabled")));
                } else {
                    misses.push(AudioClip {
                        id: (clip.id, key),
                        duration: clip.duration,
                        audio: clip.audio,
                        cache_context: None,
                    });
                }
            }
            if !misses.is_empty() {
                match self.live_client().await {
                    Ok(client) => {
                        let results = client.predict_many_uncached(misses);
                        futures::pin_mut!(results);
                        while let Some(((id, key), result)) = results.next().await {
                            let result = match result {
                                Ok(raw) => client.cache_response(key.as_deref(), raw).await,
                                Err(error) => Err(error),
                            };
                            yield (id, result);
                        }
                    }
                    Err(error) => for clip in misses {
                        yield (clip.id.0, Err(anyhow::anyhow!("{error:#}")));
                    },
                }
            }
        }
    }

    fn predict_many_uncached<'a, I: 'a>(
        &'a self,
        mut clips: Vec<AudioClip<I>>,
    ) -> impl Stream<Item = (I, Result<RawPrediction>)> + 'a {
        clips.sort_by_key(|clip| clip.duration);
        async_stream::stream! {
            let mut prepared = Box::pin(futures::stream::iter(clips)
                .map(|clip| async move { (clip.id, clip.audio.prepare().await) })
                .buffered(PREPARATIONS));
            let mut in_flight = futures::stream::FuturesUnordered::new();
            let mut ids = Vec::new();
            let mut requests = Vec::new();
            let mut done = false;
            loop {
                if !requests.is_empty() && (requests.len() == BATCH_SIZE || done) && in_flight.len() < REQUESTS {
                    let ids = std::mem::take(&mut ids);
                    let requests = std::mem::take(&mut requests);
                    in_flight.push(async move { (ids, self.send_batch(&requests).await) });
                }
                if done && requests.is_empty() && in_flight.is_empty() { break; }
                tokio::select! {
                    item = prepared.next(), if !done && requests.len() < BATCH_SIZE => {
                        match item {
                            Some((id, Ok(request))) => { ids.push(id); requests.push(request); }
                            Some((id, Err(error))) => yield (id, Err(error)),
                            None => done = true,
                        }
                    }
                    batch = in_flight.next(), if !in_flight.is_empty() => {
                        let (ids, results) = batch.expect("in-flight batch");
                        match results {
                            Ok(results) => for (id, result) in ids.into_iter().zip(results) { yield (id, result); },
                            Err(error) => {
                                let message = format!("{error:#}");
                                for id in ids { yield (id, Err(anyhow::anyhow!("{message}"))); }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Pool simultaneous individual callers into batch requests on this client.
    /// Clones share the queue; the caller supplies a Tokio runtime.
    pub async fn predict_audio(
        &self,
        audio: AudioInput,
        cache_context: Option<&str>,
    ) -> Result<RawPrediction> {
        let cache_key = self.input_cache_key(&audio, cache_context).await?;
        if let Some(key) = cache_key.as_deref() {
            if let Some(Ok(raw)) = self.cached(key).await {
                return Ok(raw);
            }
        }
        anyhow::ensure!(
            self.cache_policy != CachePolicy::CachedOnly,
            "response cache miss; cache-only mode is enabled"
        );
        let client = self.live_client().await?;
        let raw = self.queue_audio(audio).await?;
        client.cache_response(cache_key.as_deref(), raw).await
    }

    async fn queue_audio(&self, audio: AudioInput) -> Result<RawPrediction> {
        let queue = self
            .queue
            .get_or_init(|| async {
                let (tx, rx) = mpsc::channel::<QueuedClip>(BATCH_SIZE * REQUESTS);
                let mut client = self.clone();
                // The worker must not retain its own sender, so dropping the client
                // closes the queue and lets outstanding work drain.
                client.queue = Arc::new(OnceCell::new());
                tokio::spawn(async move {
                    futures::stream::unfold(rx, |mut rx| async move {
                        let first = rx.recv().await?;
                        let mut batch = vec![first];
                        tokio::time::sleep(LINGER).await;
                        while batch.len() < BATCH_SIZE {
                            match rx.try_recv() {
                                Ok(item) => batch.push(item),
                                Err(_) => break,
                            }
                        }
                        Some((batch, rx))
                    })
                    .map(|batch| {
                        let client = &client;
                        async move {
                            let clips = batch
                                .into_iter()
                                .filter(|item| !item.reply.is_closed())
                                .map(|item| AudioClip {
                                    id: item.reply,
                                    duration: Duration::ZERO,
                                    audio: item.audio,
                                    cache_context: None,
                                })
                                .collect();
                            let results = client.predict_many_uncached(clips);
                            futures::pin_mut!(results);
                            while let Some((reply, result)) = results.next().await {
                                let _ = reply.send(result);
                            }
                        }
                    })
                    .buffer_unordered(REQUESTS)
                    .collect::<Vec<_>>()
                    .await;
                });
                tx
            })
            .await;
        let (reply, result) = oneshot::channel();
        queue
            .send(QueuedClip { audio, reply })
            .await
            .map_err(|_| anyhow::anyhow!("phonemizer worker stopped"))?;
        result
            .await
            .context("phonemizer worker dropped the request")?
    }

    async fn send_batch(&self, requests: &[PredictRequest]) -> Result<Vec<Result<RawPrediction>>> {
        let mut last_error = None;
        for attempt in 1..=MAX_ATTEMPTS {
            let result = {
                // Shared by clones and the individual-call queue. Backoff does
                // not occupy a request slot or count as HTTP activity.
                let _permit = self.requests.acquire().await?;
                let _activity = self
                    .activity
                    .as_ref()
                    .map(|activity| activity.begin(attempt > 1));
                self.predict_batch_raw(requests).await
            };
            match result {
                Ok(raw) => return raw.into_predictions(),
                Err(error) if !is_transient_error(&error) => return Err(error),
                Err(error) => last_error = Some(error),
            }
            if attempt < MAX_ATTEMPTS {
                tokio::time::sleep(Duration::from_secs(5 * attempt as u64)).await;
            }
        }
        Err(last_error
            .expect("at least one attempt")
            .context(format!("phonemizer failed after {MAX_ATTEMPTS} attempts")))
    }
}

fn is_transient_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<reqwest::Error>())
        .find_map(reqwest::Error::status)
        .is_none_or(|status| matches!(status.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504))
}

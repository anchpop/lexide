use crate::pronunciation::PredictRequest;
use anyhow::{Context, Result};
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

/// Encoded audio is decoded with ffmpeg only when a batch has room for it.
#[derive(Debug)]
pub enum AudioInput {
    File(PathBuf),
    Bytes(Vec<u8>),
    /// For callers that already have float32 audio encoded for the endpoint.
    Request(PredictRequest),
}

impl AudioInput {
    pub(super) async fn prepare(self) -> Result<PredictRequest> {
        if let Self::Request(request) = self {
            return Ok(request);
        }
        tokio::task::spawn_blocking(move || {
            let samples = self.decode()?;
            Ok(request_from_samples(&samples, 16_000, 10))
        })
        .await
        .context("audio decoding task failed")?
    }

    fn decode(self) -> Result<Vec<f32>> {
        let mut command = Command::new("ffmpeg");
        command.args(["-nostdin", "-loglevel", "error", "-i"]);
        let bytes = match self {
            Self::File(path) => {
                command.arg(path);
                None
            }
            Self::Bytes(bytes) => {
                command.arg("pipe:0");
                Some(bytes)
            }
            Self::Request(_) => unreachable!(),
        };
        let mut child = command
            .args(["-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"])
            .stdin(if bytes.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawn ffmpeg for audio decoding")?;
        // Drain stdout while feeding stdin: writing first can deadlock on long clips.
        let writer = bytes.map(|bytes| {
            let mut stdin = child.stdin.take().expect("piped stdin");
            std::thread::spawn(move || stdin.write_all(&bytes))
        });
        let output = child.wait_with_output().context("wait for ffmpeg")?;
        let written = writer
            .map(|writer| {
                writer
                    .join()
                    .map_err(|_| anyhow::anyhow!("audio writer panicked"))
            })
            .transpose()?;
        anyhow::ensure!(
            output.status.success(),
            "ffmpeg decode failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Some(written) = written {
            written.context("write audio to ffmpeg")?;
        }
        anyhow::ensure!(output.stdout.len() % 4 == 0, "invalid float32 audio length");
        Ok(output
            .stdout
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
}

/// Decode encoded audio to mono 16 kHz samples for local audio checks or timing.
pub fn decode_audio_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    AudioInput::Bytes(bytes.to_vec()).decode()
}

/// Short clips need 0.6 seconds of context; frame timing uses this same padding.
pub fn min_samples(sample_rate: u32) -> usize {
    (f64::from(sample_rate) * 0.6).ceil() as usize
}

pub fn request_from_samples(samples: &[f32], sample_rate: u32, top_k: usize) -> PredictRequest {
    let needed = min_samples(sample_rate).saturating_sub(samples.len());
    let mut padded = vec![0.0; needed / 2];
    padded.extend_from_slice(samples);
    padded.resize(samples.len() + needed, 0.0);
    PredictRequest {
        sample_rate,
        top_k,
        return_frame_matrix: true,
        return_all_heads: true,
        ..PredictRequest::from_samples(&padded)
    }
}

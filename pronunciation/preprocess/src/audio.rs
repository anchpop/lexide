use anyhow::{Result, ensure};
use hound::{SampleFormat, WavReader};
use std::path::Path;

pub fn read(path: &Path) -> Result<(hound::WavSpec, Vec<f32>)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let samples = if spec.sample_format == SampleFormat::Float {
        reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?
    } else {
        let scale = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
        reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / scale))
            .collect::<Result<_, _>>()?
    };
    Ok((spec, samples))
}

pub fn stats(path: &Path) -> Result<(f64, f64)> {
    let (spec, samples) = read(path)?;
    let channel: Vec<_> = samples.iter().step_by(usize::from(spec.channels)).collect();
    let rms = if channel.is_empty() {
        0.0
    } else {
        (channel.iter().map(|v| f64::from(**v).powi(2)).sum::<f64>() / channel.len() as f64).sqrt()
    };
    Ok((channel.len() as f64 / f64::from(spec.sample_rate), rms))
}

pub fn vad(data_dir: &Path, lang: &str) -> Result<()> {
    use rayon::prelude::*;
    let dir = data_dir.join(lang);
    let rows = crate::corpus::read(&dir.join("phonemes.jsonl"))?;
    let outputs = rows
        .par_iter()
        .map(|row| -> Result<serde_json::Value> {
            let file = crate::corpus::text(row, "file")?;
            let (spec, samples) = read(&dir.join(file))?;
            ensure!(
                spec.sample_rate == 16_000 && spec.channels == 1,
                "{lang}/{file}: VAD requires 16 kHz mono audio"
            );
            let mut detector = earshot::Detector::default_boxed();
            let probs: Vec<_> = samples
                .as_chunks::<256>()
                .0
                .iter()
                .map(|frame| detector.predict_f32(frame))
                .collect();
            Ok(serde_json::json!({"file": file, "vad_probs": probs}))
        })
        .collect::<Result<Vec<_>>>()?;
    crate::corpus::write(&dir.join("vad.jsonl"), &outputs)
}

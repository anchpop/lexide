use anyhow::{Result, ensure};
use hound::{SampleFormat, WavReader};
use std::{path::Path, sync::Mutex};

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

pub fn vad(data_dir: &Path, lang: &str, incremental: bool, sources: &[String]) -> Result<()> {
    use rayon::prelude::*;
    let dir = data_dir.join(lang);
    let rows = crate::corpus::read(&dir.join("phonemes.jsonl"))?;
    // Rows stream to disk as each clip finishes: a whole language's frame
    // probabilities (62.5 per second of audio) do not belong in memory at once.
    let output = dir.join("vad.jsonl");
    let existing: std::collections::HashSet<String> = if incremental && output.exists() {
        use std::io::BufRead;
        #[derive(serde::Deserialize)]
        struct Existing {
            file: String,
        }
        std::io::BufReader::new(std::fs::File::open(&output)?)
            .lines()
            .filter_map(|line| match line {
                Ok(s) if s.trim().is_empty() => None,
                other => Some(other),
            })
            .map(|line| Ok(serde_json::from_str::<Existing>(&line?)?.file))
            .collect::<Result<_>>()?
    } else {
        Default::default()
    };
    let writer = Mutex::new(if incremental {
        crate::corpus::Writer::append(&output)?
    } else {
        crate::corpus::Writer::create(&output)?
    });
    rows.par_iter().try_for_each(|row| -> Result<()> {
        let file = crate::corpus::text(row, "file")?;
        if existing.contains(file)
            || (!sources.is_empty() && !sources.iter().any(|s| row["source"] == *s))
        {
            return Ok(());
        }
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
        writer
            .lock()
            .unwrap()
            .row(&serde_json::json!({"file": file, "vad_probs": probs}))
    })?;
    writer.into_inner().unwrap().finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incremental_vad_preserves_rows_and_selects_sources() {
        let tmp = tempfile::tempdir().unwrap();
        for lang in ["eng", "deu", "spa"] {
            let dir = tmp.path().join(lang);
            std::fs::create_dir(&dir).unwrap();
            let old = b"{ \"file\": \"old.wav\", \"vad_probs\": [0.1, 0.2] }\n";
            std::fs::write(dir.join("vad.jsonl"), old).unwrap();
            crate::corpus::write(
                &dir.join("phonemes.jsonl"),
                &[
                    serde_json::json!({"file":"old.wav","source":"mls"}),
                    serde_json::json!({"file":"new.wav","source":"cv"}),
                    serde_json::json!({"file":"not-selected.wav","source":"tts"}),
                ],
            )
            .unwrap();
            let mut wav = hound::WavWriter::create(
                dir.join("new.wav"),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16000,
                    bits_per_sample: 16,
                    sample_format: SampleFormat::Int,
                },
            )
            .unwrap();
            for _ in 0..1600 {
                wav.write_sample(1000i16).unwrap();
            }
            wav.finalize().unwrap();
            vad(tmp.path(), lang, true, &["cv".into()]).unwrap();
            let bytes = std::fs::read(dir.join("vad.jsonl")).unwrap();
            assert!(bytes.starts_with(old));
            assert_eq!(
                crate::corpus::read(&dir.join("vad.jsonl")).unwrap().len(),
                2
            );
            vad(tmp.path(), lang, true, &["cv".into()]).unwrap();
            assert_eq!(std::fs::read(dir.join("vad.jsonl")).unwrap(), bytes);
        }
    }
}

// Batch VAD: read phonemes.jsonl, for each clip compute earshot VAD probs at
// 16ms frame stride, write a parallel vad.jsonl with { file, vad_probs }.
//
// Usage:
//     vad_compute <phonemes.jsonl> <audio_dir> <vad.jsonl>
//
// vad_probs[i] is P(speech) for the i-th 256-sample frame (16 ms). Training-time
// code interpolates onto the wav2vec2 grid (320-sample / 20 ms stride). 256-sample
// frames is earshot's native stride — don't change it.

use earshot::Detector;
use hound::WavReader;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

fn process_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        return Err(format!(
            "expected 16kHz mono, got {}Hz {}ch",
            spec.sample_rate, spec.channels
        )
        .into());
    }
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|x| x as f32 / 32768.0))
        .collect::<Result<_, _>>()?;
    // One detector per file — earshot::Detector is stateful across frames,
    // but we don't want state to leak between clips.
    let mut detector = Detector::default_boxed();
    let mut probs = Vec::with_capacity(samples.len() / 256);
    for frame in samples.chunks_exact(256) {
        probs.push(detector.predict_f32(frame));
    }
    Ok(probs)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err(format!(
            "usage: {} <phonemes.jsonl> <audio_dir> <vad.jsonl>",
            args[0]
        )
        .into());
    }
    let input_jsonl = PathBuf::from(&args[1]);
    let audio_dir = PathBuf::from(&args[2]);
    let output_jsonl = PathBuf::from(&args[3]);

    // Read all entries into memory first so rayon can parallelize across them.
    let entries: Vec<String> = BufReader::new(File::open(&input_jsonl)?)
        .lines()
        .collect::<Result<_, _>>()?;
    eprintln!("Loaded {} entries from {}", entries.len(), input_jsonl.display());

    let output = Mutex::new(BufWriter::new(File::create(&output_jsonl)?));
    let processed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let start = Instant::now();

    entries.par_iter().for_each(|line| {
        let rec: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("JSON parse error: {}", e);
                failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let file_name = match rec["file"].as_str() {
            Some(s) => s,
            None => {
                failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let wav_path = audio_dir.join(file_name);
        match process_file(&wav_path) {
            Ok(probs) => {
                let out = json!({
                    "file": file_name,
                    "vad_probs": probs,
                });
                let mut w = output.lock().unwrap();
                writeln!(*w, "{}", out).unwrap();
            }
            Err(e) => {
                eprintln!("FAILED {}: {}", wav_path.display(), e);
                failed.fetch_add(1, Ordering::Relaxed);
            }
        }
        let n = processed.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 500 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!("  {} processed ({:.0} files/sec)", n, n as f64 / elapsed);
        }
    });

    let n = processed.load(Ordering::Relaxed);
    let f = failed.load(Ordering::Relaxed);
    eprintln!(
        "Done: {} processed, {} failed in {:.1}s",
        n,
        f,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

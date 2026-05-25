// Compare earshot VAD vs simple RMS energy on a 16kHz mono WAV.
// Output: CSV with columns (time_ms, rms, vad_prob).
// Both signals are computed on the same 256-sample (16ms) frame so they're directly comparable.

use earshot::Detector;
use hound::WavReader;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: vad_compare <path.wav>")?;
    let mut reader = WavReader::open(&path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        return Err(format!(
            "expected 16kHz mono WAV, got {}Hz {}ch",
            spec.sample_rate, spec.channels
        )
        .into());
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|x| x as f32 / 32768.0))
        .collect::<Result<_, _>>()?;

    // Detector is ~8 KiB on the stack — box it to be safe.
    let mut detector = Detector::default_boxed();

    println!("frame,time_ms,rms,vad_prob");
    for (i, frame) in samples.chunks_exact(256).enumerate() {
        let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
        let p = detector.predict_f32(frame);
        println!("{i},{},{:.6},{:.4}", i * 16, rms, p);
    }
    Ok(())
}

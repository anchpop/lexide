//! Speaks each request and dumps every stage's inputs and outputs as .npy for the onnxruntime parity check
//! (tests/parity.py), then times the native engine.
//!     cargo run --release --example trace -- <bundle.voca> <requests.json> <out dir> [repeats]
//! requests.json: `[{"grams": [...], "options": {...}}, ...]`, options as in `Options` (camelCase).

use std::{fs, path::Path, time::Instant};

use lexide_tts::{Data, Gram, Options, Tensor, Voice, normal};
use serde::Deserialize;

#[derive(Deserialize)]
struct Request {
    grams: Vec<Gram>,
    options: Options,
}

fn npy(path: &Path, t: &Tensor) {
    let (descr, bytes): (&str, Vec<u8>) = match &t.data {
        Data::F32(v) => ("<f4", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        Data::F64(v) => ("<f8", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        Data::I64(v) => ("<i8", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        Data::Bool(v) => ("|b1", v.iter().map(|&x| x as u8).collect()),
    };
    let dims: String = t.shape.iter().map(|d| format!("{d},")).collect();
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims}), }}");
    header.push_str(&" ".repeat(63 - (header.len() + 10) % 64));
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.as_bytes());
    out.extend(bytes);
    fs::write(path, out).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let start = Instant::now();
    let voice = Voice::load(&fs::read(&args[1]).unwrap()).unwrap();
    eprintln!("loaded in {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
    let requests: Vec<Request> = serde_json::from_slice(&fs::read(&args[2]).unwrap()).unwrap();
    let repeats: usize = args.get(4).map_or(3, |r| r.parse().unwrap());
    let mut summary = vec![];
    for (u, r) in requests.iter().enumerate() {
        let dir = Path::new(&args[3]).join(format!("u{u}"));
        fs::create_dir_all(&dir).unwrap();
        let (mut call, mut draws, mut frames) = (0, 0, 0);
        let mut wave = vec![];
        let audio = voice.speak_traced(&r.grams, &r.options, &mut |stage, inputs, outputs| {
            let d = dir.join(format!("{call:02}_{stage}"));
            fs::create_dir_all(d.join("in")).unwrap();
            fs::create_dir_all(d.join("out")).unwrap();
            // Graph value names may contain '/', which file names can't.
            let file = |n: &str| format!("{}.npy", n.replace('/', "~"));
            for (n, t) in inputs {
                npy(&d.join("in").join(file(n)), t);
            }
            for (n, t) in outputs {
                npy(&d.join("out").join(file(n)), t);
            }
            match stage {
                "encode" => draws = outputs["mean"].len(),
                "render" => {
                    (frames, wave) = (inputs["owner"].len(), outputs["wave"].f32().to_vec())
                }
                _ => {}
            }
            call += 1;
        });
        let audio = audio.unwrap();
        // The delivery draw, so onnxruntime's own chain can share it; the raw and trimmed waves.
        npy(
            &dir.join("z.npy"),
            &Tensor::new(vec![draws], normal(r.options.take, draws)),
        );
        npy(&dir.join("wave.npy"), &Tensor::new(vec![wave.len()], wave));
        npy(
            &dir.join("samples.npy"),
            &Tensor::new(vec![audio.samples.len()], audio.samples.clone()),
        );
        fs::write(
            dir.join("options.json"),
            serde_json::to_string(&serde_json::json!({
                "variety": r.options.variety, "fixEnding": r.options.fix_ending, "frames": frames,
            }))
            .unwrap(),
        )
        .unwrap();
        let mut times = vec![];
        for _ in 0..repeats {
            let t = Instant::now();
            voice.speak(&r.grams, &r.options).unwrap();
            times.push(t.elapsed().as_secs_f64());
        }
        let seconds = audio.samples.len() as f64 / audio.sample_rate as f64;
        let rtf: Vec<f64> = times
            .iter()
            .map(|t| (seconds / t * 10.0).round() / 10.0)
            .collect();
        eprintln!("u{u}: {seconds:.2} s audio, realtime x{rtf:?}");
        summary.push(serde_json::json!({"seconds": seconds, "times": times}));
    }
    fs::write(
        Path::new(&args[3]).join("native.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();
}

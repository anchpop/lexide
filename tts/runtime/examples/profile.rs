//! Native per-operator time, stage by stage, for the first request of a requests.json (see trace.rs).
//!     cargo run --release --example profile -- <bundle.voca> <requests.json>

use std::{collections::BTreeMap, fs, time::Instant};

use lexide_tts::{Gram, Options, Values, Voice};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let voice = Voice::load(&fs::read(&args[1]).unwrap()).unwrap();
    let requests: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(&args[2]).unwrap()).unwrap();
    let grams: Vec<Gram> = serde_json::from_value(requests[0]["grams"].clone()).unwrap();
    let options: Options = serde_json::from_value(requests[0]["options"].clone()).unwrap();
    let mut calls: Vec<(String, Values)> = vec![];
    let audio = voice
        .speak_traced(&grams, &options, &mut |stage, inputs, _| {
            calls.push((stage.into(), inputs.clone()))
        })
        .unwrap();
    let mut total = 0.0;
    for (stage, inputs) in &calls {
        let mut ops: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        let start = Instant::now();
        voice.graph(stage).run_with(inputs, &mut |op, kernel| {
            let t = Instant::now();
            kernel();
            let e = ops.entry(op.into()).or_default();
            e.0 += t.elapsed().as_secs_f64() * 1e3;
            e.1 += 1;
        });
        let ms = start.elapsed().as_secs_f64() * 1e3;
        total += ms;
        let mut ops: Vec<_> = ops.into_iter().collect();
        ops.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let top: Vec<String> = ops
            .iter()
            .take(8)
            .map(|(op, (t, n))| format!("{op} {t:.1}ms/{n}"))
            .collect();
        println!("{stage:8} {ms:7.1} ms  {}", top.join(", "));
    }
    println!(
        "total {total:.1} ms for {:.2} s of audio",
        audio.samples.len() as f64 / audio.sample_rate as f64
    );
}

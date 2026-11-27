//! A `.voca` voice: its tokenizer (and espeak mapping for distilled voices), the five exported stages, and the
//! chain that turns grams into playable audio.

use serde::Deserialize;

use crate::espeak::{self, Espeak};
use crate::frontend::{self, Gram, Tokenizer, Tokens};
use crate::graph::{Graph, Values};
use crate::sampler::{normal, pitch_offsets};
use crate::tensor::Tensor;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    sample_rate: u32,
    hop: usize,
    speakers: Vec<String>,
    max_frames: usize,
    max_tokens: usize,
    lexical: Vec<String>,
    /// False for distilled voices, trained without yap's pos, rarity and boundary hints.
    hints: Option<bool>,
    espeak: Option<espeak::Spec>,
    frontend: frontend::Spec,
    graphs: std::collections::BTreeMap<String, usize>,
}

/// bundle_voice.py writes the graphs in this order.
pub const STAGES: [&str; 5] = ["encode", "predict", "melody", "blend", "render"];

pub struct Voice {
    pub sample_rate: u32,
    pub speakers: Vec<String>,
    hop: usize,
    max_frames: usize,
    max_tokens: usize,
    columns: usize,
    tokenizer: Tokenizer,
    espeak: Option<Espeak>,
    graphs: Vec<Graph>,
}

/// Synthesis controls. `Default` is the delivery the web engine always used.
#[derive(Clone, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct Options {
    /// A name from `Voice::speakers`; the first when unset.
    pub speaker: Option<String>,
    /// Seed of the delivery draw: a fixed take reproduces a reading.
    pub take: u32,
    /// Seed of the source noise and pitch jitter.
    pub seed: u32,
    /// Scale of the delivery draw around the typical reading (0 is the typical reading itself).
    pub variety: f32,
    pub speed: f32,
    /// Pitch offset in the model's log-frequency units.
    pub pitch_shift: f32,
    pub pitch_range: f32,
    /// Random pitch wander, in semitones-ish units (0 is off).
    pub pitch_jitter: f32,
    /// Extra pause after commas and after sentence stops, in seconds.
    pub comma_pause: f32,
    pub stop_pause: f32,
    pub voiced_vowels: bool,
    /// Blend a rising (question) ending back toward the typical reading's.
    pub fix_ending: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            speaker: None,
            take: 0,
            seed: 0,
            variety: 1.0,
            speed: 1.0,
            pitch_shift: 0.0,
            pitch_range: 1.0,
            pitch_jitter: 0.0,
            comma_pause: 0.15,
            stop_pause: 0.3,
            voiced_vowels: true,
            fix_ending: true,
        }
    }
}

/// Mono PCM.
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// The renderer's raw output.
pub struct Synthesis {
    pub wave: Vec<f32>,
    /// Frames per token.
    pub durations: Vec<i64>,
}

/// Sees each stage's inputs and outputs (the parity harness dumps them).
pub type Trace<'a> = dyn FnMut(&str, &Values, &Values) + 'a;

impl Voice {
    pub fn load(bundle: &[u8]) -> Result<Voice, String> {
        let header = u32::from_le_bytes(bundle[..4].try_into().unwrap()) as usize;
        let m: Manifest =
            serde_json::from_slice(&bundle[4..4 + header]).map_err(|e| e.to_string())?;
        let mut offset = 4 + header;
        let graphs = STAGES
            .iter()
            .map(|name| {
                let size = m.graphs[*name];
                offset += size;
                Graph::load(&bundle[offset - size..offset]).map_err(|e| format!("{name}: {e}"))
            })
            .collect::<Result<_, _>>()?;
        let tokenizer = Tokenizer::new(m.frontend, &m.lexical, m.hints.unwrap_or(true))?;
        let espeak = m.espeak.map(|spec| Espeak::new(spec, tokenizer.phones()));
        Ok(Voice {
            sample_rate: m.sample_rate,
            speakers: m.speakers,
            hop: m.hop,
            max_frames: m.max_frames,
            max_tokens: m.max_tokens,
            columns: m.lexical.len(),
            tokenizer,
            espeak,
            graphs,
        })
    }

    pub fn graph(&self, stage: &str) -> &Graph {
        &self.graphs[STAGES.iter().position(|s| *s == stage).unwrap()]
    }

    /// The phone inventory, in the frontend's order.
    pub fn phones(&self) -> &[String] {
        self.tokenizer.phones()
    }

    /// The espeak mapping, for voices distilled to read espeak's IPA.
    pub fn espeak(&self) -> Option<&Espeak> {
        self.espeak.as_ref()
    }

    pub fn speak(&self, grams: &[Gram], options: &Options) -> Result<Audio, String> {
        self.speak_traced(grams, options, &mut |_, _, _| {})
    }

    /// Speaks text through espeak, which `phonemize` runs on each stretch between punctuation
    /// (`espeak-ng -v <espeak().spec.voice> -q --ipa --sep=_ -- <stretch>`, stdout).
    pub fn speak_text(
        &self,
        text: &str,
        phonemize: impl FnMut(&str) -> String,
        options: &Options,
    ) -> Result<Audio, String> {
        let espeak = self
            .espeak
            .as_ref()
            .ok_or("this voice reads yap grams, not espeak")?;
        let ipa: Vec<String> = espeak
            .chunks(text)
            .iter()
            .map(|c| c.as_str())
            .map(phonemize)
            .collect();
        self.speak(&espeak.grams(text, &ipa)?, options)
    }

    /// Runs the chain once so later calls don't pay for first-use allocation.
    pub fn warmup(&self) -> Result<(), String> {
        let phone = self.phones()[0].clone();
        let gram = Gram {
            words: vec!["x".into()],
            pos: vec!["X".into()],
            count: 1.0,
            phonemes: phone,
        };
        self.speak(&[gram], &Options::default()).map(drop)
    }

    pub fn speak_traced(
        &self,
        grams: &[Gram],
        options: &Options,
        trace: &mut Trace,
    ) -> Result<Audio, String> {
        let tokens = self.tokenizer.tokenize(grams)?;
        let first = tokens.first_phone.ok_or("enter text containing speech")?;
        if tokens.ids.len() > self.max_tokens {
            return Err("utterance exceeds the voice token limit".into());
        }
        let synthesis = self.synthesize(&tokens, options, trace)?;
        Ok(Audio {
            samples: self.playback(synthesis, first),
            sample_rate: self.sample_rate,
        })
    }

    fn run(&self, stage: &str, values: &Values, trace: &mut Trace) -> Values {
        let graph = self.graph(stage);
        let inputs: Values = graph
            .input_names()
            .map(|n| (n.to_string(), values[n].clone()))
            .collect();
        let outputs = graph.run(&inputs);
        trace(stage, &inputs, &outputs);
        outputs
    }

    /// Predicts durations from `draw`, expands tokens to frames and shapes the melody over them.
    fn read(&self, common: &Values, draw: Tensor, trace: &mut Trace) -> Result<Values, String> {
        let mut reading = common.clone();
        reading.insert("draw".into(), draw);
        reading.extend(self.run("predict", &reading, trace));
        let durations = reading["durations"].i64();
        let frames: i64 = durations.iter().sum();
        if frames < 1 || frames as usize > self.max_frames {
            return Err(format!("invalid predicted frame count {frames}"));
        }
        let owner: Vec<i64> = durations
            .iter()
            .enumerate()
            .flat_map(|(i, &n)| std::iter::repeat_n(i as i64, n as usize))
            .collect();
        reading.insert("owner".into(), Tensor::new(vec![1, owner.len()], owner));
        reading.extend(self.run("melody", &reading, trace));
        Ok(reading)
    }

    /// The stage chain: encode, draw a delivery, predict and shape it, blend a rising ending, render.
    pub fn synthesize(
        &self,
        tokens: &Tokens,
        o: &Options,
        trace: &mut Trace,
    ) -> Result<Synthesis, String> {
        let n = tokens.ids.len();
        assert_eq!(tokens.lexical.len(), n * self.columns);
        let speaker = match &o.speaker {
            Some(name) => self
                .speakers
                .iter()
                .position(|s| s == name)
                .ok_or_else(|| format!("no speaker {name}"))?,
            None => 0,
        };
        if !(0.25..=4.0).contains(&o.speed)
            || !(0.0..=3.0).contains(&o.pitch_range)
            || !o.pitch_shift.is_finite()
        {
            return Err("speed, pitch range or pitch shift out of range".into());
        }
        let mut common: Values = [
            ("tokens", Tensor::new(vec![1, n], tokens.ids.clone())),
            (
                "lexical",
                Tensor::new(vec![1, n, self.columns], tokens.lexical.clone()),
            ),
            ("speaker", Tensor::new(vec![1], vec![speaker as i64])),
            ("speed", Tensor::scalar(o.speed)),
            ("pitch_range", Tensor::scalar(o.pitch_range)),
            ("comma_pause", Tensor::scalar(o.comma_pause)),
            ("stop_pause", Tensor::scalar(o.stop_pause)),
            ("voiced_vowels", Tensor::scalar(o.voiced_vowels)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        common.extend(self.run("encode", &common, trace));
        let (mean, scale) = (&common["mean"], &common["scale"]);
        let z = normal(o.take, mean.len());
        let draw = mean
            .f32()
            .iter()
            .zip(scale.f32())
            .zip(z)
            .map(|((m, s), z)| m + z * (o.variety * s))
            .collect::<Vec<_>>();
        let mut reading = self.read(&common, Tensor::new(mean.shape.clone(), draw), trace)?;
        if o.fix_ending && o.variety != 0.0 && reading["rising"].bool()[0] {
            let typical = self.read(&common, common["mean"].clone(), trace)?;
            reading.insert("typical".into(), typical["prosody"].clone());
            reading.insert("typical_durations".into(), typical["durations"].clone());
            let blended = self
                .run("blend", &reading, trace)
                .remove("blended")
                .unwrap();
            reading.insert("prosody".into(), blended);
        }
        let frames = reading["owner"].len();
        let samples = frames * self.hop;
        let offsets = pitch_offsets(frames, o.pitch_shift, o.pitch_jitter, o.seed);
        reading.insert(
            "noise".into(),
            Tensor::new(vec![1, 1, samples], normal(o.seed, samples)),
        );
        reading.insert(
            "pitch_shift".into(),
            Tensor::new(vec![1, offsets.len()], offsets),
        );
        let wave = self
            .run("render", &reading, trace)
            .remove("wave")
            .unwrap()
            .into_vec();
        Ok(Synthesis {
            wave,
            durations: reading["durations"].ints(),
        })
    }

    /// Starts just before the speech (the first two loud frames, or the first phone if sooner) and keeps the peak
    /// below 0.99.
    fn playback(&self, s: Synthesis, first_phone: usize) -> Vec<f32> {
        let hop = self.hop;
        let mut onset = s.durations[..first_phone].iter().sum::<i64>() as usize;
        let mut previous = false;
        for frame in 0..=onset {
            if (frame + 1) * hop > s.wave.len() {
                break;
            }
            let power: f64 = s.wave[frame * hop..(frame + 1) * hop]
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum();
            let loud = power / hop as f64 > 1e-5;
            if previous && loud {
                onset = frame - 1;
                break;
            }
            previous = loud;
        }
        let mut samples = s.wave;
        samples.drain(..onset.saturating_sub(4) * hop);
        let peak = samples.iter().fold(0f32, |p, x| p.max(x.abs()));
        if peak > 0.99 {
            let gain = 0.99 / peak as f64;
            samples
                .iter_mut()
                .for_each(|x| *x = (*x as f64 * gain) as f32);
        }
        samples
    }
}

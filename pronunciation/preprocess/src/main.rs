//! Rust owns stage ordering and pronunciation; Python supplies corpus/audio helpers.
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

#[derive(Parser)]
#[command(about = "Prepare pronunciation training data using g2p directly from Rust")]
struct Args {
    #[arg(long, default_value_os_t = root().join("data/audio"))]
    data_dir: PathBuf,
    /// Restrict language directories; packing still includes the whole data directory.
    #[arg(long, num_args = 1..)]
    langs: Vec<String>,
    /// Python interpreter (or executable wrapper) with the existing audio dependencies.
    #[arg(long, default_value = "python3")]
    python: PathBuf,
    /// Concurrent language jobs; logs go to pronunciation/.work/preprocess_parallel.
    #[arg(long, default_value = "1", value_parser = clap::value_parser!(u16).range(1..))]
    jobs: u16,
    /// Keep existing narrowed files. Required while the aligner has untrained merged labels.
    #[arg(long)]
    skip_narrowing: bool,
    #[arg(long)]
    skip_vad: bool,
    /// Skip speaker embeddings/clustering (otherwise may call the deployed Modal app).
    #[arg(long)]
    skip_speaker_cluster: bool,
    #[arg(long)]
    allow_noncommercial: bool,
    /// Skip shared exclusions and dataset packing, useful for local label checks.
    #[arg(long)]
    no_pack: bool,
    #[arg(long, default_value_os_t = root().join(".work/pron_audio.tar"))]
    output_tar: PathBuf,
    #[arg(long, default_value_os_t = root().join("train/mixed_script_exclusions.jsonl"))]
    exclusions_output: PathBuf,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}

fn run(command: &mut Command, log: Option<&File>) -> Result<()> {
    if let Some(log) = log {
        command.stdout(Stdio::from(log.try_clone()?));
        command.stderr(Stdio::from(log.try_clone()?));
    }
    let status = command
        .status()
        .with_context(|| format!("starting {command:?}"))?;
    ensure!(status.success(), "{command:?} failed ({status})");
    Ok(())
}

impl Args {
    fn helper(&self, stage: &str) -> Command {
        let mut command = Command::new(&self.python);
        command
            .arg(root().join("train/scripts/preprocess_support.py"))
            .arg(stage)
            .arg("--data-dir")
            .arg(&self.data_dir);
        command
    }

    fn language(&self, lang: &str, vad: Option<&Path>, identity: &str) -> Result<()> {
        let log = if self.jobs > 1 {
            let dir = root().join(".work/preprocess_parallel");
            fs::create_dir_all(&dir)?;
            let path = dir.join(format!("{lang}.log"));
            eprintln!("{lang}: log {}", path.display());
            Some(File::create(path)?)
        } else {
            None
        };
        let temp = tempfile::tempdir()?;
        let prepared = temp.path().join("prepared.jsonl");
        let labels = temp.path().join("labels.jsonl");
        let mut prepare = self.helper("prepare");
        prepare
            .arg("--lang")
            .arg(lang)
            .arg("--exchange")
            .arg(&prepared);
        if self.allow_noncommercial {
            prepare.arg("--allow-noncommercial");
        }
        run(&mut prepare, log.as_ref())?;

        let mut output = BufWriter::new(File::create(&labels)?);
        for line in BufReader::new(File::open(&prepared)?).lines() {
            let input: Prepared = serde_json::from_str(&line?)?;
            let sentence = input.record["sentence"]
                .as_str()
                .context("missing sentence")?;
            let labels = match g2p::phonemize(input.language, sentence) {
                Ok(target) => {
                    // Training stores stress as 0/1/2, rather than Rust enum names.
                    let mut value = serde_json::to_value(&target)?;
                    value["stress"] =
                        json!(target.stress.iter().map(|s| s.code()).collect::<Vec<_>>());
                    value
                }
                Err(g2p::Error::Unlabelable(reason)) => json!({"exclude_reason": reason}),
                Err(error) => {
                    return Err(error).with_context(|| format!("{lang}: {}", input.record["file"]));
                }
            };
            serde_json::to_writer(
                &mut output,
                &json!({
                    "record": input.record, "language": input.language, "labels": labels,
                }),
            )?;
            writeln!(output)?;
        }
        output.flush()?;
        run(
            self.helper("finalize")
                .arg("--lang")
                .arg(lang)
                .arg("--exchange")
                .arg(&labels)
                .arg("--identity")
                .arg(identity),
            log.as_ref(),
        )?;
        if !self.skip_narrowing {
            run(self.helper("narrow").arg("--lang").arg(lang), log.as_ref())?;
        }
        if let Some(vad) = vad {
            let dir = self.data_dir.join(lang);
            run(
                Command::new(vad)
                    .arg(dir.join("phonemes.jsonl"))
                    .arg(&dir)
                    .arg(dir.join("vad.jsonl")),
                log.as_ref(),
            )?;
        }
        if !self.skip_speaker_cluster {
            run(
                self.helper("speakers").arg("--lang").arg(lang),
                log.as_ref(),
            )?;
        }
        eprintln!("{lang}: complete");
        Ok(())
    }
}

#[derive(Deserialize)]
struct Prepared {
    record: Value,
    language: g2p::Language,
}

/// Resolve Cargo's actual artifact path, including user-configured target directories.
fn build_vad() -> Result<PathBuf> {
    let output = Command::new("cargo")
        .current_dir(root().join("vad_compare"))
        .args([
            "build",
            "--release",
            "--bin",
            "vad_compute",
            "--message-format=json",
        ])
        .stderr(Stdio::inherit())
        .output()
        .context("building vad_compute")?;
    ensure!(output.status.success(), "vad_compute build failed");
    for line in output.stdout.split(|b| *b == b'\n') {
        if let Ok(value) = serde_json::from_slice::<Value>(line)
            && value["reason"] == "compiler-artifact"
            && value["target"]["name"] == "vad_compute"
            && let Some(path) = value["executable"].as_str()
        {
            return Ok(PathBuf::from(path));
        }
    }
    bail!("Cargo did not report a vad_compute executable")
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    args.data_dir = args
        .data_dir
        .canonicalize()
        .context("opening data directory")?;
    let mut languages = Vec::new();
    for entry in fs::read_dir(&args.data_dir)? {
        let entry = entry?;
        let lang = entry.file_name().to_string_lossy().into_owned();
        if lang != ".cache"
            && entry.path().is_dir()
            && entry.path().join("manifest.jsonl").is_file()
            && (args.langs.is_empty() || args.langs.contains(&lang))
        {
            languages.push(lang);
        }
    }
    languages.sort();
    for lang in &args.langs {
        ensure!(
            languages.contains(lang),
            "no manifest.jsonl for requested language {lang}"
        );
    }
    ensure!(!languages.is_empty(), "no language manifests found");
    let vad = if args.skip_vad {
        None
    } else {
        Some(build_vad()?)
    };
    let identity = g2p::identity();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs.into())
        .build()?;
    let failures: Vec<_> = pool.install(|| {
        languages
            .par_iter()
            .filter_map(
                |lang| match args.language(lang, vad.as_deref(), &identity) {
                    Ok(()) => None,
                    Err(error) => {
                        eprintln!("{lang}: FAILED: {error:#}");
                        Some(lang.clone())
                    }
                },
            )
            .collect()
    });
    ensure!(
        failures.is_empty(),
        "preprocessing failed for {}; dataset was not packed",
        failures.join(", ")
    );
    if !args.no_pack {
        run(
            args.helper("exclusions")
                .arg("--output")
                .arg(&args.exclusions_output),
            None,
        )?;
        run(
            args.helper("pack").arg("--output").arg(&args.output_tar),
            None,
        )?;
    }
    Ok(())
}

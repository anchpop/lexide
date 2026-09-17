//! Rust owns stage ordering and pronunciation; Python supplies corpus/audio helpers.
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

mod audio;
mod audit;
mod corpus;
mod french_stress;
mod language_filter;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Stage {
    Run,
    Audit,
    Stress,
    Filter,
    Labels,
    Vad,
    Speakers,
    Measure,
    Narrow,
    Pack,
    Upload,
    DeployAligner,
}

#[derive(Parser)]
#[command(about = "Prepare pronunciation training data using g2p directly from Rust")]
struct Args {
    /// Run the whole preparation/upload pipeline or a single stage.
    #[arg(value_enum)]
    stage: Stage,
    /// Show the selected stages without API calls or output writes.
    #[arg(long)]
    dry_run: bool,
    /// Load only this env file instead of the repository defaults.
    #[arg(long)]
    env_file: Option<PathBuf>,
    #[arg(long, default_value_os_t = root().join("train"))]
    train_dir: PathBuf,
    #[command(flatten)]
    audit: audit::Options,
    #[arg(long)]
    skip_audit: bool,
    #[arg(long)]
    skip_stress: bool,
    #[arg(long)]
    skip_filter: bool,
    #[arg(long)]
    skip_upload: bool,
    #[arg(long, default_value = "anchpop/lexide-pronunciation-audio")]
    hf_repo: String,
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
    /// Skip packing during a full run.
    #[arg(long)]
    no_pack: bool,
    #[arg(long, default_value_os_t = root().join(".work/pron_audio.tar"))]
    output_tar: PathBuf,
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
            .arg(&self.data_dir)
            .arg("--train-dir")
            .arg(&self.train_dir);
        command
    }

    fn labels(&self, lang: &str, identity: &str) -> Result<()> {
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
            let record: Value = serde_json::from_str(&line?)?;
            let language = corpus::language(&record, lang)?;
            let sentence = record["sentence"].as_str().context("missing sentence")?;
            let labels = match g2p::phonemize(language, sentence) {
                Ok(target) => {
                    // Training stores stress as 0/1/2, rather than Rust enum names.
                    let mut value = serde_json::to_value(&target)?;
                    value["stress"] =
                        json!(target.stress.iter().map(|s| s.code()).collect::<Vec<_>>());
                    value
                }
                Err(g2p::Error::Unlabelable(reason)) => json!({"exclude_reason": reason}),
                Err(error) => {
                    return Err(error).with_context(|| format!("{lang}: {}", record["file"]));
                }
            };
            serde_json::to_writer(
                &mut output,
                &json!({
                    "record": record, "language": language, "labels": labels,
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
        eprintln!("{lang}: complete");
        Ok(())
    }
}

impl Args {
    fn stages(&self) -> Vec<Stage> {
        if self.stage != Stage::Run {
            return vec![self.stage];
        }
        let mut stages = vec![];
        if !self.skip_audit {
            stages.push(Stage::Audit);
        }
        if !self.skip_stress {
            stages.push(Stage::Stress);
        }
        if !self.skip_filter {
            stages.push(Stage::Filter);
        }
        stages.push(Stage::Labels);
        if !self.skip_vad {
            stages.push(Stage::Vad);
        }
        if !self.skip_speaker_cluster {
            stages.push(Stage::Speakers);
        }
        if !self.skip_narrowing {
            stages.extend([Stage::Measure, Stage::Narrow]);
        }
        if !self.no_pack {
            stages.push(Stage::Pack);
        }
        if !self.skip_upload {
            stages.push(Stage::Upload);
        }
        stages
    }

    fn parallel(&self, langs: &[String], job: impl Fn(&str) -> Result<()> + Sync) -> Result<()> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.jobs.into())
            .build()?;
        let failures: Vec<_> = pool.install(|| {
            langs
                .par_iter()
                .filter_map(|lang| match job(lang) {
                    Ok(()) => None,
                    Err(error) => {
                        eprintln!("{lang}: FAILED: {error:#}");
                        Some(lang.clone())
                    }
                })
                .collect()
        });
        ensure!(
            failures.is_empty(),
            "stage failed for {}; later stages will not run",
            failures.join(", ")
        );
        Ok(())
    }

    async fn execute(&self, langs: &[String], stages: &[Stage]) -> Result<()> {
        for stage in stages {
            eprintln!("=== {stage:?} ===");
            match stage {
                Stage::Run => unreachable!(),
                Stage::Audit => {
                    audit::run(&self.data_dir, &self.train_dir, langs, &self.audit).await?
                }
                Stage::Stress => {
                    if langs.iter().any(|l| l == "fra") {
                        french_stress::run(&self.data_dir).await?;
                    }
                }
                Stage::Filter => {
                    language_filter::run(&self.data_dir, &self.train_dir, langs).await?
                }
                Stage::Labels => {
                    let identity = g2p::identity();
                    self.parallel(langs, |lang| self.labels(lang, &identity))?;
                }
                Stage::Vad => self.parallel(langs, |lang| audio::vad(&self.data_dir, lang))?,
                Stage::Speakers => {
                    for lang in langs {
                        run(self.helper("speakers").arg("--lang").arg(lang), None)?;
                    }
                }
                Stage::Measure | Stage::Narrow => {
                    // Refuse incompatible labels before importing clients or spending on alignment.
                    for lang in langs {
                        run(self.helper("guard").arg("--lang").arg(lang), None)?;
                    }
                    let name = if *stage == Stage::Measure {
                        "measure"
                    } else {
                        "narrow"
                    };
                    for lang in langs {
                        run(self.helper(name).arg("--lang").arg(lang), None)?;
                    }
                }
                Stage::Pack => {
                    run(
                        self.helper("exclusions")
                            .arg("--output")
                            .arg(self.train_dir.join("mixed_script_exclusions.jsonl")),
                        None,
                    )?;
                    run(
                        self.helper("pack").arg("--output").arg(&self.output_tar),
                        None,
                    )?;
                }
                Stage::Upload => {
                    run(
                        Command::new(&self.python)
                            .arg(root().join("scripts/upload_audio_to_hf.py"))
                            .arg("--env-file")
                            .arg("/dev/null")
                            .arg("--audio-root")
                            .arg(&self.data_dir)
                            .arg("--repo")
                            .arg(&self.hf_repo)
                            .arg("--large"),
                        None,
                    )?;
                }
                Stage::DeployAligner => {
                    run(
                        Command::new(&self.python)
                            .current_dir(root().join("espeak_audit"))
                            .args(["-m", "modal", "deploy", "modal_aligner.py"]),
                        None,
                    )?;
                }
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    let stages = args.stages();
    if args.dry_run {
        for stage in &stages {
            println!("{stage:?}");
        }
        return Ok(());
    }
    // Load before constructing worker threads; match the former shell's local override.
    let env_files = args
        .env_file
        .clone()
        .map(|p| vec![p])
        .unwrap_or_else(|| vec![root().parent().unwrap().join(".env"), root().join(".env")]);
    for path in env_files {
        if args.env_file.is_some() || path.exists() {
            dotenvy::from_path_override(&path)
                .with_context(|| format!("loading {}", path.display()))?;
        }
    }
    let mut languages = Vec::new();
    if args.stage != Stage::DeployAligner {
        args.data_dir = args
            .data_dir
            .canonicalize()
            .context("opening data directory")?;
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
    }
    tokio::runtime::Runtime::new()?.block_on(args.execute(&languages, &stages))
}

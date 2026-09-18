use anyhow::{Context, Result, bail};
use g2p::Language;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

pub fn read(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|s| serde_json::from_str(s).with_context(|| format!("JSON in {}", path.display())))
        .collect()
}

/// Validate Python-produced training rows against the same inventory as G2P
/// and inference. Keep the rest of the row opaque, including acoustic metadata.
pub fn validate_phonemes(path: &Path) -> Result<()> {
    use std::io::BufRead;
    #[derive(serde::Deserialize)]
    struct Labels {
        phonemes: Vec<g2p::Phoneme>,
    }
    for (index, line) in std::io::BufReader::new(fs::File::open(path)?)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Labels = serde_json::from_str(&line).with_context(|| {
            format!(
                "invalid phoneme inventory in {} line {}",
                path.display(),
                index + 1
            )
        })?;
        anyhow::ensure!(
            !row.phonemes.is_empty(),
            "empty phonemes in {} line {}",
            path.display(),
            index + 1
        );
    }
    Ok(())
}

/// Streams JSONL rows into a buffered temp file beside `path`, which only
/// replaces the previous file on `finish`. A failed run drops the temp file,
/// so readers never see a partial output.
pub struct Writer {
    path: PathBuf,
    temp: BufWriter<tempfile::NamedTempFile>,
}

impl Writer {
    pub fn create(path: &Path) -> Result<Self> {
        let parent = path.parent().context("missing output directory")?;
        fs::create_dir_all(parent)?;
        Ok(Self {
            path: path.to_owned(),
            temp: BufWriter::new(tempfile::NamedTempFile::new_in(parent)?),
        })
    }

    pub fn row(&mut self, row: &impl Serialize) -> Result<()> {
        serde_json::to_writer(&mut self.temp, row)?;
        writeln!(self.temp)?;
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        self.temp.into_inner()?.persist(&self.path)?;
        Ok(())
    }
}

pub fn write(path: &Path, rows: &[Value]) -> Result<()> {
    let mut writer = Writer::create(path)?;
    for row in rows {
        writer.row(row)?;
    }
    writer.finish()
}

/// A subset run replaces only those languages, retaining other exclusion rows.
pub fn write_scoped(path: &Path, mut rows: Vec<Value>, langs: &[String]) -> Result<()> {
    if path.exists() {
        rows.extend(
            read(path)?
                .into_iter()
                .filter(|r| !langs.iter().any(|l| r["lang"] == *l)),
        );
    }
    write(path, &rows)
}

pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn text<'a>(row: &'a Value, key: &str) -> Result<&'a str> {
    row[key]
        .as_str()
        .with_context(|| format!("missing string field {key}"))
}

/// Decode corpus dialect metadata once, shared by labeling and ASR comparisons.
pub fn language(row: &Value, code: &str) -> Result<Language> {
    if let Some(value) = row
        .get("g2p_language")
        .filter(|v| !v.is_null() && **v != "")
    {
        return serde_json::from_value(value.clone()).context("invalid g2p_language");
    }
    let mut variety = row["variety"]
        .as_str()
        .filter(|s| !s.is_empty())
        .unwrap_or("default");
    if variety == "default" && row["variety"].as_str().filter(|s| !s.is_empty()).is_none() {
        if let Some(voice) = row["espeak_voice"].as_str().filter(|s| !s.is_empty()) {
            variety = match (code, voice) {
                ("spa", "es") | ("por", "pt") => "european",
                ("spa", "es-419") => "latin_american",
                ("por", "pt-br") => "brazilian",
                ("spa" | "por", _) => bail!("unknown historical voice {voice} for {code}"),
                _ => "default",
            };
        } else if code == "spa" {
            if row["source"] == "fleurs" {
                variety = "latin_american";
            } else if row["source"] == "tts"
                && (row["tts_backend"].is_null() || row["tts_backend"] == "chirp3")
            {
                let voice = row["voice"].as_str().unwrap_or("");
                if voice.starts_with("es-US-Chirp3-HD-") {
                    variety = "latin_american";
                }
                if voice.starts_with("es-ES-Chirp3-HD-") {
                    variety = "european";
                }
            }
        }
    }
    Ok(match (code, variety) {
        ("spa", "default" | "european") => Language::SpanishEuro,
        ("spa", "latin_american") => Language::SpanishLatinAmerica,
        ("por", "default" | "brazilian") => Language::PortugueseBrazil,
        ("por", "european") => Language::PortugueseEuro,
        (_, "default") => {
            Language::from_code(code).with_context(|| format!("unsupported language {code}"))?
        }
        _ => bail!("unsupported variety {variety} for {code}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recording_dialects_and_explicit_overrides() {
        for (record, code, expected) in [
            (
                json!({"source":"fleurs"}),
                "spa",
                Language::SpanishLatinAmerica,
            ),
            (json!({"source":"tatoeba"}), "spa", Language::SpanishEuro),
            (
                json!({"espeak_voice":"pt"}),
                "por",
                Language::PortugueseEuro,
            ),
            (
                json!({"espeak_voice":"pt-br"}),
                "por",
                Language::PortugueseBrazil,
            ),
            (
                json!({"variety":"latin_american"}),
                "spa",
                Language::SpanishLatinAmerica,
            ),
            (
                json!({"g2p_language":"spa-419", "espeak_voice":"es"}),
                "spa",
                Language::SpanishLatinAmerica,
            ),
            (
                json!({"source":"tts","voice":"es-US-Chirp3-HD-Kore"}),
                "spa",
                Language::SpanishLatinAmerica,
            ),
            (
                json!({"source":"tts","voice":"es-US-Chirp3-HD-Kore","tts_backend":"gemini"}),
                "spa",
                Language::SpanishEuro,
            ),
        ] {
            assert_eq!(language(&record, code).unwrap(), expected);
        }
    }
}

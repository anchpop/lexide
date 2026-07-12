//! Wiktionary lemma tables in a compact, load-fast format — the Rust port of
//! `tagger/lemma_lookup.py` (the out-of-distribution lemma floor).
//!
//! The Python side keeps `{pos: {form: [candidate lemmas]}}` JSON (~4-35 MB per language)
//! and picks a candidate at lookup time. Candidate selection only depends on the form
//! (`min` by `(|len(c)-len(form)| in chars, c)`), so the builder resolves it once and the
//! runtime table maps straight to the winning lemma.
//!
//! File layout (`wikt_{lang}.fst`):
//!   8 bytes  magic `LEXLEM1\0`
//!   8 bytes  u64 LE: fst section length
//!   fst      `fst::Map` keyed by `{POS}\0{form}`, value = blob_offset << 16 | lemma_byte_len
//!   blob     deduplicated UTF-8 lemma bytes
//!
//! Build with `cargo run --features local --bin build-lemma-fst`.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;

use anyhow::{bail, Context, Result};

const MAGIC: &[u8; 8] = b"LEXLEM1\0";

/// Open-class POS where lemmatization is a real transformation and Wiktionary helps.
/// Proper nouns / closed-class words copy the surface form instead (see lemma_lookup.py).
pub const CONTENT_POS: [&str; 4] = ["NOUN", "VERB", "ADJ", "ADV"];

pub struct LemmaTable {
    map: fst::Map<Vec<u8>>,
    blob: Vec<u8>,
}

impl LemmaTable {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("failed to read lemma table {}", path.display()))?;
        if data.len() < 16 || &data[..8] != MAGIC {
            bail!("{} is not a lexide lemma table (bad magic)", path.display());
        }
        let fst_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        if 16 + fst_len > data.len() {
            bail!("{} is truncated", path.display());
        }
        let map = fst::Map::new(data[16..16 + fst_len].to_vec())
            .with_context(|| format!("failed to parse fst in {}", path.display()))?;
        let blob = data[16 + fst_len..].to_vec();
        Ok(Self { map, blob })
    }

    /// Return the table lemma for (form, pos), or None if the table shouldn't fire.
    /// Non-content POS never fire: the builder only emits content-POS keys.
    pub fn lookup(&self, form: &str, pos: &str) -> Option<&str> {
        let mut key = Vec::with_capacity(pos.len() + 1 + form.len());
        key.extend_from_slice(pos.as_bytes());
        key.push(0);
        key.extend_from_slice(form.as_bytes());
        let v = self.map.get(key)?;
        let off = (v >> 16) as usize;
        let len = (v & 0xffff) as usize;
        std::str::from_utf8(self.blob.get(off..off + len)?).ok()
    }

    /// Layered fallback, mirroring `LemmaTable.resolve`: keep a confident (non-copy) model
    /// lemma; otherwise, for a content form the model punted on, use the table if it has a
    /// real lemma; last resort is the model lemma or the form itself.
    pub fn resolve(&self, form: &str, pos: &str, model_lemma: &str) -> String {
        if !model_lemma.is_empty() && model_lemma != form {
            return model_lemma.to_string(); // model produced a real transformation -> trust it
        }
        if let Some(table_lemma) = self.lookup(form, pos) {
            return table_lemma.to_string(); // fill the OOD floor
        }
        if model_lemma.is_empty() {
            form.to_string()
        } else {
            model_lemma.to_string()
        }
    }
}

/// Build the compact table from parsed Wiktionary JSON (`{pos: {form: [lemmas]}}`),
/// resolving multi-candidate entries the way `lemma_lookup.LemmaTable.lookup` does:
/// prefer the lemmatization the training data uses — first how training lemmatized this
/// exact form, then overall lemma frequency (`priors` =
/// `{pos: {"forms": {form: {lemma: n}}, "lemmas": {lemma: n}}}` from
/// `tagger/build_lemma_priors.py`; e.g. eng "love" over the obsolete homograph "lofe") —
/// then the candidate closest in char length to the form, ties broken by string order.
pub fn build_table<W: Write>(
    json: &serde_json::Value,
    priors: Option<&serde_json::Value>,
    mut out: W,
) -> Result<(usize, usize)> {
    let mut entries: BTreeMap<Vec<u8>, (String, u64)> = BTreeMap::new();
    let mut blob: Vec<u8> = Vec::new();
    let mut interned: HashMap<String, u64> = HashMap::new();

    let obj = json
        .as_object()
        .context("lemma table JSON must be an object keyed by POS")?;
    for pos in CONTENT_POS {
        let Some(forms) = obj.get(pos).and_then(|v| v.as_object()) else {
            continue;
        };
        let pos_priors = priors.and_then(|p| p.get(pos));
        let form_priors = pos_priors.and_then(|p| p.get("forms")).and_then(|v| v.as_object());
        let lemma_priors = pos_priors.and_then(|p| p.get("lemmas")).and_then(|v| v.as_object());
        let count = |table: Option<&serde_json::Map<String, serde_json::Value>>, key: &str| {
            table
                .and_then(|t| t.get(key))
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
        };
        for (form, cands) in forms {
            let cands: Vec<&str> = cands
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            if cands.is_empty() {
                continue;
            }
            let this_form = form_priors
                .and_then(|f| f.get(form.as_str()))
                .and_then(|v| v.as_object());
            let form_len = form.chars().count() as i64;
            let lemma = *cands
                .iter()
                .min_by_key(|c| {
                    (
                        -count(this_form, c),
                        -count(lemma_priors, c),
                        (c.chars().count() as i64 - form_len).abs(),
                        c.as_bytes(),
                    )
                })
                .unwrap();
            if lemma.len() > 0xffff {
                continue; // value packs the byte length into 16 bits; nothing real is this long
            }
            let off = *interned.entry(lemma.to_string()).or_insert_with(|| {
                let off = blob.len() as u64;
                blob.extend_from_slice(lemma.as_bytes());
                off
            });
            let mut key = Vec::with_capacity(pos.len() + 1 + form.len());
            key.extend_from_slice(pos.as_bytes());
            key.push(0);
            key.extend_from_slice(form.as_bytes());
            entries.insert(key, (lemma.to_string(), off << 16 | lemma.len() as u64));
        }
    }

    let mut builder = fst::MapBuilder::new(Vec::new()).context("fst builder")?;
    for (key, (_, value)) in &entries {
        builder.insert(key, *value).context("fst insert")?;
    }
    let fst_bytes = builder.into_inner().context("fst finish")?;

    out.write_all(MAGIC)?;
    out.write_all(&(fst_bytes.len() as u64).to_le_bytes())?;
    out.write_all(&fst_bytes)?;
    out.write_all(&blob)?;
    Ok((entries.len(), 16 + fst_bytes.len() + blob.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_table() -> LemmaTable {
        let json = serde_json::json!({
            "NOUN": {
                "Hunde": ["Hund"],
                "cats": ["cat", "catl", "ca"],       // deltas 1, 0, 2 -> closest length wins: "catl"
                "chats": ["chat", "chas"],           // equal delta -> lexicographically smaller "chas"
            },
            "VERB": {
                "went": ["go"],
                "love": ["lofe", "love"],            // form prior: training lemmatizes love->love
                "runs": ["rune", "run"],             // lemma-frequency prior beats closest-length
            },
            "PROPN": {"Berlin": ["Berlino"]},        // non-content POS: dropped at build
        });
        let priors = serde_json::json!({"VERB": {
            "forms": {"love": {"love": 1516}},
            "lemmas": {"run": 900}
        }});
        let mut buf = Vec::new();
        let (n, _) = build_table(&json, Some(&priors), &mut buf).unwrap();
        assert_eq!(n, 6); // PROPN entry excluded
        // round-trip through a temp file to exercise load(). The filename must be unique per
        // call: tests run in parallel within one process (same PID), so a PID-only name lets
        // two callers share — and delete — the same file, a flaky "No such file" on load.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join("lexide-lemma-test");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dir.join(format!("t{}-{}.fst", std::process::id(), unique));
        std::fs::write(&path, &buf).unwrap();
        let table = LemmaTable::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        table
    }

    #[test]
    fn lookup_matches_python_selection() {
        let t = sample_table();
        assert_eq!(t.lookup("Hunde", "NOUN"), Some("Hund"));
        // min by (|len delta in chars|, string): "catl" (delta 0) wins over "cat" and "ca"
        assert_eq!(t.lookup("cats", "NOUN"), Some("catl"));
        // equal delta (4 vs 5 for both): tie broken by byte order -> "chas"
        assert_eq!(t.lookup("chats", "NOUN"), Some("chas"));
        assert_eq!(t.lookup("went", "VERB"), Some("go"));
        // form-level training prior beats the lexicographic rule ("lofe" would otherwise win)
        assert_eq!(t.lookup("love", "VERB"), Some("love"));
        // lemma-frequency prior beats closest-length ("rune", delta 0, would otherwise win)
        assert_eq!(t.lookup("runs", "VERB"), Some("run"));
        assert_eq!(t.lookup("Berlin", "PROPN"), None);
        assert_eq!(t.lookup("unknown", "NOUN"), None);
    }

    #[test]
    fn resolve_layering() {
        let t = sample_table();
        // model produced a real transformation -> trusted over the table
        assert_eq!(t.resolve("Hunde", "NOUN", "hund"), "hund");
        // model punted to copy -> table floor fires
        assert_eq!(t.resolve("Hunde", "NOUN", "Hunde"), "Hund");
        // no table entry -> keep the model copy
        assert_eq!(t.resolve("Katzen", "NOUN", "Katzen"), "Katzen");
        // empty model lemma, no table entry -> form
        assert_eq!(t.resolve("Katzen", "NOUN", ""), "Katzen");
        // non-content POS: model copy stays even if a homograph exists elsewhere
        assert_eq!(t.resolve("went", "AUX", "went"), "went");
    }
}

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use std::{collections::HashMap, fs, path::PathBuf};

const LANGUAGES: &[&str] = &[
    "deu", "eng", "fra", "hin", "ita", "jpn", "kor", "por", "rus", "spa",
];
const GAPS: &[&str] = &[
    " ",
    "  ",
    "\t",
    "\n",
    "\n\n",
    "\r\n",
    "\r\n\r\n",
    "\n\t",
    "\u{00a0}",
    "\u{2003}",
    "\u{2009}",
    "\u{2028}",
    "\u{2029}",
    "\u{200b}",
    "\u{2060}",
    "\n---\n",
    "\n***\n",
    "\n···\n",
    "\n§ § §\n",
    "<!-- break -->",
    "\n[... ]\n",
    "",
];
const LEADERS: &[&str] = &["", "• ", "- ", "— ", "1. ", "(a) ", "※ ", "> "];
const WRAPPERS: &[(&str, &str)] = &[
    ("", ""),
    ("\"", "\""),
    ("“", "”"),
    ("‘", "’"),
    ("«", "»"),
    ("‹", "›"),
    ("「", "」"),
    ("『", "』"),
    ("（", "）"),
];
const OUTSIDE: &[&str] = &[
    "",
    "\n",
    "\n\n",
    "###\n",
    "§\n",
    "[document]\n",
    "<hr>\n",
    "\u{feff}",
    "\t\n",
];

#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn usize(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }
    fn pick<'a>(&mut self, values: &'a [&'a str]) -> &'a str {
        values[self.usize(values.len())]
    }
}

#[derive(Debug, Serialize)]
struct Section {
    #[serde(rename = "type")]
    kind: &'static str,
    content: String,
}
#[derive(Debug, Serialize)]
struct Record {
    id: String,
    lang: String,
    source: &'static str,
    text: String,
    sections: Vec<Section>,
}

fn push(sections: &mut Vec<Section>, kind: &'static str, content: String) {
    if content.is_empty() {
        return;
    }
    if let Some(last) = sections.last_mut() {
        if last.kind == kind && kind == "gap" {
            last.content.push_str(&content);
            return;
        }
    }
    sections.push(Section { kind, content });
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let data_root = PathBuf::from(args.next().unwrap_or_else(|| "../data/big".into()));
    let output = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "data/mechanical-augmented.jsonl".into()),
    );
    let per_language: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(2_000);
    let mut pools: HashMap<&str, Vec<String>> = HashMap::new();
    for &lang in LANGUAGES {
        let path = data_root
            .join(lang)
            .join("target_language_sentences_tokenization.jsonl");
        let input =
            fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let pool = input
            .lines()
            .filter_map(|line| {
                let value: Value = serde_json::from_str(line).ok()?;
                let sentence = value.get("sentence")?.as_str()?.to_owned();
                (!sentence.is_empty()).then_some(sentence)
            })
            .collect::<Vec<_>>();
        if pool.is_empty() {
            bail!("no sentences for {lang}")
        }
        pools.insert(lang, pool);
    }

    let mut out = String::new();
    for (lang_index, &lang) in LANGUAGES.iter().enumerate() {
        let pool = &pools[lang];
        let mut rng = Rng::new(0x9e3779b97f4a7c15 ^ ((lang_index as u64 + 1) * 0x100000001b3));
        for sample in 0..per_language {
            let mut sections = Vec::new();
            push(&mut sections, "gap", rng.pick(OUTSIDE).to_owned());
            let sentence_count = 2 + rng.usize(15);
            for index in 0..sentence_count {
                let sentence = &pool[rng.usize(pool.len())];
                let leader = rng.pick(LEADERS);
                let (open, close) = WRAPPERS[rng.usize(WRAPPERS.len())];
                push(
                    &mut sections,
                    "sentence",
                    format!("{leader}{open}{sentence}{close}"),
                );
                if index + 1 < sentence_count {
                    push(&mut sections, "gap", rng.pick(GAPS).to_owned());
                }
            }
            push(&mut sections, "gap", rng.pick(OUTSIDE).to_owned());
            let text: String = sections.iter().map(|s| s.content.as_str()).collect();
            if text
                != sections
                    .iter()
                    .map(|s| s.content.as_str())
                    .collect::<String>()
            {
                unreachable!()
            }
            let record = Record {
                id: format!("mechanical-{lang}-{sample:05}"),
                lang: lang.to_owned(),
                source: "mechanical-sentence-composition-v1",
                text,
                sections,
            };
            out.push_str(&serde_json::to_string(&record)?);
            out.push('\n');
        }
    }
    fs::write(&output, out).with_context(|| format!("write {}", output.display()))?;
    println!(
        "Wrote {} examples ({} per language) to {}",
        per_language * LANGUAGES.len(),
        per_language,
        output.display()
    );
    Ok(())
}

//! The voice's tokenizer (voca/text.py's `tokenize`): yap grams to symbol ids and per-token lexical features.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

/// One word or phrase as yap annotates it.
#[derive(Clone, Debug, PartialEq, Deserialize, serde::Serialize)]
pub struct Gram {
    pub words: Vec<String>,
    pub pos: Vec<String>,
    /// Corpus frequency, bucketed into the rarity feature.
    pub count: f64,
    /// Space-separated phones, words split by the frontend's word break.
    pub phonemes: String,
}

/// The tokenizer spec bundled in the voice manifest (`frontend_spec()` in voca/text.py).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Spec {
    version: u32,
    symbol_table: Vec<String>,
    pub phones: Vec<String>,
    punctuation: Vec<String>,
    aliases: HashMap<String, String>,
    clitics: Vec<String>,
    elision_endings: Vec<String>,
    attached_tags: Vec<String>,
    pos: HashMap<String, i64>,
    other_pos: i64,
    stress: HashMap<String, i64>,
    rarity_buckets: u32,
    word_break: String,
    phrase: String,
    nucleus: String,
}

/// Token ids, one lexical row per token (flattened), and the index of the first phone.
pub struct Tokens {
    pub ids: Vec<i64>,
    pub lexical: Vec<i64>,
    pub first_phone: Option<usize>,
}

pub struct Tokenizer {
    spec: Spec,
    ids: HashMap<String, i64>,
    phones: HashSet<String>,
    /// Which feature fills each lexical column; None for a column the voice was trained without (hints off).
    columns: Vec<Option<Feature>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Feature {
    Stress,
    Tone,
    Accent,
    Pos,
    Rarity,
    Boundary,
}

#[derive(Default)]
struct Features {
    stress: i64,
    tone: i64,
    accent: i64,
    pos: i64,
    rarity: i64,
    boundary: i64,
}

/// Python's str.split() whitespace, which includes the ASCII separators U+001C..U+001F.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub fn words(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_space).filter(|w| !w.is_empty())
}

struct Spoken {
    pos: i64,
    after: Vec<String>,
}

impl Tokenizer {
    pub fn new(spec: Spec, columns: &[String], hints: bool) -> Result<Tokenizer, String> {
        if spec.version != 4 {
            return Err(format!("unsupported tokenizer version {}", spec.version));
        }
        let columns = columns
            .iter()
            .map(|c| {
                let f = match c.as_str() {
                    "stress" => Feature::Stress,
                    "tone" => Feature::Tone,
                    "accent" => Feature::Accent,
                    "pos" => Feature::Pos,
                    "rarity" => Feature::Rarity,
                    "boundary" => Feature::Boundary,
                    _ => return Err(format!("unknown lexical column {c}")),
                };
                Ok(
                    (hints || !matches!(f, Feature::Pos | Feature::Rarity | Feature::Boundary))
                        .then_some(f),
                )
            })
            .collect::<Result<_, String>>()?;
        let ids = spec
            .symbol_table
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as i64))
            .collect();
        let phones = spec.phones.iter().cloned().collect();
        Ok(Tokenizer {
            spec,
            ids,
            phones,
            columns,
        })
    }

    pub fn phones(&self) -> &[String] {
        &self.spec.phones
    }

    fn alias<'a>(&'a self, s: &'a str) -> &'a str {
        self.spec.aliases.get(s).map_or(s, String::as_str)
    }

    fn is_mark(&self, s: &str) -> bool {
        self.spec.punctuation.iter().any(|p| p == s)
    }

    /// The punctuation marks a written word stands for.
    fn punctuation(&self, words: &[String]) -> Vec<String> {
        words
            .iter()
            .flat_map(|word| {
                let alias = self.alias(word);
                if self.is_mark(alias) {
                    return vec![alias.to_string()];
                }
                alias
                    .chars()
                    .map(|c| {
                        let mut b = [0; 4];
                        self.alias(c.encode_utf8(&mut b)).to_string()
                    })
                    .filter(|c| self.is_mark(c))
                    .collect()
            })
            .collect()
    }

    /// Groups the gram's words into the spoken words its phoneme groups stand for, with trailing punctuation.
    fn spoken_words(&self, gram: &Gram, attach: bool) -> (Vec<String>, Vec<Spoken>) {
        let (mut leading, mut spoken) = (vec![], Vec::<Spoken>::new());
        let mut join: Option<&str> = None;
        for (word, tag) in gram.words.iter().zip(&gram.pos) {
            let tag_id = self
                .spec
                .pos
                .get(tag)
                .copied()
                .unwrap_or(self.spec.other_pos);
            let joinable = spoken.last().is_some_and(|p| p.after.is_empty());
            if word == "-" && !spoken.is_empty() {
                join = Some("-");
            } else if !word.chars().any(char::is_alphanumeric) {
                let marks = self.punctuation(std::slice::from_ref(word));
                spoken
                    .last_mut()
                    .map_or(&mut leading, |p| &mut p.after)
                    .extend(marks);
            } else if joinable
                && (join.is_some()
                    || self
                        .spec
                        .clitics
                        .iter()
                        .any(|c| word.to_lowercase().starts_with(c.as_str()))
                    || (attach && self.spec.attached_tags.contains(tag)))
            {
                if join == Some("") {
                    spoken.last_mut().unwrap().pos = tag_id;
                }
                join = None;
            } else {
                spoken.push(Spoken {
                    pos: tag_id,
                    after: vec![],
                });
                join = self
                    .spec
                    .elision_endings
                    .iter()
                    .any(|e| word.ends_with(e.as_str()))
                    .then_some("");
            }
        }
        (leading, spoken)
    }

    fn add(&self, t: &mut Tokens, symbol: &str, f: Features) -> Result<(), String> {
        t.ids.push(
            *self
                .ids
                .get(symbol)
                .ok_or_else(|| format!("unknown symbol {symbol}"))?,
        );
        t.lexical.extend(self.columns.iter().map(|c| match c {
            Some(Feature::Stress) => f.stress,
            Some(Feature::Tone) => f.tone,
            Some(Feature::Accent) => f.accent,
            Some(Feature::Pos) => f.pos,
            Some(Feature::Rarity) => f.rarity,
            Some(Feature::Boundary) => f.boundary,
            None => 0,
        }));
        Ok(())
    }

    /// Punctuation tokens of a boundary, or a silent pause where there is none.
    fn pause(&self, t: &mut Tokens, marks: &[String], kind: i64) -> Result<(), String> {
        let f = || Features {
            boundary: kind,
            ..Default::default()
        };
        if marks.is_empty() {
            return self.add(t, "<sp>", f());
        }
        marks.iter().try_for_each(|m| self.add(t, m, f()))
    }

    pub fn tokenize(&self, grams: &[Gram]) -> Result<Tokens, String> {
        let mut t = Tokens {
            ids: vec![],
            lexical: vec![],
            first_phone: None,
        };
        self.add(&mut t, "<bos>", Features::default())?;
        let mut spoken_before = false;
        for gram in grams {
            if gram.words.len() != gram.pos.len() {
                return Err("expected one POS tag per word".into());
            }
            let sequence = words(&gram.phonemes).collect::<Vec<_>>().join(" ");
            if sequence.is_empty() {
                for mark in self.punctuation(&gram.words) {
                    self.pause(&mut t, &[mark], 2)?;
                }
                continue;
            }
            let groups: Vec<&str> = sequence
                .split(&format!(" {} ", self.spec.word_break))
                .collect();
            let plain = self.spoken_words(gram, false);
            let (leading, spoken) =
                if groups.len() != plain.1.len() && !self.spec.attached_tags.is_empty() {
                    self.spoken_words(gram, true)
                } else {
                    plain
                };
            let last = &self.spec.symbol_table[*t.ids.last().unwrap() as usize];
            if !leading.is_empty() || (spoken_before && !self.is_mark(last)) {
                self.pause(&mut t, &leading, 2)?;
            }
            let ceiling = 2f64.powi(self.spec.rarity_buckets as i32 - 1);
            let rarity = 1 + ceiling.min(gram.count + 1.0).log2().floor() as i64;
            let matched = groups.len() == spoken.len();
            for (i, group) in groups.iter().enumerate() {
                if i > 0 {
                    self.pause(&mut t, if matched { &spoken[i - 1].after } else { &[] }, 1)?;
                }
                let pos = if matched || groups.len() == 1 {
                    spoken.get(i).map_or(0, |s| s.pos)
                } else {
                    0
                };
                let mut start = 0;
                for token in group.split(' ') {
                    if token == self.spec.phrase {
                        start = 1;
                        continue;
                    }
                    let first = token.chars().next().map(String::from).unwrap_or_default();
                    let stress = self.spec.stress.get(&first).copied().unwrap_or(0);
                    let token = if stress != 0 {
                        &token[first.len()..]
                    } else {
                        token
                    };
                    let core = token.strip_suffix(self.spec.nucleus.as_str());
                    let token = core.unwrap_or(token);
                    let phone = token.trim_end_matches(|c: char| c.is_ascii_digit());
                    if !self.phones.contains(phone) {
                        return Err(format!("phone outside the voice inventory: {phone}"));
                    }
                    t.first_phone.get_or_insert(t.ids.len());
                    let tone = token[phone.len()..].parse().unwrap_or(0);
                    let accent = start | if core.is_some() { 2 } else { 0 };
                    self.add(
                        &mut t,
                        phone,
                        Features {
                            stress,
                            tone,
                            accent,
                            pos,
                            rarity,
                            ..Default::default()
                        },
                    )?;
                    start = 0;
                }
            }
            for mark in spoken.last().map_or(&[][..], |s| &s.after) {
                self.pause(&mut t, std::slice::from_ref(mark), 2)?;
            }
            spoken_before = true;
        }
        self.add(&mut t, "<eos>", Features::default())?;
        Ok(t)
    }
}

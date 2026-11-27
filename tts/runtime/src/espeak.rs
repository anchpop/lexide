//! Text to grams through a caller-supplied espeak-ng (plain/stock_espeak.py): punctuation comes from the text,
//! phones from espeak's `--ipa --sep=_` trace mapped into the voice inventory.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

use crate::frontend::{Gram, is_space, words};

#[derive(Deserialize)]
pub struct Spec {
    /// The espeak voice name the model was trained with (en-us for English).
    pub voice: String,
    keep: Vec<String>,
    vowels: Vec<String>,
    rhotic: Vec<String>,
    split: HashMap<String, Vec<String>>,
    fold: HashMap<String, String>,
    rename: HashMap<String, String>,
}

pub struct Espeak {
    pub spec: Spec,
    inventory: HashSet<String>,
    /// Inventory phones by their canonical decomposition, so espeak's composed or decomposed forms both match.
    decomposed: HashMap<String, String>,
}

/// A stretch of text to phonemize, or punctuation between stretches.
enum Piece<'a> {
    Text(&'a str),
    Mark(&'a str),
}

/// Length in chars of the punctuation unit at `i`, 0 if none: one of the stock_espeak.py MARKS, an apostrophe not
/// inside an ASCII word, or a dash run standing alone between spaces.
fn mark_at(chars: &[char], i: usize) -> usize {
    let (prev, next) = (
        i.checked_sub(1).map(|j| chars[j]),
        chars.get(i + 1).copied(),
    );
    let letter = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphabetic());
    match chars[i] {
        '.' | ',' | ';' | ':' | '!' | '?' | '…' | '—' | '–' | '(' | ')' | '"' | '“' | '”' | '‘'
        | '«' | '»' | '[' | ']' => 1,
        '\'' | '’' if !letter(prev) || !letter(next) => 1,
        '-' if prev.is_some_and(is_space) => {
            let run = chars[i..].iter().take_while(|&&c| c == '-').count();
            if chars.get(i + run).is_some_and(|&c| is_space(c)) {
                run
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// Splits text into maximal punctuation runs and the stretches between them.
fn pieces(text: &str) -> Vec<Piece<'_>> {
    let chars: Vec<char> = text.chars().collect();
    let bytes: Vec<usize> = text
        .char_indices()
        .map(|(b, _)| b)
        .chain([text.len()])
        .collect();
    let (mut out, mut start, mut i) = (vec![], 0, 0);
    while i < chars.len() {
        let mut end = i;
        while end < chars.len() && mark_at(&chars, end) > 0 {
            end += mark_at(&chars, end);
        }
        if end == i {
            i += 1;
            continue;
        }
        out.push(Piece::Text(&text[bytes[start]..bytes[i]]));
        out.push(Piece::Mark(&text[bytes[i]..bytes[end]]));
        (start, i) = (end, end);
    }
    out.push(Piece::Text(&text[bytes[start]..]));
    out
}

fn speaks(s: &str) -> bool {
    s.chars().any(|c| c.is_alphanumeric() || c == '_')
}

/// What espeak is given: curly apostrophes inside ASCII words become straight ones.
fn espeak_input(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let letter = |j: Option<&char>| j.is_some_and(|c| c.is_ascii_alphabetic());
    chars
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if c == '’' && letter(i.checked_sub(1).map(|j| &chars[j])) && letter(chars.get(i + 1))
            {
                '\''
            } else {
                c
            }
        })
        .collect()
}

/// Drops espeak's language-switch markers such as `(en)`.
fn strip_switches(ipa: &str) -> String {
    let mut out = String::new();
    let mut rest = ipa;
    while let Some(open) = rest.find('(') {
        let inner = &rest[open + 1..];
        let len = inner
            .find(|c: char| !(c.is_ascii_lowercase() || c == '-'))
            .unwrap_or(inner.len());
        let switch = len > 0 && inner[len..].starts_with(')');
        out.push_str(&rest[..open]);
        if !switch {
            out.push('(');
        }
        rest = if switch { &inner[len + 1..] } else { inner };
    }
    out + rest
}

#[derive(Clone)]
struct Unit {
    stress: String,
    phone: String,
}

impl Espeak {
    pub fn new(spec: Spec, phones: &[String]) -> Espeak {
        let decomposed = phones
            .iter()
            .map(|p| (p.nfd().collect(), p.clone()))
            .collect();
        Espeak {
            spec,
            inventory: phones.iter().cloned().collect(),
            decomposed,
        }
    }

    /// The stretches of `text` the caller must phonemize, in order.
    pub fn chunks(&self, text: &str) -> Vec<String> {
        pieces(text)
            .into_iter()
            .filter_map(|p| match p {
                Piece::Text(s) if speaks(s) => Some(espeak_input(s)),
                _ => None,
            })
            .collect()
    }

    /// Grams for `text`, given espeak's output for each of its `chunks`.
    pub fn grams(&self, text: &str, ipa: &[String]) -> Result<Vec<Gram>, String> {
        let mut ipa = ipa.iter();
        let mut out = vec![];
        for piece in pieces(text) {
            let (written, tag, phonemes) = match piece {
                Piece::Mark(m) => (vec![m.to_string()], "PUNCT", String::new()),
                Piece::Text(s) if speaks(s) => {
                    let trace =
                        strip_switches(ipa.next().ok_or("fewer phonemizations than chunks")?);
                    let phones: Vec<String> = words(&trace).map(|w| self.word_phones(w)).collect();
                    (
                        words(s).map(String::from).collect(),
                        "X",
                        phones.join(" # "),
                    )
                }
                Piece::Text(_) => continue,
            };
            out.push(Gram {
                pos: vec![tag.into(); written.len()],
                words: written,
                count: 0.0,
                phonemes,
            });
        }
        if ipa.next().is_some() {
            return Err("more phonemizations than chunks".into());
        }
        Ok(out)
    }

    fn fold(&self, p: &str) -> String {
        self.spec
            .fold
            .get(p)
            .cloned()
            .unwrap_or_else(|| p.to_string())
    }

    fn vowel(&self, p: &str) -> bool {
        p.chars()
            .next()
            .is_some_and(|c| self.spec.vowels.iter().any(|v| v.chars().eq([c])))
    }

    /// An inventory phone, or the longest inventory prefixes the unit splits into (the unit itself if none fit).
    fn inventory_phones(&self, unit: &str) -> Vec<String> {
        let p = self.spec.rename.get(unit).map_or(unit, String::as_str);
        if let Some(phone) = self
            .inventory
            .get(p)
            .or_else(|| self.decomposed.get(&p.nfd().collect::<String>()))
        {
            return vec![phone.clone()];
        }
        let bounds: Vec<usize> = p.char_indices().map(|(b, _)| b).chain([p.len()]).collect();
        let (mut out, mut i) = (vec![], 0);
        while i + 1 < bounds.len() {
            let Some(end) = (i + 1..bounds.len())
                .rev()
                .find(|&e| self.inventory.contains(&p[bounds[i]..bounds[e]]))
            else {
                return vec![p.to_string()];
            };
            out.push(p[bounds[i]..bounds[end]].to_string());
            i = end;
        }
        out
    }

    /// The model's units for one espeak unit: folds, explicit splits, geminates, and vowel-led clusters.
    fn units(&self, unit: &str) -> Vec<String> {
        let p = self.fold(unit);
        if let Some(split) = self.spec.split.get(&p) {
            return split.iter().map(|s| self.fold(s)).collect();
        }
        let chars: Vec<char> = p.chars().collect();
        if chars.len() == 2 && chars[0] == chars[1] && !self.vowel(&p) {
            return vec![format!("{}ː", chars[0])];
        }
        if !self.vowel(&p) || self.spec.keep.contains(&p) {
            return vec![p];
        }
        let mut out: Vec<String> = vec![];
        for c in chars {
            let s = c.to_string();
            match out.last_mut() {
                Some(last) if !self.vowel(&s) => last.push(c),
                _ => out.push(self.fold(&s)),
            }
        }
        out.iter().map(|s| self.fold(s)).collect()
    }

    fn word_phones(&self, word: &str) -> String {
        let mut units: Vec<Unit> = vec![];
        for raw in word.split('_') {
            let stress: String = raw.chars().filter(|c| matches!(c, 'ˈ' | 'ˌ')).collect();
            let bare: String = raw
                .chars()
                .filter(|c| !matches!(c, 'ˈ' | 'ˌ' | '-'))
                .collect();
            for piece in self.units(&bare) {
                let split = self.inventory_phones(&piece);
                let carrier = split.iter().position(|p| self.vowel(p)).unwrap_or(0);
                for (i, phone) in split.into_iter().enumerate() {
                    let inherited = match units.last() {
                        Some(prev) if self.vowel(&prev.phone) && self.vowel(&phone) => {
                            prev.stress.clone()
                        }
                        _ => String::new(),
                    };
                    let stress = if i == carrier && !stress.is_empty() {
                        stress.clone()
                    } else {
                        inherited
                    };
                    units.push(Unit { stress, phone });
                }
            }
        }
        let mut out: Vec<Unit> = vec![];
        for (i, unit) in units.iter().enumerate() {
            match out.last_mut() {
                // English coda vowel + ɹ is one phone, unless a vowel follows.
                Some(prev)
                    if unit.phone == "ɹ"
                        && self.spec.rhotic.contains(&prev.phone)
                        && !units.get(i + 1).is_some_and(|n| self.vowel(&n.phone)) =>
                {
                    prev.phone.push('ɹ')
                }
                Some(prev)
                    if unit.phone == prev.phone
                        && !self.vowel(&unit.phone)
                        && unit.stress.is_empty() =>
                {
                    prev.phone.push('ː')
                }
                _ => out.push(unit.clone()),
            }
        }
        out.iter()
            .map(|u| format!("{}{}", u.stress, u.phone))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<String> {
        pieces(text)
            .iter()
            .map(|p| match p {
                Piece::Text(s) => format!("t:{s}"),
                Piece::Mark(m) => format!("m:{m}"),
            })
            .collect()
    }

    #[test]
    fn punctuation_runs() {
        assert_eq!(
            split("Wait, are you sure?!"),
            ["t:Wait", "m:,", "t: are you sure", "m:?!", "t:"]
        );
        assert_eq!(
            split("don't 'quote' a -- b a-b"),
            [
                "t:don't ", "m:'", "t:quote", "m:'", "t: a ", "m:--", "t: b a-b"
            ]
        );
        assert_eq!(espeak_input("Don’t ’x"), "Don't ’x");
        assert_eq!(strip_switches("(en)hɛloʊ (x1) a(b"), "hɛloʊ (x1) a(b");
    }
}

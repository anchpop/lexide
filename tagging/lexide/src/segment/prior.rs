//! Boundary priors — a per-byte proposal fed to the byte model alongside the text.
//!
//! The model has to decide where words start from raw bytes. With whitespace that is
//! nearly free: an *unseen* German word is still 97% correct because the spaces hand over
//! both boundaries. Japanese has no such signal, so every rare word must be known rather
//! than copied — measured on the shipped tokenizer, a Japanese token seen fewer than 500
//! times was 24-30% wrong against 0% for German. Feeding a lexicon-backed proposal lifted
//! Japanese from 86.6 to 94.5 and Korean from 91.8 to 95.2.
//!
//! The proposal is not the answer. It contains our boundaries — 99.5% of gold token starts
//! including 98.8% of never-seen words — while over-proposing about 1.18x, so the model's
//! job becomes deleting boundaries rather than finding them.
//!
//! This is a port of `tagger/prior.py`, and it must stay one: training data is generated
//! with *this* implementation (via the `emit-priors` binary) so the priors a model trains
//! on and the priors it sees at inference cannot drift apart.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use super::unidic::UniDic;

/// Per-byte prior ids. NONE covers BOS/EOS/padding, so 0 is a safe fill.
///
/// `B` and `B_SOFT` are separate because a prior's worth is its precision, not its
/// coverage. Whitespace marks a real Korean boundary 100% of the time; a Viterbi bank
/// finds far more boundaries and is right 92.6% of the time. Encoding both as one symbol
/// forces a single average trust level, and measurably cost 3.3 F1 on Korean when a bank
/// replaced whitespace. Kept apart, the model can trust one absolutely and weigh the other.
pub const PRIOR_NONE: u8 = 0;
pub const PRIOR_O: u8 = 1;
pub const PRIOR_B: u8 = 2;
pub const PRIOR_I: u8 = 3;
pub const PRIOR_B_SOFT: u8 = 4;
pub const PRIOR_VOCAB: usize = 5;

/// Character classes, following MeCab's `char.def`: a run of one script is usually one
/// word. The limit is how long an *unknown* run of that class may be grouped into a single
/// proposal — katakana loanwords and numbers run long, kanji compounds are usually 2-3, and
/// a run of hiragana is grammar rather than one word so it is never grouped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CharType {
    Katakana,
    Hiragana,
    Kanji,
    Digit,
    Latin,
    Other,
}

impl CharType {
    pub fn of(ch: char) -> Self {
        let o = ch as u32;
        if (0x30A0..=0x30FF).contains(&o) || o == 0x30FC {
            CharType::Katakana
        } else if (0x3040..=0x309F).contains(&o) {
            CharType::Hiragana
        } else if (0x4E00..=0x9FFF).contains(&o) || (0x3400..=0x4DBF).contains(&o) {
            CharType::Kanji
        } else if ch.is_ascii_digit() {
            CharType::Digit
        } else if ch.is_ascii_alphabetic() {
            CharType::Latin
        } else {
            CharType::Other
        }
    }

    pub fn unk_max_len(self) -> usize {
        match self {
            CharType::Katakana | CharType::Latin | CharType::Digit => 24,
            CharType::Kanji => 3,
            CharType::Hiragana | CharType::Other => 1,
        }
    }
}

/// Anything that can propose word boundaries inside one whitespace-free run.
///
/// Whitespace handling, the hard/soft distinction and the byte alignment are the same for
/// every source, so they live here once; an implementor only has to segment a run. Today
/// that is the corpus [`Wordbank`] and the bundled [`UniDic`](super::unidic::UniDic).
pub trait Proposer {
    /// Minimum-cost segmentation of `chars`, as (start, end) character indices.
    fn segment_run(&self, chars: &[char]) -> Vec<(usize, usize)>;
}

/// A unigram wordbank plus Viterbi — the same proposal a morphological analyzer gives,
/// from a file small enough to ship beside the weights.
pub struct Wordbank {
    cost: HashMap<String, f64>,
    max_len: usize,
    unk: f64,
    unk_len_penalty: f64,
    group_unknown: bool,
}

impl Wordbank {
    pub fn from_counts(counts: HashMap<String, u64>, group_unknown: bool) -> Self {
        let total: u64 = counts.values().sum::<u64>().max(1);
        let max_len = counts.keys().map(|w| w.chars().count()).max().unwrap_or(1);
        let cost = counts
            .into_iter()
            .map(|(w, c)| (w, -((c as f64) / (total as f64)).ln()))
            .collect();
        // The scan window must also admit an unknown run at its full grouped length —
        // otherwise the longest *known* entry silently caps how much unseen katakana can
        // be proposed as one word.
        let max_len = max_len.max(CharType::Katakana.unk_max_len());
        Self {
            cost,
            max_len,
            unk: -(1.0 / (total as f64 * 100.0)).ln(),
            unk_len_penalty: 0.5,
            group_unknown,
        }
    }

    /// `word<TAB>count` per line, as emitted by `tagger/build_wordbanks.py`.
    pub fn load(path: &Path, group_unknown: bool) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading wordbank {}", path.display()))?;
        let mut counts = HashMap::new();
        for line in text.lines() {
            if let Some((word, count)) = line.split_once('\t') {
                if let Ok(n) = count.trim().parse::<u64>() {
                    if !word.is_empty() {
                        counts.insert(word.to_string(), n);
                    }
                }
            }
        }
        Ok(Self::from_counts(counts, group_unknown))
    }

    fn segment_chars(&self, chars: &[char]) -> Vec<(usize, usize)> {
        let n = chars.len();
        let types: Vec<CharType> = chars.iter().map(|&c| CharType::of(c)).collect();
        let mut best = vec![f64::INFINITY; n + 1];
        let mut back = vec![0usize; n + 1];
        best[0] = 0.0;
        let mut buf = String::new();
        for i in 1..=n {
            let lo = i.saturating_sub(self.max_len);
            for j in lo..i {
                buf.clear();
                buf.extend(&chars[j..i]);
                let c = match self.cost.get(&buf) {
                    Some(&c) => c,
                    None => {
                        let span = i - j;
                        let t = types[j];
                        let limit = if self.group_unknown { t.unk_max_len() } else { 1 };
                        if span <= limit && types[j..i].iter().all(|&x| x == t) {
                            // one unknown word, not one per character
                            self.unk + self.unk_len_penalty * (span - 1) as f64
                        } else {
                            continue;
                        }
                    }
                };
                if best[j] + c < best[i] {
                    best[i] = best[j] + c;
                    back[i] = j;
                }
            }
        }
        let mut spans = Vec::new();
        let mut i = n;
        while i > 0 {
            let j = back[i];
            spans.push((j, i));
            i = j;
        }
        spans.reverse();
        spans
    }
}

impl Proposer for Wordbank {
    fn segment_run(&self, chars: &[char]) -> Vec<(usize, usize)> {
        self.segment_chars(chars)
    }
}

/// Whitespace is a hard boundary; the proposer only splits *inside* a run.
///
/// One mechanism for every language. Japanese has no spaces, so the sentence is a single
/// run and this is plain Viterbi. Korean gets the eojeol boundary free from whitespace and
/// the proposer splits inside it (나는 밥을 -> 나|는|밥|을), which whitespace cannot see and
/// where Korean's remaining errors lived.
pub fn segment_constrained(p: &dyn Proposer, chars: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !chars[j].is_whitespace() {
            j += 1;
        }
        spans.extend(p.segment_run(&chars[i..j]).into_iter().map(|(a, b)| (i + a, i + b)));
        i = j;
    }
    spans
}

/// Indices where whitespace *guarantees* a token begins — the run starts.
///
/// Every proposal source shares this rule, so `PRIOR_B` means the same thing in every
/// language: a boundary we are certain of. Anything a dictionary merely proposes is
/// `PRIOR_B_SOFT`, whatever proposed it — including the Japanese analyzer, whose 84.4%
/// precision is the lowest of the three sources we ship.
fn hard_starts(chars: &[char]) -> Vec<bool> {
    let mut hard = vec![false; chars.len()];
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        hard[i] = true;
        while i < chars.len() && !chars[i].is_whitespace() {
            i += 1;
        }
    }
    hard
}

fn proposer_char_labels(p: &dyn Proposer, chars: &[char]) -> Vec<u8> {
    let mut out = vec![PRIOR_O; chars.len()];
    let hard = hard_starts(chars);
    for (a, b) in segment_constrained(p, chars) {
        if chars[a..b].iter().any(|c| !c.is_whitespace()) {
            out[a] = if hard[a] { PRIOR_B } else { PRIOR_B_SOFT };
            for slot in out.iter_mut().take(b).skip(a + 1) {
                *slot = PRIOR_I;
            }
        }
    }
    out
}

/// B on the first character of each whitespace-delimited run, I inside, O on the space.
fn whitespace_char_labels(chars: &[char]) -> Vec<u8> {
    let mut out = Vec::with_capacity(chars.len());
    let mut at_start = true;
    for &ch in chars {
        if ch.is_whitespace() {
            out.push(PRIOR_O);
            at_start = true;
        } else {
            out.push(if at_start { PRIOR_B } else { PRIOR_I });
            at_start = false;
        }
    }
    out
}

/// Per-byte prior ids aligned to `[BOS] + utf8(text) + [EOS]`.
///
/// Mirrors `dataset.encode_bytes_and_labels`: one leading slot for BOS, each character's
/// label on its first byte with continuation bytes marked I inside a proposed word, one
/// trailing slot for EOS — so the prior and the byte stream stay aligned byte for byte.
pub fn prior_ids(text: &str, proposer: Option<&dyn Proposer>, max_bytes: usize) -> Vec<u8> {
    let chars: Vec<char> = text.chars().collect();
    let labels = match proposer {
        Some(p) => proposer_char_labels(p, &chars),
        None => whitespace_char_labels(&chars),
    };
    let mut ids = Vec::with_capacity(text.len() + 2);
    ids.push(PRIOR_NONE);
    for (ch, &lab) in chars.iter().zip(labels.iter()) {
        let n_bytes = ch.len_utf8();
        ids.push(lab);
        if n_bytes > 1 {
            let cont = if lab == PRIOR_B || lab == PRIOR_I || lab == PRIOR_B_SOFT {
                PRIOR_I
            } else {
                PRIOR_O
            };
            for _ in 1..n_bytes {
                ids.push(cont);
            }
        }
    }
    ids.push(PRIOR_NONE);
    ids.truncate(max_bytes);
    ids
}

/// The proposal itself, as character spans — for diagnostics and parity testing.
pub fn proposal_spans(text: &str, proposer: Option<&dyn Proposer>) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    match proposer {
        Some(p) => segment_constrained(p, &chars),
        None => {
            let labels = whitespace_char_labels(&chars);
            let mut spans = Vec::new();
            let mut start: Option<usize> = None;
            for (i, &lab) in labels.iter().enumerate() {
                match lab {
                    PRIOR_B | PRIOR_B_SOFT => {
                        if let Some(s) = start {
                            spans.push((s, i));
                        }
                        start = Some(i);
                    }
                    PRIOR_O => {
                        if let Some(s) = start.take() {
                            spans.push((s, i));
                        }
                    }
                    _ => {}
                }
            }
            if let Some(s) = start {
                spans.push((s, chars.len()));
            }
            spans
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bank(pairs: &[(&str, u64)], group_unknown: bool) -> Wordbank {
        Wordbank::from_counts(
            pairs.iter().map(|(w, c)| (w.to_string(), *c)).collect(),
            group_unknown,
        )
    }

    #[test]
    fn whitespace_is_a_hard_boundary() {
        let wb = bank(&[("나", 100), ("는", 100), ("밥", 50), ("을", 100)], true);
        let chars: Vec<char> = "나는 밥을".chars().collect();
        // the eojeol boundary comes from whitespace, the split inside it from the bank
        assert_eq!(segment_constrained(&wb, &chars), vec![(0, 1), (1, 2), (3, 4), (4, 5)]);
    }

    #[test]
    fn unknown_runs_group_by_script() {
        let wb = bank(&[("が", 100), ("好き", 50)], true);
        let chars: Vec<char> = "ブロックチェーンが好き".chars().collect();
        let spans = segment_constrained(&wb, &chars);
        // the unseen katakana loanword stays whole rather than shattering
        assert_eq!(chars[spans[0].0..spans[0].1].iter().collect::<String>(), "ブロックチェーン");

        let shattered = bank(&[("が", 100), ("好き", 50)], false);
        let spans = segment_constrained(&shattered, &chars);
        assert_eq!(chars[spans[0].0..spans[0].1].iter().collect::<String>(), "ブ");
    }

    #[test]
    fn prior_ids_align_with_the_byte_stream() {
        let text = "手紙をtranslate";
        let ids = prior_ids(text, None, 512);
        // one slot per byte, plus BOS and EOS
        assert_eq!(ids.len(), text.len() + 2);
        assert_eq!(ids[0], PRIOR_NONE);
        assert_eq!(*ids.last().unwrap(), PRIOR_NONE);
        // 手 is three bytes: B then two continuation slots marked I
        assert_eq!(&ids[1..4], &[PRIOR_B, PRIOR_I, PRIOR_I]);
    }

    #[test]
    fn whitespace_prior_marks_each_run() {
        let ids = prior_ids("ab cd", None, 512);
        assert_eq!(
            ids,
            vec![PRIOR_NONE, PRIOR_B, PRIOR_I, PRIOR_O, PRIOR_B, PRIOR_I, PRIOR_NONE]
        );
    }
}

/// The proposal sources that ship with a model: the bundled UniDic for Japanese, plus a
/// corpus wordbank for each language that has one.
///
/// Which languages carry a bank is a training-time decision — a bank finds more boundaries
/// than whitespace but is less precise, so it is not a free win everywhere — and the answer
/// is recorded by which files the release contains. Loading is therefore a directory scan
/// rather than a hardcoded list: the model and its priors cannot disagree.
#[derive(Default)]
pub struct PriorSet {
    unidic: Option<UniDic>,
    banks: HashMap<String, Wordbank>,
}

impl PriorSet {
    /// Reads `jpn-unidic.bin` and `wordbanks/*.tsv` from a model directory. Missing files
    /// are not an error — the languages they cover fall back to whitespace.
    pub fn load(dir: &Path) -> Result<Self> {
        let mut set = Self::default();
        let dict = dir.join("jpn-unidic.bin");
        if dict.exists() {
            set.unidic = Some(UniDic::load(&dict)?);
        }
        let banks_dir = dir.join("wordbanks");
        if let Ok(entries) = std::fs::read_dir(&banks_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("tsv") {
                    continue;
                }
                let Some(lang) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                set.banks
                    .insert(lang.to_string(), Wordbank::load(&path, true)?);
            }
        }
        Ok(set)
    }

    pub fn is_empty(&self) -> bool {
        self.unidic.is_none() && self.banks.is_empty()
    }

    /// The best proposer for a language, or `None` to fall back to plain whitespace.
    pub fn for_lang(&self, lang: Option<&str>) -> Option<&dyn Proposer> {
        match lang {
            Some("jpn") if self.unidic.is_some() => {
                self.unidic.as_ref().map(|d| d as &dyn Proposer)
            }
            Some(l) => self.banks.get(l).map(|b| b as &dyn Proposer),
            None => None,
        }
    }

    /// Per-byte prior ids for `text`, ready to hand to the model.
    pub fn ids(&self, text: &str, lang: Option<&str>, max_bytes: usize) -> Vec<u8> {
        prior_ids(text, self.for_lang(lang), max_bytes)
    }
}

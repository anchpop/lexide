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
//! job becomes mostly deleting boundaries rather than finding them. See [`WHY_RECALL`] for
//! why that direction is the safe one to err in, and why it is not the tautology it looks
//! like.
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
/// There was briefly a fifth symbol, `B_SOFT`, separating a boundary whitespace guarantees
/// from one a dictionary merely proposes. It measured flat (91.80 against 92.24), and
/// counting showed why: with no wordbanks shipped it appeared only in Japanese, where a
/// sentence is a single whitespace-free run — the first token `B`, every other `B_SOFT`,
/// 1.07 `B` per sentence. It encoded "not sentence-initial", and across languages "is
/// Japanese", which the language token already carries. A language with both whitespace and
/// a dictionary would give it real content; that is the configuration the same measurement
/// rejects.
/// Why a proposal is judged on recall far more than on precision.
///
/// This was long stated as "the model deletes boundaries but cannot invent them", which is
/// false as written: the network is a per-byte O/B/I classifier and the prior is just
/// another input feature, so nothing stops it emitting `B` where the proposal says `I`.
/// Two true things stand behind the slogan, and only the second is fundamental.
///
/// 1. **It learns to defer.** Measured on the v11 weights, trained with the prior available
///    from step 0: Japanese scores 93.3 with its dictionary and **21.2** with a whitespace
///    proposal. That is a property of the training regime, not of the architecture, and
///    `--prior-warmup-frac` / `--prior-dropout` exist to keep an independent route alive.
/// 2. **Deleting and inventing differ in difficulty.** An over-proposed boundary is a
///    locally marked, learnable correction — the proposal says `B`, the gold says `I`, and
///    the bytes carry the evidence. An under-proposed one asks the model to know that
///    苛立ち is a word from the bytes alone, which is exactly the lexical knowledge the
///    prior was added to supply. A hint that omits boundaries therefore contributes nothing
///    precisely where it fails, while a hint that adds spurious ones stays useful.
///
/// Measured consequence (2026-08-06): jieba and the Thai National Corpus list *under*-propose
/// at 0.83–0.90x the gold token count and lose to a corpus bank by 19–27 points of
/// out-of-domain boundary recall, while UniDic's 84.4% precision costs Japanese nothing.
pub const WHY_RECALL: &str =
    "a proposal may over-propose freely; omitting a boundary is the costly direction";

pub const PRIOR_NONE: u8 = 0;
pub const PRIOR_O: u8 = 1;
pub const PRIOR_B: u8 = 2;
pub const PRIOR_I: u8 = 3;
pub const PRIOR_VOCAB: usize = 4;

/// Character classes, following MeCab's `char.def`: a run of one script is usually one
/// word. The limit is how long an *unknown* run of that class may be grouped into a single
/// proposal — katakana loanwords and numbers run long, kanji compounds are usually 2-3, and
/// a run of hiragana is grammar rather than one word so it is never grouped.
///
/// Thai is its own class rather than `Other`, whose limit of 1 would shatter an unknown run
/// into single characters — and Thai, having no whitespace, is one long run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CharType {
    Katakana,
    Hiragana,
    Kanji,
    Thai,
    Digit,
    Latin,
    Other,
}

impl CharType {
    /// Every variant, in discriminant order. `segment::unidic` indexes an array by
    /// `CharType as usize`, so that array has to be built from this — inserting a variant
    /// mid-enum silently shifts every later discriminant, which is how adding `Thai` first
    /// pushed `Other` off the end of a hand-written six-element table.
    pub const ALL: [CharType; 7] = [
        CharType::Katakana,
        CharType::Hiragana,
        CharType::Kanji,
        CharType::Thai,
        CharType::Digit,
        CharType::Latin,
        CharType::Other,
    ];

    pub fn of(ch: char) -> Self {
        let o = ch as u32;
        if (0x30A0..=0x30FF).contains(&o) || o == 0x30FC {
            CharType::Katakana
        } else if (0x3040..=0x309F).contains(&o) {
            CharType::Hiragana
        } else if (0x4E00..=0x9FFF).contains(&o) || (0x3400..=0x4DBF).contains(&o) {
            CharType::Kanji
        } else if (0x0E00..=0x0E7F).contains(&o) {
            CharType::Thai
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
            CharType::Thai => 8,
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
    /// Optional first line of a bank file, e.g. `#!group_unknown=0`. Whether unknown runs
    /// group or shatter is a property *of the bank*, not of the caller: it was measured
    /// per language (Chinese wants shattering — 97.05 boundary recall against 90.33
    /// grouped — while Thai is flat either way), and one global default cannot hold two
    /// answers. Keeping it in the file means a bank and its setting cannot be separated in
    /// a release. Must stay identical to `Wordbank.HEADER` in tagger/prior.py.
    pub const HEADER: &'static str = "#!group_unknown=";

    /// `group_unknown` is the default; a `#!group_unknown=` header in the file overrides it.
    pub fn load(path: &Path, group_unknown: bool) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading wordbank {}", path.display()))?;
        let mut group_unknown = group_unknown;
        let mut counts = HashMap::new();
        for line in text.lines() {
            if let Some(v) = line.strip_prefix(Self::HEADER) {
                group_unknown = !matches!(v.trim(), "0" | "false");
                continue;
            }
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

/// B on each proposed token start, I inside it, O on characters no token covers.
fn proposer_char_labels(p: &dyn Proposer, chars: &[char]) -> Vec<u8> {
    let mut out = vec![PRIOR_O; chars.len()];
    for (a, b) in segment_constrained(p, chars) {
        if chars[a..b].iter().any(|c| !c.is_whitespace()) {
            out[a] = PRIOR_B;
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
            let cont = if lab == PRIOR_B || lab == PRIOR_I {
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
                    PRIOR_B => {
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

    /// `segment::unidic` builds a per-CharType array indexed by `t as usize`, so ALL has
    /// to list every variant in discriminant order. Inserting `Thai` before `Digit` once
    /// shifted `Other` to 6 against a hand-written 6-element table and panicked on the
    /// first Japanese sentence with a punctuation mark in it.
    #[test]
    fn char_type_all_is_every_variant_in_discriminant_order() {
        for (i, &t) in CharType::ALL.iter().enumerate() {
            assert_eq!(t as usize, i, "{t:?} is not at index {i}");
        }
        // every char class `of` can return must be in ALL
        for ch in ['ア', 'あ', '漢', 'ก', '7', 'x', '。'] {
            assert!(CharType::ALL.contains(&CharType::of(ch)), "{ch} missing from ALL");
        }
    }

    #[test]
    fn wordbank_header_overrides_the_caller_default() {
        let dir = std::env::temp_dir().join("lexide_wb_header_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zho-hans.tsv");
        std::fs::write(&path, format!("{}0\n猫\t10\n", Wordbank::HEADER)).unwrap();
        // caller says group, the file says shatter — the file wins, because grouping was
        // measured per language and travels with the bank
        let wb = Wordbank::load(&path, true).unwrap();
        assert!(!wb.group_unknown);
        // and a bank with no header keeps the caller's default
        let plain = dir.join("plain.tsv");
        std::fs::write(&plain, "猫\t10\n").unwrap();
        assert!(Wordbank::load(&plain, true).unwrap().group_unknown);
        std::fs::remove_dir_all(&dir).ok();
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

/// Language for the *prior* when the caller supplied none.
///
/// Training always hands the prior the true language, even for the 15% of examples whose
/// language *token* is dropped — so "Japanese text with a whitespace proposal" is a
/// combination the model has never seen. A caller may pass no language at all, and
/// whitespace on Japanese does not merely say nothing: it asserts that the whole sentence
/// is one word, and the model, having learned to lean on the proposal, obeys. Measured on
/// the v11 weights that collapses a Japanese sentence to a single token.
///
/// Script recovers what the prior needs, so this restores the training distribution rather
/// than papering over it. Must stay identical to `infer_lang` in tagger/prior.py.
///
/// Only the three whitespace-free languages are guessed, because they are the ones where
/// guessing wrong is catastrophic rather than merely unhelpful. Kana is decisive for
/// Japanese and Thai script for Thai; Han with no kana reads as Chinese, which is the
/// common case — Japanese prose without a single kana is rare, and the alternative hands
/// every Chinese sentence to a Japanese dictionary.
pub fn infer_lang(text: &str) -> Option<&'static str> {
    let mut seen_han = false;
    for c in text.chars() {
        match CharType::of(c) {
            CharType::Hiragana | CharType::Katakana => return Some("jpn"),
            CharType::Thai => return Some("tha"),
            CharType::Kanji => seen_han = true,
            _ => {}
        }
    }
    seen_han.then_some("zho-hans")
}

/// Languages that do not delimit words with whitespace. A whitespace proposal for these is
/// not a weak answer but a wrong one — it asserts the sentence is a single word, and a
/// model trained to lean on the proposal obeys (measured on jpn: 21.2 F1 against 93.3). So
/// with no lexicon available they get `PRIOR_NONE`, which says nothing, and which the model
/// is trained for (`--prior-dropout`).
pub const NO_WHITESPACE_LANGS: [&str; 3] = ["jpn", "tha", "zho-hans"];

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

    /// Install the Japanese dictionary after construction — for callers that fetch it
    /// lazily rather than finding it on disk, such as the wasm demo, where 87MB is a
    /// deliberate opt-in and only Japanese benefits.
    pub fn set_unidic(&mut self, dict: UniDic) {
        self.unidic = Some(dict);
    }

    /// Whether a Japanese proposal can be produced at all.
    pub fn has_japanese(&self) -> bool {
        self.unidic.is_some() || self.banks.contains_key("jpn")
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

    /// Per-byte prior ids for `text`, ready to hand to the model. A caller who supplies no
    /// language gets one inferred from the script — see [`infer_lang`].
    ///
    /// With no lexicon loaded, a whitespace-free language gets an all-`NONE` proposal
    /// rather than the whitespace one — see [`NO_WHITESPACE_LANGS`]. This is how a build
    /// with no room for the 87MB artifact, such as the wasm demo, degrades.
    pub fn ids(&self, text: &str, lang: Option<&str>, max_bytes: usize) -> Vec<u8> {
        let lang = lang.or_else(|| infer_lang(text));
        let proposer = self.for_lang(lang);
        if proposer.is_none() && lang.is_some_and(|l| NO_WHITESPACE_LANGS.contains(&l)) {
            let n = (text.len() + 2).min(max_bytes);
            return vec![PRIOR_NONE; n];
        }
        prior_ids(text, proposer, max_bytes)
    }
}

#[cfg(test)]
mod infer_lang_tests {
    use super::*;

    #[test]
    fn script_recovers_the_whitespace_free_languages() {
        // kana is decisive for Japanese, wherever in the string it appears
        assert_eq!(infer_lang("私は猫が好きです。"), Some("jpn"));
        assert_eq!(infer_lang("ブロックチェーン"), Some("jpn"));
        assert_eq!(infer_lang("漢字だけ"), Some("jpn"));
        // Han with no kana reads as Chinese — the common case, and the safe one
        assert_eq!(infer_lang("我喜欢猫。"), Some("zho-hans"));
        assert_eq!(infer_lang("日本語"), Some("zho-hans"));
        assert_eq!(infer_lang("ผมชอบแมว"), Some("tha"));
        // a spaced language claims nothing: whitespace is already the right proposal
        assert_eq!(infer_lang("Eine Fundgrube."), None);
        assert_eq!(infer_lang("고양이가 좋아요."), None);
        assert_eq!(infer_lang("мама"), None);
    }
}

#[cfg(test)]
mod priorset_tests {
    use super::*;

    #[test]
    fn a_whitespace_free_language_without_a_lexicon_says_nothing_rather_than_something_false() {
        let empty = PriorSet::default();
        // whitespace would mark the run as one token — an assertion, not an absence
        for (text, lang) in [
            ("これはペンです", "jpn"),
            ("我喜欢猫", "zho-hans"),
            ("ผมชอบแมว", "tha"),
        ] {
            let ids = empty.ids(text, Some(lang), 512);
            assert!(ids.iter().all(|&p| p == PRIOR_NONE), "{lang}: got {ids:?}");
            // inferred from script too, with no language supplied
            assert!(empty.ids(text, None, 512).iter().all(|&p| p == PRIOR_NONE), "{lang} inferred");
        }
        // spaced languages still get their exact, free proposal
        let deu = empty.ids("Ein Haus", Some("deu"), 512);
        assert!(deu.contains(&PRIOR_B) && deu.contains(&PRIOR_O), "got {deu:?}");
    }
}

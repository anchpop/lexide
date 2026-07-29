//! A MeCab-compatible Viterbi over the bundled UniDic artifact.
//!
//! The corpus wordbank in `prior.rs` gets Japanese most of the way there from ~0.4MB, but a
//! real dictionary measured better where it counts: UniDic proposes a boundary at 99.4% of
//! gold token starts against the bank's 98.8%, worth about 1.1 F1. So for Japanese we ship
//! the dictionary.
//!
//! MeCab's cost is `sum(word cost) + sum(transition cost)`, with the transition looked up
//! as `matrix[prev.right_id + cur.left_id * lsize]` (connector.cpp). That means the entry
//! that wins for a surface is not knowable until the search runs — the same surface has
//! several entries with different context ids — so the search carries a frontier keyed by
//! right-context id rather than a single best cost per position.
//!
//! Built by `tagger/build_unidic_artifact.py`; verified at 100% span agreement with fugashi
//! on the Japanese test split, which is what let us drop the analyzer dependency.

use std::path::Path;

use anyhow::{bail, Context, Result};

use super::prior::{CharType, Proposer};

const MAGIC: &[u8; 8] = b"LXUNIDIC";
const ENTRY_BYTES: usize = 6; // i16 cost, u16 left_id, u16 right_id

#[derive(Clone, Copy, Debug)]
struct Entry {
    cost: i16,
    left: u16,
    right: u16,
}

struct Category {
    group_len: usize,
    first: usize, // index into `unk_entries`
    count: usize,
}

pub struct UniDic {
    raw: Vec<u8>,
    lsize: usize,
    max_chars: usize,
    n_surf: usize,
    /// byte offsets into `raw` of each section
    surf_off: usize,
    ent_off: usize,
    blob: usize,
    entries: usize,
    matrix: usize,
    unk_entries: Vec<Entry>,
    /// parallel to `CharType`'s discriminants, plus a DEFAULT fallback
    cats: Vec<Option<Category>>,
    default_cat: Option<Category>,
}

fn u32_at(raw: &[u8], off: usize) -> usize {
    u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as usize
}

fn entry_at(raw: &[u8], off: usize) -> Entry {
    Entry {
        cost: i16::from_le_bytes(raw[off..off + 2].try_into().unwrap()),
        left: u16::from_le_bytes(raw[off + 2..off + 4].try_into().unwrap()),
        right: u16::from_le_bytes(raw[off + 4..off + 6].try_into().unwrap()),
    }
}

/// The unk.def category each character class is analysed as.
fn category_name(t: CharType) -> &'static str {
    match t {
        CharType::Katakana => "KATAKANA",
        CharType::Hiragana => "HIRAGANA",
        CharType::Kanji => "KANJI",
        CharType::Digit => "NUMERIC",
        CharType::Latin => "ALPHA",
        CharType::Other => "SYMBOL",
    }
}

impl UniDic {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path)
            .with_context(|| format!("reading unidic artifact {}", path.display()))?;
        Self::from_bytes(raw)
    }

    pub fn from_bytes(raw: Vec<u8>) -> Result<Self> {
        if raw.len() < 40 || &raw[..8] != MAGIC {
            bail!("not a lexide unidic artifact (bad magic)");
        }
        let version = u32_at(&raw, 8);
        if version != 1 {
            bail!("unidic artifact version {version}, expected 1");
        }
        let lsize = u32_at(&raw, 12);
        let rsize = u32_at(&raw, 16);
        let max_chars = u32_at(&raw, 20);
        let n_surf = u32_at(&raw, 24);
        let n_ent = u32_at(&raw, 28);
        let blob_len = u32_at(&raw, 32);
        let n_cats = u32_at(&raw, 36);

        let surf_off = 40;
        let ent_off = surf_off + (n_surf + 1) * 4;
        let blob = ent_off + (n_surf + 1) * 4;
        let entries = blob + blob_len;
        let cats_at = entries + n_ent * ENTRY_BYTES;

        // unknown-word entries, keyed by character category
        let mut unk_entries = Vec::new();
        let mut by_name: Vec<(String, Category)> = Vec::new();
        let mut p = cats_at;
        for _ in 0..n_cats {
            let name_len = raw[p] as usize;
            p += 1;
            let name = String::from_utf8_lossy(&raw[p..p + name_len]).into_owned();
            p += name_len;
            let group_len = u16::from_le_bytes(raw[p..p + 2].try_into().unwrap()) as usize;
            p += 2;
            let count = u32_at(&raw, p);
            p += 4;
            let first = unk_entries.len();
            for k in 0..count {
                unk_entries.push(entry_at(&raw, p + k * ENTRY_BYTES));
            }
            p += count * ENTRY_BYTES;
            by_name.push((name, Category { group_len, first, count }));
        }
        let matrix = p;
        if raw.len() < matrix + lsize * rsize * 2 {
            bail!("unidic artifact truncated: matrix does not fit");
        }

        let take = |want: &str| -> Option<Category> {
            by_name
                .iter()
                .find(|(n, _)| n == want)
                .map(|(_, c)| Category { group_len: c.group_len, first: c.first, count: c.count })
        };
        let cats = [
            CharType::Katakana,
            CharType::Hiragana,
            CharType::Kanji,
            CharType::Digit,
            CharType::Latin,
            CharType::Other,
        ]
        .iter()
        .map(|&t| take(category_name(t)))
        .collect();

        Ok(Self {
            raw,
            lsize,
            max_chars,
            n_surf,
            surf_off,
            ent_off,
            blob,
            entries,
            matrix,
            unk_entries,
            cats,
            default_cat: take("DEFAULT"),
        })
    }

    fn surface(&self, i: usize) -> &[u8] {
        let a = self.blob + u32_at(&self.raw, self.surf_off + i * 4);
        let b = self.blob + u32_at(&self.raw, self.surf_off + (i + 1) * 4);
        &self.raw[a..b]
    }

    /// Index of `cand` in the byte-sorted surface array.
    fn find(&self, cand: &[u8]) -> Option<usize> {
        let (mut lo, mut hi) = (0usize, self.n_surf);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.surface(mid) < cand {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo < self.n_surf && self.surface(lo) == cand).then_some(lo)
    }

    fn lexicon_entries(&self, i: usize, out: &mut Vec<Entry>) {
        let a = u32_at(&self.raw, self.ent_off + i * 4);
        let b = u32_at(&self.raw, self.ent_off + (i + 1) * 4);
        for k in a..b {
            out.push(entry_at(&self.raw, self.entries + k * ENTRY_BYTES));
        }
    }

    fn category(&self, t: CharType) -> Option<&Category> {
        self.cats[t as usize].as_ref().or(self.default_cat.as_ref())
    }

    /// `matrix[prev_right + left * lsize]`, exactly as mecab indexes it.
    fn transition(&self, prev_right: u16, left: u16) -> i32 {
        let idx = prev_right as usize + left as usize * self.lsize;
        let off = self.matrix + idx * 2;
        i16::from_le_bytes(self.raw[off..off + 2].try_into().unwrap()) as i32
    }
}

/// One surviving path into a position, keyed by the right-context id it ends with.
#[derive(Clone, Copy)]
struct Node {
    right: u16,
    cost: i32,
    back_pos: usize,
    back_right: u16,
}

impl Proposer for UniDic {
    fn segment_run(&self, chars: &[char]) -> Vec<(usize, usize)> {
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        let types: Vec<CharType> = chars.iter().map(|&c| CharType::of(c)).collect();

        // byte offsets so a candidate span can be sliced without re-encoding
        let mut text = String::with_capacity(n * 3);
        let mut starts = Vec::with_capacity(n + 1);
        for &c in chars {
            starts.push(text.len());
            text.push(c);
        }
        starts.push(text.len());
        let bytes = text.as_bytes();

        // BOS is context id 0
        let mut best: Vec<Vec<Node>> = vec![Vec::new(); n + 1];
        best[0].push(Node { right: 0, cost: 0, back_pos: usize::MAX, back_right: 0 });

        let mut cands: Vec<Entry> = Vec::new();
        for i in 1..=n {
            for j in i.saturating_sub(self.max_chars)..i {
                if best[j].is_empty() {
                    continue;
                }
                cands.clear();
                match self.find(&bytes[starts[j]..starts[i]]) {
                    Some(k) => self.lexicon_entries(k, &mut cands),
                    None => {
                        // unknown word: only as one run of a single character class, and
                        // only up to that class's grouping length (char.def's idea — a run
                        // of one script is usually one word)
                        let t = types[j];
                        let Some(cat) = self.category(t) else { continue };
                        if i - j > cat.group_len || !types[j..i].iter().all(|&x| x == t) {
                            continue;
                        }
                        cands.extend_from_slice(
                            &self.unk_entries[cat.first..cat.first + cat.count],
                        );
                    }
                }

                for e in &cands {
                    // cheapest way in, over every context the previous position ended with
                    let mut pick: Option<(i32, u16)> = None;
                    for p in &best[j] {
                        let c = p.cost + self.transition(p.right, e.left) + e.cost as i32;
                        if pick.is_none_or(|(bc, _)| c < bc) {
                            pick = Some((c, p.right));
                        }
                    }
                    let Some((cost, back_right)) = pick else { continue };
                    match best[i].iter_mut().find(|m| m.right == e.right) {
                        Some(m) if cost < m.cost => {
                            m.cost = cost;
                            m.back_pos = j;
                            m.back_right = back_right;
                        }
                        Some(_) => {}
                        None => best[i].push(Node {
                            right: e.right,
                            cost,
                            back_pos: j,
                            back_right,
                        }),
                    }
                }
            }
        }

        if best[n].is_empty() {
            return vec![(0, n)];
        }
        // close to EOS, which is also context id 0
        let end = best[n]
            .iter()
            .min_by_key(|m| m.cost + self.transition(m.right, 0))
            .unwrap();

        let mut spans = Vec::new();
        let (mut i, mut right) = (n, end.right);
        while i > 0 {
            let node = best[i].iter().find(|m| m.right == right).copied();
            let Some(node) = node else { break };
            spans.push((node.back_pos, i));
            i = node.back_pos;
            right = node.back_right;
        }
        spans.reverse();
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::prior::segment_constrained;
    use crate::segment::test_support::model_file;

    fn seg(dict: &UniDic, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        segment_constrained(dict, &chars)
            .into_iter()
            .map(|(a, b)| chars[a..b].iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("|")
    }

    /// The artifact is ~83MB and lives with the model, not in the repo; skip without it.
    #[test]
    fn reproduces_mecab_segmentation() {
        let Some(path) = model_file("jpn-unidic.bin") else {
            eprintln!("skipping: jpn-unidic.bin not found (set LEXIDE_MODEL_DIR)");
            return;
        };
        let dict = UniDic::load(&path).unwrap();

        // Plain sentence: the dictionary splits predicate from auxiliary, which is UniDic's
        // policy rather than ours — the model learns the mapping, the prior just has to
        // know these are words at all.
        assert_eq!(seg(&dict, "これはペンです"), "これ|は|ペン|です");
        // A known compound splits into its known parts — UniDic has both halves.
        assert_eq!(seg(&dict, "ブロックチェーンが好き"), "ブロック|チェーン|が|好き");
        // A genuinely out-of-dictionary katakana loanword stays whole instead of
        // shattering, because char.def groups a run of one script. This is the case a
        // per-character unknown cost gets wrong, and unseen words are where Japanese
        // errors actually live.
        assert_eq!(seg(&dict, "ヴォンゴレビアンコが食べたい"), "ヴォンゴレビアンコ|が|食べ|たい");
        // Whitespace stays a hard boundary even here.
        assert_eq!(seg(&dict, "本 です"), "本|です");
    }
}

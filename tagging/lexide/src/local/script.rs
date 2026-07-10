//! Lemma edit scripts — the inverse of `lemma_script` in `tagger/data_prep.py`.
//!
//! A script `"p|s|ins"` means: keep the first `p` chars and last `s` chars of the surface
//! form, replace the middle with `ins`. The sentinel `COPY` (and the legacy `"0|0|"`)
//! means the lemma is the form unchanged. Indices are chars (code points), not bytes.

pub const COPY_SCRIPT: &str = "COPY";
const LEGACY_COPY: &str = "0|0|";

/// Apply an edit script to a surface form, mirroring `data_prep.apply_script` exactly:
/// a script whose prefix+suffix exceeds the form's length falls back to copying the form.
pub fn apply_script(form: &str, script: &str) -> String {
    if script == COPY_SCRIPT || script == LEGACY_COPY {
        return form.to_string();
    }
    let mut parts = script.splitn(3, '|');
    let (Some(p), Some(s), Some(ins)) = (parts.next(), parts.next(), parts.next()) else {
        return form.to_string();
    };
    let (Ok(p), Ok(s)) = (p.parse::<usize>(), s.parse::<usize>()) else {
        return form.to_string();
    };
    let chars: Vec<char> = form.chars().collect();
    if p + s > chars.len() {
        return form.to_string(); // malformed for this form; fall back to copy
    }
    let mut out: String = chars[..p].iter().collect();
    out.push_str(ins);
    out.extend(&chars[chars.len() - s..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_variants() {
        assert_eq!(apply_script("Hunde", COPY_SCRIPT), "Hunde");
        assert_eq!(apply_script("Hunde", "0|0|"), "Hunde");
    }

    #[test]
    fn suffix_replacement() {
        // cats -> cat: keep 3 chars, drop the rest, insert nothing
        assert_eq!(apply_script("cats", "3|0|"), "cat");
        // sleeping -> sleep
        assert_eq!(apply_script("sleeping", "5|0|"), "sleep");
    }

    #[test]
    fn infix_and_case_edits() {
        // Was -> sein style full replacement
        assert_eq!(apply_script("war", "0|0|sein"), "sein");
        // lowercase first char: Die -> die (keep 0 prefix, keep 2 suffix, insert "d")
        assert_eq!(apply_script("Die", "0|2|d"), "die");
    }

    #[test]
    fn char_not_byte_indices() {
        // Cyrillic: доверяю -> доверять (keep 6 chars "довер я"... indices are chars)
        assert_eq!(apply_script("доверяю", "6|0|ть"), "доверять");
        // CJK
        assert_eq!(apply_script("好きです", "2|0|"), "好き");
    }

    #[test]
    fn malformed_scripts_fall_back_to_copy() {
        assert_eq!(apply_script("ab", "5|5|x"), "ab"); // p+s > len
        assert_eq!(apply_script("ab", "junk"), "ab");
        assert_eq!(apply_script("ab", "a|b|c"), "ab");
    }

    #[test]
    fn insert_may_contain_pipe() {
        // splitn(3) keeps any further '|' inside the insertion text
        assert_eq!(apply_script("xy", "1|0|a|b"), "xa|b");
    }
}

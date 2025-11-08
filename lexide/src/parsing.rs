/// Shared parsing logic for both local and remote inference
use crate::{Lemma, Text, Token, Tokenization};
use anyhow::{Context, Result};

/// Create the prompt in the expected format
pub fn create_prompt(sentence: &str, language: crate::Language) -> String {
    format!(
        "Language: {}\nSentence: {}\nTask: Analyze tokens (idx,token,ws,POS,lemma,dep,head)\n\nAnalysis:\n",
        language, sentence
    )
}

/// Parse the model's response into structured data
pub fn parse_response(response: &str, sentence: &str) -> Result<Tokenization> {
    let mut tokens = Vec::new();

    // Skip the conversational prefix if present
    let response = if response.contains("Here's the token analysis:") {
        response
            .split("Here's the token analysis:")
            .last()
            .unwrap_or(response)
    } else {
        response
    };

    // Remove the </analysis> end marker if present
    let response = if response.contains("</analysis>") {
        response.split("</analysis>").next().unwrap_or(response)
    } else {
        response
    };

    let response = response.trim();

    for line in response.lines() {
        let line = line.trim();
        // Skip empty lines and sentence separators
        if line.is_empty() || line == "-----" {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 7 {
            let text = Text {
                text: parts[1].to_string(),
            };
            // Decode whitespace representation
            let whitespace = match parts[2] {
                "_" => " ".to_string(),
                "nbsp" => "\u{00A0}".to_string(),
                "narnbsp" => "\u{202F}".to_string(),
                "\u{202F} " => "\u{202F}".to_string(), // TODO: remove this once model is updated
                "  " => "\u{202F}".to_string(), // TODO: remove this once model is updated
                "\u{a0} " => "\u{202F}".to_string(), // TODO: remove this once model is updated
                "thinsp" => "\u{2009}".to_string(),
                "hairsp" => "\u{200A}".to_string(),
                "zwsp" => "\u{200B}".to_string(),
                "ideogrp" => "\u{3000}".to_string(),
                "none" => "".to_string(),
                other => other.to_string(), // Pass through unknown values
            };
            let pos = parts[3].to_string();
            let lemma = Lemma {
                lemma: parts[4].to_string(),
            };
            let dep = parts[5].to_string();
            let head = parts[6].parse::<i32>().unwrap_or(-1);

            let pos = serde_plain::from_str(&pos)
                .context(format!("Failed to parse POS in line: {}", line))?;
            let dep = serde_plain::from_str(&dep)
                .context(format!("Failed to parse dependency in line: {}", line))?;

            tokens.push(Token {
                text,
                whitespace,
                pos,
                lemma,
                dep,
                head,
            });
        }
    }

    // Reverse the tokenization-aware transformation:
    // Move leading spaces from token text to previous token's whitespace field
    for i in 0..tokens.len() {
        if tokens[i].text.text.starts_with(' ') {
            // Strip the leading space from current token
            tokens[i].text.text = tokens[i].text.text[1..].to_string();

            // Add it to the previous token's whitespace (if there is a previous token)
            if i > 0 {
                tokens[i - 1].whitespace = " ".to_string();
            }
        }
    }

    // Attempt to fix reconstruction issues
    if !fix_reconstruction(&mut tokens, sentence) {
        let reconstructed_text = tokens
            .iter()
            .map(|token| format!("{}{}", token.text.text, token.whitespace))
            .collect::<String>();
        anyhow::bail!(
            "Reconstructed text does not match the original sentence ({} != {}) in response: {:?}",
            reconstructed_text,
            sentence,
            response
        );
    }

    Ok(Tokenization { tokens })
}

/// Normalize Unicode characters for comparison (remove accents, normalize punctuation)
fn normalize_unicode(s: &str) -> String {
    // Use NFD (Canonical Decomposition) to separate base characters from combining marks
    use unicode_normalization::UnicodeNormalization;

    s.nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .collect::<String>()
        // Normalize various Unicode punctuation and whitespace characters
        .replace('\u{2019}', "'") // Right single quotation mark
        .replace('\u{2018}', "'") // Left single quotation mark
        .replace('\u{201C}', "\"") // Left double quotation mark
        .replace('\u{201D}', "\"") // Right double quotation mark
        .replace('\u{2013}', "-") // En dash
        .replace('\u{2014}', "-") // Em dash
        .replace('\u{2026}', "...") // Horizontal ellipsis
        .replace('\u{00A0}', " ") // Non-breaking space
        .replace('\u{202F}', " ") // Narrow non-breaking space
        .replace('\u{2009}', " ") // Thin space
        .replace('\u{200A}', " ") // Hair space
        .replace('\u{200B}', "") // Zero-width space
        .replace('\u{3000}', " ") // Ideographic space
}

/// Attempt to fix reconstruction mismatches
fn fix_reconstruction(tokens: &mut Vec<Token>, sentence: &str) -> bool {
    let reconstructed: String = tokens
        .iter()
        .map(|token| format!("{}{}", token.text.text, token.whitespace))
        .collect();

    if reconstructed == sentence {
        return true;
    }

    // Fix 1: Remove extra period at the end
    if tokens.len() > 1 {
        let last_idx = tokens.len() - 1;

        // Sometimes the model will add a punctuation mark at the end of the sentence that were not in the input.
        let reconstructed_without_last_token: String = tokens
            .iter()
            .take(tokens.len() - 1)
            .map(|token| format!("{}{}", token.text.text, token.whitespace))
            .collect();

        if reconstructed_without_last_token == sentence {
            tokens.remove(last_idx);
            return true;
        }
        if reconstructed_without_last_token.trim_end_matches(&[' ', '\u{202F}']) == sentence {
            tokens.remove(last_idx);
            tokens[last_idx - 1].whitespace = "".to_string();
            return true;
        }
    }

    // Fix 2: Normalize accents and special Unicode characters
    let normalized_reconstructed = normalize_unicode(&reconstructed);
    let normalized_sentence = normalize_unicode(sentence);

    if normalized_reconstructed == normalized_sentence {
        // The texts match after normalization, so we need to adjust tokens to match the original input
        let sentence_chars: Vec<char> = sentence.chars().collect();
        let mut sentence_pos = 0;

        for token in tokens.iter_mut() {
            // Match token text
            let token_len = token.text.text.chars().count();
            let original_text: String = sentence_chars
                .iter()
                .skip(sentence_pos)
                .take(token_len)
                .collect();

            if !original_text.is_empty() {
                token.text.text = original_text;
                sentence_pos += token_len;
            }

            // Match whitespace
            let ws_len = token.whitespace.chars().count();
            let original_ws: String = sentence_chars
                .iter()
                .skip(sentence_pos)
                .take(ws_len)
                .collect();

            token.whitespace = original_ws;
            sentence_pos += ws_len;
        }

        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dep::DependencyRelation, pos::PartOfSpeech};

    /// Helper function to create a test token
    fn create_test_token(
        text: &str,
        whitespace: &str,
        pos: PartOfSpeech,
        lemma: &str,
        dep: DependencyRelation,
        head: i32,
    ) -> Token {
        Token {
            text: Text {
                text: text.to_string(),
            },
            whitespace: whitespace.to_string(),
            pos,
            lemma: Lemma {
                lemma: lemma.to_string(),
            },
            dep,
            head,
        }
    }

    #[test]
    fn test_normalize_unicode_removes_accents() {
        assert_eq!(normalize_unicode("café"), "cafe");
        assert_eq!(normalize_unicode("naïve"), "naive");
        assert_eq!(normalize_unicode("résumé"), "resume");
        assert_eq!(normalize_unicode("Zürich"), "Zurich");
        assert_eq!(normalize_unicode("crème brûlée"), "creme brulee");
    }

    #[test]
    fn test_normalize_unicode_normalizes_punctuation() {
        // Smart quotes
        assert_eq!(normalize_unicode("don\u{2019}t"), "don't");
        assert_eq!(normalize_unicode("\u{2018}hello\u{2019}"), "'hello'");
        assert_eq!(normalize_unicode("\u{201C}hello\u{201D}"), "\"hello\"");

        // Dashes
        assert_eq!(normalize_unicode("hello–world"), "hello-world"); // en dash
        assert_eq!(normalize_unicode("hello—world"), "hello-world"); // em dash

        // Ellipsis
        assert_eq!(normalize_unicode("hello…"), "hello...");
    }

    #[test]
    fn test_normalize_unicode_normalizes_whitespace() {
        // Non-breaking space
        assert_eq!(normalize_unicode("hello\u{00A0}world"), "hello world");
        // Narrow non-breaking space
        assert_eq!(normalize_unicode("hello\u{202F}world"), "hello world");
        // Thin space
        assert_eq!(normalize_unicode("hello\u{2009}world"), "hello world");
        // Zero-width space
        assert_eq!(normalize_unicode("hello\u{200B}world"), "helloworld");
    }

    #[test]
    fn test_fix_reconstruction_exact_match() {
        let mut tokens = vec![
            create_test_token(
                "Hello",
                " ",
                PartOfSpeech::Noun,
                "hello",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                "world",
                "",
                PartOfSpeech::Noun,
                "world",
                DependencyRelation::Nmod,
                1,
            ),
        ];

        assert!(fix_reconstruction(&mut tokens, "Hello world"));
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_fix_reconstruction_removes_extra_punctuation() {
        let mut tokens = vec![
            create_test_token(
                "Hello",
                " ",
                PartOfSpeech::Noun,
                "hello",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                "world",
                "",
                PartOfSpeech::Noun,
                "world",
                DependencyRelation::Nmod,
                1,
            ),
            create_test_token(
                ".",
                "",
                PartOfSpeech::Punct,
                ".",
                DependencyRelation::Punct,
                1,
            ),
        ];

        // Should remove the extra period
        assert!(fix_reconstruction(&mut tokens, "Hello world"));
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[1].text.text, "world");
    }

    #[test]
    fn test_fix_reconstruction_removes_trailing_whitespace_with_punctuation() {
        let mut tokens = vec![
            create_test_token(
                "Hello",
                " ",
                PartOfSpeech::Noun,
                "hello",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                "world",
                " ",
                PartOfSpeech::Noun,
                "world",
                DependencyRelation::Nmod,
                1,
            ),
            create_test_token(
                ".",
                "",
                PartOfSpeech::Punct,
                ".",
                DependencyRelation::Punct,
                1,
            ),
        ];

        // Should remove the extra period and trailing whitespace
        assert!(fix_reconstruction(&mut tokens, "Hello world"));
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[1].whitespace, "");
    }

    #[test]
    fn test_fix_reconstruction_with_accents() {
        let mut tokens = vec![
            create_test_token(
                "Le",
                " ",
                PartOfSpeech::Det,
                "le",
                DependencyRelation::Det,
                2,
            ),
            create_test_token(
                "cafe",
                " ",
                PartOfSpeech::Noun,
                "cafe",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                "est",
                " ",
                PartOfSpeech::Aux,
                "être",
                DependencyRelation::Cop,
                4,
            ),
            create_test_token(
                "ouvert",
                "",
                PartOfSpeech::Adj,
                "ouvert",
                DependencyRelation::Root,
                0,
            ),
        ];

        // Model returned "Le cafe est ouvert" but input was "Le café est ouvert"
        assert!(fix_reconstruction(&mut tokens, "Le café est ouvert"));

        // Verify the accent was restored
        assert_eq!(tokens[1].text.text, "café");
    }

    #[test]
    fn test_fix_reconstruction_with_smart_quotes() {
        let mut tokens = vec![
            create_test_token(
                "It's",
                " ",
                PartOfSpeech::Pron,
                "it",
                DependencyRelation::Nsubj,
                2,
            ),
            create_test_token(
                "great",
                "",
                PartOfSpeech::Adj,
                "great",
                DependencyRelation::Root,
                0,
            ),
        ];

        // Model returned "It's great" but input was "It's great" (smart apostrophe)
        assert!(fix_reconstruction(&mut tokens, "It\u{2019}s great"));

        // Verify the smart quote was restored
        assert_eq!(tokens[0].text.text, "It\u{2019}s");
    }

    #[test]
    fn test_fix_reconstruction_with_special_whitespace() {
        let mut tokens = vec![
            create_test_token(
                "Hello",
                " ",
                PartOfSpeech::Noun,
                "hello",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                "world",
                "",
                PartOfSpeech::Noun,
                "world",
                DependencyRelation::Nmod,
                1,
            ),
        ];

        // Model returned "Hello world" but input had a narrow non-breaking space
        assert!(fix_reconstruction(&mut tokens, "Hello\u{202F}world"));

        // Verify the special whitespace was restored
        assert_eq!(tokens[0].whitespace, "\u{202F}");
    }

    #[test]
    fn test_fix_reconstruction_multiple_accent_differences() {
        let mut tokens = vec![
            create_test_token(
                "Creme",
                " ",
                PartOfSpeech::Noun,
                "creme",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                "brulee",
                "",
                PartOfSpeech::Noun,
                "brulee",
                DependencyRelation::Nmod,
                1,
            ),
        ];

        // Model returned "Creme brulee" but input was "Crème brûlée"
        assert!(fix_reconstruction(&mut tokens, "Crème brûlée"));

        // Verify both accents were restored
        assert_eq!(tokens[0].text.text, "Crème");
        assert_eq!(tokens[1].text.text, "brûlée");
    }

    #[test]
    fn test_fix_reconstruction_fails_on_real_mismatch() {
        let mut tokens = vec![
            create_test_token(
                "Hello",
                " ",
                PartOfSpeech::Noun,
                "hello",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                "world",
                "",
                PartOfSpeech::Noun,
                "world",
                DependencyRelation::Nmod,
                1,
            ),
        ];

        // This is a real mismatch that shouldn't be fixed
        assert!(!fix_reconstruction(&mut tokens, "Goodbye world"));

        // Tokens should remain unchanged
        assert_eq!(tokens[0].text.text, "Hello");
    }

    #[test]
    fn test_fix_reconstruction_dibujala_vs_dibujela() {
        // Test that "Dibujala." and "Dibújela." are correctly distinguished
        // (different words, not just accent differences)
        let mut tokens_dibujala = vec![
            create_test_token(
                "Dibujala",
                "",
                PartOfSpeech::Verb,
                "dibujar",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                ".",
                "",
                PartOfSpeech::Punct,
                ".",
                DependencyRelation::Punct,
                1,
            ),
        ];

        // "Dibujala" should NOT match "Dibújela" even after normalization
        // because they normalize to different strings:
        // "Dibujala" -> "Dibujala" and "Dibújela" -> "Dibujela"
        assert!(!fix_reconstruction(&mut tokens_dibujala, "Dibújela."));

        // But it should work if the input actually was "Dibujala."
        let mut tokens_dibujala2 = vec![
            create_test_token(
                "Dibujala",
                "",
                PartOfSpeech::Verb,
                "dibujar",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                ".",
                "",
                PartOfSpeech::Punct,
                ".",
                DependencyRelation::Punct,
                1,
            ),
        ];
        assert!(fix_reconstruction(&mut tokens_dibujala2, "Dibujala."));
    }

    #[test]
    fn test_fix_reconstruction_same_word_different_accents() {
        // Test that if the model returns "Dibujala" but input was "Dibújala" (same word, just accent difference)
        // it should restore the accent
        let mut tokens = vec![
            create_test_token(
                "Dibujala",
                "",
                PartOfSpeech::Verb,
                "dibujar",
                DependencyRelation::Root,
                0,
            ),
            create_test_token(
                ".",
                "",
                PartOfSpeech::Punct,
                ".",
                DependencyRelation::Punct,
                1,
            ),
        ];

        // Model returned "Dibujala." but input was "Dibújala."
        assert!(fix_reconstruction(&mut tokens, "Dibújala."));

        // Verify the accent was restored
        assert_eq!(tokens[0].text.text, "Dibújala");
    }
}

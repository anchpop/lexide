use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported languages for analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    English,
    French,
    Spanish,
    Korean,
    German,
    Italian,
    Portuguese,
    Russian,
    Japanese,
    Hindi,
    Thai,
    ChineseSimplified,
}

impl Language {
    /// Short code used to select the parsley lemma table for this language.
    pub fn code(&self) -> &'static str {
        match self {
            Language::English => "eng",
            Language::French => "fra",
            Language::Spanish => "spa",
            Language::Korean => "kor",
            Language::German => "deu",
            Language::Italian => "ita",
            Language::Portuguese => "por",
            Language::Russian => "rus",
            Language::Japanese => "jpn",
            Language::Hindi => "hin",
            Language::Thai => "tha",
            Language::ChineseSimplified => "zho-hans",
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Language::English => write!(f, "English"),
            Language::French => write!(f, "French"),
            Language::Spanish => write!(f, "Spanish"),
            Language::Korean => write!(f, "Korean"),
            Language::German => write!(f, "German"),
            Language::Italian => write!(f, "Italian"),
            Language::Portuguese => write!(f, "Portuguese"),
            Language::Russian => write!(f, "Russian"),
            Language::Japanese => write!(f, "Japanese"),
            Language::Hindi => write!(f, "Hindi"),
            Language::Thai => write!(f, "Thai"),
            Language::ChineseSimplified => write!(f, "Chinese (Simplified)"),
        }
    }
}

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
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Language::English => write!(f, "English"),
            Language::French => write!(f, "French"),
            Language::Spanish => write!(f, "Spanish"),
            Language::Korean => write!(f, "Korean"),
            Language::German => write!(f, "German"),
        }
    }
}

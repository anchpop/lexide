use std::fmt;

#[derive(
    Clone, Debug, serde::Serialize, serde::Deserialize, Hash, Eq, PartialEq, Ord, PartialOrd, Copy,
)]
pub enum PartOfSpeech {
    #[serde(rename = "ADJ")]
    Adj, // adjective
    #[serde(rename = "ADP")]
    Adp, // adposition
    #[serde(rename = "ADV")]
    Adv, // adverb
    #[serde(rename = "AUX")]
    Aux, // auxiliary
    #[serde(rename = "CCONJ")]
    Cconj, // coordinating conjunction
    #[serde(rename = "DET")]
    Det, // determiner
    #[serde(rename = "INTJ")]
    Intj, // interjection
    #[serde(rename = "NOUN")]
    Noun, // noun
    #[serde(rename = "NUM")]
    Num, // numeral
    #[serde(rename = "PART")]
    Part, // particle
    #[serde(rename = "PRON")]
    Pron, // pronoun
    #[serde(rename = "PROPN")]
    Propn, // proper noun
    #[serde(rename = "PUNCT")]
    Punct, // punctuation
    #[serde(rename = "SCONJ")]
    Sconj, // subordinating conjunction
    #[serde(rename = "SYM")]
    Sym, // symbol
    #[serde(rename = "VERB")]
    Verb, // verb
    #[serde(rename = "SPACE")]
    Space, // space
    #[serde(rename = "X")]
    X, // other
}

impl fmt::Display for PartOfSpeech {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_plain::to_string(self).unwrap())
    }
}

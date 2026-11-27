//! Checks the tokenizer and espeak mapping against Python fixtures on stdin (tts/web/test/parity.py and
//! espeak-parity.py write them).
//!     tokens: first line [frontend spec, columns], then [id, grams, ids, lexical] per row.
//!     espeak: first line [espeak spec, frontend spec, columns], then [text, [[chunk, ipa], ...], grams, [ids,
//!             lexical] or null for text outside the inventory] per row.

use std::io::BufRead;

use lexide_tts::{Espeak, Gram, Tokenizer};
use serde_json::{Value, from_value};

fn main() {
    let mode = std::env::args().nth(1).expect("tokens or espeak");
    let mut lines = std::io::stdin()
        .lock()
        .lines()
        .map(|l| serde_json::from_str::<Value>(&l.unwrap()).unwrap());
    let head = lines.next().unwrap();
    let espeak = mode == "espeak";
    let (spec, columns) = if espeak {
        (&head[1], &head[2])
    } else {
        (&head[0], &head[1])
    };
    let tokenizer = Tokenizer::new(
        from_value(spec.clone()).unwrap(),
        &from_value::<Vec<String>>(columns.clone()).unwrap(),
        !espeak,
    )
    .unwrap();
    let parser =
        espeak.then(|| Espeak::new(from_value(head[0].clone()).unwrap(), tokenizer.phones()));
    let mut rows = 0;
    for row in lines {
        let (grams, expected) = match &parser {
            Some(p) => {
                let text = row[0].as_str().unwrap();
                let calls: Vec<(String, String)> = from_value(row[1].clone()).unwrap();
                let chunks = p.chunks(text);
                assert_eq!(
                    chunks,
                    calls.iter().map(|c| c.0.clone()).collect::<Vec<_>>(),
                    "chunks of {text:?}"
                );
                let ipa: Vec<String> = calls.into_iter().map(|c| c.1).collect();
                let grams = p.grams(text, &ipa).unwrap();
                assert_eq!(
                    grams,
                    from_value::<Vec<Gram>>(row[2].clone()).unwrap(),
                    "grams of {text:?}"
                );
                (grams, row[3].clone())
            }
            None => (
                from_value(row[1].clone()).unwrap(),
                Value::Array(vec![row[2].clone(), row[3].clone()]),
            ),
        };
        match tokenizer.tokenize(&grams) {
            Ok(t) if !expected.is_null() => {
                let lexical: Vec<Vec<i64>> = from_value(expected[1].clone()).unwrap();
                assert_eq!(
                    t.ids,
                    from_value::<Vec<i64>>(expected[0].clone()).unwrap(),
                    "ids of {}",
                    row[0]
                );
                assert_eq!(t.lexical, lexical.concat(), "lexical of {}", row[0]);
            }
            Err(e) if expected.is_null() => {
                assert!(e.contains("outside the voice inventory"), "{e}")
            }
            other => panic!("{}: {:?}", row[0], other.map(|t| t.ids)),
        }
        rows += 1;
    }
    eprintln!("{rows} rows match");
}

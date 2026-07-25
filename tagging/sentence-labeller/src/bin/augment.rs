use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use std::{collections::HashMap, fs, path::PathBuf};

const LANGUAGES: &[&str] = &[
    "deu", "eng", "fra", "hin", "ita", "jpn", "kor", "por", "rus", "spa",
];
const GAPS: &[&str] = &[
    " ",
    "  ",
    "\t",
    "\n",
    "\n\n",
    "\r\n",
    "\r\n\r\n",
    "\n\t",
    "\u{00a0}",
    "\u{2003}",
    "\u{2009}",
    "\u{2028}",
    "\u{2029}",
    "\u{200b}",
    "\u{2060}",
    "\n---\n",
    "\n***\n",
    "\n···\n",
    "\n§ § §\n",
    "<!-- break -->",
    "\n[... ]\n",
    "",
];
const LEADERS: &[&str] = &["", "• ", "- ", "— ", "1. ", "(a) ", "※ ", "> "];
const WRAPPERS: &[(&str, &str)] = &[
    ("", ""),
    ("\"", "\""),
    ("“", "”"),
    ("‘", "’"),
    ("«", "»"),
    ("‹", "›"),
    ("「", "」"),
    ("『", "』"),
    ("（", "）"),
];
const OUTSIDE: &[&str] = &[
    "",
    "\n",
    "\n\n",
    "###\n",
    "§\n",
    "[document]\n",
    "<hr>\n",
    "\u{feff}",
    "\t\n",
];

/// Per-language material for the v2 hard cases the LLM-labelled data barely covers:
/// abbreviation-bearing sentences (periods that do NOT end the sentence), quoted speech
/// with an attribution tail (one sentence, not two), and heading-like gap content.
/// `{N}` slots are filled with names mined from the language's own sentence pool
/// (capitalized non-initial words), or `names` for caseless scripts.
struct LangPack {
    abbr_templates: &'static [&'static str],
    /// (before, after) wrapped around a pool sentence; the whole thing is ONE sentence.
    attributions: &'static [(&'static str, &'static str)],
    headings: &'static [&'static str],
    /// Fixed name pool for languages where capitalization can't identify names.
    names: &'static [&'static str],
}

fn lang_pack(lang: &str) -> LangPack {
    match lang {
        "eng" => LangPack {
            abbr_templates: &[
                "Mr. {N} met Mrs. {N} at 3 p.m. on Friday.",
                "Dr. {N} works at the N.A.S.A. laboratory on Elm St. near the river.",
                "Prof. {N} cited pp. 12–14, i.e. the appendix.",
                "{N} joined Acme Inc. in 1998, e.g. as a clerk.",
                "St. {N} was born ca. 300 A.D. in a small town.",
                "Capt. {N} and Sgt. {N} arrived at 6 a.m. sharp.",
                "The U.S.A. sent Dr. {N} to the U.N. summit.",
                "See Fig. 3 and Vol. II, ch. 4, for details.",
                "{N} earned a Ph.D. under Prof. {N} last year.",
                "They live at No. 22 Baker St., near Mt. Hope.",
                "Mr. and Mrs. {N} of number four were proud to say that they were normal.",
                "The F.B.I. interviewed {N} for approx. two hours.",
            ],
            attributions: &[
                ("\"", "\" she asked."),
                ("\"", "\" he said quietly."),
                ("\"", "\" said Mr. {N}."),
                ("“", "” she whispered."),
                ("“", "” shouted {N}."),
                ("\"", "\" they replied."),
            ],
            headings: &[
                "Chapter Two", "Chapter 13", "PART ONE", "Prologue", "III.",
                "2. Introduction", "Epilogue",
            ],
            names: &[],
        },
        "deu" => LangPack {
            abbr_templates: &[
                "Dr. {N} traf Hrn. {N} um 15 Uhr, d. h. am Nachmittag.",
                "Prof. {N} zitierte S. 12 ff., vgl. Abb. 3.",
                "Sie wohnt in der Hauptstr. 5, Nr. 12, bei Fr. {N}.",
                "Das kostet ca. 20 Euro, z. B. im Supermarkt.",
                "Wir brauchen Mehl, Zucker, Eier usw. für den Kuchen.",
                "Die Fa. {N} GmbH liefert u. a. nach Österreich.",
                "Hr. {N} kommt evtl. am Mo. oder Di. vorbei.",
                "Das Treffen ist am 3. Okt. um 9 Uhr, bzw. etwas später.",
            ],
            attributions: &[
                ("„", "“, sagte sie."),
                ("„", "“, fragte Hr. {N}."),
                ("«", "», flüsterte er."),
                ("„", "“, rief {N}."),
            ],
            headings: &["Kapitel Zwei", "2. Kapitel", "Teil Eins", "Prolog", "IV."],
            names: &[],
        },
        "fra" => LangPack {
            abbr_templates: &[
                "M. {N} a rencontré Mme {N} à 15 h, c.-à-d. l'après-midi.",
                "Le Dr {N} habite au n° 5, av. de la République.",
                "Voir p. ex. les pp. 12-14, cf. chap. 3.",
                "L'événement date du IIIe s. av. J.-C. environ.",
                "M. {N} travaille chez {N} S.A. depuis 1998.",
                "MM. {N} et {N} sont arrivés vers 18 h, etc.",
            ],
            attributions: &[
                ("«\u{a0}", "\u{a0}» demanda-t-elle."),
                ("«\u{a0}", "\u{a0}», dit M. {N}."),
                ("— ", " murmura-t-elle."),
                ("«\u{a0}", "\u{a0}» répondit {N}."),
            ],
            headings: &["Chapitre deux", "Chapitre 13", "Prologue", "Première partie", "II."],
            names: &[],
        },
        "ita" => LangPack {
            abbr_templates: &[
                "Il sig. {N} ha incontrato la sig.ra {N} alle 15.",
                "Il dott. {N} lavora in via Roma n. 5.",
                "Vedi ad es. le pagg. 12-14, cfr. cap. 3.",
                "Il prof. {N} cita il vol. II, pag. 44, ecc.",
                "La ditta {N} S.p.A. consegna il 3 ott. circa.",
            ],
            attributions: &[
                ("«", "» chiese lei."),
                ("\"", "\" disse il sig. {N}."),
                ("«", "» rispose {N}."),
                ("“", "” sussurrò lui."),
            ],
            headings: &["Capitolo due", "Parte prima", "Prologo", "III."],
            names: &[],
        },
        "spa" => LangPack {
            abbr_templates: &[
                "El Sr. {N} saludó a la Sra. {N} a las 3 p. m.",
                "La Dra. {N} vive en la avda. Mayor, núm. 5.",
                "Véase p. ej. las págs. 12-14, cf. cap. 3.",
                "EE. UU. envió al Dr. {N} a la cumbre.",
                "El prof. {N} llegó a las 9 a. m. aprox.",
            ],
            attributions: &[
                ("«", "», preguntó ella."),
                ("—", " —dijo el Sr. {N}—."),
                ("«", "», respondió {N}."),
                ("“", "”, susurró él."),
            ],
            headings: &["Capítulo dos", "Segunda parte", "Prólogo", "XI."],
            names: &[],
        },
        "por" => LangPack {
            abbr_templates: &[
                "O Sr. {N} encontrou a Sra. {N} às 15 h.",
                "O Dr. {N} mora na Av. Paulista, n.º 5.",
                "Veja p. ex. as págs. 12-14, cf. cap. 3.",
                "A Cia. de teatro chegou às 20 h, i.e. atrasada.",
                "O prof. {N} nasceu no séc. XIX, em S. Paulo.",
            ],
            attributions: &[
                ("«", "», perguntou ela."),
                ("“", "”, disse o Sr. {N}."),
                ("—", " — respondeu {N}."),
            ],
            headings: &["Capítulo dois", "Parte um", "Prólogo", "VII."],
            names: &[],
        },
        "rus" => LangPack {
            abbr_templates: &[
                "Г-н {N} встретил г-жу {N} в 15 ч., т. е. днём.",
                "Д-р {N} живёт на ул. Ленина, д. 5, кв. 12.",
                "Он купил хлеб, молоко и т. д. в магазине.",
                "В 1998 г. компания выросла до 5 тыс. человек.",
                "См. напр. стр. 12–14 и др. источники.",
            ],
            attributions: &[
                ("«", "» — спросила она."),
                ("— ", " — сказал г-н {N}."),
                ("«", "» — ответил {N}."),
                ("«", "» — прошептала она."),
            ],
            headings: &["Глава вторая", "Часть I", "Пролог", "Глава 13"],
            names: &[],
        },
        "hin" => LangPack {
            abbr_templates: &[
                "डॉ. {N} सुबह 9 बजे अस्पताल पहुँचे।",
                "प्रो. {N} ने पृ. 12 का हवाला दिया।",
                "श्री {N} और डॉ. {N} कल मिलेंगे।",
            ],
            attributions: &[
                ("\"", "\" उसने पूछा।"),
                ("“", "” {N} ने कहा।"),
                ("\"", "\" उसने धीरे से कहा।"),
            ],
            headings: &["अध्याय दो", "भाग एक", "अध्याय 13"],
            names: &["राम", "सीता", "अर्जुन", "मीरा", "कृष्ण", "राधा"],
        },
        "jpn" => LangPack {
            abbr_templates: &[
                "{N}はN.A.S.A.の研究員だ。",
                "会議は午前9時、つまりA.M.9時に始まった。",
            ],
            attributions: &[
                ("「", "」と彼女は尋ねた。"),
                ("「", "」と{N}は言った。"),
                ("「", "」と彼はつぶやいた。"),
            ],
            headings: &["第二章", "第1部", "プロローグ"],
            names: &["田中", "佐藤", "鈴木", "山田"],
        },
        "kor" => LangPack {
            abbr_templates: &[
                "{N}은 N.A.S.A.에서 일한다.",
                "회의는 오전 9시, 즉 A.M. 9시에 시작한다.",
            ],
            attributions: &[
                ("\"", "\"라고 그녀가 물었다."),
                ("“", "”라고 {N}이 말했다."),
                ("\"", "\"라고 그가 속삭였다."),
            ],
            headings: &["제2장", "1부", "프롤로그"],
            names: &["민준", "서연", "지훈", "하은"],
        },
        _ => unreachable!("unknown language {lang}"),
    }
}

/// Names mined from the pool: capitalized words not at sentence start (sentence-initial
/// capitals are usually not names). Falls back to the pack's fixed list for caseless
/// scripts or thin pools.
fn mine_names(pool: &[String], pack: &LangPack) -> Vec<String> {
    let mut names: Vec<String> = pool
        .iter()
        .flat_map(|s| {
            s.split_whitespace().skip(1).filter_map(|w| {
                let w = w.trim_matches(|c: char| !c.is_alphabetic());
                let mut chars = w.chars();
                let first = chars.next()?;
                (first.is_uppercase()
                    && w.chars().count() >= 3
                    && w.chars().count() <= 12
                    && chars.all(|c| c.is_lowercase()))
                .then(|| w.to_owned())
            })
        })
        .take(4000)
        .collect();
    names.extend(pack.names.iter().map(|s| s.to_string()));
    if names.is_empty() {
        names.push("Alex".to_owned()); // unreachable in practice; keeps fill() total
    }
    names
}

fn fill(template: &str, names: &[String], rng: &mut Rng) -> String {
    let mut out = template.to_owned();
    while let Some(pos) = out.find("{N}") {
        out.replace_range(pos..pos + 3, &names[rng.usize(names.len())]);
    }
    out
}

#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn usize(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }
    fn pick<'a>(&mut self, values: &'a [&'a str]) -> &'a str {
        values[self.usize(values.len())]
    }
}

#[derive(Debug, Serialize)]
struct Section {
    #[serde(rename = "type")]
    kind: &'static str,
    content: String,
}
#[derive(Debug, Serialize)]
struct Record {
    id: String,
    lang: String,
    source: &'static str,
    text: String,
    sections: Vec<Section>,
}

fn push(sections: &mut Vec<Section>, kind: &'static str, content: String) {
    if content.is_empty() {
        return;
    }
    if let Some(last) = sections.last_mut() {
        if last.kind == kind && kind == "gap" {
            last.content.push_str(&content);
            return;
        }
    }
    sections.push(Section { kind, content });
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let data_root = PathBuf::from(args.next().unwrap_or_else(|| "../data/big".into()));
    let output = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "data/mechanical-augmented.jsonl".into()),
    );
    let per_language: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(2_000);
    let mut pools: HashMap<&str, Vec<String>> = HashMap::new();
    for &lang in LANGUAGES {
        let path = data_root
            .join(lang)
            .join("target_language_sentences_tokenization.jsonl");
        let input =
            fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let pool = input
            .lines()
            .filter_map(|line| {
                let value: Value = serde_json::from_str(line).ok()?;
                let sentence = value.get("sentence")?.as_str()?.to_owned();
                (!sentence.is_empty()).then_some(sentence)
            })
            .collect::<Vec<_>>();
        if pool.is_empty() {
            bail!("no sentences for {lang}")
        }
        pools.insert(lang, pool);
    }

    let mut out = String::new();
    for (lang_index, &lang) in LANGUAGES.iter().enumerate() {
        let pool = &pools[lang];
        let pack = lang_pack(lang);
        let names = mine_names(pool, &pack);
        let mut rng = Rng::new(0x9e3779b97f4a7c15 ^ ((lang_index as u64 + 1) * 0x100000001b3));
        for sample in 0..per_language {
            let mut sections = Vec::new();
            push(&mut sections, "gap", rng.pick(OUTSIDE).to_owned());
            let sentence_count = 2 + rng.usize(15);
            for index in 0..sentence_count {
                // ~20% abbreviation-template sentence, ~15% quote+attribution around a
                // pool sentence, else the v1 leader/wrapper composition. The first two
                // are the ". does not end the sentence" counter-examples the LLM data
                // barely covers; they stay flourish-free so the pattern is clean.
                let roll = rng.usize(100);
                let content = if roll < 20 {
                    let template = pack.abbr_templates[rng.usize(pack.abbr_templates.len())];
                    let leader = rng.pick(LEADERS);
                    format!("{leader}{}", fill(template, &names, &mut rng))
                } else if roll < 35 {
                    let sentence = &pool[rng.usize(pool.len())];
                    let (before, after) = pack.attributions[rng.usize(pack.attributions.len())];
                    format!(
                        "{}{sentence}{}",
                        fill(before, &names, &mut rng),
                        fill(after, &names, &mut rng)
                    )
                } else {
                    let sentence = &pool[rng.usize(pool.len())];
                    let leader = rng.pick(LEADERS);
                    let (open, close) = WRAPPERS[rng.usize(WRAPPERS.len())];
                    format!("{leader}{open}{sentence}{close}")
                };
                push(&mut sections, "sentence", content);
                if index + 1 < sentence_count {
                    // ~12% heading-like gap (Chapter Two etc.), else the v1 gap set.
                    let gap = if rng.usize(100) < 12 {
                        let heading = pack.headings[rng.usize(pack.headings.len())];
                        format!("\n\n{heading}\n\n")
                    } else {
                        rng.pick(GAPS).to_owned()
                    };
                    push(&mut sections, "gap", gap);
                }
            }
            push(&mut sections, "gap", rng.pick(OUTSIDE).to_owned());
            let text: String = sections.iter().map(|s| s.content.as_str()).collect();
            if text
                != sections
                    .iter()
                    .map(|s| s.content.as_str())
                    .collect::<String>()
            {
                unreachable!()
            }
            let record = Record {
                id: format!("mechanical-{lang}-{sample:05}"),
                lang: lang.to_owned(),
                source: "mechanical-sentence-composition-v2",
                text,
                sections,
            };
            out.push_str(&serde_json::to_string(&record)?);
            out.push('\n');
        }
    }
    fs::write(&output, out).with_context(|| format!("write {}", output.display()))?;
    println!(
        "Wrote {} examples ({} per language) to {}",
        per_language * LANGUAGES.len(),
        per_language,
        output.display()
    );
    Ok(())
}

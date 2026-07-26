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
    ("(", ")"),
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
///
/// v3 adds `dash_templates`: mid-sentence dash continuations (especially after an
/// abbreviation — "Dr. X arrived at 3 p.m. - and he wasn't late." is ONE sentence,
/// even though "- " also appears in LEADERS as a list-item starter) and
/// dash/paren parentheticals with internal terminal punctuation ("— can you
/// believe it? —"), which the field eval showed orphan 1-char fragments. For jpn
/// it also covers the 「…」だ。 copula tail that orphaned だ in Pale Lights.
struct LangPack {
    abbr_templates: &'static [&'static str],
    /// One-sentence templates exercising dashes: abbreviation + dash continuation,
    /// dash-interrupted parentheticals, parens with inner !/?.
    dash_templates: &'static [&'static str],
    /// One-sentence templates with non-terminal periods that are NOT abbreviations:
    /// decimals, version numbers, URLs, emails, mid-sentence ellipses, colons.
    tricky_templates: &'static [&'static str],
    /// Sentences that END with an abbreviation period — the counter-case to
    /// abbr_templates, so "abbrev period" doesn't learn to mean "never a boundary".
    /// Empty for scripts whose sentences don't end in ASCII periods (hin/jpn/kor).
    abbr_final_templates: &'static [&'static str],
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
            dash_templates: &[
                "Dr. {N} arrived at 3 p.m. - and he wasn't late.",
                "Mr. {N} left at 6 a.m. — without saying a word.",
                "{N} packed rope, maps, tinned food, etc. - everything but water.",
                "The meeting — can you believe it? — ran until 9 p.m.",
                "She was — or so Mrs. {N} claimed — never once wrong.",
                "He paused — what else could he do? — and kept walking.",
                "The results (see Fig. 3!) surprised even Prof. {N}.",
                "It cost $3.50 – i.e. half the usual price – at the corner shop.",
                "The plan - simple enough, no? - fell apart by noon.",
                "Her answer — a flat \"no.\" — ended the discussion.",
            ],
            tricky_templates: &[
                "The update to version 2.4.1 fixed nothing.",
                "Pi is roughly 3.14159, give or take.",
                "Visit www.example.com for the full schedule.",
                "Write to info@example.org before Friday.",
                "There was one rule: never be late.",
                "He waited... and waited... and nothing happened.",
                "It weighs about 3.5 kg, more or less.",
            ],
            abbr_final_templates: &[
                "They moved to No. 22 Baker St.",
                "He finally earned his Ph.D.",
                "The parcel came from Acme Inc.",
                "The lecture covered the history of the U.S.A.",
                "She was born around 300 A.D.",
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
            dash_templates: &[
                "Dr. {N} kam um 15 Uhr an - und er war nicht zu spät.",
                "Hr. {N} ging um 6 Uhr — ohne ein Wort zu sagen.",
                "Sie kaufte Mehl, Zucker, Eier usw. - aber kein Salz.",
                "Das Treffen — man glaubt es kaum! — dauerte bis 21 Uhr.",
                "Sie war — so sagte Fr. {N} — niemals unpünktlich.",
                "Die Ergebnisse (vgl. Abb. 3!) überraschten Prof. {N}.",
                "Der Plan - ganz einfach, oder? - scheiterte am Mittag.",
            ],
            tricky_templates: &[
                "Version 2.4.1 hat gar nichts geändert.",
                "Besuchen Sie www.beispiel.de für den Plan.",
                "Schreiben Sie an info@beispiel.de bis Freitag.",
                "Es gilt eine Regel: niemals zu spät kommen.",
                "Er wartete... und wartete... und nichts geschah.",
            ],
            abbr_final_templates: &[
                "Der Termin ist am 3. Okt.",
                "Er kommt am Mo. oder Di.",
                "Wir brauchen Mehl, Zucker, Eier usw.",
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
            dash_templates: &[
                "M. {N} est arrivé à 15 h - et il n'était pas en retard.",
                "Mme {N} est partie à 6 h — sans dire un mot.",
                "Il a tout acheté, cartes, cordes, etc. - sauf l'eau.",
                "La réunion — qui l'eût cru ? — a duré jusqu'à 21 h.",
                "Elle était — d'après Mme {N} — toujours à l'heure.",
                "Les résultats (voir fig. 3 !) ont surpris le Dr {N}.",
                "Le plan - simple, non ? - a échoué avant midi.",
            ],
            tricky_templates: &[
                "La version 2.4.1 n'a rien changé.",
                "Visitez www.exemple.fr pour le programme.",
                "Écrivez à info@exemple.fr avant vendredi.",
                "Une seule règle : ne jamais être en retard.",
                "Il attendit... encore et encore... sans résultat.",
            ],
            abbr_final_templates: &[
                "Il travaille pour la société {N} S.A.",
                "L'accord date du IIIe s. av. J.-C.",
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
            dash_templates: &[
                "Il sig. {N} è arrivato alle 15 - e non era in ritardo.",
                "La sig.ra {N} è uscita alle 6 — senza dire una parola.",
                "Ha comprato corde, mappe, viveri, ecc. - ma non l'acqua.",
                "La riunione — chi l'avrebbe detto? — è durata fino alle 21.",
                "Lei era — così diceva il dott. {N} — sempre puntuale.",
                "I risultati (vedi fig. 3!) sorpresero il prof. {N}.",
            ],
            tricky_templates: &[
                "La versione 2.4.1 non ha cambiato nulla.",
                "Visita www.esempio.it per il programma.",
                "Scrivi a info@esempio.it entro venerdì.",
                "Una sola regola: mai arrivare in ritardo.",
                "Aspettò... e aspettò... e non accadde nulla.",
            ],
            abbr_final_templates: &[
                "La ditta si chiama {N} S.p.A.",
                "La sede è in via Roma n. 5.",
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
            dash_templates: &[
                "El Sr. {N} llegó a las 3 p. m. - y no llegó tarde.",
                "La Sra. {N} salió a las 6 a. m. — sin decir palabra.",
                "Compró mapas, cuerdas, provisiones, etc. - pero no agua.",
                "La reunión —¿quién lo diría?— duró hasta las 9 p. m.",
                "Ella era —según la Dra. {N}— siempre puntual.",
                "Los resultados (véase fig. 3, ¡increíble!) sorprendieron al prof. {N}.",
            ],
            tricky_templates: &[
                "La versión 2.4.1 no cambió nada.",
                "Visita www.ejemplo.es para el programa.",
                "Escribe a info@ejemplo.es antes del viernes.",
                "Solo hay una regla: nunca llegar tarde.",
                "Esperó... y esperó... y no pasó nada.",
            ],
            abbr_final_templates: &[
                "Él trabaja en los EE. UU.",
                "La cita es a las 9 a. m.",
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
            dash_templates: &[
                "O Sr. {N} chegou às 15 h - e não estava atrasado.",
                "A Sra. {N} saiu às 6 h — sem dizer palavra.",
                "Comprou mapas, cordas, mantimentos, etc. - mas não água.",
                "A reunião — quem diria? — durou até às 21 h.",
                "Ela era — segundo o Dr. {N} — sempre pontual.",
                "Os resultados (ver fig. 3!) surpreenderam o prof. {N}.",
            ],
            tricky_templates: &[
                "A versão 2.4.1 não mudou nada.",
                "Visite www.exemplo.pt para o programa.",
                "Escreva para info@exemplo.pt até sexta.",
                "Só há uma regra: nunca se atrasar.",
                "Esperou... e esperou... e nada aconteceu.",
            ],
            abbr_final_templates: &[
                "Ele mora na Av. Paulista.",
                "A empresa chama-se {N} Ltda.",
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
            dash_templates: &[
                "Г-н {N} пришёл в 15 ч. - и он не опоздал.",
                "Г-жа {N} ушла в 6 ч. — не сказав ни слова.",
                "Он купил карты, верёвки, хлеб и т. д. - но не воду.",
                "Собрание — кто бы мог подумать! — длилось до 21 ч.",
                "Она была — так говорила г-жа {N} — всегда пунктуальна.",
                "Результаты (см. рис. 3!) удивили даже д-ра {N}.",
            ],
            tricky_templates: &[
                "Версия 2.4.1 ничего не изменила.",
                "Подробности на сайте www.primer.ru всегда открыты.",
                "Пишите на info@primer.ru до пятницы.",
                "Правило одно: никогда не опаздывать.",
                "Он ждал... и ждал... и ничего не происходило.",
            ],
            abbr_final_templates: &[
                "Он живёт на ул. Ленина, д. 5.",
                "Компания выросла до 5 тыс.",
                "Он купил хлеб, молоко и т. д.",
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
            dash_templates: &[
                "डॉ. {N} सुबह 9 बजे पहुँचे - और वे देर से नहीं आए।",
                "श्री {N} सुबह 6 बजे निकले — बिना कुछ कहे।",
                "बैठक - कौन जानता था? - देर रात तक चली।",
                "वह — प्रो. {N} के अनुसार — हमेशा समय पर आती थी।",
            ],
            tricky_templates: &[
                "संस्करण 2.4.1 से कुछ नहीं बदला।",
                "विवरण के लिए www.example.com देखें।",
                "नियम एक ही है: कभी देर न करना।",
                "वह इंतज़ार करता रहा... करता रहा... पर कुछ नहीं हुआ।",
            ],
            abbr_final_templates: &[],
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
            dash_templates: &[
                "{N}は午前9時に着いた――遅刻ではなかった。",
                "会議は――信じられないことに――夜9時まで続いた。",
                "彼のモットーは「急がば回れ」だ。",
                "合言葉は「前へ」だ。",
                "返ってきた答えは「知らない」だった。",
                "彼女の口癖は「なるようになる」だそうだ。",
                "その看板の文字は「立入禁止」だったのだ。",
            ],
            tricky_templates: &[
                "バージョン2.4.1では何も変わらなかった。",
                "詳細はwww.example.comを見てください。",
                "規則はただ一つ、遅れないことだ。",
                "彼は待った……ずっと待った……何も起こらなかった。",
            ],
            abbr_final_templates: &[],
            attributions: &[
                ("「", "」と彼女は尋ねた。"),
                ("答えは「", "」だった。"),
                ("口癖は「", "」だ。"),
                ("「", "」――それが彼の答えだった。"),
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
            dash_templates: &[
                "{N}은 오전 9시에 도착했다 - 늦지 않았다.",
                "회의는 - 믿기 어렵겠지만 - 밤 9시까지 이어졌다.",
                "그의 좌우명은 \"천천히 서두르라\"다.",
                "돌아온 대답은 \"모른다\"였다.",
                "그녀는 — {N}의 말에 따르면 — 늘 정확했다.",
            ],
            tricky_templates: &[
                "버전 2.4.1은 아무것도 바꾸지 못했다.",
                "자세한 내용은 www.example.com을 보세요.",
                "규칙은 하나다: 절대 늦지 않는 것.",
                "그는 기다리고... 또 기다렸다... 아무 일도 없었다.",
            ],
            abbr_final_templates: &[],
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
                // ~12% abbreviation template, ~12% dash template, ~8% tricky-period
                // template, ~12% quote+attribution around a pool sentence, ~7%
                // multi-sentence quote (two sentences sharing one wrapper pair,
                // split mid-quote — matches how the LLM labels them), ~5% sentence
                // ENDING in an abbreviation (followed by a forced plain gap: with
                // an empty/zero-width gap the boundary would be unknowable), else
                // the v1 leader/wrapper composition (5% of which get one internal
                // hard-wrap \n, mirroring wrapped prose in the real data). The
                // templates are the counter-examples the LLM data barely covers
                // (". does not end the sentence", "mid-sentence dash is not a list
                // leader"); they stay flourish-free so the pattern is clean. Dash
                // templates get no leader: a leading "- " would blur exactly the
                // leader-vs-continuation distinction they exist to teach.
                let roll = rng.usize(100);
                let abbr_final_ok = !pack.abbr_final_templates.is_empty();
                let content = if roll < 12 {
                    let template = pack.abbr_templates[rng.usize(pack.abbr_templates.len())];
                    let leader = rng.pick(LEADERS);
                    format!("{leader}{}", fill(template, &names, &mut rng))
                } else if roll < 24 {
                    let template = pack.dash_templates[rng.usize(pack.dash_templates.len())];
                    fill(template, &names, &mut rng)
                } else if roll < 32 {
                    pack.tricky_templates[rng.usize(pack.tricky_templates.len())].to_owned()
                } else if roll < 44 {
                    let sentence = &pool[rng.usize(pool.len())];
                    let (before, after) = pack.attributions[rng.usize(pack.attributions.len())];
                    format!(
                        "{}{sentence}{}",
                        fill(before, &names, &mut rng),
                        fill(after, &names, &mut rng)
                    )
                } else if roll < 51 {
                    // Multi-sentence quote: push both sentences and the inner gap
                    // here, then fall through to the normal between-sentence gap.
                    let (open, close) = WRAPPERS[1 + rng.usize(WRAPPERS.len() - 1)];
                    let a = &pool[rng.usize(pool.len())];
                    let b = &pool[rng.usize(pool.len())];
                    push(&mut sections, "sentence", format!("{open}{a}"));
                    push(&mut sections, "gap", " ".to_owned());
                    format!("{b}{close}")
                } else if roll < 56 && abbr_final_ok {
                    let t = pack.abbr_final_templates[rng.usize(pack.abbr_final_templates.len())];
                    let filled = fill(t, &names, &mut rng);
                    push(&mut sections, "sentence", filled);
                    if index + 1 < sentence_count {
                        push(&mut sections, "gap", if rng.usize(2) == 0 { " " } else { "\n" }.to_owned());
                    }
                    continue;
                } else {
                    let sentence = &pool[rng.usize(pool.len())];
                    let leader = rng.pick(LEADERS);
                    let (open, close) = WRAPPERS[rng.usize(WRAPPERS.len())];
                    let mut composed = format!("{leader}{open}{sentence}{close}");
                    if rng.usize(100) < 5 {
                        // Hard-wrap: turn one interior space into a newline.
                        let space_positions: Vec<usize> = composed
                            .char_indices()
                            .skip(1)
                            .filter(|&(i, c)| c == ' ' && i + 1 < composed.len())
                            .map(|(i, _)| i)
                            .collect();
                        if !space_positions.is_empty() {
                            let at = space_positions[rng.usize(space_positions.len())];
                            composed.replace_range(at..at + 1, "\n");
                        }
                    }
                    composed
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
                source: "mechanical-sentence-composition-v3",
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

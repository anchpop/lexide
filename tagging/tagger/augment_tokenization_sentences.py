"""Generate abbreviation-dense + multiword-noun-phrase-rich sentences for the Gemma
teacher to tokenize — the tokenization silver data's blind spot (the char tokenizer
drops words around abbreviations and wobbles on multi-word proper nouns because the
pools barely contain them; same root cause the segmenter's augment v2 fixed).

    python3 tagger/augment_tokenization_sentences.py            # -> data/aug_sentences/<lang>.txt

Names / multi-word entities are mined from each language's own silver pool (capitalized
non-initial words / runs), topped up with curated per-language entities (Eiffel Tower
etc.) and dotted acronyms. Output is plain sentences; label them with
`lexide/src/bin/label_tokenization.rs` against the Gemma serve, hold out
`--holdout` per language for eval, and feed the rest to data_prep via
data/big/<lang>/target_language_sentences_tokenization_augmented.jsonl.
"""
import argparse
import json
import random
import re
from pathlib import Path

LANGS = ["deu", "eng", "fra", "hin", "ita", "jpn", "kor", "por", "rus", "spa"]

ACRONYMS = ["N.A.S.A.", "U.S.A.", "F.B.I.", "U.N.", "B.B.C.", "C.I.A.", "M.I.T.", "U.K."]

# {N}=name  {M}=multi-word entity  {A}=dotted acronym  {NUM}=small number  {YEAR}=year
PACKS = {
    "eng": {
        "mwes": ["Eiffel Tower", "New York Times", "United Nations", "Mount Everest",
                 "Harry Potter", "New York City", "World Health Organization",
                 "Statue of Liberty", "Great Wall of China", "Silicon Valley",
                 "Wall Street", "Central Park", "Oxford University", "Amazon River",
                 "Golden Gate Bridge", "Nobel Prize"],
        "names": [],
        "templates": [
            "Mr. {N} met Mrs. {N} near the {M} at 3 p.m.",
            "Dr. {N} joined the {M} in {YEAR}, e.g. as a clerk.",
            "The {M} hired {N} for approx. {NUM} dollars an hour.",
            "{N} visited the {M} with Prof. {N} on Jan. {NUM}.",
            "She read about the {M} and the {M} in the {M}.",
            "The {A} sent Capt. {N} to the {M} at 6 a.m.",
            "My friend Dr. {N}, the director of the {M}, lives on Baker St.",
            "See Fig. {NUM}, Vol. II, pp. 12–14 of the {M} report.",
            "{N} earned a Ph.D. from the {M} under Prof. {N}.",
            "St. {N} and the {M} appear in ch. {NUM}, i.e. the appendix.",
            "The {M}, the {M}, and the {M} signed the deal at 5 p.m.",
            "According to the {A}, Mr. {N} owns No. {NUM} Baker St.",
            "The {M} stood beside the {M} in the rain.",
            "{N} works at the {M} headquarters near the {M}.",
            "Sgt. {N} of the {A} arrived at {NUM} p.m. sharp, didn't he?",
            "Mrs. {N} bought approx. {NUM} kg of flour, sugar, eggs, etc.",
            "Prof. {N} spoke first; Dr. {N} answered at 4 p.m.",
            "Did Mr. {N} ever visit the {M}?",
            "\"Meet me at the {M} at 5 p.m.,\" said Dr. {N}.",
            "Guests: Mr. {N}, Mrs. {N}, Dr. {N}, and Prof. {N}.",
            "The {M} (est. {YEAR}) employs approx. {NUM} people.",
            "The {M} opened at 9 a.m. and closed at 6 p.m.",
            "Mr. {N}'s trip to the {M} ended on Dec. {NUM}.",
            "The {M} — according to Dr. {N} — reopened in {YEAR}.",
            "It is approx. 3.5 km from the {M} to the {M}.",
            "Was the {M} named after St. {N} or Mrs. {N}?",
            "He cited the {M}, the {A}, and Prof. {N} (p. {NUM}).",
        ],
    },
    "deu": {
        "mwes": ["Neue Zürcher Zeitung", "Vereinten Nationen", "Brandenburger Tor",
                 "Harry Potter", "Rotes Kreuz", "Deutsche Bahn", "Mount Everest",
                 "Eiffelturm in Paris", "Kölner Dom", "Alte Oper"],
        "names": [],
        "templates": [
            "Dr. {N} traf Hrn. {N} um 15 Uhr am {M}.",
            "Prof. {N} zitierte S. {NUM} ff., vgl. Abb. {NUM}, aus der {M}.",
            "Fr. {N} wohnt in der Hauptstr. {NUM}, Nr. {NUM}, nahe dem {M}.",
            "Das kostet ca. {NUM} Euro, z. B. bei der {M}.",
            "Wir brauchen Mehl, Zucker, Eier usw. für Hrn. {N}.",
            "Die Fa. {N} GmbH beliefert u. a. die {M}.",
            "Hr. {N} kommt evtl. am Mo. oder Di. zum {M}.",
            "Das Treffen mit Dr. {N} ist am {NUM}. Okt. um 9 Uhr bzw. später.",
            "Die {M} und das {M} gehören zum Weltkulturerbe.",
            "{N} las in der {M} über die {M}, d. h. am Morgen.",
            "Laut {A} reiste Prof. {N} im Jahr {YEAR} zur {M}.",
            "Prof. {N} sprach zuerst; Dr. {N} antwortete gegen 16 Uhr.",
            "Hat Hr. {N} je die {M} besucht?",
            "„Wir treffen uns um 17 Uhr an der {M}“, sagte Dr. {N}.",
            "Gäste: Hr. {N}, Fr. {N}, Dr. {N} u. a.",
            "Die {M} (gegr. {YEAR}) beschäftigt ca. {NUM} Leute.",
            "Die {M} öffnet um 9 Uhr, d. h. vor der {M}.",
            "Es sind ca. 3,5 km von der {M} bis zur {M}.",
            "Die {M} — so Dr. {N} — wurde {YEAR} eröffnet, vgl. S. {NUM}.",
        ],
    },
    "fra": {
        "mwes": ["tour Eiffel", "Nations Unies", "Mont Blanc", "Harry Potter",
                 "Arc de Triomphe", "Croix-Rouge", "Union européenne",
                 "Champs-Élysées", "Louvre à Paris", "Académie française"],
        "names": [],
        "templates": [
            "M. {N} a rencontré Mme {N} près de la {M} à 15 h.",
            "Le Dr {N} habite au n° {NUM}, av. de la République, face à la {M}.",
            "Voir p. ex. les pp. 12-14, cf. chap. {NUM}, sur la {M}.",
            "L'{M} date du IIIe s. av. J.-C. environ.",
            "M. {N} travaille pour la {M} depuis {YEAR}.",
            "MM. {N} et {N} ont visité la {M} vers 18 h, etc.",
            "Mon ami M. {N}, directeur de la {M}, arrive le {NUM} janv.",
            "Selon la {A}, Mme {N} a photographié la {M} et le {M}.",
            "La {M}, le {M} et la {M} ferment à 17 h.",
            "Le prof. {N} a cité la {M}, c.-à-d. page {NUM}.",
            "M. {N} a parlé d'abord ; le Dr {N} a répondu vers 16 h.",
            "M. {N} a-t-il jamais visité la {M} ?",
            "« Rendez-vous à la {M} à 17 h », dit M. {N}.",
            "Invités : M. {N}, Mme {N}, le Dr {N}, etc.",
            "La {M} (fondée en {YEAR}) emploie env. {NUM} personnes.",
            "La {M} ouvre à 9 h, c.-à-d. avant le {M}.",
            "Il y a env. 3,5 km de la {M} au {M}.",
            "La {M} — selon le Dr {N} — a rouvert en {YEAR}, cf. p. {NUM}.",
        ],
    },
    "ita": {
        "mwes": ["Torre Eiffel", "Nazioni Unite", "Monte Bianco", "Harry Potter",
                 "Croce Rossa", "Unione Europea", "Piazza Navona", "Cappella Sistina",
                 "Divina Commedia", "Via Appia"],
        "names": [],
        "templates": [
            "Il sig. {N} ha incontrato la sig.ra {N} vicino alla {M} alle 15.",
            "Il dott. {N} lavora in via Roma n. {NUM}, presso la {M}.",
            "Vedi ad es. le pagg. 12-14, cfr. cap. {NUM}, sulla {M}.",
            "Il prof. {N} cita il vol. II, pag. {NUM}, ecc.",
            "La ditta {N} S.p.A. rifornisce la {M} dal {YEAR}.",
            "La {M}, la {M} e la {M} chiudono alle 17.",
            "Il mio amico dott. {N}, direttore della {M}, arriva il {NUM} ott.",
            "Secondo la {A}, il sig. {N} ha visitato la {M}.",
            "La {M} apparve nel sec. XIX, a.C. escluso, dice il prof. {N}.",
            "Il prof. {N} parlò per primo; il dott. {N} rispose alle 16.",
            "Il sig. {N} ha mai visitato la {M}?",
            "«Ci vediamo alla {M} alle 17», disse il dott. {N}.",
            "Ospiti: il sig. {N}, la sig.ra {N}, il dott. {N}, ecc.",
            "La {M} (fondata nel {YEAR}) impiega ca. {NUM} persone.",
            "La {M} apre alle 9, cioè prima della {M}.",
            "Sono ca. 3,5 km dalla {M} alla {M}.",
            "La {M} — secondo il dott. {N} — riaprì nel {YEAR}, cfr. pag. {NUM}.",
        ],
    },
    "spa": {
        "mwes": ["Torre Eiffel", "Naciones Unidas", "Nueva York", "Harry Potter",
                 "Cruz Roja", "Unión Europea", "América Latina", "Casa Blanca",
                 "Museo del Prado", "Sagrada Familia"],
        "names": [],
        "templates": [
            "El Sr. {N} saludó a la Sra. {N} junto a la {M} a las 3 p. m.",
            "La Dra. {N} vive en la avda. Mayor, núm. {NUM}, cerca de la {M}.",
            "Véase p. ej. las págs. 12-14, cf. cap. {NUM}, sobre la {M}.",
            "EE. UU. envió al Dr. {N} a la {M} en {YEAR}.",
            "El prof. {N} llegó a las 9 a. m. aprox. a la {M}.",
            "La {M}, la {M} y la {M} cierran a las 17 h.",
            "Mi amigo el Sr. {N}, director de la {M}, llega el {NUM} de oct.",
            "Según la {A}, la Sra. {N} fotografió la {M}.",
            "El Dr. {N} compró harina, azúcar, huevos, etc. en la {M}.",
            "El prof. {N} habló primero; la Dra. {N} respondió a las 4 p. m.",
            "¿Visitó alguna vez el Sr. {N} la {M}?",
            "«Nos vemos en la {M} a las 5 p. m.», dijo la Dra. {N}.",
            "Invitados: el Sr. {N}, la Sra. {N}, el Dr. {N}, etc.",
            "La {M} (fund. {YEAR}) emplea aprox. {NUM} personas.",
            "La {M} abre a las 9 a. m., es decir, antes de la {M}.",
            "Hay aprox. 3,5 km de la {M} a la {M}.",
            "La {M} —según el Dr. {N}— reabrió en {YEAR}, cf. pág. {NUM}.",
        ],
    },
    "por": {
        "mwes": ["Torre Eiffel", "Nações Unidas", "Nova Iorque", "Harry Potter",
                 "Cruz Vermelha", "União Europeia", "São Paulo", "Rio de Janeiro",
                 "Cristo Redentor", "Museu do Louvre"],
        "names": [],
        "templates": [
            "O Sr. {N} encontrou a Sra. {N} perto da {M} às 15 h.",
            "O Dr. {N} mora na Av. Paulista, n.º {NUM}, junto à {M}.",
            "Veja p. ex. as págs. 12-14, cf. cap. {NUM}, sobre a {M}.",
            "A Cia. de teatro chegou à {M} às 20 h, i.e. atrasada.",
            "O prof. {N} nasceu no séc. XIX, em {M}.",
            "A {M}, a {M} e a {M} fecham às 17 h.",
            "Meu amigo o Dr. {N}, diretor da {M}, chega em {NUM} de out.",
            "Segundo a {A}, o Sr. {N} visitou a {M} em {YEAR}.",
            "O prof. {N} falou primeiro; a Dra. {N} respondeu às 16 h.",
            "O Sr. {N} alguma vez visitou a {M}?",
            "«Encontramo-nos na {M} às 17 h», disse a Dra. {N}.",
            "Convidados: o Sr. {N}, a Sra. {N}, o Dr. {N}, etc.",
            "A {M} (fund. {YEAR}) emprega aprox. {NUM} pessoas.",
            "A {M} abre às 9 h, i.e. antes da {M}.",
            "São aprox. 3,5 km da {M} até a {M}.",
            "A {M} — segundo o Dr. {N} — reabriu em {YEAR}, cf. pág. {NUM}.",
        ],
    },
    "rus": {
        "mwes": ["Эйфелева башня", "Организация Объединённых Наций", "Гарри Поттер",
                 "Красный Крест", "Нью-Йорк Таймс", "Московский государственный университет",
                 "Красная площадь", "Зимний дворец", "Большой театр"],
        "names": [],
        "templates": [
            "Г-н {N} встретил г-жу {N} у {M} в 15 ч., т. е. днём.",
            "Д-р {N} живёт на ул. Ленина, д. {NUM}, кв. {NUM}, рядом с {M}.",
            "Он купил хлеб, молоко и т. д. по дороге к {M}.",
            "В {YEAR} г. компания выросла до {NUM} тыс. человек, пишет {M}.",
            "См. напр. стр. 12–14 и др. источники о {M}.",
            "{M} и {M} закрываются в 17 ч.",
            "Мой друг д-р {N}, директор {M}, приедет {NUM} окт.",
            "По данным {M}, г-н {N} посетил {M} в {YEAR} г.",
            "Проф. {N} выступил первым; д-р {N} ответил в 16 ч.",
            "Бывал ли г-н {N} когда-нибудь у {M}?",
            "«Встретимся у {M} в 17 ч.», — сказал д-р {N}.",
            "Гости: г-н {N}, г-жа {N}, д-р {N} и др.",
            "{M} (осн. в {YEAR} г.) насчитывает ок. {NUM} сотрудников.",
            "{M} открывается в 9 ч., т. е. раньше {M}.",
            "От {M} до {M} ок. 3,5 км.",
            "{M} — по словам д-ра {N} — открылась в {YEAR} г., см. стр. {NUM}.",
        ],
    },
    "hin": {
        "mwes": ["ताज महल", "संयुक्त राष्ट्र", "हैरी पॉटर", "एफिल टावर",
                 "दिल्ली विश्वविद्यालय", "लाल किला", "गेटवे ऑफ इंडिया", "इंडिया गेट"],
        "names": ["राम", "सीता", "अर्जुन", "मीरा", "कृष्ण", "राधा", "विजय", "अनु"],
        "templates": [
            "डॉ. {N} सुबह 9 बजे {M} पहुँचे।",
            "प्रो. {N} ने पृ. {NUM} पर {M} का हवाला दिया।",
            "श्री {N} और डॉ. {N} कल {M} देखने जाएँगे।",
            "{M} और {M} शाम 5 बजे बंद हो जाते हैं।",
            "मेरे मित्र डॉ. {N}, {M} के निदेशक, {NUM} अक्तूबर को आएँगे।",
            "{N} ने {YEAR} में {M} की यात्रा की।",
            "प्रो. {N} ने पहले बात की; डॉ. {N} ने शाम 4 बजे जवाब दिया।",
            "क्या श्री {N} कभी {M} गए हैं?",
            "\"शाम 5 बजे {M} पर मिलते हैं,\" डॉ. {N} ने कहा।",
            "अतिथि: श्री {N}, डॉ. {N}, प्रो. {N} आदि।",
            "{M} ({YEAR} में स्थापित) में लगभग {NUM} लोग काम करते हैं।",
            "{M} सुबह 9 बजे खुलता है, यानी {M} से पहले।",
            "{M} से {M} तक लगभग 3.5 कि.मी. है।",
        ],
    },
    "jpn": {
        "mwes": ["ハリー・ポッター", "エッフェル塔", "東京大学", "国際連合",
                 "ニューヨーク・タイムズ", "万里の長城", "自由の女神", "赤十字"],
        "names": ["田中", "佐藤", "鈴木", "山田", "高橋", "伊藤"],
        "templates": [
            "{N}さんは{M}をN.A.S.A.の友人と見学した。",
            "会議は午前9時、つまりA.M.9時に{M}で始まった。",
            "{N}教授は{M}について第{NUM}章で述べた。",
            "{M}と{M}は午後5時に閉まる。",
            "{N}さんは{YEAR}年に{M}を訪れた。",
            "友人の{N}博士は{M}の所長だ。",
            "{N}教授が先に話し、{N}博士が午後4時に答えた。",
            "{N}さんは{M}に行ったことがありますか。",
            "「午後5時に{M}で会いましょう」と{N}博士は言った。",
            "出席者：{N}さん、{N}教授、{N}博士など。",
            "{M}（{YEAR}年設立）には約{NUM}人が働いている。",
            "{M}は午前9時に開く。つまり{M}より早い。",
            "{M}から{M}まで約3.5kmだ。",
        ],
    },
    "kor": {
        "mwes": ["해리 포터", "에펠탑", "서울 대학교", "국제 연합",
                 "뉴욕 타임스", "만리장성", "자유의 여신상", "적십자"],
        "names": ["민준", "서연", "지훈", "하은", "도윤", "지우"],
        "templates": [
            "{N}은 {M}에서 A.M. 9시에 일을 시작했다.",
            "{N} 박사는 N.A.S.A.와 {M}에서 근무했다.",
            "{N} 교수는 제{NUM}장에서 {M}를 인용했다.",
            "{M}와 {M}는 오후 5시에 문을 닫는다.",
            "{N}은 {YEAR}년에 {M}를 방문했다.",
            "내 친구 {N} 박사는 {M}의 소장이다.",
            "{N} 교수가 먼저 말했고, {N} 박사가 오후 4시에 답했다.",
            "{N}은 {M}에 가 본 적이 있나요?",
            "\"오후 5시에 {M}에서 만나요\"라고 {N} 박사가 말했다.",
            "참석자: {N}, {N} 교수, {N} 박사 등.",
            "{M}({YEAR}년 설립)에는 약 {NUM}명이 일한다.",
            "{M}는 오전 9시에 연다. 즉 {M}보다 이르다.",
            "{M}에서 {M}까지 약 3.5km이다.",
        ],
    },
}

CASED = {"eng", "deu", "fra", "ita", "spa", "por", "rus"}


def mine(pool_path, limit=120000):
    """(names, mwes) from capitalized non-initial words / 2-3 word runs in the pool."""
    names, mwes = set(), set()
    word_re = re.compile(r"^[^\W\d_]+$", re.UNICODE)
    with open(pool_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                sent = json.loads(line)["sentence"]
            except (json.JSONDecodeError, KeyError):
                continue
            words = sent.split()
            run = []
            for w in words[1:]:
                core = w.strip(".,;:!?»«\"'()[]—–-")
                cap = (core and word_re.match(core) and core[0].isupper()
                       and 2 <= len(core) <= 14)
                if cap:
                    run.append(core)
                else:
                    if len(run) == 1 and run[0][1:].islower():
                        names.add(run[0])
                    elif 2 <= len(run) <= 3:
                        mwes.add(" ".join(run))
                    run = []
            if len(run) == 1 and run[0][1:].islower():
                names.add(run[0])
            elif 2 <= len(run) <= 3:
                mwes.add(" ".join(run))
    return sorted(names), sorted(mwes)


def fill(template, rng, names, mwes):
    out = template
    while "{N}" in out:
        out = out.replace("{N}", rng.choice(names), 1)
    while "{M}" in out:
        out = out.replace("{M}", rng.choice(mwes), 1)
    while "{A}" in out:
        out = out.replace("{A}", rng.choice(ACRONYMS), 1)
    while "{NUM}" in out:
        out = out.replace("{NUM}", str(rng.randint(2, 99)), 1)
    while "{YEAR}" in out:
        out = out.replace("{YEAR}", str(rng.randint(1950, 2025)), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-dir", default="data/big")
    ap.add_argument("--out-dir", default="data/aug_sentences")
    ap.add_argument("--per-lang", type=int, default=8000)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260724)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in LANGS:
        pack = PACKS[lang]
        rng = random.Random(args.seed + hash(lang) % 1000)
        names, mwes = list(pack["names"]), list(pack["mwes"])
        if lang in CASED:
            pool = Path(args.big_dir) / lang / "target_language_sentences_tokenization.jsonl"
            mined_names, mined_mwes = mine(pool)
            names += mined_names
            # mined runs are noisier than the curated list; keep curated dominant-ish
            mwes += mined_mwes[: len(mined_mwes) // 2] or mined_mwes
        if not names:
            names = ["Alex"]

        total = args.per_lang + args.holdout
        seen = set()
        # cap attempts: template space is combinatorial, but dedup can saturate for the
        # fixed-list languages — don't loop forever
        attempts = 0
        while len(seen) < total and attempts < total * 30:
            attempts += 1
            seen.add(fill(rng.choice(pack["templates"]), rng, names, mwes))
        sents = sorted(seen)
        rng.shuffle(sents)
        holdout, train = sents[: args.holdout], sents[args.holdout:]
        (out_dir / f"{lang}.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
        (out_dir / f"{lang}_holdout.txt").write_text("\n".join(holdout) + "\n", encoding="utf-8")
        print(f"{lang}: {len(train)} train + {len(holdout)} holdout "
              f"(names={len(names)} mwes={len(mwes)}, attempts={attempts})")


if __name__ == "__main__":
    main()

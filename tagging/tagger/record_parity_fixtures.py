"""Record parity fixtures from the live parsley endpoint.

Fetches reference outputs for a fixed multilingual sentence set (all 10 languages, plus
regression cases like eng "love" for the training-prior candidate selection) and writes
them to `lexide/tests/fixtures/parsley_reference.json`, where `tests/parsley_parity.rs`
asserts the Rust local pipeline reproduces every token exactly.

Run AFTER deploying the serve you want to pin (release.sh does this in order):

    python3 tagger/record_parity_fixtures.py            # from tagging/
    python3 tagger/record_parity_fixtures.py --url https://... --out path.json
"""
import argparse
import json
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://anchpop--lexide-parsley-parsley-tag.modal.run"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "lexide/tests/fixtures/parsley_reference.json"

SENTENCES = {
    "deu": ["Eine Fundgrube.", "Die Kinder spielten gestern im Garten.",
            "Ich weiß, dass es 3,5 km sind."],
    "eng": ["The cats were sleeping.", "She had already finished her homework.",
            "Don't touch that — it's mine!",
            "I love programming."],  # love/lofe: the training-prior candidate-selection case
    "fra": ["L'homme n'est pas venu, n'est-ce pas ?", "Les oiseaux chantaient dans les arbres.",
            "Je voudrais un café, s'il vous plaît."],
    "spa": ["¿Dónde está la biblioteca?", "Los niños corrieron hacia la playa.",
            "Me gustaría viajar a España el año que viene."],
    "ita": ["I gatti dormivano sul divano.", "Domani andremo al mare con gli amici."],
    "por": ["Vamos à praia amanhã!", "As crianças brincavam no parque."],
    "rus": ["Я им доверяю — правда.", "Дети играли в парке вчера вечером."],
    "kor": ["고양이가 좋아요.", "아이들이 공원에서 놀고 있었어요."],
    "hin": ["मुझे बिल्लियाँ पसंद हैं।", "बच्चे कल बगीचे में खेल रहे थे।"],
    "jpn": ["私は猫が好きです。", "子供たちは公園で遊んでいました。"],
    "tha": ["ผมชอบแมวมาก", "เด็กๆ เล่นอยู่ในสวนเมื่อวานนี้"],
    "zho-hans": ["我喜欢猫。", "孩子们昨天在公园里玩。"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = []
    for lang, sents in SENTENCES.items():
        body = json.dumps({"sentences": sents, "lang": lang}).encode()
        req = urllib.request.Request(args.url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            results = json.load(r)["results"]
        for sent, toks in zip(sents, results):
            out.append({"lang": lang, "text": sent, "tokens": toks})
        print(f"[fixtures] {lang}: {[len(t) for t in results]} tokens")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[fixtures] wrote {len(out)} sentences to {args.out}")


if __name__ == "__main__":
    main()

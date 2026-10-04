"""Record Rust parity fixtures from CPU fp32 predict_joint, never the GPU serve."""
import argparse
import json
from pathlib import Path
import torch
from predict_joint import load_checkpoint, predict_batch
from verify_joint_onnx import sample_records
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "lexide/tests/fixtures/parsley_reference.json"

# Includes French narrow/NBSP gaps and a supplementary-plane Japanese character.
SENTENCES = {
    "deu": ["Eine Fundgrube.", "Die Kinder spielten gestern im Garten.",
            "Ich weiß, dass es 3,5 km sind."],
    "eng": ["The cats were sleeping.", "She had already finished her homework.",
            "Don't touch that — it's mine!",
            "I love programming... really?!"],
    "fra": ["L'homme n'est pas venu, n'est-ce pas ?", "Les oiseaux chantaient dans les arbres.",
            "Je voudrais un café ; s'il vous plaît !"],
    "spa": ["¿Dónde está la biblioteca?", "Los niños corrieron hacia la playa.",
            "Me gustaría viajar a España el año que viene."],
    "ita": ["I gatti dormivano sul divano.", "Domani andremo al mare con gli amici."],
    "por": ["Vamos à praia amanhã!", "As crianças brincavam no parque."],
    "rus": ["Я им доверяю — правда.", "Дети играли в парке вчера вечером."],
    "kor": ["고양이가 좋아요.", "아이들이 공원에서 놀고 있었어요."],
    "hin": ["मुझे बिल्लियाँ पसंद हैं।", "बच्चे कल बगीचे में खेल रहे थे।"],
    "jpn": ["私は猫が好きです。", "\U00020bb7野家で食べる。"],
    "tha": ["ผมชอบแมวมาก", "เด็กๆ เล่นอยู่ในสวนเมื่อวานนี้"],
    "zho-hans": ["我喜欢猫。", "孩子们昨天在公园里玩。"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test", default=str(DEFAULT_OUT.parents[3] / "data/processed-joint/test.jsonl"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, tokenizer, vocab = load_checkpoint(args.checkpoint, "cpu")
    records = sample_records([json.loads(line) for line in Path(args.test).read_text().splitlines()], 21)
    for lang, texts in SENTENCES.items():
        records.extend(dict(lang=lang, text=text) for text in texts)
        records.append(dict(lang=lang, text=""))
    out = []
    for record in records:
        predictions, _, _ = predict_batch(model, tokenizer, vocab, [record], "cpu")
        p = predictions[0]
        for i, token in enumerate(p["tokens"]):
            token["text"] = p["text"][token["start"]:token["end"]]
            next_start = p["tokens"][i+1]["start"] if i+1 < len(p["tokens"]) else len(p["text"])
            token["whitespace"] = p["text"][token["end"]:next_start]
        out.append(p)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[fixtures] wrote {len(out)} sentences to {args.out}")


if __name__ == "__main__":
    main()

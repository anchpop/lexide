"""Token-span F1 of one or more char-tokenizer checkpoints on the augmented holdout
(data/aug_holdout/<lang>.jsonl, raw silver schema — never seen in training).

    .venv-seg/bin/python tagger/eval_tokenizer_holdout.py \
        --ckpt old=/path/to/old.pt new=/path/to/new.pt [--holdout-dir data/aug_holdout]

Reports per-language and overall F1 per checkpoint (with language token where the
checkpoint supports it), plus how often a checkpoint drops text entirely (chars of
non-whitespace outside every predicted token — the worst failure mode).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import normalize_sentence  # noqa: E402
from dataset import encode_bytes_and_labels  # noqa: E402
from predict import byte_encode, load_char_model, spans_from_byte_labels  # noqa: E402
from train_tokenizer import spans_from_labels  # noqa: E402


@torch.no_grad()
def evaluate(model, records, device):
    tp = fp = fn = 0
    dropped_chars = 0
    total_chars = 0
    for lang, text, tokens in records:
        gold_ids, gold_labels = encode_bytes_and_labels(text, tokens, max_bytes=512)
        x = torch.tensor([byte_encode(text, lang, model)[:512]], device=device)
        pred = model(x)[0].argmax(-1).tolist()
        gold_spans = spans_from_labels(gold_labels)
        pred_spans = spans_from_labels(pred)
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
        # text-drop measure: non-space chars not covered by any predicted char span
        covered = set()
        for s, e in spans_from_byte_labels(text, pred):
            covered.update(range(s, e))
        for i, ch in enumerate(text):
            if ch.isspace():
                continue
            total_chars += 1
            if i not in covered:
                dropped_chars += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"f1": round(f1 * 100, 2), "prec": round(prec * 100, 2),
            "rec": round(rec * 100, 2),
            "dropped_char_pct": round(dropped_chars / max(1, total_chars) * 100, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--holdout-dir", default="data/aug_holdout")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    per_lang = {}
    for path in sorted(Path(args.holdout_dir).glob("*.jsonl")):
        lang = path.stem
        # Leakage guard: regenerations can put a sentence in one run's train file and a
        # later run's holdout — drop any holdout sentence present in the train jsonl.
        trained = set()
        train_path = Path("data/big") / lang / \
            "target_language_sentences_tokenization_augmented.jsonl"
        if train_path.exists():
            for line in train_path.open(encoding="utf-8"):
                try:
                    trained.add(json.loads(line)["sentence"])
                except (json.JSONDecodeError, KeyError):
                    pass
        records = []
        leaked = 0
        for line in path.open(encoding="utf-8"):
            obj = json.loads(line)
            if obj.get("sentence") in trained:
                leaked += 1
                continue
            norm = normalize_sentence(obj)
            if norm is None:
                continue
            text, toks = norm
            records.append((lang, text, [{"start": t["start"], "end": t["end"]} for t in toks]))
        if leaked:
            print(f"[{lang}] dropped {leaked} holdout sentences that leaked into train")
        per_lang[lang] = records

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model = load_char_model(path, device)
        print(f"\n=== {name} ({path}) vocab={model.emb.num_embeddings} ===")
        all_records = []
        for lang, records in sorted(per_lang.items()):
            m = evaluate(model, records, device)
            all_records.extend(records)
            print(f"  {lang}: {m}")
        print(f"  OVERALL: {evaluate(model, all_records, device)}")


if __name__ == "__main__":
    main()

"""System-independent exact-span scoring. Offsets are Python/Unicode character indices."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

METRICS = ("token", "pos", "lemma", "uas", "las")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keyed(records):
    result = {}
    for r in records:
        key = (r["lang"], r["text"])
        if key in result:
            raise ValueError(f"Duplicate sentence: {key!r}")
        result[key] = r
    return result


def token_keys(record):
    tokens, text = record["tokens"], record["text"]
    spans = [(t["start"], t["end"]) for t in tokens]
    previous = 0
    for start, end in spans:
        if not (previous <= start < end <= len(text)):
            raise ValueError(f"Invalid/overlapping token span {(start, end)}")
        previous = end
    keys = {name: set() for name in METRICS}
    for t, span in zip(tokens, spans):
        head = t["head"]
        if not isinstance(head, int) or not 0 <= head <= len(tokens):
            raise ValueError(f"Invalid head {head}")
        head_span = spans[head - 1] if head else None
        keys["token"].add(span)
        keys["pos"].add((span, t["pos"]))
        keys["lemma"].add((span, t["lemma"]))
        keys["uas"].add((span, head_span))
        keys["las"].add((span, head_span, t["dep"]))
    return keys


def score(gold, predictions):
    gold, predictions = keyed(gold), keyed(predictions)
    extra = predictions.keys() - gold.keys()
    if extra:
        raise ValueError(f"{len(extra)} predictions have no gold sentence")
    counts = defaultdict(lambda: {m: [0, 0, 0] for m in METRICS})
    missing = defaultdict(int)
    for key, record in gold.items():
        g = token_keys(record)
        if key not in predictions:
            missing[key[0]] += 1
        p = token_keys(predictions.get(key, {**record, "tokens": []}))
        for m in METRICS:
            c = counts[key[0]][m]
            c[0] += len(g[m] & p[m])
            c[1] += len(p[m])
            c[2] += len(g[m])
    result = {"languages": {}, "missing_sentences": dict(missing)}
    for lang, metrics in sorted(counts.items()):
        result["languages"][lang] = {}
        for name, (tp, predicted, actual) in metrics.items():
            result["languages"][lang][name] = {
                "precision": tp / predicted if predicted else 0.,
                "recall": tp / actual if actual else 0.,
                "f1": 2 * tp / (predicted + actual) if predicted + actual else 0.,
                "correct": tp, "predicted": predicted, "gold": actual,
            }
    langs = result["languages"]
    result["macro"] = {m: {k: sum(v[m][k] for v in langs.values()) / max(1, len(langs))
                            for k in ("precision", "recall", "f1")} for m in METRICS}
    return result


def markdown(result):
    lines = ["| Language | Token F1 | POS F1 | Lemma F1 | UAS F1 | LAS F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for lang, values in list(result["languages"].items()) + [("**macro**", result["macro"])]:
        lines.append("| " + lang + " | " + " | ".join(f"{values[m]['f1'] * 100:.2f}" for m in METRICS) + " |")
    if result["missing_sentences"]:
        lines.append("\nMissing predictions (counted as false negatives): " + json.dumps(result["missing_sentences"]))
    return "\n".join(lines) + "\n"


def write_scores(result, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    path.with_suffix(".md").write_text(markdown(result))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True, help="Output stem for .json and .md")
    args = ap.parse_args()
    result = score(read_jsonl(args.gold), read_jsonl(args.predictions))
    write_scores(result, args.output)
    print(markdown(result))

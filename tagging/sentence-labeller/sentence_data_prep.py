"""Convert LLM/mechanically labelled passages into per-example sentence spans.

The sentence-labeller emits, for each passage, an ordered list of `sections`:
`sentence` sections (one complete sentence, punctuation included) and `gap` sections
(whitespace, headings, separators — text belonging to no sentence). We flatten those
into character spans and re-use the exact schema the byte-boundary tagger already trains
on: `{text, tokens: [{start, end}, ...]}`, where here each "token" is one *sentence*
span. The byte model then learns per-byte O/B/I with B = sentence start, I = inside a
sentence, O = gap — the same shape as the token boundary tagger, just at sentence scale.

Split policy: the mechanically-augmented passages (`mechanical-augmented.jsonl`) are cheap
synthetic composition — good for teaching gap/wrapper/leader variety, but not a quality
yardstick, so they go entirely into train. The LLM-labelled passages (real Harry Potter
text + synthetic prose) are the true signal, so val/test are drawn only from them
(80/10/10 by a deterministic id hash), giving a meaningful held-out sentence-splitting F1.
"""
import argparse
import hashlib
import json
import os
from collections import Counter

# (path relative to this file's dir, is_real). Real sources feed val/test; synthetic → train.
SOURCES = [
    ("data/labelled-all-languages.jsonl", True),
    ("data/labelled-large.jsonl", True),
    ("data/synthetic-labelled.jsonl", True),
    ("data/labelled.jsonl", True),
    ("data/mechanical-augmented.jsonl", False),
]


def sentence_spans(sections):
    """sections -> (text, [(start, end) char spans of sentence sections]).

    Character indices are code points (Python str indexing), matching
    dataset.encode_bytes_and_labels which slices `text[s:e]` and enumerates chars.
    """
    spans = []
    cursor = 0
    parts = []
    for sec in sections:
        content = sec["content"]
        parts.append(content)
        length = len(content)
        if sec["type"] == "sentence":
            spans.append((cursor, cursor + length))
        cursor += length
    return "".join(parts), spans


def bucket(example_id, val_frac, test_frac):
    """Deterministic per-id split: hash -> [0,1) -> {train,val,test}."""
    h = hashlib.md5(example_id.encode("utf-8")).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < test_frac:
        return "test"
    if x < test_frac + val_frac:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    stats = {"train": Counter(), "val": Counter(), "test": Counter()}
    skipped = 0
    seen_text = set()

    for rel, is_real in SOURCES:
        path = os.path.join(args.data_dir, rel)
        if not os.path.exists(path):
            print(f"[skip] {rel} not found")
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sections = r.get("sections")
                if not sections:
                    skipped += 1
                    continue
                text, spans = sentence_spans(sections)
                # Reconstruction must be exact, else the char spans are meaningless.
                if text != r.get("text", text):
                    skipped += 1
                    continue
                if not text or not spans:
                    skipped += 1
                    continue
                # Drop exact-duplicate passages (keep first) to avoid train/eval leakage.
                if text in seen_text:
                    continue
                seen_text.add(text)

                rec = {
                    "id": r["id"],
                    "lang": r.get("lang", "und"),
                    "text": text,
                    "tokens": [{"start": s, "end": e} for s, e in spans],
                }
                split = "train" if not is_real else bucket(r["id"], args.val_frac, args.test_frac)
                splits[split].append(rec)
                stats[split][rec["lang"]] += 1
                n += 1
        print(f"[read] {rel}: {n} usable ({'real' if is_real else 'synthetic'})")

    for split, recs in splits.items():
        recs.sort(key=lambda r: r["id"])
        out = os.path.join(out_dir, f"{split}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_sent = sum(len(r["tokens"]) for r in recs)
        print(f"[write] {split}: {len(recs)} passages, {n_sent} sentences -> {out}")
        print(f"        by lang: {dict(sorted(stats[split].items()))}")
    if skipped:
        print(f"[note] skipped {skipped} records (no sections / bad reconstruction / empty)")


if __name__ == "__main__":
    main()

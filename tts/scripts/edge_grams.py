"""Build English edge-case scripts from data/edge and the yap TTS rows.

    python scripts/edge_grams.py data/eng_sentences.jsonl

The written form finds yap's grams. If it differs from what was recorded, or yap lacks a
sentence, phonemize the spoken form with g2p-jsonl (cargo build --release in g2p-jsonl/).
"""
import collections
import hashlib
import itertools
import json
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path


def curl(t):
    out, open_ = [], True
    for c in t:
        if c == '"': out.append("“" if open_ else "”"); open_ = not open_
        else: out.append(c)
    return "".join(out)
def apos(t):  # ’ between letters, or after s before a non-letter (possessive), is an apostrophe
    t = re.sub(r"(?<=[A-Za-z])’(?=[A-Za-z])", "'", t)
    t = re.sub(r"(?<=s)’(?=[\s,.;:!?—]|$)", "'", t) if "‘" not in t else t
    return re.sub(r"(?<=\s)’(?=(?:cause|em|n|til|tis|twas|round|bout)\b)", "'", t)
def split(t):
    parts, depth, start = [], 0, 0
    for i, c in enumerate(t):
        if c == "“": depth += 1
        elif c == "”": depth = max(0, depth - 1)
        if c in ".!?…" and depth == 0:
            j = i + 1
            while j < len(t) and t[j] in ".!?…”’)": j += 1
            if j < len(t) and t[j] == " " and j + 1 < len(t) and (t[j+1].isupper() or t[j+1] in "“‘\"("):
                if not re.search(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|e\.g|i\.e|Ave|Mt|No|Prof|Gen|Col|Capt|Lt|U\.S|a\.m|p\.m|[A-Z])\.$", t[start:j]):
                    parts.append(t[start:j].strip()); start = j + 1
    parts.append(t[start:].strip())
    return [p for p in parts if p]


def select():
    def key(text):
        return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

    scripts = Path("data/scripts")
    seen = {key(json.loads(line)["text"]) for path in scripts.glob("eng_*.jsonl")
            if not path.name.startswith("eng_edge_") for line in path.read_text().splitlines()}
    rows = collections.defaultdict(list)
    for seed, kinds in [(7, ("heteronyms", "punctuation", "intonation", "rare")), (8, ("books", "nonfiction"))]:
        rng = random.Random(seed)
        for kind in kinds:
            for line in Path(f"data/edge/{kind}.jsonl").read_text().splitlines():
                source = json.loads(line)
                spoken = source.get("spoken") or source["text"]
                normalized = key(spoken)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                voice = "achernar" if rng.random() < 0.82 else rng.choice(("kore", "leda", "puck", "iapetus", "orus"))
                rows[voice].append({"id": hashlib.sha256(spoken.encode()).hexdigest()[:16],
                                    "text": spoken, "written": source["text"], "split": "train", "role": "edge",
                                    "kind": f"{kind}/{source['kind']}",
                                    **{k: source[k] for k in ("source", "heteronyms") if k in source}})
    return rows


def from_g2p(text, p):
    """One gram per g2p word. g2p's spans are UTF-8 byte offsets."""
    chars = [i for i, c in enumerate(text) for _ in c.encode()] + [len(text)]     # byte offset -> character offset
    grams, at = [], 0
    for (a, b), (s, e) in zip(p["word_spans"], ((chars[s], chars[e]) for s, e in p["word_text_spans"])):
        gap = text[at:s]
        marks = [c for c in gap if not c.isspace() and unicodedata.category(c)[0] == "P"]
        if marks: grams.append({"words": ["".join(marks)], "pos": ["PUNCT"], "count": 0, "phonemes": ""})
        tokens = []
        for i in range(a, b):
            stressed = p["stress"][i] in ("Primary", "Secondary") and not (i > a and p["stress"][i - 1] == p["stress"][i])
            tokens.append(("ˈ" if p["stress"][i] == "Primary" else "ˌ") * stressed + p["phonemes"][i])
        grams.append({"words": [text[s:e]], "pos": ["X"], "count": 0, "phonemes": " ".join(tokens)}); at = e
    marks = [c for c in text[at:] if not c.isspace() and unicodedata.category(c)[0] == "P"]
    if marks: grams.append({"words": ["".join(marks)], "pos": ["PUNCT"], "count": 0, "phonemes": ""})
    return grams


def initialisms(grams, written):
    """An A inside a spelled-out initialism takes its letter name."""
    def letter(gram):
        return len(gram["words"]) == 1 and bool(re.fullmatch("[A-Z]", gram["words"][0]))
    for letters, group in itertools.groupby(grams, letter):
        if letters:
            group = list(group)
            acronym = "".join(g["words"][0] for g in group)
            if len(acronym) >= 2 and re.search(r"\b" + acronym + r"\b", written):
                for g in group:
                    if g["words"] == ["A"] and g["phonemes"] == "ə":
                        g["phonemes"] = "ˈeɪ"


def main(rows_path):
    scripts = select()
    parts = {r["written"]: split(apos(curl(r["written"]))) for rows in scripts.values() for r in rows}
    wanted = {sentence for sentences in parts.values() for sentence in sentences}
    with open(rows_path, encoding="utf-8") as stream:
        pack = {r["text"]: r["grams"] for r in map(json.loads, stream) if r["text"] in wanted}

    def has_grams(row):
        return row["text"] == row["written"] and all(s in pack for s in parts[row["written"]])

    missing = [r for rows in scripts.values() for r in rows if not has_grams(r)]
    binary = Path(__file__).resolve().parents[1] / "g2p-jsonl/target/release/g2p-jsonl"
    requests = ({"id": r["id"], "language": "eng", "text": r["text"]} for r in missing)
    out = subprocess.run([binary], input="".join(json.dumps(r) + "\n" for r in requests),
                         capture_output=True, text=True, check=True).stdout
    fallback = {r["id"]: r.get("g2p", {}) for r in map(json.loads, out.splitlines())}
    counts = collections.Counter()
    for voice, rows in scripts.items():
        with open(f"data/scripts/eng_edge_{voice}.jsonl", "w", encoding="utf-8") as stream:
            for r in rows:
                if has_grams(r):
                    r["grams"] = [g for s in parts[r["written"]] for g in pack[s]]
                    r["grams_from"] = "yap"
                elif fallback[r["id"]].get("word_text_spans"):
                    r["grams"] = from_g2p(r["text"], fallback[r["id"]])
                    if r["kind"] == "rare/acronyms":
                        initialisms(r["grams"], r["written"])
                    r["grams_from"] = "g2p"
                else:
                    counts["dropped"] += 1
                    continue
                counts[r["grams_from"]] += 1
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(counts))


if __name__ == "__main__":
    main(sys.argv[1])

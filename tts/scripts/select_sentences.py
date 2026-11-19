"""Pick a TTS script from a pack dump: plain sentences of moderate length,
greedy on phone-bigram coverage, until an estimated duration is reached.

    python select_sentences.py data/eng_sentences.jsonl data/eng_script.jsonl --hours 1

``--extend`` grows an existing script: its sentences (and its validation
holdout) are kept, and new ones are picked for what they still add.
"""
import argparse, heapq, json, random, re, hashlib

from voca.text import bare_phones

parser = argparse.ArgumentParser()
parser.add_argument("dump")
parser.add_argument("output")
parser.add_argument("--hours", type=float, default=1.0, help="total, including an --extend script")
parser.add_argument("--chars-per-second", type=float, default=12.6, help="Gemini Achernar speaks ~12.6")
parser.add_argument("--holdout", type=int, default=60)
parser.add_argument("--extend", help="an earlier script to keep and grow")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

PLAIN = re.compile(r"^[A-Za-z][A-Za-z ,.'?!-]*[.?!]$")
rows, seen = [], set()
for line in open(args.dump):
    row = json.loads(line)
    text = row["text"]
    words = len(text.split())
    if not 4 <= words <= 25 or not PLAIN.match(text) or text in seen:
        continue
    seen.add(text)
    phones = bare_phones(row["grams"])
    row["bigrams"] = set(zip(["#"] + phones, phones + ["#"]))
    rows.append(row)
random.Random(args.seed).shuffle(rows)
print(f"{len(rows)} candidate sentences")

chosen, holdout = [], rows[60000:60000 + args.holdout]
if args.extend:
    kept = [json.loads(l) for l in open(args.extend)]
    by_text = {r["text"]: r for r in rows}
    chosen = [by_text[r["text"]] for r in kept if r["split"] == "train"]
    holdout = [by_text[r["text"]] for r in kept if r["split"] == "validation"]
taken = {r["text"] for r in chosen + holdout}
covered = {}
for row in chosen:
    for b in row["bigrams"]:
        covered[b] = covered.get(b, 0) + 1

# Lazy greedy: a sentence's gain only falls as coverage grows, so a stale heap
# entry is an upper bound and only the top needs rescoring.
gain = lambda r: sum(1 / (1 + covered.get(b, 0)) for b in r["bigrams"]) / len(r["text"]) ** 0.5
heap = [(-gain(r), i) for i, r in enumerate(rows) if r["text"] not in taken]
heapq.heapify(heap)
budget, used = args.hours * 3600 * args.chars_per_second, sum(len(r["text"]) for r in chosen)
while used < budget and heap:
    _, i = heapq.heappop(heap)
    score = gain(rows[i])
    if heap and score < -heap[0][0]:
        heapq.heappush(heap, (-score, i))
        continue
    for b in rows[i]["bigrams"]:
        covered[b] = covered.get(b, 0) + 1
    chosen.append(rows[i])
    used += len(rows[i]["text"])
with open(args.output, "w") as out:
    for split, items in (("train", chosen), ("validation", holdout)):
        for row in items:
            row = {k: v for k, v in row.items() if k != "bigrams"}
            row["id"] = hashlib.sha256(row["text"].encode()).hexdigest()[:16]
            row["split"] = split
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{len(chosen)} train sentences (~{used / args.chars_per_second / 3600:.2f} h), {len(holdout)} validation, {len(covered)} bigrams covered")

"""Single words and short phrases for a language's TTS data, from its pack dump: what yap's lessons show on their own.

    python scripts/select_words.py data/deu_sentences.jsonl data/scripts/deu_words.jsonl --count 6000

Candidates are the pack's grams of one to four words (with the phonemes yap sends), written as they stand in a
sentence of the dump: in the language's script, without digits, punctuation or proper nouns, every phone known to the aligner.
Chosen: the most common ones (what a learner meets first), then others at random for variety. ``--phrases`` is the
share of multi-word ones; ``--full-stop`` the share written with a full stop (a learner also hears a word ending a
sentence). ``--holdout`` texts are validation.
"""
import argparse, collections, hashlib, json, pathlib, random, re, sys

from select_sentences import plain
from voca.build import FrameMatrixRecognizer
from voca.text import bare_phones

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("dump", type=pathlib.Path)
parser.add_argument("output", type=pathlib.Path)
parser.add_argument("--count", type=int, default=6000)
parser.add_argument("--phrases", type=float, default=0.35)
parser.add_argument("--full-stop", type=float, default=0.3)
parser.add_argument("--holdout", type=int, default=100)
parser.add_argument("--common", type=float, default=0.6, help="share taken from the most common")
parser.add_argument("--aligner-labels", type=pathlib.Path, default=pathlib.Path("data/eng_gemini_achernar/frames/0000eef7710dae6d.json"))
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
language = args.dump.name.removesuffix("_sentences.jsonl")
aligner = FrameMatrixRecognizer(json.loads(args.aligner_labels.read_text()))
rng = random.Random(args.seed)


def alignable(gram):
    try:
        [aligner.label(p) for p in bare_phones([gram])]
        return True
    except KeyError:
        return False


seen, surface, grams = collections.Counter(), {}, {}
for line in open(args.dump, encoding="utf-8"):
    sentence = json.loads(line)
    for gram in sentence["grams"]:
        words = tuple(gram["words"])
        if not 1 <= len(words) <= 4 or not gram["phonemes"] or not all(any(c.isalnum() for c in w) for w in words) or "PROPN" in gram["pos"]:
            continue
        seen[words] += 1
        if words not in grams:
            grams[words] = gram
        if words not in surface:
            match = re.search(r"\s*".join(map(re.escape, words)), sentence["text"], re.IGNORECASE)
            if match:
                surface[words] = match.group()
pool = {"word": [], "phrase": []}
for words, _ in seen.most_common():
    text = surface.get(words)
    if text and plain(text, language) and not any(c in ",.?!;:…" for c in text) and len(text) > 1 and alignable(grams[words]):
        pool["word" if len(words) == 1 else "phrase"].append(words)


def pick(candidates, count):
    """The first ``common`` share, the rest at random from the next three times as many (rarer words stay out)."""
    common = int(count * args.common)
    rest = candidates[common:common + 3 * count]
    return candidates[:common] + rng.sample(rest, min(count - common, len(rest)))


phrases = min(int(args.count * args.phrases), len(pool["phrase"]))
chosen = [("phrase", w) for w in pick(pool["phrase"], phrases)] + [("word", w) for w in pick(pool["word"], args.count - phrases)]
rows, used = [], set()
stop = "。" if language in ("jpn", "zho-hans") else "."
rng.shuffle(chosen)
for role, words in chosen:
    text, gram = surface[words], grams[words]
    if text.casefold() in used:
        continue
    used.add(text.casefold())
    full = rng.random() < args.full_stop
    entry = [{k: gram[k] for k in ("words", "pos", "count", "phonemes")}]
    if full:
        text += stop
        entry.append({"words": [stop], "pos": ["PUNCT"], "count": 0, "phonemes": ""})
    rows.append({"id": hashlib.sha256(("word " + text).encode()).hexdigest()[:16], "text": text, "grams": entry, "split": "train", "role": role})
for row in rng.sample(rows, args.holdout):
    row["split"] = "validation"
args.output.parent.mkdir(parents=True, exist_ok=True)
with open(args.output, "w", encoding="utf-8") as stream:
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{language}: {len(rows)} rows ({sum(r['role'] == 'phrase' for r in rows)} phrases), {len(pool['word'])} words and {len(pool['phrase'])} phrases to choose from")
print("  " + " | ".join(r["text"] for r in rng.sample(rows, 24)), file=sys.stderr)

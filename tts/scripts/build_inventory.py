"""Write voca/inventory.json: every phone a whole pack dump uses.

    python build_inventory.py data/eng_sentences.jsonl voca/inventory.json --language eng

Append-only against an existing inventory: known phones keep their positions
(checkpoints depend on them), new ones are added at the end.
"""
import argparse, collections, json, pathlib, unicodedata

from voca.text import bare_phones

parser = argparse.ArgumentParser()
parser.add_argument("dump")
parser.add_argument("output", type=pathlib.Path)
parser.add_argument("--language", required=True)
args = parser.parse_args()

VOWEL_LETTERS = set("aeiouyæɑɒɐəɘɛɜɞɪʊʌɔøœɤɯɨʉɵᵻɚɝɶʏ")
VOICELESS_LETTERS = set("ptkqcfθsʃhxχçɕʂʔɸħɬ")
counts = collections.Counter()
for line in open(args.dump, encoding="utf-8"):
    counts.update(bare_phones(json.loads(line)["grams"]))

known = json.loads(args.output.read_text())["phones"] if args.output.exists() else []
phones = known + [p for p, _ in counts.most_common() if p not in known]
vowels = [p for p in phones if VOWEL_LETTERS & set(p) or "̩" in unicodedata.normalize("NFD", p)]
voiceless = [p for p in phones if p not in vowels and (p[0] in VOICELESS_LETTERS or "ʰ" in p)]
args.output.write_text(json.dumps({"language": args.language, "source": pathlib.Path(args.dump).name, "phones": phones,
                                   "vowels": vowels, "voiceless": voiceless, "counts": dict(counts)}, ensure_ascii=False, indent=1) + "\n")
print(f"{len(phones)} phones ({len(phones) - len(known)} new), {len(vowels)} vowels, {len(voiceless)} voiceless")
print("rare:", [p for p in phones if counts[p] < 20])

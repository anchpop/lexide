"""Write voca/inventories/<language>.json: every phone a language's whole pack dump uses.

    python scripts/build_inventory.py data/jpn_sentences.jsonl

The dump is named ``<language>_sentences.jsonl`` (yap's language code). Append-only
against an existing inventory: known phones keep their positions (checkpoints
depend on them), new ones are added at the end, most frequent first.

``always_voiced`` lists the phones whose frames training and synthesis mark
voiced whatever the pitch tracker says (the tracker misses creaky vowels; a
model trained on that drops whole vowels): every vowel g2p does not write as
devoiced (ring diacritic: Japanese ``ɯ̥ᵝ``). ``sonorants`` (nasals, liquids,
glides) are voiced too, but training only forces their frames voiced where the
audio is voiced-like (``voca.data``): they devoice more often than vowels do.
"""
import argparse, collections, json, pathlib, unicodedata

from voca.text import bare_phones

parser = argparse.ArgumentParser()
parser.add_argument("dump", type=pathlib.Path)
parser.add_argument("--output", type=pathlib.Path, help="default: voca/inventories/<language>.json")
args = parser.parse_args()
language = args.dump.name.removesuffix("_sentences.jsonl")
output = args.output or pathlib.Path(__file__).parents[1] / "voca" / "inventories" / f"{language}.json"

VOWEL_LETTERS = set("aeiouyæɑɒɐəɘɛɜɞɪʊʌɔøœɤɯɨʉɵᵻɚɝɶʏ")
VOICELESS_LETTERS = set("ptʈkqcfθsʃhxχçɕʂʔɸħɬ")
SONORANT_LETTERS = set("mnŋɲɴɳlɫɭʎɹɻrɾɽwjɥʋɰ")     # nasals, liquids, glides
DEVOICED = ("̥", "̊")   # combining ring below / above
nfd = lambda p: unicodedata.normalize("NFD", p)   # g2p writes some nasal vowels precomposed (Portuguese ẽ)
counts = collections.Counter()
for line in open(args.dump, encoding="utf-8"):
    counts.update(bare_phones(json.loads(line)["grams"]))

known = json.loads(output.read_text())["phones"] if output.exists() else []
phones = known + [p for p, _ in counts.most_common() if p not in known]
vowels = [p for p in phones if VOWEL_LETTERS & set(nfd(p)) or "̩" in nfd(p)]
voiceless = [p for p in phones if p not in vowels and (p[0] in VOICELESS_LETTERS or "ʰ" in p)]
always_voiced = [p for p in vowels if not set(DEVOICED) & set(nfd(p))]
sonorants = [p for p in phones if p not in vowels and p not in voiceless and nfd(p)[0] in SONORANT_LETTERS and not set(DEVOICED) & set(nfd(p))]
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps({"language": language, "source": args.dump.name, "phones": phones, "vowels": vowels, "voiceless": voiceless,
                              "always_voiced": always_voiced, "sonorants": sonorants, "counts": dict(counts)}, ensure_ascii=False, indent=1) + "\n")
print(f"{language}: {len(phones)} phones ({len(phones) - len(known)} new), {len(vowels)} vowels, {len(voiceless)} voiceless, "
      f"{len(always_voiced)} always voiced; rare: {[p for p in phones if counts[p] < 20]}")

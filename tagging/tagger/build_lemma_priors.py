"""Build training-data lemma priors for Wiktionary candidate selection.

When a Wiktionary table has several candidate lemmas for a (form, POS) — e.g. eng
"love" -> ["lofe", "love"], where "lofe" is an obsolete homograph — the selection
should prefer the lemmatization our training data actually uses. This script builds
two prior levels from the silver training labels, per language and content POS,
keeping only what competes in some multi-candidate Wiktionary entry (the only place
a prior changes anything). Output sits next to the tables:

    data/lemma_tables/wikt_priors_{lang}.json
    {pos: {"forms": {form: {lemma: count}},   # how training lemmatized THIS form
           "lemmas": {lemma: count}}}          # overall lemma frequency (fallback)

Both consumers use the same rule — pick by (form-count desc, lemma-count desc,
|len(c)-len(form)|, c) — so the parsley serve (lemma_lookup.py) and the Rust fst
builder (build-lemma-fst) stay in token-for-token agreement:

    python tagger/build_lemma_priors.py   # from tagging/, after data_prep + tables
"""
import argparse
import collections
import json
import os

from data_prep import apply_script
from lemma_lookup import CONTENT_POS


def competing(tables_dir, lang):
    """The multi-candidate entries of this language's table: {pos: {form: set(cands)}}."""
    path = os.path.join(tables_dir, f"wikt_{lang}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    return {
        pos: {form: set(cands) for form, cands in table.get(pos, {}).items() if len(cands) > 1}
        for pos in CONTENT_POS
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--vocab", default="data/processed/vocab.json")
    ap.add_argument("--tables-dir", default="data/lemma_tables")
    args = ap.parse_args()

    with open(args.vocab, encoding="utf-8") as f:
        scripts = json.load(f)["lemma_scripts"]

    langs = sorted(
        n[len("wikt_"):-len(".json")]
        for n in os.listdir(args.tables_dir)
        if n.startswith("wikt_") and n.endswith(".json") and "priors" not in n
    )
    comp_by_lang = {lang: competing(args.tables_dir, lang) for lang in langs}
    # union of competing candidates per (lang, pos), for the O(1) lemma-frequency check
    comp_lemma_sets = {
        lang: {pos: set().union(*comp[pos].values()) if comp[pos] else set()
               for pos in CONTENT_POS}
        for lang, comp in comp_by_lang.items()
    }
    print(f"[priors] languages: {langs}")

    # Two prior levels over the training silver labels, restricted to what competes in the
    # Wiktionary table (the only place a prior changes anything):
    #   forms[pos][form][lemma]  how training lemmatized THIS form (strongest signal)
    #   lemmas[pos][lemma]       overall lemma frequency (homograph fallback)
    form_counts = {l: {p: collections.defaultdict(collections.Counter) for p in CONTENT_POS}
                   for l in langs}
    lemma_counts = {l: {p: collections.Counter() for p in CONTENT_POS} for l in langs}
    n_lines = 0
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if n_lines % 500_000 == 0:
                print(f"[priors] {n_lines} sentences...")
            rec = json.loads(line)
            lang = rec.get("lang")
            comp = comp_by_lang.get(lang)
            if comp is None:
                continue
            text = rec["text"]
            for tok in rec["tokens"]:
                pos = tok["pos"]
                if pos not in comp:
                    continue
                si = tok.get("lemma_script")
                if si is None or not (0 <= si < len(scripts)):
                    continue
                form = text[tok["start"]:tok["end"]]
                lemma = apply_script(form, scripts[si])
                cands = comp[pos].get(form)
                if cands and lemma in cands:
                    form_counts[lang][pos][form][lemma] += 1
                if lemma in comp_lemma_sets[lang][pos]:
                    lemma_counts[lang][pos][lemma] += 1

    for lang in langs:
        out = {}
        for pos in CONTENT_POS:
            fc = {form: dict(c) for form, c in form_counts[lang][pos].items()}
            lc = dict(lemma_counts[lang][pos])
            if fc or lc:
                out[pos] = {"forms": fc, "lemmas": lc}
        path = os.path.join(args.tables_dir, f"wikt_priors_{lang}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        n_forms = sum(len(p["forms"]) for p in out.values())
        n_lemmas = sum(len(p["lemmas"]) for p in out.values())
        print(f"[priors] {lang}: {n_forms} form priors, {n_lemmas} lemma priors "
              f"-> {os.path.getsize(path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()

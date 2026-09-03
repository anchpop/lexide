"""Find strings our own corpus segments more than one way, ranked by what it would cost
to canonicalize them.

Half the tokenizer's remaining errors in the whitespace-free languages are not errors
against a policy — they are the model reproducing one of the two policies the training data
itself uses for the same string. Measured on the byte-v14 disagreement blocks: 51% of
Japanese, 41-47% of Chinese and 41% of Thai errors land on a string the corpus writes both
ways, and in a third of all Japanese errors the model's form is the corpus *majority* while
the test label is the minority (`何か` x250 against `何|か` x1032).

This dumps that list. For every surface that appears somewhere as a single token, it counts
every way the corpus segments that same character run, so the output says: this string
occurs N times, M of them under a non-majority segmentation, and here is each variant.
Sorting by M gives the order to fix them in — M is exactly the number of token instances a
majority-vote canonicalization would flip.

    .venv-seg/bin/python tagger/find_ambiguity.py \
        --data-dir data/processed --out-dir data/ambiguity

Scope and caveats:

  * All three splits are scanned, because a canonicalization pass would rewrite all three.
    The per-split counts are kept so a fix can be checked for changing the eval as well as
    the training data.
  * Only runs written *contiguously* are compared. `New York` (a token containing a space)
    is not the same question as `不要` and is not counted here.
  * A string tracked here has to appear as one token at least once. A run that is only ever
    split, but split inconsistently (`食べ|させ|られ` against `食べさせ|られ`), is invisible to
    this — catching those needs every n-gram, not every n-gram that is also a word.
  * Japanese is compared after `merge_jpn_inflection`, i.e. against the policy the tokenizer
    is actually trained for, not against what is on disk.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import maybe_merge_inflection  # noqa: E402

SPLITS = ("train", "val", "test")
MAX_NGRAM = 6


def iter_records(data_dir, langs=None):
    """(split, lang, text, [(surface, start, end)]) over every split.

    The lang field is first in each record, so it is read off the raw line and the
    json.loads is skipped entirely for languages we are not asked about — worth ~3x on the
    2.2GB train file when a single language is requested.
    """
    for split in SPLITS:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                lang = line[10:line.index('"', 10)]
                if langs and lang not in langs:
                    continue
                r = json.loads(line)
                text = r["text"]
                toks = [(text[t["start"]:t["end"]], t["start"], t["end"])
                        for t in maybe_merge_inflection(r, True)]
                yield split, lang, text, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/ambiguity")
    ap.add_argument("--langs", default=None,
                    help="comma-separated subset (default: every language in the data)")
    ap.add_argument("--min-total", type=int, default=2,
                    help="skip surfaces occurring fewer than this many times in total")
    ap.add_argument("--max-ngram", type=int, default=MAX_NGRAM)
    args = ap.parse_args()
    langs = set(args.langs.split(",")) if args.langs else None

    # ---- pass 1: which character runs are a single token somewhere? -------------------
    whole = defaultdict(Counter)          # lang -> surface -> times written as one token
    for split, lang, _, toks in iter_records(args.data_dir, langs):
        c = whole[lang]
        for surface, _, _ in toks:
            c[surface] += 1
    for lang, c in sorted(whole.items()):
        print(f"[pass1] {lang}: {len(c)} token types, {sum(c.values())} tokens")

    # ---- pass 2: every segmentation of those same runs --------------------------------
    # forms[lang][surface][segmentation] = count, where a segmentation is the pieces joined
    # by "|" — the one-token case included, so the counts in a row sum to the run's total.
    forms = {lang: defaultdict(Counter) for lang in whole}
    per_split = {lang: defaultdict(Counter) for lang in whole}
    for split, lang, text, toks in iter_records(args.data_dir, langs):
        known = whole[lang]
        f, ps = forms[lang], per_split[lang]
        n = len(toks)
        for i in range(n):
            run = ""
            for j in range(i, min(i + args.max_ngram, n)):
                if j > i and toks[j][1] != toks[j - 1][2]:
                    break               # written apart — a different question
                run += toks[j][0]
                if run in known:
                    seg = "|".join(t[0] for t in toks[i:j + 1])
                    f[run][seg] += 1
                    ps[run][split] += 1

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for lang in sorted(forms):
        rows = []
        for surface, segs in forms[lang].items():
            if len(segs) < 2:
                continue
            total = sum(segs.values())
            if total < args.min_total:
                continue
            ordered = segs.most_common()
            majority, maj_n = ordered[0]
            rows.append({
                "surface": surface,
                "total": total,
                "minority": total - maj_n,
                "majority": majority,
                "majority_n": maj_n,
                "n_forms": len(segs),
                "forms": ordered,
                "splits": dict(per_split[lang][surface]),
            })
        rows.sort(key=lambda r: (-r["minority"], -r["total"], r["surface"]))

        path = os.path.join(args.out_dir, f"{lang}.tsv")
        with open(path, "w", encoding="utf-8") as out:
            out.write("# strings this corpus segments more than one way, worst first.\n")
            out.write("# minority = token instances that would change if the majority "
                      "form were made canonical.\n")
            out.write("# forms lists every observed segmentation as form:count, "
                      "majority first.\n")
            out.write("surface\ttotal\tminority\tmajority\tforms\tsplits\n")
            for r in rows:
                out.write(f"{r['surface']}\t{r['total']}\t{r['minority']}\t"
                          f"{r['majority']}\t"
                          f"{' '.join(f'{s}:{n}' for s, n in r['forms'])}\t"
                          f"{','.join(f'{k}={v}' for k, v in sorted(r['splits'].items()))}\n")

        tot_tokens = sum(whole[lang].values())
        minority = sum(r["minority"] for r in rows)
        # how much of the list is carried by its head — the "few hundred strings" claim
        head = sum(r["minority"] for r in rows[:200])
        summary[lang] = {
            "ambiguous_surfaces": len(rows),
            "minority_instances": minority,
            "pct_of_all_tokens": round(100 * minority / max(1, tot_tokens), 2),
            "top200_covers_pct_of_minority": round(100 * head / max(1, minority), 1),
            "path": path,
        }
        print(f"[write] {path}: {len(rows)} surfaces, {minority} minority instances "
              f"({summary[lang]['pct_of_all_tokens']}% of tokens); "
              f"top 200 cover {summary[lang]['top200_covers_pct_of_minority']}%")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

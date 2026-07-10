"""Does the Wiktionary lemma table actually help on out-of-training forms?

Holdout protocol (mimics deployment): build the training form-vocabulary from all but the
last HOLDOUT sentences of the raw silver; evaluate on tokens in the tail whose surface form
never appeared in training (OOV). For those OOV tokens we compare, against Gemma's lemma as
reference:
  - copy-the-form   : the naive floor (what the tagger falls back to on a form it can't handle)
  - wiktionary      : the table lemma when it covers (form, POS), else copy
so the question is simply: does table-else-copy beat copy on the OOV tail, and how often does
the table agree with Gemma where it does fire.
"""
import argparse
import json
from collections import Counter, defaultdict

BIG = "/data/coding/lexide/tagging/data/big"


def unwrap(v, k):
    return v[k] if isinstance(v, dict) else v


def load_raw(lang):
    rows = []  # (sentence_idx, form, pos, lemma)
    path = f"{BIG}/{lang}/target_language_sentences_tokenization.jsonl"
    for si, line in enumerate(open(path, encoding="utf-8")):
        o = json.loads(line)
        toks = []
        for t in o["tokens"]:
            form = unwrap(t["text"], "text")
            lem = unwrap(t.get("lemma"), "lemma")
            if form and lem:
                toks.append((form, t["pos"], lem))
        rows.append(toks)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="deu")
    ap.add_argument("--table", required=True)
    ap.add_argument("--holdout", type=int, default=20000)
    ap.add_argument("--dump-disagreements", type=int, default=40)
    ap.add_argument("--table-pos", default="NOUN,VERB,ADJ,ADV",
                    help="only apply the table for these POS (proper nouns should copy, not normalize)")
    args = ap.parse_args()
    TABLE_POS = set(args.table_pos.split(","))

    table = json.load(open(args.table, encoding="utf-8"))  # {pos: {form: [lemmas]}}
    sents = load_raw(args.lang)
    print(f"[{args.lang}] sentences={len(sents)}")
    train_sents = sents[:-args.holdout]
    eval_sents = sents[-args.holdout:]

    train_forms = set()
    for toks in train_sents:
        for form, pos, lem in toks:
            train_forms.add(form)
    print(f"train forms (vocab): {len(train_forms)}")

    def lookup(form, pos):
        cands = table.get(pos, {}).get(form)
        return cands  # list or None

    # evaluate on OOV eval tokens (skip punctuation/sym/num — lemma is trivial there)
    SKIP = {"PUNCT", "SYM", "NUM", "X", "SPACE"}
    stats = Counter()
    per_pos = defaultdict(Counter)
    disagreements = []
    covered_unique_vs_gemma = Counter()  # agree / disagree when table has a UNIQUE lemma
    for toks in eval_sents:
        for form, pos, gem in toks:
            if pos in SKIP:
                continue
            if form in train_forms:
                continue  # in-training, not OOV
            stats["oov_tokens"] += 1
            per_pos[pos]["n"] += 1
            copy_ok = (gem == form)
            stats["copy_correct"] += copy_ok
            per_pos[pos]["copy"] += copy_ok

            # DESIGN metric: use the table only for allowed (content) POS, else copy
            use_table = pos in TABLE_POS
            cands_design = lookup(form, pos) if use_table else None
            if cands_design:
                design_ok = (cands_design[0] == gem)
            else:
                design_ok = copy_ok
            stats["design_correct"] += design_ok
            per_pos[pos]["design"] += design_ok
            if not use_table:
                stats["nontable_tokens"] += 1

            cands = lookup(form, pos)
            if cands:
                stats["table_covered"] += 1
                per_pos[pos]["covered"] += 1
                table_ok = (gem in cands)
                # "table-else-copy" prediction: if unique candidate use it, else... take the
                # candidate that matches gemma? No — at inference we don't know gemma. Use the
                # first candidate as the deterministic pick (unique case is the honest one).
                pick = cands[0]
                pred_ok = (pick == gem)
                stats["tablepick_correct_on_covered"] += pred_ok
                if len(cands) == 1:
                    stats["covered_unique"] += 1
                    covered_unique_vs_gemma["agree" if table_ok else "disagree"] += 1
                    if not table_ok and len(disagreements) < 5000:
                        disagreements.append((args.lang, form, pos, gem, cands))
                # combined floor: table pick when covered
                stats["combined_correct"] += pred_ok
            else:
                # not covered -> fall back to copy
                stats["combined_correct"] += copy_ok

    n = stats["oov_tokens"]
    if n == 0:
        print("no OOV tokens?!")
        return
    print(f"\n=== OOV tail: {n} tokens (forms never seen in training) ===")
    print(f"table covers (form,POS): {stats['table_covered']}/{n} = {stats['table_covered']/n*100:.1f}%")
    print(f"  of covered, unique-lemma: {stats['covered_unique']}/{stats['table_covered']}")
    print(f"\n--- lemma accuracy vs Gemma on the OOV tail ---")
    print(f"copy-the-form (floor)     : {stats['copy_correct']/n*100:5.1f}%")
    print(f"table-else-copy (all POS) : {stats['combined_correct']/n*100:5.1f}%")
    print(f"DESIGN (table {sorted(TABLE_POS)} only, else copy): {stats['design_correct']/n*100:5.1f}%   <- the proposed floor")
    # same, but restricted to the content tokens the table actually targets (exclude PROPN etc.)
    content_n = sum(per_pos[p]['n'] for p in TABLE_POS if p in per_pos)
    content_copy = sum(per_pos[p]['copy'] for p in TABLE_POS if p in per_pos)
    content_design = sum(per_pos[p]['design'] for p in TABLE_POS if p in per_pos)
    if content_n:
        print(f"\n  on content POS only ({content_n} tokens): copy {content_copy/content_n*100:.1f}% "
              f"-> table {content_design/content_n*100:.1f}%  (+{(content_design-content_copy)/content_n*100:.1f})")
    cov = stats["table_covered"]
    if cov:
        print(f"table pick, on covered only: {stats['tablepick_correct_on_covered']/cov*100:5.1f}% "
              f"(vs copy on same tokens: {sum(1 for _ in ())})")
    u = stats["covered_unique"]
    if u:
        ag = covered_unique_vs_gemma["agree"]
        print(f"\nwhere table gives a UNIQUE lemma ({u} tokens): agrees with Gemma "
              f"{ag}/{u} = {ag/u*100:.1f}%")

    print(f"\n--- per-POS (OOV tail) ---")
    print(f"{'POS':6} {'n':>6} {'cover%':>7} {'copy%':>7} {'combined%':>10}")
    for pos in sorted(per_pos, key=lambda p: -per_pos[p]['n']):
        c = per_pos[pos]
        if c["n"] < 20:
            continue
        comb = 0
        # recompute combined per pos quickly is messy; show copy & coverage (main levers)
        print(f"{pos:6} {c['n']:>6} {c['covered']/c['n']*100:>6.1f}% {c['copy']/c['n']*100:>6.1f}%")

    # dump a sample of unique-lemma disagreements for manual/self adjudication
    import random
    random.seed(0)
    random.shuffle(disagreements)
    print(f"\n=== sample of table-vs-Gemma disagreements (unique-lemma cases) ===")
    for lang, form, pos, gem, cands in disagreements[:args.dump_disagreements]:
        print(f"  [{pos}] {form!r}: gemma={gem!r}  wiktionary={cands}")
    out = "/tmp/claude-1000/-data-coding-lexide-tagging/68294016-c301-4f43-9008-4a9d1852a080/scratchpad/ood_disagreements.json"
    json.dump(disagreements, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n({len(disagreements)} disagreements saved -> {out})")


if __name__ == "__main__":
    main()

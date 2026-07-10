"""Normalize raw silver+gold tagging data into a unified schema for the multi-task tagger.

Unified per-sentence record (one JSON object per line):
    {
      "lang": "deu",
      "text": "<canonical string>",              # == concat(token.text + token.whitespace)
      "tokens": [
        {"start": int, "end": int,               # char offsets into text (end exclusive)
         "pos": "NOUN", "dep": "root", "head": 0, # head: 0=ROOT, i=1-indexed token position
         "lemma_script": int}                     # index into the lemma-script vocab
      ]
    }

Also writes vocab.json with the POS / DEP / lemma-script label spaces.

The big silver data nests text/lemma as {"text": {"text": ...}} / {"lemma": {"lemma": ...}};
the gold `cleaned_*.jsonl` uses flat token["text"] / token["lemma"]. Both are handled.
Char offsets are computed by construction (reconstruction is exact), so subword->word
pooling downstream never needs fuzzy alignment.
"""
import argparse
import collections
import json
import os
import random
from pathlib import Path

LANGS = ["deu", "eng", "fra", "hin", "ita", "jpn", "kor", "por", "rus", "spa"]

# spaCy sometimes emits a SPACE upos for whitespace-only tokens; keep it in the space.
UPOS = ["NOUN", "PUNCT", "VERB", "PRON", "DET", "ADP", "ADV", "ADJ", "AUX", "PROPN",
        "SCONJ", "PART", "CCONJ", "INTJ", "NUM", "X", "SYM", "SPACE"]

# Sentinel lemma "script" meaning: lemma == surface form, verbatim. It is NOT a real
# "p|s|ins" script (lemma_script never emits it), so it can't collide with one. Used as the
# fallback for tokens whose true edit script is out of vocab. "0|0|" was the old sentinel
# but it decodes to "" (form[-0:] is empty), so apply_script treats it as COPY for back-compat.
COPY_SCRIPT = "COPY"
LEMMA_IDENTITY = COPY_SCRIPT
_LEGACY_COPY = "0|0|"


def unwrap(v, inner_key):
    return v[inner_key] if isinstance(v, dict) else v


def get_text(tok):
    return unwrap(tok["text"], "text")


def get_lemma(tok):
    return unwrap(tok.get("lemma"), "lemma")


def lemma_script(form, lemma):
    """Invertible edit script mapping form -> lemma via a common-prefix/suffix diff.

    Returns "p|s|ins": keep first p chars and last s chars of the form, replace the
    middle with `ins`. apply_script(form, script) reconstructs lemma exactly.
    Case changes fall out naturally as prefix/suffix edits, so no separate casing flag.
    """
    if lemma is None:
        lemma = form
    n, m = len(form), len(lemma)
    p = 0
    while p < n and p < m and form[p] == lemma[p]:
        p += 1
    s = 0
    while s < (n - p) and s < (m - p) and form[n - 1 - s] == lemma[m - 1 - s]:
        s += 1
    ins = lemma[p:m - s]
    return f"{p}|{s}|{ins}"


def apply_script(form, script):
    # COPY (and the legacy "0|0|" sentinel) mean: lemma is the form unchanged.
    if script == COPY_SCRIPT or script == _LEGACY_COPY:
        return form
    p_str, s_str, ins = script.split("|", 2)
    p, s = int(p_str), int(s_str)
    if p + s > len(form):
        return form  # malformed for this form; fall back to copy
    return form[:p] + ins + form[len(form) - s:]


def iter_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_sentence(obj):
    """-> (text, tokens) where tokens carry char offsets + raw string labels, or None to skip."""
    toks = obj.get("tokens") or obj.get("doc") or []
    if not toks:
        return None
    text_parts = []
    out = []
    cursor = 0
    for t in toks:
        txt = get_text(t)
        if txt is None:
            return None
        ws = t.get("whitespace") or ""
        start = cursor
        end = start + len(txt)
        cursor = end + len(ws)
        text_parts.append(txt)
        text_parts.append(ws)
        head = t.get("head", 0)
        out.append({
            "start": start,
            "end": end,
            "pos": t.get("pos", "X"),
            "dep": t.get("dep", "dep"),
            "head": head,
            "lemma": get_lemma(t),
            "form": txt,
        })
    text = "".join(text_parts)
    T = len(out)
    # clip out-of-range heads (0.015% of tokens) to ROOT rather than dropping the sentence
    for tok in out:
        if not (0 <= tok["head"] <= T):
            tok["head"] = 0
    return text, out


def build(args):
    random.seed(1234)
    big_root = Path(args.big_dir)
    gold_root = Path(args.gold_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: build label vocabs (POS fixed; DEP + lemma-script from frequency) ----
    dep_counter = collections.Counter()
    script_counter = collections.Counter()
    sources = []
    for lang in LANGS:
        big = big_root / lang / "target_language_sentences_tokenization.jsonl"
        if big.exists():
            sources.append((lang, big, "silver"))
        gold = gold_root / f"cleaned_{lang}.jsonl"
        if gold.exists() and gold.stat().st_size > 0:
            sources.append((lang, gold, "gold"))

    print(f"[vocab] scanning {len(sources)} source files...")
    for lang, path, kind in sources:
        for i, obj in enumerate(iter_records(path)):
            norm = normalize_sentence(obj)
            if norm is None:
                continue
            _, toks = norm
            for tk in toks:
                dep_counter[tk["dep"]] += 1
                script_counter[lemma_script(tk["form"], tk["lemma"])] += 1
            if kind == "silver" and i >= args.vocab_scan_cap:
                break

    dep_labels = [d for d, _ in dep_counter.most_common()]
    scripts = [s for s, _ in script_counter.most_common(args.max_lemma_scripts)]
    if LEMMA_IDENTITY not in scripts:
        scripts.append(LEMMA_IDENTITY)
    vocab = {
        "pos": UPOS,
        "dep": dep_labels,
        "lemma_scripts": scripts,
        "langs": LANGS,
    }
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    script_to_id = {s: i for i, s in enumerate(scripts)}
    identity_id = script_to_id[LEMMA_IDENTITY]
    dep_to_id = {d: i for i, d in enumerate(dep_labels)}
    pos_to_id = {p: i for i, p in enumerate(UPOS)}
    print(f"[vocab] POS={len(UPOS)} DEP={len(dep_labels)} lemma_scripts={len(scripts)} "
          f"(covering {sum(c for s,c in script_counter.most_common(args.max_lemma_scripts))/max(1,sum(script_counter.values()))*100:.1f}% of tokens)")

    # ---- pass 2: emit unified records into per-split shards, stratified by language ----
    writers = {sp: open(out_dir / f"{sp}.jsonl", "w", encoding="utf-8") for sp in ("train", "val", "test")}
    counts = collections.Counter()
    n_scanned = n_kept = 0
    unk_pos = 0
    for lang, path, kind in sources:
        for obj in iter_records(path):
            n_scanned += 1
            norm = normalize_sentence(obj)
            if norm is None:
                continue
            text, toks = norm
            rec_tokens = []
            ok = True
            for tk in toks:
                if tk["pos"] not in pos_to_id:
                    unk_pos += 1
                    tk["pos"] = "X"
                if tk["dep"] not in dep_to_id:
                    tk["dep"] = "dep" if "dep" in dep_to_id else dep_labels[0]
                sid = script_to_id.get(lemma_script(tk["form"], tk["lemma"]), identity_id)
                rec_tokens.append({
                    "start": tk["start"], "end": tk["end"],
                    "pos": tk["pos"], "dep": tk["dep"], "head": tk["head"],
                    "lemma_script": sid,
                })
            if not ok or not rec_tokens:
                continue
            rec = {"lang": lang, "kind": kind, "text": text, "tokens": rec_tokens}
            r = random.random()
            if r < args.val_frac:
                sp = "val"
            elif r < args.val_frac + args.test_frac:
                sp = "test"
            else:
                sp = "train"
            # cap val/test per (lang) so eval is balanced and small
            if sp in ("val", "test") and counts[(sp, lang)] >= args.eval_cap_per_lang:
                sp = "train"
            writers[sp].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[(sp, lang)] += 1
            counts[sp] += 1
            n_kept += 1
    for w in writers.values():
        w.close()
    print(f"[emit] scanned={n_scanned} kept={n_kept} unk_pos_coerced={unk_pos}")
    print(f"[emit] train={counts['train']} val={counts['val']} test={counts['test']}")
    for lang in LANGS:
        print(f"        {lang}: train={counts[('train',lang)]} val={counts[('val',lang)]} test={counts[('test',lang)]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-dir", default="data/big")
    ap.add_argument("--gold-dir", default="train/data")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--max-lemma-scripts", type=int, default=4000)
    ap.add_argument("--vocab-scan-cap", type=int, default=250000,
                    help="max silver sentences per language to scan when building vocab")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    ap.add_argument("--eval-cap-per-lang", type=int, default=1500)
    args = ap.parse_args()
    build(args)

"""Stream Wiktextract JSONL (stdin) -> compact (POS -> form -> [lemmas]) table (stdout path arg).

Each Wiktextract line is a headword entry: its `word` is the lemma, `pos` its part of speech,
and `forms[]` its inflected surface forms. We invert that into form->lemma keyed by UPOS, so a
tagger can look up the lemma of an inflected surface form it never saw in training.
"""
import json
import sys
import unicodedata
from collections import defaultdict


def norm(s):
    """Strip combining stress marks (Russian/Ukrainian mark stress with U+0301/U+0300, which the
    silver forms don't carry) so table strings match the tagger's unaccented tokens."""
    d = unicodedata.normalize("NFD", s)
    d = "".join(c for c in d if c not in ("́", "̀"))
    return unicodedata.normalize("NFC", d)

# Wiktextract pos string -> UPOS (our tag set). Unmapped pos are skipped.
POS_MAP = {
    "noun": "NOUN", "name": "PROPN", "proper noun": "PROPN",
    "verb": "VERB", "adj": "ADJ", "adv": "ADV", "pron": "PRON",
    "det": "DET", "article": "DET", "num": "NUM",
    "prep": "ADP", "postp": "ADP", "adp": "ADP",
    "particle": "PART", "intj": "INTJ",
    "conj": "CCONJ", "cconj": "CCONJ", "sconj": "SCONJ",
}
# form-entry tags that mean "not a real inflected surface form" -> skip
SKIP_TAGS = {"romanization", "table-tags", "inflection-template", "class",
             "auxiliary", "no-table-tags", "hyphenation"}
SKIP_FORM_VALUES = {"", "-", "—", "no-table-tags"}


def clean_form(f):
    if not isinstance(f, str):
        return None
    f = f.strip()
    if f in SKIP_FORM_VALUES or " " in f or "\n" in f:
        return None
    if any(ch.isdigit() for ch in f):
        return None
    return norm(f)


def main(out_path, lang_code):
    table = defaultdict(lambda: defaultdict(set))  # upos -> form -> set(lemma)
    n_lines = n_entries = n_pairs = 0
    for line in sys.stdin:
        n_lines += 1
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("lang_code") != lang_code:
            continue
        pos = POS_MAP.get(e.get("pos", ""))
        if pos is None:
            continue
        word = e.get("word")
        if not word or not isinstance(word, str):
            continue
        word = word.strip()
        n_entries += 1
        wf = clean_form(word)

        # Distinguish a LEMMA headword from a FORM-OF entry (where `word` is an inflected form
        # and the real lemma lives in sense.form_of / sense.alt_of). Mapping a form-of entry's
        # word to itself would poison the table with identity lemmas (посылают -> посылают).
        targets = []
        has_real_sense = False
        for s in e.get("senses", []) or []:
            fo = s.get("form_of") or s.get("alt_of")
            if fo:
                for t in fo:
                    w = t.get("word") if isinstance(t, dict) else None
                    if w:
                        targets.append(w.strip())
            elif s.get("glosses") or s.get("raw_glosses"):
                has_real_sense = True

        if targets and not has_real_sense:
            # pure form-of entry: `word` is an inflected form of each target lemma
            if wf:
                for tgt in {norm(t) for t in targets if t}:
                    table[pos][wf].add(tgt)
                    n_pairs += 1
            continue

        # lemma headword: map the headword and its inflection-table forms to itself
        lemma = norm(word)
        if wf:
            table[pos][wf].add(lemma)
            n_pairs += 1
        for fe in e.get("forms", []) or []:
            tags = set(fe.get("tags", []) or [])
            if tags & SKIP_TAGS:
                continue
            fv = clean_form(fe.get("form"))
            if not fv:
                continue
            table[pos][fv].add(lemma)
            n_pairs += 1

    # serialize compactly: {pos: {form: [lemmas]}}
    out = {pos: {form: sorted(lemmas) for form, lemmas in forms.items()}
           for pos, forms in table.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    forms_total = sum(len(v) for v in out.values())
    print(f"[parse] lines={n_lines} entries={n_entries} pairs={n_pairs} "
          f"unique_forms={forms_total} -> {out_path}", file=sys.stderr)
    for pos in sorted(out):
        print(f"        {pos}: {len(out[pos])} forms", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "de")

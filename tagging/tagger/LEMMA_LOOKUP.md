# Wiktionary lemma lookup — the out-of-distribution quality floor

**Question:** can a Wiktionary `(form, POS) → lemma` table give the tagger correct lemmas for
words it never saw in training — extra coverage + a base quality level for the long tail?

**Answer: yes, decisively**, for open-class content words. Validated on German and Russian.

## Protocol

Holdout that mimics deployment (`validate_ood.py`): build the training form-vocabulary from
all but the last 20k sentences of the silver, then evaluate on tokens in the tail whose surface
form **never appeared in training** (true OOV). Reference = Gemma's lemma. Compare:
- **copy-the-form** — the naive floor the tagger falls back to on a form it can't handle
- **table-else-copy** — the Wiktionary lemma when it covers `(form, POS)`, else copy

## Result (OOV content-word tokens: NOUN / VERB / ADJ / ADV)

| lang | copy floor | with table | lift | table-pick acc | agrees-with-Gemma (unique) |
|------|-----------|-----------|------|----------------|-----------------------------|
| German  | 52.9% | 76.1% | **+23.2** | 90.2% | 92.8% |
| Russian | 15.8% | 55.3% | **+39.4** | 92.4% | 95.6% |

Russian's lift is larger because it inflects so heavily that copy-the-form is nearly useless —
i.e. the dictionary floor helps most exactly where the model is weakest. The unique-lemma
Gemma-agreement (93–96%) is an **under**count: hand-adjudicating the German disagreements, the
table is *correct and Gemma wrong* in the majority of them (e.g. `wiedergab→wiedergeben`,
`kloppen`, `Charakteristika→Charakteristikum`), so the table also quietly fixes silver errors on
OOV forms.

## Design decisions (load-bearing)

- **Content POS only.** Applied to NOUN/VERB/ADJ/ADV. Proper nouns and closed-class words copy
  the surface form — Wiktionary otherwise normalizes spelling/transliteration variants
  (`Shiva→Schiwa`, `Jann→Jan`) that we want verbatim. PROPN copy is already ~90–97% correct and
  table coverage there is ~3–5%, so excluding it loses ~nothing and avoids those errors.
- **Layered fallback, never an override** (`lemma_lookup.LemmaTable.resolve`): a confident
  (non-copy) model lemma wins; the table only fills in where the model punted to copy on a
  content form; copy is the last resort. So the table is strictly a floor.
- **Ambiguous `(form,POS)`:** prefer the lemmatization the training data uses
  (`build_lemma_priors.py` → `wikt_priors_{lang}.json`): first how training lemmatized this
  exact form, then overall training frequency of the candidate lemma, then the candidate
  closest in length to the form. Without the priors, closest-length + lexicographic tie-break
  occasionally picked obsolete homographs (eng `love→lofe`, spa `mejor→mejor` instead of the
  in-policy `bien`). The same rule runs in the Rust `build-lemma-fst` builder, so the parsley
  serve and the lexide `local` backend stay token-for-token identical.

## Two bugs found during validation (both fixed in `parse_wiktextract.py`)

1. **form-of entries.** Russian Wiktextract stores most inflected forms as standalone entries
   whose headword *is* the inflected form; the real lemma is in `sense.form_of`. Naively taking
   `word` as the lemma mapped `посылают→посылают` (identity) and the table did nothing (+0.8).
   Fix: follow `form_of`/`alt_of` for pure form-of entries. → Russian jumped to +39.
2. **stress marks.** Wiktionary writes Russian lemmas with combining stress accents
   (`сади́ть` = U+0301) the silver doesn't carry, so correct lemmas read as mismatches. Fix:
   strip combining U+0301/U+0300 (`norm()`).

Lesson: validate on a second, typologically different language — German alone hid both bugs.

## Files

- `parse_wiktextract.py` — stream kaikki.org Wiktextract JSONL → `{POS: {form: [lemmas]}}`.
  `curl -sL https://kaikki.org/dictionary/<Lang>/kaikki.org-dictionary-<Lang>.jsonl | python3 parse_wiktextract.py out.json <lang_code>`
- `build_lemma_priors.py` — training-data lemma counts for candidate selection
  (`wikt_priors_{lang}.json`, next to the tables; rerun after rebuilding tables or training data).
- `lemma_lookup.py` — `LemmaTable.load(path, priors_path=...).resolve(form, pos, model_lemma)`;
  wired into `predict.py` via `Pipeline(..., lemma_table_path=...)` and the parsley serve.
- `validate_ood.py` — the holdout validation above.
- Tables live in `data/lemma_tables/` (gitignored; regenerate with `parse_wiktextract.py`).

## Where this fits / next

This is **layer 3** of the lemma stack (OOD floor). Complementary layers, not yet built:
- **Layer 2 — Gemma-majority table**: `(form,POS)→majority lemma` from the silver itself, to fix
  random errors on *seen* forms while staying 100% in-policy (no external data).
- Coverage is richest for European languages (great OOV tail there) and thinner for jpn/kor
  inflected forms — partial floor there. Not yet validated for jpn/kor/hin.
- Real end-to-end gain vs the *trained model* (not just vs copy) needs a Lambda eval run with
  the model in the loop; the +23/+39 here is vs the copy floor, which is the conservative bound.

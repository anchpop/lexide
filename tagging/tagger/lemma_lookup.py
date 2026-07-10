"""Wiktionary-backed lemma lookup — the out-of-distribution quality floor (layer 3).

Validated on German (see validate_ood.py): on out-of-training content-word forms the lookup
lifts lemma accuracy from a 52.9% copy-the-form floor to 73.7% (+20.7), and where it returns a
unique lemma it agrees with (or corrects) Gemma ~94-98% of the time. It is applied ONLY to
open-class content POS — proper nouns and closed-class words copy the surface form, because
Wiktionary would otherwise normalize spelling/transliteration variants (Shiva -> Schiwa) that
we want to keep verbatim.

Intended use as the last layer of a fallback stack, so it never overrides a confident model
prediction — only fills in where the model would otherwise punt to copy:

    model prediction (in-distribution)  ->  Gemma-majority table (seen forms)  ->
    THIS (unseen content forms)         ->  copy the form (last resort)
"""
import json

# Open-class POS where lemmatization is a real transformation and Wiktionary helps.
# Proper nouns / closed-class copy instead (see module docstring).
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


class LemmaTable:
    def __init__(self, table, content_pos=CONTENT_POS):
        self.table = table  # {pos: {form: [lemmas]}}
        self.content_pos = content_pos

    @classmethod
    def load(cls, path, content_pos=CONTENT_POS):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), content_pos)

    def lookup(self, form, pos):
        """Return a lemma for (form, pos), or None if the table shouldn't fire.

        Only content POS are served. Among multiple candidates, prefer the one closest to the
        surface form (smallest length delta), which is almost always the correct morphological
        base rather than a homograph from another paradigm.
        """
        if pos not in self.content_pos:
            return None
        cands = self.table.get(pos, {}).get(form)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        return min(cands, key=lambda c: (abs(len(c) - len(form)), c))

    def resolve(self, form, pos, model_lemma):
        """Layered fallback: keep a confident (non-copy) model lemma; otherwise, for a content
        form the model punted on (predicted copy), use the table if it has a real lemma."""
        if model_lemma and model_lemma != form:
            return model_lemma  # model produced a real transformation -> trust it
        table_lemma = self.lookup(form, pos)
        if table_lemma is not None:
            return table_lemma  # fill the OOD floor
        return model_lemma or form

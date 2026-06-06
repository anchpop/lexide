"""Stage 3 of the phonetizer pipeline: NARROWING broad espeak labels.

Two narrowings, each applied where it's justified and reliable:

  - NASAL (contextual, deterministic): an oral vowel before a CODA nasal →
    its nasalized form, in the 6 non-French languages. Justified by the
    population acoustics (Phase-0: every such vowel is coarticulatorily
    nasalized, A1-P0 depressed vs oral in deu/eng/ita/spa/rus/por), but NOT
    gated per-token — at-scale automated A1-P0 has a ~24% oral false-positive
    floor (formant-tracking + alignment noise), so per-token gating would
    inject bad labels. The acoustics pick the *context*; the rule marks it.
    French is excluded: it already phonemicized its pre-nasal vowels.

  - FLAP (acoustic, bimodal): English intervocalic /t,d/ → [ɾ] when voiced-
    through + burstless. voiced_frac/burst ARE cleanly separable per-token
    (ear-validated), so this one is gated on the cached measurements.

Reads broad phonemes.jsonl (immutable input); writes data/audio/<lang>/
phonemes_narrowed.jsonl (separate artifact, keeps `phonemes_broad`). Flap uses
the measure_corpus cache (model-pinned); nasal needs no measurements.

  narrow.py [--langs deu eng ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "espeak_audit"))
from measure_corpus import MODEL, CACHE, AUDIO  # noqa: E402

VOW = set("aeiouɛɔøœəɐɨʊɪyɯɤʌɒæɑ")
NASAL_C = {"n", "m", "ŋ", "ɲ", "ɴ"}
TILDE = "̃"
NASALIZE_LANGS = {"eng", "deu", "ita", "spa", "rus", "por"}  # French already phonemicized


def is_vowel(s): return bool(s) and s[0] in VOW
def is_nasal_vowel(s): return TILDE in unicodedata.normalize("NFD", s or "")
def oral_vowel(s): return is_vowel(s) and not is_nasal_vowel(s)
def nasalize(s): return s + TILDE


def classify_flap(base, voiced_frac, burst):
    """'ɾ' to flap, else None. Only /t/ — a flap is voiced-through with no burst,
    which voiced_frac/burst separate cleanly. /d/ is intentionally NOT flapped:
    voicing can't distinguish [d] from [ɾ] (both voiced), and the only other cue
    (duration) isn't usable — the raw CTC span is peaky (~1 frame, ~20 ms always),
    so a duration gate is vacuous (it flapped ~every intervocalic /d/). [d]≈[ɾ]
    acoustically, so under-flapping /d/ is cheap; a /d/ rule would have to be
    contextual, not acoustic."""
    if base == "t" and voiced_frac is not None \
            and voiced_frac >= 0.60 and burst is not None and burst < 5.0:
        return "ɾ"
    return None


def narrow_clip(phonemes, segments, lang):
    """Return (narrowed_phonemes, [relabel records]). segments may be None
    (nasal still applies; flap is skipped without measurements)."""
    by_idx = {s["idx"]: s for s in (segments or []) if "idx" in s}
    out = list(phonemes)
    relabels = []
    n = len(phonemes)
    for i, p in enumerate(phonemes):
        # FLAP — acoustic, English intervocalic /t,d/ (needs a measured segment)
        if lang == "eng" and p in ("t", "d") and 0 < i < n - 1 \
                and is_vowel(phonemes[i - 1]) and is_vowel(phonemes[i + 1]):
            s = by_idx.get(i)
            if s is not None:
                new = classify_flap(p, s.get("voiced_frac"), s.get("burst_hi_lo_db"))
                if new and new != p:
                    out[i] = new
                    relabels.append({"idx": i, "from": p, "to": new, "why": "flap"})
                    continue
        # NASAL — contextual, oral vowel before a coda nasal
        if lang in NASALIZE_LANGS and oral_vowel(p) and i + 1 < n and phonemes[i + 1] in NASAL_C:
            nxt2 = phonemes[i + 2] if i + 2 < n else None
            if nxt2 is None or not is_vowel(nxt2):           # coda nasal, not onset
                out[i] = nasalize(p)
                relabels.append({"idx": i, "from": p, "to": out[i], "why": "nasal"})
    return out, relabels


def load_cache_index():
    """{(lang,file): path} of flap measurements under the pinned model."""
    idx = {}
    for p in glob.glob(str(CACHE / "**" / "*.json"), recursive=True):
        try:
            d = json.loads(Path(p).read_text())
            idx[(d["lang"], d["file"])] = p
        except Exception:
            continue
    return idx


def run(langs):
    cache_idx = load_cache_index()
    new_symbols = set()
    for lang in langs:
        pf = AUDIO / lang / "phonemes.jsonl"
        if not pf.exists():
            continue
        out_path = AUDIO / lang / "phonemes_narrowed.jsonl"
        n_clips = n_narrowed = n_flap = n_nasal = 0
        with out_path.open("w") as out:
            for line in pf.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                n_clips += 1
                cp = cache_idx.get((lang, row["file"]))
                segs = json.loads(Path(cp).read_text())["segments"] if cp else None
                narrowed, relabels = narrow_clip(row["phonemes"], segs, lang)
                # Write EVERY clip (narrowed or not) so this file is a complete
                # drop-in training target — broad labels stand where nothing fired.
                new = dict(row)
                new["phonemes_broad"] = row["phonemes"]
                new["phonemes"] = narrowed
                if relabels:
                    n_narrowed += 1
                    n_flap += sum(r["why"] == "flap" for r in relabels)
                    n_nasal += sum(r["why"] == "nasal" for r in relabels)
                    new_symbols.update(r["to"] for r in relabels if r["why"] == "nasal")
                    new["narrow_relabels"] = relabels
                out.write(json.dumps(new, ensure_ascii=False) + "\n")
        flap_note = f", {n_flap} flap" if lang == "eng" else ""
        print(f"{lang}: {n_clips} clips → {n_narrowed} narrowed ({n_nasal} nasal{flap_note}) "
              f"→ {out_path.name}")
    print(f"\nnasalized symbols produced: {sorted(new_symbols)}")
    print("(ensure all are in preprocess.VOCAB_EXTENSIONS or the tokenizer vocab)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=None,
                    help="Default: every lang with a phonemes.jsonl. A narrowed file "
                         "is written for ALL langs (a broad copy where nothing narrows) "
                         "so phonemes_narrowed.jsonl is a uniform, complete drop-in.")
    args = ap.parse_args()
    langs = args.langs or sorted(d.name for d in AUDIO.iterdir()
                                 if (d / "phonemes.jsonl").exists())
    print(f"narrowing {len(langs)} langs (nasal=contextual where applicable, "
          f"flap=acoustic from {MODEL})")
    run(langs)


if __name__ == "__main__":
    main()

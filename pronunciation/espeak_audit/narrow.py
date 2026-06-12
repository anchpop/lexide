"""Stage 3 of the phonetizer pipeline: NARROWING broad espeak labels.

Two narrowings, each applied where it's justified and reliable:

  - NASAL: an oral vowel before a CODA nasal → its nasalized form, in the 6
    non-French languages. French is excluded (already phonemicized). Default mode
    is ACOUSTIC:

      acoustic (default): mark only the ones whose harmonic A1-P0 is depressed
        >= DEPTH dB below that speaker's own oral tokens of the same vowel
        (abstain -> leave oral). Coarticulatory nasalization is a CONTINUUM
        (pre-nasal median ~-2 to -5 dB, phonemic nasals ~-7), so this keeps the
        nasal symbol acoustically sharp and consistent with French's strong
        phonemic nasals. The harmonic A1-P0 is recomputed locally each run from
        the measure cache's boundaries + audio (see compute_harmonic — it's a
        local DSP pass, deliberately NOT cached); baselines are within-speaker,
        same-vowel (speaker resolved by speaker_group_of).

      contextual: mark EVERY such vowel (population coarticulation). Kept as an
        option for comparison; not the default.

  - FLAP (acoustic, bimodal): English intervocalic /t,d/ → [ɾ] when voiced-
    through + burstless. voiced_frac/burst ARE cleanly separable per-token
    (ear-validated), so this one is gated on the cached measurements.

Reads broad phonemes.jsonl (immutable input); writes data/audio/<lang>/
<out-name> (default phonemes_narrowed.jsonl — the canonical file training reads),
keeping `phonemes_broad`. Both narrowings read alignment boundaries from the
measure_corpus cache (the one persisted cache; matched per-clip by exact
phon_key). Run measure_corpus first so the cache exists.

  narrow.py [--mode acoustic|contextual] [--nasal-depth 4.0] [--langs deu eng ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
import unicodedata
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "espeak_audit"))
from measure_corpus import MODEL, CACHE, AUDIO, cache_path  # noqa: E402
from nasal_acoustic import _measure_clip  # noqa: E402  (harmonic A1-P0 per clip)

VOW = set("aeiouɛɔøœəɐɨʊɪyɯɤʌɒæɑ")
NASAL_C = {"n", "m", "ŋ", "ɲ", "ɴ"}
TILDE = "̃"
NASALIZE_LANGS = {"eng", "deu", "ita", "spa", "rus", "por"}  # French already phonemicized
DEFAULT_DEPTH = 4.0
MIN_BASELINE = 3  # min same-vowel oral tokens to trust a speaker baseline


def _phon_key(phonemes):
    """phon_key matching measure_corpus.load_targets — selects the cache entry
    measured for THESE exact phonemes (so changed labels never read a stale one)."""
    return hashlib.sha256("\x00".join(phonemes).encode()).hexdigest()[:16]


def is_vowel(s): return bool(s) and s[0] in VOW
def is_nasal_vowel(s): return TILDE in unicodedata.normalize("NFD", s or "")
def oral_vowel(s): return is_vowel(s) and not is_nasal_vowel(s)
def nasalize(s): return s + TILDE
def base_vowel(s): return s[0] if s else ""


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


def is_coda_nasal_ctx(phonemes, i):
    """True if phonemes[i] is an oral vowel before a CODA nasal (not onset)."""
    n = len(phonemes)
    if not (oral_vowel(phonemes[i]) and i + 1 < n and phonemes[i + 1] in NASAL_C):
        return False
    nxt2 = phonemes[i + 2] if i + 2 < n else None
    return nxt2 is None or not is_vowel(nxt2)


def narrow_clip(phonemes, segments, lang, nasal_ok=None):
    """Return (narrowed_phonemes, [relabel records]). segments may be None
    (flap is skipped without measurements). `nasal_ok(i, phoneme)` decides each
    pre-nasal vowel: contextual passes None (=> always nasalize); acoustic
    passes a per-token depression test."""
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
        # NASAL — oral vowel before a coda nasal (decision per `nasal_ok`)
        if lang in NASALIZE_LANGS and is_coda_nasal_ctx(phonemes, i):
            if nasal_ok is None or nasal_ok(i, p):
                out[i] = nasalize(p)
                relabels.append({"idx": i, "from": p, "to": out[i], "why": "nasal"})
    return out, relabels


# ---------------------------------------------------------------------------
# Acoustic nasal: harmonic A1-P0 (recomputed locally) + within-speaker baselines
# ---------------------------------------------------------------------------

def load_clip_segs(lang, rows):
    """{file: segments} from the measure cache, keyed by each clip's CURRENT
    phonemes (exact phon_key) so a stale entry from changed labels is never used.
    Supplies alignment boundaries for both the flap and the harmonic step."""
    out = {}
    for row in rows:
        cp = cache_path(lang, row["file"], _phon_key(row["phonemes"]))
        if cp.exists():
            out[row["file"]] = json.loads(cp.read_text())["segments"]
    return out


def compute_harmonic(lang, clip_segs, workers=8):
    """{(file, idx): a1p0_h} computed on the fly from cached boundaries + audio.

    Deliberately NOT cached: it's a local DSP pass, and a persisted harmonic cache
    would need its own (phon_key-aware) invalidation — complexity + staleness bugs
    for no Modal saving. The threshold/baselines (the tunable part) are recomputed
    here anyway, so re-running narrow.py is the single, always-fresh local step."""
    keys, jobs = [], []
    for file, segs in clip_segs.items():
        spans = [(t["start"], t["end"]) for t in segs if t.get("start") is not None]
        vowels = [(t["idx"], t["win_start"], t["win_end"], t["start"], t["end"])
                  for t in segs if t.get("kind") == "vowel" and t.get("win_start") is not None]
        if not vowels:
            continue
        keys.append(file)
        jobs.append((str(AUDIO / lang / file), segs[0].get("ceiling") or 5500.0, vowels, spans))
    harm = {}
    if jobs:
        with Pool(workers) as pool:
            for file, rows in zip(keys, pool.imap(_measure_clip, jobs, chunksize=8)):
                for r in rows:
                    if r.get("a1p0_h") is not None:
                        harm[(file, r["idx"])] = r["a1p0_h"]
    return harm


def speaker_group_of(lang):
    """{file: group} for within-speaker baselines, by best available speaker id:
      1. speaker_cluster  — FLEURS/Pimsleur (embedding-clustered speakers)
      2. voice            — Tatoeba contributor username + TTS voice name; both
                            are ground-truth per-speaker ids
      3. clip:<file>      — last resort (no speaker info → effectively abstains)
    """
    out = {}
    mf = AUDIO / lang / "manifest.jsonl"
    if mf.exists():
        for line in mf.open():
            if not line.strip():
                continue
            d = json.loads(line)
            g = d.get("speaker_cluster") or d.get("voice")
            out[d["file"]] = g if g else f"clip:{d['file']}"
    return out


def build_nasal_baselines(rows, harm, groups):
    """Median harmonic A1-P0 of CLEAN ORAL vowels (no nasal neighbor) per
    (speaker_group, base_vowel)."""
    cat = defaultdict(list)
    for d in rows:
        f = d["file"]
        g = groups.get(f, f"clip:{f}")
        ph = d["phonemes"]
        for i, p in enumerate(ph):
            if not oral_vowel(p):
                continue
            prevn = i > 0 and ph[i - 1] in NASAL_C
            nextn = i + 1 < len(ph) and ph[i + 1] in NASAL_C
            if prevn or nextn:
                continue
            a = harm.get((f, i))
            if a is not None:
                cat[(g, base_vowel(p))].append(a)
    return {k: st.median(v) for k, v in cat.items() if len(v) >= MIN_BASELINE}


def make_nasal_ok(file, harm, groups, baselines, depth):
    """Closure: (i, phoneme) -> True iff this pre-nasal vowel's harmonic A1-P0 is
    >= `depth` dB below its speaker/vowel baseline. Abstains (False) when the
    token wasn't measured or has no baseline."""
    g = groups.get(file, f"clip:{file}")

    def ok(i, p):
        a = harm.get((file, i))
        if a is None:
            return False
        med = baselines.get((g, base_vowel(p)))
        if med is None:
            return False
        return (a - med) <= -depth
    return ok


def run(langs, mode="acoustic", depth=DEFAULT_DEPTH, out_name="phonemes_narrowed.jsonl", workers=8):
    new_symbols = set()
    for lang in langs:
        pf = AUDIO / lang / "phonemes.jsonl"
        if not pf.exists():
            continue
        rows = [json.loads(l) for l in pf.read_text().splitlines() if l.strip()]
        # Boundaries from the measure cache (flap + harmonic both read these).
        clip_segs = load_clip_segs(lang, rows)

        # Acoustic nasal: recompute harmonic A1-P0 locally + within-speaker baselines
        # (only the 6 nasal langs nasalize; others get flap/pass-through either way).
        harm = groups = baselines = None
        if mode == "acoustic" and lang in NASALIZE_LANGS:
            groups = speaker_group_of(lang)
            harm = compute_harmonic(lang, clip_segs, workers)
            baselines = build_nasal_baselines(rows, harm, groups)
            if not baselines:
                print(f"{lang}: WARNING — no baselines (measure cache empty for these "
                      f"phonemes? run measure_corpus). Nasal will all abstain → oral.")

        out_path = AUDIO / lang / out_name
        n_clips = n_narrowed = n_flap = n_nasal = 0
        n_ctx = n_abstain = 0  # acoustic: pre-nasal contexts seen / left oral
        with out_path.open("w") as out:
            for row in rows:
                n_clips += 1
                segs = clip_segs.get(row["file"])
                ph = row["phonemes"]
                if harm is not None:
                    nasal_ok = make_nasal_ok(row["file"], harm, groups, baselines, depth)
                    ctxs = [i for i in range(len(ph)) if is_coda_nasal_ctx(ph, i)]
                    n_ctx += len(ctxs)
                    n_abstain += sum(0 if nasal_ok(i, ph[i]) else 1 for i in ctxs)
                else:
                    nasal_ok = None
                narrowed, relabels = narrow_clip(ph, segs, lang, nasal_ok=nasal_ok)
                # Write EVERY clip (narrowed or not) so this file is a complete
                # drop-in training target — broad labels stand where nothing fired.
                new = dict(row)
                new["phonemes_broad"] = ph
                new["phonemes"] = narrowed
                if relabels:
                    n_narrowed += 1
                    n_flap += sum(r["why"] == "flap" for r in relabels)
                    n_nasal += sum(r["why"] == "nasal" for r in relabels)
                    new_symbols.update(r["to"] for r in relabels if r["why"] == "nasal")
                    new["narrow_relabels"] = relabels
                out.write(json.dumps(new, ensure_ascii=False) + "\n")
        flap_note = f", {n_flap} flap" if lang == "eng" else ""
        acoustic_note = ""
        if harm is not None and n_ctx:
            acoustic_note = (f"  [acoustic: {n_nasal}/{n_ctx} pre-nasal flagged, "
                             f"{n_abstain} left oral, {len(baselines)} baselines]")
        print(f"{lang}: {n_clips} clips → {n_narrowed} narrowed ({n_nasal} nasal{flap_note}) "
              f"→ {out_path.name}{acoustic_note}")
    print(f"\nnasalized symbols produced: {sorted(new_symbols)}")
    print("(ensure all are in preprocess.VOCAB_EXTENSIONS or the tokenizer vocab)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=None,
                    help="Default: every lang with a phonemes.jsonl. A narrowed file "
                         "is written for ALL langs (a broad copy where nothing narrows) "
                         "so the output is a uniform, complete drop-in.")
    ap.add_argument("--mode", choices=("contextual", "acoustic"), default="acoustic",
                    help="nasal narrowing: acoustic (default — mark only measurably-"
                         "depressed pre-nasal vowels; abstain->oral) or contextual "
                         "(mark all pre-nasal vowels).")
    ap.add_argument("--nasal-depth", type=float, default=DEFAULT_DEPTH,
                    help="acoustic mode: min A1-P0 depression (dB) below the speaker's "
                         "same-vowel oral baseline to mark a vowel nasalized.")
    ap.add_argument("--out-name", default="phonemes_narrowed.jsonl",
                    help="output filename per lang dir (the canonical narrowed file "
                         "training reads).")
    ap.add_argument("--workers", type=int, default=8, help="local DSP processes")
    args = ap.parse_args()
    langs = args.langs or sorted(d.name for d in AUDIO.iterdir()
                                 if (d / "phonemes.jsonl").exists())
    nasal_desc = (f"acoustic@{args.nasal_depth}dB" if args.mode == "acoustic" else "contextual")
    print(f"narrowing {len(langs)} langs (nasal={nasal_desc}, flap=acoustic from {MODEL}) "
          f"→ {args.out_name}")
    run(langs, mode=args.mode, depth=args.nasal_depth, out_name=args.out_name, workers=args.workers)


if __name__ == "__main__":
    main()

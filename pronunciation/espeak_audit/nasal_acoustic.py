"""Acoustic per-token nasalization detector — the A/B alternative to the
contextual rule in narrow.py.

The idea: a pre-nasal-coda vowel should be marked nasalized only if its A1-P0 is
depressed (toward phonemic-nasal depth) relative to that speaker's OWN oral
tokens of the SAME vowel height. A1-P0 scales with F1, so the baseline must be
per-vowel-category, and it carries a speaker offset, so it must be within-speaker.

`calibrate()` measures the ceiling of this approach on the single-speaker TTS
audit segments, where French/Portuguese PHONEMIC nasals are ground-truth
positives and vowels with no nasal neighbor are clean negatives. It reports
recall/false-positive at several thresholds, the per-vowel-category FP breakdown,
and a sample of the false positives themselves — so the "is this detector viable
or is the 24% a bug" question can actually be audited.

Round 1 (cached band-max A1-P0 from phonetics.py) measured 24% oral FP at 73%
recall. A diagnostic pass showed the FPs are NOT irreducible noise: they cluster
on vowels next to VOICELESS consonants (32-36% FP vs 10-13% next to voiced) and
carry wide B1 — i.e. partially devoiced/weak tokens where A1 collapses for
non-nasal reasons. The fix is to measure differently, not threshold differently:

  --remeasure   per-frame HARMONIC A1-P0 on voiced frames only (P0 = amplitude
                of the actual harmonic(s) in 150-350 Hz from the frame's F0, A1
                = spectral peak at F1, zero-padded FFT so harmonics resolve),
                plus voiced-frame B1. Abstains (None) when <2 voiced frames —
                the devoiced-/s/ FP population abstains instead of polluting.
                Writes out/<name>.nasal2.jsonl. Reads audio from data/audio/.
  --calibrate   ceiling report; uses .nasal2.jsonl when present (falls back to
                the cached band-max metric otherwise) and reports both, plus the
                B1 second cue and the combined rule.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import unicodedata
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUDIT = REPO / "espeak_audit" / "out"
AUDIO = REPO / "data" / "audio"

VOW = set("aeiouɛɔøœəɐɨʊɪyɯɤʌɒæɑ")
NASAL_C = {"n", "m", "ŋ", "ɲ", "ɴ"}
TILDE = "̃"


def is_vowel(s): return bool(s) and s[0] in VOW
def is_nasal_vowel(s): return TILDE in unicodedata.normalize("NFD", s or "")
def oral_vowel(s): return is_vowel(s) and not is_nasal_vowel(s)
def base_vowel(s): return s[0] if s else ""        # ɔ̃ -> ɔ, ɑː -> ɑ


def _clips(path):
    by = defaultdict(list)
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            by[d["key"]].append(d)
    for toks in by.values():
        toks.sort(key=lambda t: t["idx"])
        yield toks


# ---------------------------------------------------------------------------
# Harmonic re-measurement (--remeasure)
# ---------------------------------------------------------------------------

def _measure_clip(job):
    """Per-frame harmonic A1-P0 + voiced-frame B1 for one clip's vowels.

    Frames are kept only when (a) F0 is tracked (voiced), (b) F1 is tracked and
    >= 400 Hz (P0 band would confound lower F1), and (c) the frame doesn't fall
    inside a NEIGHBORING token's raw span (no murmur/frication bleed). Token
    abstains (None) below 2 qualifying frames.
    """
    import numpy as np
    import parselmouth
    from parselmouth.praat import call

    path, ceiling, vowels, all_spans = job
    try:
        snd = parselmouth.Sound(path)
        if snd.n_channels > 1:
            snd = snd.convert_to_mono()
    except Exception as e:
        return [{"idx": idx, "error": f"load: {e}"} for idx, *_ in vowels]
    x = snd.values[0].astype(np.float64)
    fs = 1.0 / snd.dx
    pitch = snd.to_pitch(time_step=0.005)
    formant = snd.to_formant_burg(
        time_step=0.0025, max_number_of_formants=5, maximum_formant=ceiling,
        window_length=0.025, pre_emphasis_from=50.0)

    NFFT = 8192
    HALF = 0.015  # 30 ms analysis frame
    win = np.hanning(int(2 * HALF * fs))

    out = []
    for idx, ws, we, s, e in vowels:
        # neighbor spans = every other token's raw span; exclude bleed frames
        others = [(a, b) for (a, b) in all_spans if (a, b) != (s, e)]
        vals, b1s = [], []
        for t in np.arange(ws + 0.005, we, 0.005):
            if any(a <= t <= b for (a, b) in others):
                continue
            f0 = pitch.get_value_at_time(t)
            if f0 is None or np.isnan(f0) or f0 <= 0:
                continue
            f1 = formant.get_value_at_time(1, t)
            if f1 is None or np.isnan(f1) or f1 < 400.0:
                continue
            i0 = int((t - HALF) * fs)
            i1 = i0 + win.size
            if i0 < 0 or i1 > x.size:
                continue
            spec = np.fft.rfft(x[i0:i1] * win, NFFT)
            amp = 20.0 * np.log10(np.abs(spec) + 1e-20)
            freqs = np.fft.rfftfreq(NFFT, 1.0 / fs)

            def peak(lo, hi):
                m = (freqs >= lo) & (freqs <= hi)
                return float(np.max(amp[m])) if m.any() else None

            # P0: amplitude at the actual harmonic(s) landing in 150-350 Hz
            hw = max(15.0, 0.25 * f0)            # search half-width per harmonic
            cands = [k * f0 for k in (1, 2, 3) if 150.0 <= k * f0 <= 350.0]
            if not cands:
                continue
            p0 = max(p for p in (peak(c - hw, c + hw) for c in cands) if p is not None)
            a1 = peak(0.75 * f1, 1.25 * f1)      # strongest harmonic at F1
            if a1 is None:
                continue
            vals.append(a1 - p0)
            b1 = call(formant, "Get bandwidth at time", 1, t, "hertz", "linear")
            if b1 is not None and not np.isnan(b1):
                b1s.append(float(b1))
        row = {"idx": idx, "n_frames": len(vals)}
        if len(vals) >= 2:
            row["a1p0_h"] = float(np.median(vals))
            row["b1_v"] = float(np.median(b1s)) if b1s else None
        out.append(row)
    return out


def remeasure(workers=8):
    for path in sorted(glob.glob(str(AUDIT / "*_tts_*.segments.jsonl"))):
        lang = os.path.basename(path).split("_")[0]
        jobs, keys = [], []
        for toks in _clips(path):
            spans = [(t["start"], t["end"]) for t in toks if t.get("start") is not None]
            vowels = [(t["idx"], t["win_start"], t["win_end"], t["start"], t["end"])
                      for t in toks
                      if t.get("kind") == "vowel" and t.get("win_start") is not None]
            if not vowels:
                continue
            ceiling = toks[0].get("ceiling") or 5500.0
            jobs.append((str(AUDIO / lang / toks[0]["key"]), ceiling, vowels, spans))
            keys.append(toks[0]["key"])
        dst = path.replace(".segments.jsonl", ".nasal2.jsonl")
        with Pool(workers) as pool, open(dst, "w") as f:
            for key, rows in zip(keys, pool.imap(_measure_clip, jobs, chunksize=8)):
                for r in rows:
                    r["key"] = key
                    f.write(json.dumps(r) + "\n")
        print(f"{os.path.basename(dst)}: {len(keys)} clips")


def _load_nasal2(path):
    dst = path.replace(".segments.jsonl", ".nasal2.jsonl")
    if not os.path.exists(dst):
        return {}
    out = {}
    for line in open(dst):
        if line.strip():
            d = json.loads(line)
            out[(d["key"], d["idx"])] = d
    return out


def calibrate(thresholds=(2.0, 2.5, 3.0, 3.5, 4.0, 5.0), n_fp_examples=12):
    """Same-vowel-category, within-speaker normalization; clean negatives.

    Uses the harmonic A1-P0 (.nasal2.jsonl) when present, else the cached
    band-max a1_p0 from phonetics.py. Reports recall/FP for the chosen metric,
    plus the B1-bandwidth cue and the AND-combined rule. Prints, returns nothing.
    """
    using_h = False
    # pos/neg hold dicts: {a1p0, b1, base}. pre is the application set (oral, next-nasal).
    pos, neg, pre = [], [], []
    speakers = 0
    for path in sorted(glob.glob(str(AUDIT / "*_tts_*.segments.jsonl"))):
        speakers += 1
        nasal2 = _load_nasal2(path)
        if nasal2:
            using_h = True

        def metric(t):
            """(a1p0, b1) for token t, harmonic if available else band-max."""
            if nasal2:
                d = nasal2.get((t["key"], t["idx"]))
                if not d or d.get("a1p0_h") is None:
                    return None, None
                return d["a1p0_h"], d.get("b1_v")
            return t.get("a1_p0"), t.get("b1")

        # within-speaker, per-base-vowel oral baselines (truly-oral tokens only)
        cat_a, cat_b = defaultdict(list), defaultdict(list)
        toks_all = []
        for toks in _clips(path):
            sb = {t["idx"]: t for t in toks}
            for t in toks:
                if t.get("kind") != "vowel":
                    continue
                a, b1 = metric(t)
                if a is None:
                    continue
                prevn = sb.get(t["idx"] - 1, {}).get("symbol") in NASAL_C
                nextn = sb.get(t["idx"] + 1, {}).get("symbol") in NASAL_C
                toks_all.append((t, a, b1, prevn, nextn))
                if oral_vowel(t["symbol"]) and not prevn and not nextn:
                    cat_a[base_vowel(t["symbol"])].append(a)
                    if b1 is not None:
                        cat_b[base_vowel(t["symbol"])].append(b1)
        meda = {b: st.median(v) for b, v in cat_a.items() if len(v) >= 3}
        medb = {b: st.median(v) for b, v in cat_b.items() if len(v) >= 3}
        for (t, a, b1, prevn, nextn) in toks_all:
            base = base_vowel(t["symbol"])
            if base not in meda:
                continue
            rec = {"dep": a - meda[base],
                   "bw": (b1 - medb[base]) if (b1 is not None and base in medb) else None,
                   "base": base, "sym": t["symbol"]}
            if is_nasal_vowel(t["symbol"]):
                pos.append(rec)
            elif not prevn and not nextn:
                neg.append(rec)
            elif nextn and oral_vowel(t["symbol"]):
                pre.append(rec)

    metric_name = "HARMONIC a1p0_h" if using_h else "band-max a1_p0 (cached)"
    print(f"metric = {metric_name}")
    print(f"speakers={speakers}  phonemic-nasal(pos)={len(pos)}  clean-oral(neg)={len(neg)}  "
          f"pre-nasal-oral={len(pre)}")

    print(f"\nA1-P0 depth threshold:\n{'depth':>6} {'recall':>8} {'oral FP':>8}")
    for thr in thresholds:
        r = sum(1 for x in pos if x["dep"] <= -thr) / max(len(pos), 1)
        f = sum(1 for x in neg if x["dep"] <= -thr) / max(len(neg), 1)
        print(f"{thr:6.1f} {r:8.1%} {f:8.1%}")

    # Second cue: F1 bandwidth widening (nasal vowels have broader B1).
    posb = [x for x in pos if x["bw"] is not None]
    negb = [x for x in neg if x["bw"] is not None]
    if posb and negb:
        print(f"\nB1-widening threshold (Δb1 >= X Hz vs same-vowel oral):"
              f"\n{'+Hz':>6} {'recall':>8} {'oral FP':>8}")
        for thr in (20, 40, 60, 80, 100):
            r = sum(1 for x in posb if x["bw"] >= thr) / len(posb)
            f = sum(1 for x in negb if x["bw"] >= thr) / len(negb)
            print(f"{thr:6d} {r:8.1%} {f:8.1%}")

        # Combined AND rule: both A1-P0 depressed and B1 widened.
        print(f"\nAND rule (A1-P0<=-D AND Δb1>=B):")
        for D in (2.5, 3.5):
            for B in (40, 60):
                pp = [x for x in pos if x["bw"] is not None]
                nn = [x for x in neg if x["bw"] is not None]
                r = sum(1 for x in pp if x["dep"] <= -D and x["bw"] >= B) / max(len(pp), 1)
                f = sum(1 for x in nn if x["dep"] <= -D and x["bw"] >= B) / max(len(nn), 1)
                print(f"  D={D} B={B}: recall {r:6.1%}  oral FP {f:6.1%}")

    print("\noral FP rate by base vowel at depth 3.5 (n>=30):")
    by = defaultdict(lambda: [0, 0])
    for x in neg:
        by[x["base"]][1] += 1
        if x["dep"] <= -3.5:
            by[x["base"]][0] += 1
    for b, (fp, tot) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        if tot >= 30:
            print(f"  {b:3} {fp/tot:6.1%}  ({fp}/{tot})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remeasure", action="store_true",
                    help="recompute harmonic A1-P0 + voiced B1 on TTS audit -> .nasal2.jsonl")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.remeasure:
        remeasure(workers=args.workers)
    if args.calibrate or not args.remeasure:
        calibrate()


if __name__ == "__main__":
    main()

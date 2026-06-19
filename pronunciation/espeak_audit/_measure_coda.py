"""Test the 'narrow the consonant too' theory: when we nasalize a pre-nasal vowel,
is the following coda nasal actually REALIZED (own murmur) or ABSORBED into the
vowel (→ a bare nasal vowel, like French)?

Signal: a realized nasal consonant has an oral closure → it radiates only through
the nose → measurably LOWER intensity than its vowel (a "drop"). An absorbed coda
has no closure → the nasalized vowel just continues → ~no drop.

We compare the coda-nasal intensity drop for:
  - FLAGGED   pre-nasal vowels (we nasalized them) — the theory says these absorb more
  - ORAL      pre-nasal vowels (we left oral)      — coda should be a real consonant
  - ONSET     nasals (N before V)                  — calibration: a definitely-realized nasal

All from the existing measure cache (no recompute). Split by source (Pimsleur =
connected speech should absorb more than FLEURS read speech).

  _measure_coda.py [langs...]
"""
import sys, json, hashlib, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure_corpus import cache_path, AUDIO
import narrow as N

NASAL_C = N.NASAL_C


def phon_key(ph):
    return hashlib.sha256("\x00".join(ph).encode()).hexdigest()[:16]


def med(xs):
    return st.median(xs) if xs else float("nan")


def run(langs):
    # buckets: drop = vowel_intensity_db - nasal_intensity_db (higher = nasal quieter = realized)
    flagged = defaultdict(list)   # source -> [coda drop] for nasalized pre-nasal vowels
    oral = defaultdict(list)      # source -> [coda drop] for non-flagged pre-nasal vowels
    onset = defaultdict(list)     # source -> [drop] for onset nasals (realized reference)
    flagged_all, onset_all = [], []
    # alignment-based realization signal: dur (s) + align_score (model log-prob).
    # absorbed coda → model finds little evidence → short span / low score.
    align = defaultdict(lambda: {"flag_dur": [], "flag_sc": [], "onset_dur": [], "onset_sc": []})

    for lang in langs:
        src = {}
        mf = AUDIO / lang / "manifest.jsonl"
        if mf.exists():
            for line in mf.open():
                if line.strip():
                    d = json.loads(line)
                    src[d["file"]] = d.get("source", "?")
        nf = AUDIO / lang / "phonemes_narrowed.jsonl"
        if not nf.exists():
            continue
        n_clips = n_cached = 0
        for line in nf.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            ph = d.get("phonemes_broad") or d["phonemes"]
            n_clips += 1
            cp = cache_path(lang, d["file"], phon_key(ph))
            if not cp.exists():
                continue
            n_cached += 1
            try:
                segs = {s["idx"]: s for s in json.loads(cp.read_text())["segments"]}
            except Exception:
                continue
            relabels = {r["idx"] for r in d.get("narrow_relabels", []) if r["why"] == "nasal"}
            s = src.get(d["file"], "?")
            for i, p in enumerate(ph):
                # pre-nasal vowel + coda nasal
                if N.is_vowel(p) and i + 1 < len(ph) and ph[i + 1] in NASAL_C:
                    v, n = segs.get(i), segs.get(i + 1)
                    if v and n and v.get("intensity_db") is not None and n.get("intensity_db") is not None:
                        drop = v["intensity_db"] - n["intensity_db"]
                        (flagged if i in relabels else oral)[s].append(drop)
                        if i in relabels:
                            flagged_all.append(drop)
                    if n and i in relabels:
                        if n.get("dur") is not None:
                            align[s]["flag_dur"].append(n["dur"])
                        if n.get("align_score") is not None:
                            align[s]["flag_sc"].append(n["align_score"])
                # onset nasal (realized reference): N before V
                if p in NASAL_C and i + 1 < len(ph) and N.is_vowel(ph[i + 1]):
                    n, v = segs.get(i), segs.get(i + 1)
                    if v and n and v.get("intensity_db") is not None and n.get("intensity_db") is not None:
                        onset[s].append(v["intensity_db"] - n["intensity_db"])
                        onset_all.append(v["intensity_db"] - n["intensity_db"])
                    if n:
                        if n.get("dur") is not None:
                            align[s]["onset_dur"].append(n["dur"])
                        if n.get("align_score") is not None:
                            align[s]["onset_sc"].append(n["align_score"])
        print(f"{lang}: {n_clips} clips, {n_cached} cached")

    sources = sorted(set(list(flagged) + list(oral) + list(onset)))
    print(f"\nintensity DROP (dB) = vowel − following-nasal  (higher = nasal has own quieter murmur = REALIZED;")
    print(f"                                                 near 0 = ABSORBED into the nasal vowel)")
    print(f"\n{'source':10} {'ONSET-N (ref, realized)':26} {'ORAL pre-nasal coda':24} {'FLAGGED pre-nasal coda':24}")
    for s in sources:
        def cell(b):
            xs = b.get(s, [])
            return f"med={med(xs):5.1f} n={len(xs):5}" if xs else "n=0"
        print(f"{s:10} {cell(onset):26} {cell(oral):24} {cell(flagged):24}")

    # Calibrate 'absorbed' off the onset reference, report fraction of FLAGGED codas that look absorbed.
    if onset_all and flagged_all:
        onset_all.sort()
        thr = onset_all[len(onset_all) // 4]   # 25th pctile of realized-nasal drops
        absorbed = sum(1 for d in flagged_all if d < thr) / len(flagged_all)
        print(f"\nrealized-nasal (onset) drop: median {med(onset_all):.1f} dB, p25 {thr:.1f} dB")
        print(f"FLAGGED coda drop: median {med(flagged_all):.1f} dB, p25 {sorted(flagged_all)[len(flagged_all)//4]:.1f}")
        print(f"→ fraction of FLAGGED codas below the realized-nasal p25 ({thr:.1f} dB) = "
              f"{absorbed:.0%}  (candidate 'absorbed' → would relabel bare Ṽ)")

    # Alignment-based realization: a realized nasal gets its own span + decent score;
    # an absorbed one collapses to the ~20ms frame floor with low score.
    print(f"\nALIGNMENT signal — coda nasal span (ms) + align_score, FLAGGED vs realized ONSET:")
    print(f"{'source':10} {'ONSET dur/score':28} {'FLAGGED-coda dur/score':28}")
    for s in sources:
        a = align[s]
        def fmt(durs, scs):
            return (f"dur med={med(durs)*1000:4.0f}ms n={len(durs):5}  sc={med(scs):5.2f}"
                    if durs else "n=0")
        print(f"{s:10} {fmt(a['onset_dur'], a['onset_sc']):28} {fmt(a['flag_dur'], a['flag_sc']):28}")
    # frame-floor (≈absorbed/deleted) rate among flagged codas vs onset nasals
    all_flag_dur = [d for s in sources for d in align[s]["flag_dur"]]
    all_onset_dur = [d for s in sources for d in align[s]["onset_dur"]]
    if all_flag_dur and all_onset_dur:
        floor = min(all_onset_dur + all_flag_dur)
        ff = lambda xs: sum(1 for d in xs if d <= floor + 0.005) / len(xs)
        print(f"\nframe-floor (≤{ (floor+0.005)*1000:.0f}ms, ~deleted) rate: "
              f"onset {ff(all_onset_dur):.0%}  vs  FLAGGED-coda {ff(all_flag_dur):.0%}")


if __name__ == "__main__":
    run(sys.argv[1:] or ["eng", "deu", "ita", "spa", "rus", "por"])

"""Prototype per-token flap detector for English /t,d/ → [ɾ], with audio cutouts
for human (ear) review.

The 4-source audit established that intervocalic /t,d/ flapping is acoustically
SEPARABLE per token (voiced_frac ~bimodal, ~85% of tokens confident). This turns
that aggregate signal into a per-token DECISION with a confidence gate, and —
crucially, since there's no phonetician ground truth — emits a sample of cut
audio clips so a human can verify the calls by ear before we trust the detector.

Decision (intervocalic /t,d/ only):
  /t/ (voiceless underlyingly): a flap is VOICED with no release burst.
      voiced_frac >= 0.60 and burst_hi_lo_db < 5  -> [ɾ]   (high if vf>=0.75)
      voiced_frac <= 0.35                          -> keep [t]
      else                                          -> ambiguous (keep [t])
  /d/ (voiced underlyingly): voicing doesn't separate [d] from [ɾ] (both voiced),
      so use DURATION — a flap is very short.
      phone_dur <= 45 ms and voiced_frac >= 0.5    -> [ɾ]
      else                                          -> keep [d]

This is a PROTOTYPE: thresholds are first-guess and must be calibrated against
the ear-reviewed sample before being trusted / applied to training labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"


def classify(sym_base: str, voiced_frac, burst, phone_dur):
    """Return (label, decision, confidence) for an intervocalic /t/ or /d/.
    decision ∈ {flap, keep, ambiguous}. Missing required measurements never
    satisfy a flap gate — they fall through to keep/ambiguous so a failed
    measurement can't be emitted as a confident flap."""
    if sym_base == "t":
        # A flap is voiced AND has no burst — BOTH must be measured to claim it.
        if voiced_frac is None:
            return "t", "ambiguous", "low"
        if voiced_frac >= 0.60 and burst is not None and burst < 5.0:
            return "ɾ", "flap", ("high" if voiced_frac >= 0.75 else "medium")
        if voiced_frac <= 0.35:
            return "t", "keep", "high"
        return "t", "ambiguous", "low"
    if sym_base == "d":
        # Voicing can't separate [d] from [ɾ]; use duration. Missing dur -> keep.
        if phone_dur is not None and phone_dur <= 0.045 and (voiced_frac or 0) >= 0.5:
            return "ɾ", "flap", "medium"
        return "d", "keep", "medium"
    raise ValueError(f"classify() only handles /t,d/, got {sym_base!r}")


def collect(tag: str, lang: str):
    segs = A.load_segments(tag)
    by_clip = A.clip_sequences(segs)  # annotates phone_dur
    rows = []
    for key, seq in by_clip.items():
        for i, s in enumerate(seq):
            b = A._base(s["symbol"])
            if b not in ("t", "d") or s.get("oov"):
                continue
            pv = i > 0 and A.is_vowel(seq[i - 1]["symbol"])
            nv = i + 1 < len(seq) and A.is_vowel(seq[i + 1]["symbol"])
            if not (pv and nv):
                continue
            label, decision, conf = classify(b, s.get("voiced_frac"), s.get("burst_hi_lo_db"), s.get("phone_dur"))
            rows.append({
                "key": key, "sentence": s["sentence"], "idx": s["idx"],
                "base": b, "symbol": s["symbol"], "label": label,
                "decision": decision, "confidence": conf,
                "voiced_frac": s.get("voiced_frac"), "burst": s.get("burst_hi_lo_db"),
                "phone_dur": s.get("phone_dur"), "word_idx": s.get("word_idx"),
                "start": s.get("start"), "end": s.get("end"),
                # vowel-to-vowel context (defines the flap; spans word boundaries
                # for cases like "that is" -> tha[ɾ]is) — for audible cutouts.
                "ctx_start": seq[i - 1].get("start"), "ctx_end": seq[i + 1].get("end"),
                "seq": seq,
            })
    return rows


def word_window(row):
    """Audible window: the vowel-to-vowel context around the token (so the flap
    is heard, even across a word boundary), generously padded so it sounds
    natural."""
    cs, ce = row.get("ctx_start"), row.get("ctx_end")
    if cs is not None and ce is not None:
        return max(0.0, cs - 0.18), ce + 0.18
    return max(0.0, row["start"] - 0.28), row["end"] + 0.28


def cut_audio(row, lang, dest: Path):
    wav = REPO / "data" / "audio" / lang / row["key"]
    audio, sr = sf.read(str(wav))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    ws, we = word_window(row)
    seg = audio[int(ws * sr):int(we * sr)]
    sf.write(str(dest), seg, sr)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "eng_tatoeba_CK"
    lang = sys.argv[2] if len(sys.argv) > 2 else "eng"
    rows = collect(tag, lang)

    # overall stats
    from collections import Counter
    dec = Counter(r["decision"] for r in rows)
    t_rows = [r for r in rows if r["base"] == "t"]
    d_rows = [r for r in rows if r["base"] == "d"]
    print(f"=== {tag}: {len(rows)} intervocalic /t,d/ tokens ===")
    print(f"  /t/ n={len(t_rows)}  /d/ n={len(d_rows)}")
    print(f"  decisions: flap={dec['flap']}  keep={dec['keep']}  ambiguous={dec['ambiguous']}")
    tf = [r for r in t_rows if r['decision'] == 'flap']
    print(f"  /t/ → [ɾ]: {len(tf)}/{len(t_rows)} ({100*len(tf)/max(1,len(t_rows)):.0f}%)")

    # sample across the voiced_frac range for ear review (only /t/, the clean case),
    # diversifying by word so we don't get the same word repeated.
    def word_of(r):
        ws = r["sentence"].split()
        return ws[r["word_idx"]].lower().strip(".,!?\"'") if 0 <= r["word_idx"] < len(ws) else "?"

    def diverse(cands, k):
        out, seen = [], set()
        for r in cands:
            w = word_of(r)
            if w in seen:
                continue
            seen.add(w); out.append(r)
            if len(out) >= k:
                break
        return out

    tv = sorted([r for r in t_rows if r["voiced_frac"] is not None], key=lambda r: -r["voiced_frac"])
    picks = []
    picks += [("FLAP", r) for r in diverse(tv, 4)]                                   # highest vf, distinct words
    picks += [("AMBIG", r) for r in diverse([r for r in tv if 0.35 < r["voiced_frac"] < 0.6], 2)]
    picks += [("STOP", r) for r in diverse(list(reversed(tv)), 2)]                   # lowest vf, distinct words
    dv = sorted([r for r in d_rows if r.get("phone_dur")], key=lambda r: r["phone_dur"])
    picks += [("d?", r) for r in diverse(dv, 2)]

    sample_dir = OUT / "flap_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    print("\n=== ear-review sample (listen and judge: is it a flap [ɾ] or a true stop?) ===")
    for n, (tier, r) in enumerate(picks, 1):
        fname = f"{n:02d}_{tier}_{r['base']}_vf{(r['voiced_frac'] or 0):.2f}.wav"
        cut_audio(r, lang, sample_dir / fname)
        # text word
        words = r["sentence"].split()
        wtxt = words[r["word_idx"]] if 0 <= r["word_idx"] < len(words) else "?"
        manifest.append({"file": fname, "pred": r["label"], "decision": r["decision"],
                         "voiced_frac": r["voiced_frac"], "burst": r["burst"],
                         "phone_dur_ms": round((r["phone_dur"] or 0) * 1000, 0),
                         "word": wtxt, "sentence": r["sentence"]})
        print(f"  {fname}")
        print(f"     pred=[{r['label']}] ({r['decision']}/{r['confidence']}) "
              f"vf={r['voiced_frac']} burst={r['burst']:.0f}dB dur={round((r['phone_dur'] or 0)*1000)}ms "
              f"| word={wtxt!r}")
    (sample_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nwrote {len(picks)} clips + manifest -> {sample_dir}")


if __name__ == "__main__":
    main()

"""Aggregate per-segment acoustics into within-speaker summaries + detect where
espeak's broad/citation transcription systematically diverges from the audio.

Everything anchors WITHIN speaker: a vowel is judged against that speaker's own
other vowels, never textbook Hz. Outputs:
  out/<tag>.summary.json  machine-readable aggregates + flagged phenomena
  out/<tag>.report.txt    human-readable

Detectors:
  EN: intervocalic /t,d/ flapping; /u/-fronting; unstressed reduction.
  FR: nasal-vowel realization (A1-P0 / B1); schwa-ə vs ø overlap; schwa deletion.
  ALL: over-specification (espeak phones the aligner squeezes to ~0 frames).
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

VOWELS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒ")


def _base(sym: str) -> str:
    return sym[0] if sym else ""


def is_vowel(sym: str) -> bool:
    return bool(sym) and sym[0] in VOWELS


def load_segments(tag: str) -> list[dict]:
    p = OUT / f"{tag}.segments.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def clip_sequences(segs: list[dict]) -> dict[str, list[dict]]:
    by_clip = defaultdict(list)
    for s in segs:
        by_clip[s["key"]].append(s)
    for k in by_clip:
        by_clip[k].sort(key=lambda s: s["idx"])
        _annotate_phone_dur(by_clip[k])
    return by_clip


def _annotate_phone_dur(seq: list[dict]) -> None:
    """Real phone durations from CTC alignment: split the blank gaps between
    consecutive emitted spans at their midpoint, so each phone gets contiguous
    coverage. The raw CTC span is peaky (~1 frame); this recovers usable
    durations for reduction and deletion analysis. Sets s['phone_dur'] (sec)
    on every segment with a span; leaves it absent on OOV phones.
    """
    placed = [s for s in seq if s.get("start") is not None]
    for i, s in enumerate(placed):
        left = (placed[i - 1]["end"] + s["start"]) / 2.0 if i > 0 else s["start"]
        right = (s["end"] + placed[i + 1]["start"]) / 2.0 if i + 1 < len(placed) else s["end"]
        s["phone_dur"] = max(0.0, right - left)


def _stats(xs: list[float]) -> dict | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs.sort()
    return {
        "n": len(xs),
        "median": round(st.median(xs), 1),
        "mean": round(st.fmean(xs), 1),
        "sd": round(st.pstdev(xs), 1) if len(xs) > 1 else 0.0,
        "p10": round(xs[max(0, int(0.1 * len(xs)) - 1)], 1),
        "p90": round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 1),
    }


def vowel_table(segs: list[dict]) -> dict:
    """Per espeak vowel symbol: F1/F2/F3/B1/A1P0/dur aggregates, split by stress."""
    by = defaultdict(lambda: defaultdict(list))
    for s in segs:
        if s.get("kind") != "vowel" or s.get("oov"):
            continue
        key = s["symbol"]
        for f in ("f1", "f2", "f3", "b1", "a1_p0", "dur"):
            by[key][f].append(s.get(f))
        by[key]["stress"].append(s.get("stress"))
    out = {}
    for sym, d in by.items():
        out[sym] = {f: _stats(d[f]) for f in ("f1", "f2", "f3", "b1", "a1_p0", "dur")}
        out[sym]["n_tokens"] = len(d["f1"])
    return out


# ---------------------------------------------------------------------------
# English detectors
# ---------------------------------------------------------------------------

def detect_flapping(by_clip: dict) -> dict:
    """Intervocalic /t,d/ (incl. across word boundary): AmE flaps -> short,
    voiced, no burst. Compare to non-intervocalic /t,d/."""
    inter, other = [], []
    for seq in by_clip.values():
        for i, s in enumerate(seq):
            if _base(s["symbol"]) not in ("t", "d") or s.get("oov"):
                continue
            prev_v = i > 0 and is_vowel(seq[i - 1]["symbol"])
            next_v = i + 1 < len(seq) and is_vowel(seq[i + 1]["symbol"])
            rec = {
                "symbol": s["symbol"], "dur": s.get("dur"),
                "voiced_frac": s.get("voiced_frac"),
                "burst_hi_lo_db": s.get("burst_hi_lo_db"),
                "key": s["key"], "idx": s["idx"],
            }
            (inter if (prev_v and next_v) else other).append(rec)

    def summarize(rows):
        return {
            "n": len(rows),
            "dur_ms": _stats([r["dur"] * 1000 for r in rows if r["dur"]]),
            "voiced_frac": _stats([r["voiced_frac"] for r in rows]),
            "burst_hi_lo_db": _stats([r["burst_hi_lo_db"] for r in rows]),
        }

    # A flap-like intervocalic token: short (<60ms) AND substantially voiced (>0.5).
    flaplike = [r for r in inter if r["dur"] and r["dur"] < 0.060 and (r["voiced_frac"] or 0) > 0.5]
    return {
        "intervocalic": summarize(inter),
        "non_intervocalic": summarize(other),
        "intervocalic_flaplike_frac": round(len(flaplike) / len(inter), 3) if inter else None,
        "examples_flaplike": flaplike[:12],
    }


def detect_u_fronting(vt: dict) -> dict:
    """AmE /u/ is fronted: F2 well above the ~900-1100 Hz espeak's back /u/ implies."""
    out = {}
    for sym in ("uː", "u", "ʊ"):
        if sym in vt and vt[sym]["f2"]:
            out[sym] = {"f2": vt[sym]["f2"], "f1": vt[sym]["f1"], "n": vt[sym]["n_tokens"]}
    return out


def detect_reduction(by_clip: dict) -> dict:
    """Unstressed vowels should shorten + centralise vs stressed. Uses phone_dur
    (midpoint-boundary durations), not the peaky raw span."""
    stressed, unstressed = [], []
    for seq in by_clip.values():
        for s in seq:
            if s.get("kind") != "vowel" or s.get("oov") or s.get("f1") is None:
                continue
            rec = (s["f1"], s.get("f2"), s.get("phone_dur"))
            (stressed if s.get("stress") else unstressed).append(rec)
    def agg(rows):
        return {
            "n": len(rows),
            "f1": _stats([r[0] for r in rows]),
            "f2": _stats([r[1] for r in rows]),
            "dur_ms": _stats([r[2] * 1000 for r in rows if r[2] is not None]),
        }
    return {"stressed": agg(stressed) if stressed else None,
            "unstressed": agg(unstressed) if unstressed else None}


# ---------------------------------------------------------------------------
# French detectors
# ---------------------------------------------------------------------------

NASAL_VOWELS = {"ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "ã", "õ", "ẽ"}


def detect_nasalization(vt: dict, segs: list[dict]) -> dict:
    """Nasal vowels: wide B1 + low A1-P0 vs the speaker's oral vowels.

    Anchors against the speaker's OWN oral vowels (mean B1 / A1-P0)."""
    oral_b1, oral_a1p0 = [], []
    for s in segs:
        if s.get("kind") != "vowel" or s.get("oov"):
            continue
        if s["symbol"] in NASAL_VOWELS:
            continue
        if s.get("b1") is not None:
            oral_b1.append(s["b1"])
        if s.get("a1_p0") is not None:
            oral_a1p0.append(s["a1_p0"])
    out = {"oral_ref": {"b1": _stats(oral_b1), "a1_p0": _stats(oral_a1p0)}, "nasals": {}}
    for sym in vt:
        if sym in NASAL_VOWELS:
            out["nasals"][sym] = {"b1": vt[sym]["b1"], "a1_p0": vt[sym]["a1_p0"],
                                  "f1": vt[sym]["f1"], "f2": vt[sym]["f2"],
                                  "n": vt[sym]["n_tokens"]}
    return out


def detect_schwa_vs_round(vt: dict) -> dict:
    """Does the speaker's ə collapse toward ø? Compare F1/F2 clusters (the
    chat's ə≡ø finding). Decisive dim is F1 (height) and F2 (front/round)."""
    out = {}
    for sym in ("ə", "ø", "œ", "œ̃", "e", "ɛ"):
        if sym in vt and vt[sym]["f1"]:
            out[sym] = {"f1": vt[sym]["f1"], "f2": vt[sym]["f2"], "n": vt[sym]["n_tokens"]}
    return out


# ---------------------------------------------------------------------------
# Generic cross-language detectors
# ---------------------------------------------------------------------------

VOICED_STOPS = set("bdɡg")
VOICELESS_STOPS = set("ptk")


def detect_obstruents(by_clip: dict) -> dict:
    """Voicing + burst of stops by position. One pass catches several espeak
    blind spots:
      - final devoicing (DE/RU): word-final voiced /b d ɡ/ with low voiced_frac
      - intervocalic voicing/lenition (ES /b d ɡ/ -> [β ð ɣ]; EN /t d/ -> flap)
      - aspiration proxies via burst tilt
    voiced_frac/burst were measured on the RAW span (stop core)."""
    buckets = defaultdict(lambda: {"voiced": [], "burst": [], "dur": []})

    def pos_of(seq, i):
        s = seq[i]
        pv = i > 0 and is_vowel(seq[i - 1]["symbol"])
        nv = i + 1 < len(seq) and is_vowel(seq[i + 1]["symbol"])
        word_final = (i + 1 >= len(seq)) or (seq[i + 1].get("word_idx") != s.get("word_idx"))
        if pv and nv:
            return "intervocalic"
        if word_final:
            return "word_final"
        if nv:
            return "onset"
        return "coda_other"

    for seq in by_clip.values():
        for i, s in enumerate(seq):
            b = _base(s["symbol"])
            if s.get("oov") or b not in (VOICED_STOPS | VOICELESS_STOPS):
                continue
            voi = "voiced" if b in VOICED_STOPS else "voiceless"
            key = (voi, pos_of(seq, i))
            buckets[key]["voiced"].append(s.get("voiced_frac"))
            buckets[key]["burst"].append(s.get("burst_hi_lo_db"))
            buckets[key]["dur"].append(s.get("phone_dur"))

    out = {}
    for (voi, pos), d in buckets.items():
        out[f"{voi}/{pos}"] = {
            "n": len(d["voiced"]),
            "voiced_frac": _stats(d["voiced"]),
            "burst_hi_lo_db": _stats(d["burst"]),
            "dur_ms": _stats([x * 1000 for x in d["dur"] if x is not None]),
        }
    return out


def detect_model_agreement(tag: str) -> dict:
    """How often the espeak-trained model's greedy reading equals espeak (it
    should be ~always — confirming the model can't arbitrate). Surfaces the rare
    systematic model!=espeak substitutions via difflib alignment."""
    import difflib
    p = OUT / f"{tag}.clips.jsonl"
    if not p.exists():
        return {}
    clips = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    exact = 0
    subs = Counter()
    dels = Counter()
    ins = Counter()
    for c in clips:
        e, m = c.get("espeak", []), c.get("model_reading", [])
        if e == m:
            exact += 1
        sm = difflib.SequenceMatcher(a=e, b=m, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "replace":
                for a, bb in zip(e[i1:i2], m[j1:j2]):
                    subs[f"{a}->{bb}"] += 1
            elif op == "delete":
                for a in e[i1:i2]:
                    dels[a] += 1
            elif op == "insert":
                for bb in m[j1:j2]:
                    ins[bb] += 1
    return {
        "n_clips": len(clips),
        "exact_match_frac": round(exact / len(clips), 3) if clips else None,
        "top_substitutions": subs.most_common(12),
        "top_model_deletions_of_espeak": dels.most_common(8),
        "top_model_insertions": ins.most_common(8),
    }


# ---------------------------------------------------------------------------
# Over-specification (any language)
# ---------------------------------------------------------------------------

def detect_overspecification(by_clip: dict) -> dict:
    """espeak phones the aligner could barely place: tiny phone_dur AND low
    alignment score => strong candidates for phones espeak posits that aren't
    really in the audio (deletions/epenthesis). phone_dur (midpoint boundaries)
    is a real duration, so unlike the raw peaky span it isn't ~20ms for
    everything. We require BOTH short duration and poor score to avoid the CTC
    peakiness artifact."""
    deleted = []
    by_sym = defaultdict(lambda: {"n": 0, "deleted": 0})
    for seq in by_clip.values():
        for s in seq:
            if s.get("oov") or s.get("phone_dur") is None:
                continue
            by_sym[s["symbol"]]["n"] += 1
            score = s.get("align_score")
            is_deleted = s["phone_dur"] < 0.025 and (score is not None and score < -0.5)
            if is_deleted:
                by_sym[s["symbol"]]["deleted"] += 1
                deleted.append({"symbol": s["symbol"], "phone_dur_ms": round(s["phone_dur"] * 1000, 1),
                                "score": round(score, 2), "key": s["key"], "idx": s["idx"]})
    ranked = sorted(
        ((sym, d["deleted"] / d["n"], d["n"]) for sym, d in by_sym.items() if d["n"] >= 4),
        key=lambda x: -x[1],
    )
    return {
        "n_deleted_candidates": len(deleted),
        "deletion_rate_by_symbol": [{"symbol": s, "deletion_rate": round(r, 3), "n": n}
                                    for s, r, n in ranked[:15] if r > 0],
        "examples": deleted[:20],
    }


def analyze(tag: str, lang: str) -> dict:
    segs = load_segments(tag)
    by_clip = clip_sequences(segs)
    vt = vowel_table(segs)
    summary = {
        "tag": tag, "lang": lang,
        "n_clips": len(by_clip), "n_segments": len(segs),
        "vowel_table": vt,
        # generic detectors run for EVERY language:
        "reduction": detect_reduction(by_clip),
        "obstruents": detect_obstruents(by_clip),
        "nasalization": detect_nasalization(vt, segs),
        "overspecification": detect_overspecification(by_clip),
        "model_agreement": detect_model_agreement(tag),
    }
    # language-specific extras:
    if lang == "eng":
        summary["flapping"] = detect_flapping(by_clip)
        summary["u_fronting"] = detect_u_fronting(vt)
    if lang == "fra":
        summary["schwa_vs_round"] = detect_schwa_vs_round(vt)
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lang", required=True)
    args = ap.parse_args()
    summary = analyze(args.tag, args.lang)
    (OUT / f"{args.tag}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

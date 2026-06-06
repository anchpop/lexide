"""Acoustic measurement of phone segments via parselmouth (Praat DSP).

This is the *independent arbiter* for the espeak audit. espeak gives a broad,
citation-form phonemic transcription; a model trained on espeak labels inherits
that broadness. Only the signal itself can tell us what was actually produced,
so every "narrow" claim ultimately grounds in numbers measured here.

Methodology mirrors the hand-analysis that motivated this project:

  - Formants via Burg LPC (vocal tract = all-pole filter; poles = formants).
    The one setting that matters is the ceiling (5000 Hz male / 5500 Hz female);
    set it wrong and the tracker renumbers every formant. We pick it per clip
    from median F0.
  - Measure the steady centre of a segment (central 60%), not the edges where
    coarticulatory transitions live. Use medians, robust to tracker outliers.
  - Nasalization: F1 bandwidth widening + the A1-P0 spectral metric (a low nasal
    pole near ~250 Hz emerges, dropping A1-P0 by 6-8 dB at matched F1).
  - Stops: VOT/aspiration, closure voicing, and burst spectral tilt (place).

Functions return plain dicts of floats (or None when a measurement is
unreliable / not applicable) so the output serialises straight to JSONL.
"""

from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

# ---------------------------------------------------------------------------
# Loading & whole-clip contours
# ---------------------------------------------------------------------------

def load_sound(path: str) -> parselmouth.Sound:
    snd = parselmouth.Sound(str(path))
    # Force mono; the dataset is 16k mono already but be safe.
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    return snd


def sound_from_array(values, sr) -> parselmouth.Sound:
    """Build a Sound from an in-memory mono sample array (no file read).

    Used by the at-scale measure-on-Modal path, where the audio is already in
    hand from the alignment call — avoids a disk round-trip per clip."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[1] else 1)
    return parselmouth.Sound(arr, sampling_frequency=float(sr))


def measure_segments(audio, sr, phonemes, stress, word_of, ok, keep, spans) -> list[dict]:
    """Acoustic measurement of every espeak phone from an in-memory clip + its
    aligned spans. Pure (no I/O, no clip-identity fields) so it runs identically
    locally or on a Modal container. Mirrors run_audit.measure_clip exactly;
    `spans` are [start_s, end_s, score] in `keep` order.
    """
    snd = sound_from_array(audio, sr)
    f0 = clip_f0_stats(snd)
    ceiling = formant_ceiling_for(f0["f0_median"])

    if len(spans) != len(keep):
        # Alignment skipped / mismatched: measure nothing, don't misattribute.
        return [{
            "idx": j, "symbol": sym, "stress": stress[j], "word_idx": word_of[j],
            "oov": not ok[j], "start": None, "end": None, "dur": None,
            "align_score": None, "align_skipped": True,
        } for j, sym in enumerate(phonemes)]
    span_by_j = {j: spans[k] for k, j in enumerate(keep)}

    MAXD = 0.030  # dilate peaky CTC spans into surrounding blanks for measurement
    win_by_j = {}
    for k, (s, e, _sc) in enumerate(spans):
        prev_e = spans[k - 1][1] if k > 0 else s
        next_s = spans[k + 1][0] if k + 1 < len(spans) else e
        ws = s - min(MAXD, max(0.0, (s - prev_e) / 2.0))
        we = e + min(MAXD, max(0.0, (next_s - e) / 2.0))
        win_by_j[keep[k]] = (ws, we)

    rows = []
    for j, sym in enumerate(phonemes):
        base = {
            "idx": j, "symbol": sym, "stress": stress[j], "word_idx": word_of[j],
            "oov": not ok[j], "f0_median_clip": f0["f0_median"], "ceiling": ceiling,
        }
        if not ok[j] or j not in span_by_j:
            base.update({"start": None, "end": None, "dur": None, "align_score": None})
            rows.append(base)
            continue
        s, e, score = span_by_j[j]
        ws, we = win_by_j[j]
        base.update({"start": s, "end": e, "dur": e - s, "align_score": score,
                     "win_start": ws, "win_end": we})
        try:
            if is_vowel(sym):
                base["kind"] = "vowel"
                base.update(measure_vowel(snd, ws, we, ceiling))
            elif is_stop(sym):
                base["kind"] = "stop"      # RAW span, not dilated (see run_audit)
                base.update(measure_stop(snd, s, e))
            else:
                base["kind"] = "other"
                base["intensity_db"] = _mean_intensity(snd, s, e)
        except Exception as ex:  # measurement failure shouldn't kill the clip
            base["measure_error"] = str(ex)
        rows.append(base)
    return rows


def clip_f0_stats(snd: parselmouth.Sound) -> dict:
    """Median/min/max F0 over the whole clip (voiced frames only).

    Used both to set the formant ceiling and to characterise the speaker.
    """
    pitch = snd.to_pitch(time_step=0.005)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return {"f0_median": None, "f0_min": None, "f0_max": None, "voiced_frac": 0.0}
    return {
        "f0_median": float(np.median(voiced)),
        "f0_min": float(np.min(voiced)),
        "f0_max": float(np.max(voiced)),
        "voiced_frac": float(voiced.size / f0.size),
    }


def formant_ceiling_for(f0_median: float | None) -> float:
    """5500 Hz for higher-pitched (likely female) voices, 5000 for lower.

    This is the standard Praat heuristic. A boundary at ~170 Hz median F0
    separates the two regimes well enough; the exact value matters far less
    than not being off by a whole formant.
    """
    if f0_median is None:
        return 5500.0
    return 5500.0 if f0_median > 170.0 else 5000.0


# ---------------------------------------------------------------------------
# Vowel measurement
# ---------------------------------------------------------------------------

def _central_window(t0: float, t1: float, frac: float = 0.6) -> tuple[float, float]:
    """Central `frac` of [t0, t1]; trims coarticulatory edges."""
    dur = t1 - t0
    pad = (1.0 - frac) / 2.0 * dur
    return t0 + pad, t1 - pad


def measure_vowel(
    snd: parselmouth.Sound,
    t0: float,
    t1: float,
    ceiling: float,
) -> dict:
    """Median F1/F2/F3 + bandwidths over the central 60%, plus nasalization.

    Returns dict with formants (Hz), bandwidths (Hz), A1-P0 (dB), duration,
    and per-formant frame-to-frame stddev (a tracker-reliability flag). Values
    that can't be measured come back as None.
    """
    c0, c1 = _central_window(t0, t1)
    out: dict = {"win_dur": float(t1 - t0)}

    formant = snd.to_formant_burg(
        time_step=0.0025,
        max_number_of_formants=5,
        maximum_formant=ceiling,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )

    # Sample formant tracks across the central window.
    times = [t for t in np.arange(c0, c1, 0.0025) if t < c1]
    if len(times) < 3:
        # Segment too short to trust; sample whatever we can at the midpoint.
        times = [(t0 + t1) / 2.0]

    for fi, key in ((1, "f1"), (2, "f2"), (3, "f3")):
        vals = []
        bws = []
        for t in times:
            v = formant.get_value_at_time(fi, t)
            b = call(formant, "Get bandwidth at time", fi, t, "hertz", "linear")
            if v is not None and not np.isnan(v):
                vals.append(v)
            if b is not None and not np.isnan(b):
                bws.append(b)
        out[key] = float(np.median(vals)) if vals else None
        out[key + "_sd"] = float(np.std(vals)) if len(vals) > 1 else None
        out["b" + str(fi)] = float(np.median(bws)) if bws else None

    out["a1_p0"] = _a1_p0(snd, c0, c1, out.get("f1"))
    out["intensity_db"] = _mean_intensity(snd, c0, c1)
    return out


def _mean_intensity(snd: parselmouth.Sound, t0: float, t1: float) -> float | None:
    try:
        intensity = snd.to_intensity(time_step=0.005)
        vals = [
            intensity.get_value(t)
            for t in np.arange(t0, t1, 0.005)
            if intensity.get_value(t) is not None
        ]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else None
    except Exception:
        return None


def _a1_p0(snd: parselmouth.Sound, t0: float, t1: float, f1: float | None) -> float | None:
    """A1 - P0 nasalization metric (Chen 1997), in dB.

    A1 = spectral level near F1. P0 = level of the low nasal pole (~180-350 Hz).
    Nasalisation introduces a low-frequency pole that raises P0 and damps A1,
    so a *small or negative* A1-P0 signals nasal coupling. Computed on a smoothed
    spectrum (Welch) of the central vowel. Returns None if F1 unknown or F1 sits
    inside the P0 band (metric undefined for very low vowels).
    """
    if f1 is None or f1 < 400.0:
        # F1 inside / too near the P0 band -> metric confounded. Skip.
        return None
    try:
        from scipy.signal import welch
        seg = snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)
        x = seg.values[0].astype(np.float64)
        fs = 1.0 / snd.dx
        if x.size < 256:
            return None
        nperseg = min(1024, x.size)
        freqs, psd = welch(x, fs=fs, nperseg=nperseg)
        psd_db = 10.0 * np.log10(psd + 1e-20)

        def peak_db(lo, hi):
            m = (freqs >= lo) & (freqs <= hi)
            return float(np.max(psd_db[m])) if m.any() else None

        p0 = peak_db(180.0, 350.0)
        a1 = peak_db(f1 - 150.0, f1 + 150.0)
        if p0 is None or a1 is None:
            return None
        return float(a1 - p0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stop / consonant measurement
# ---------------------------------------------------------------------------

def measure_stop(snd: parselmouth.Sound, t0: float, t1: float) -> dict:
    """Closure voicing + burst spectral tilt for an oral stop.

    Distinguishes flap [ɾ] (very short, fully voiced, no burst) from a true
    [t]/[d], and aspirated [tʰ] (long high-freq burst) from unaspirated.
      - dur: total segment duration (flaps are ~20-40 ms)
      - voiced_frac: fraction of frames with detectable F0 (voicing through
        closure -> [d]/[ɾ]; voiceless -> [t])
      - burst_hi_lo_db: energy in 3-6 kHz minus 0.3-1.5 kHz at the release.
        Alveolar bursts are high-frequency (>0); labial bursts diffuse/low (<0).
    """
    out: dict = {"win_dur": float(t1 - t0)}
    try:
        pitch = snd.to_pitch(time_step=0.005)
        times = list(np.arange(t0, t1, 0.005))
        voiced = 0
        for t in times:
            v = pitch.get_value_at_time(t)
            if v is not None and not np.isnan(v) and v > 0:
                voiced += 1
        out["voiced_frac"] = float(voiced / len(times)) if times else None
    except Exception:
        out["voiced_frac"] = None

    # Burst: spectrum of the last ~25 ms of the segment (the release).
    out["burst_hi_lo_db"] = _band_tilt(snd, max(t0, t1 - 0.025), t1)
    return out


def _band_tilt(snd: parselmouth.Sound, t0: float, t1: float) -> float | None:
    try:
        from scipy.signal import welch
        seg = snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)
        x = seg.values[0].astype(np.float64)
        fs = 1.0 / snd.dx
        if x.size < 64:
            return None
        nperseg = min(256, x.size)
        freqs, psd = welch(x, fs=fs, nperseg=nperseg)
        psd_db = 10.0 * np.log10(psd + 1e-20)

        def band(lo, hi):
            m = (freqs >= lo) & (freqs <= hi)
            return float(np.mean(psd_db[m])) if m.any() else None

        hi = band(3000.0, 6000.0)
        lo = band(300.0, 1500.0)
        if hi is None or lo is None:
            return None
        return float(hi - lo)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# IPA helpers
# ---------------------------------------------------------------------------

# Vowel base characters we care about (after stripping length/nasal diacritics).
_VOWEL_CHARS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒ")


def is_vowel(symbol: str) -> bool:
    """True if the espeak phoneme's base char is a vowel."""
    if not symbol:
        return False
    return symbol[0] in _VOWEL_CHARS


def is_stop(symbol: str) -> bool:
    base = symbol[0] if symbol else ""
    return base in set("ptkbdɡg")


def is_nasal_vowel(symbol: str) -> bool:
    return is_vowel(symbol) and ("̃" in symbol or "̃" in symbol)

"""Interactive audit of nasal-vowel failures + acoustic proxy.

For each clip in the input file:
  - Compute a simple acoustic nasalization proxy (low-band / F1-band energy ratio)
  - Play the audio via afplay (macOS)
  - Show expected/predicted phonemes, ask user to rate (n/o/?)
  - Write results to CSV (append-only, resumable)

Input file format (one entry per line):
    path/to/file.wav  exp=[p ɛ̃]  pred=[t a]

Or just paths (no exp/pred):
    path/to/file.wav

Usage:
    python -m scripts.audit_nasals fails.txt --output ratings.csv
    python -m scripts.audit_nasals fails.txt --resume   # continue past prior ratings
    python -m scripts.audit_nasals fails.txt --proxy-only   # no audio, just compute proxy
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

LINE_RE = re.compile(r"^(\S+)(?:\s+exp=\[([^\]]*)\])?(?:\s+pred=\[([^\]]*)\])?\s*$")


def parse_failures(path: Path):
    entries = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.match(line)
            if not m:
                # Be lenient — also accept lines that are just a path.
                if Path(line).exists():
                    entries.append({"path": line, "exp": [], "pred": []})
                continue
            entries.append({
                "path": m.group(1),
                "exp": (m.group(2) or "").split(),
                "pred": (m.group(3) or "").split(),
            })
    return entries


def nasal_proxy(audio: np.ndarray, sr: int) -> dict:
    """Rough acoustic nasalization estimate.

    Nasalized vowels typically show:
      - extra spectral peak around 200-350 Hz (the nasal pole, ~velar port resonance)
      - reduced/widened F1 (~400-1000 Hz depending on vowel)

    We compute energy in those two bands on the loudest 60% of the clip
    (a crude proxy for the voiced/vowel segment) and return the nasal-to-F1
    ratio. Higher = more nasal. Threshold between nasal and oral varies
    by vowel; use scores RELATIVELY across clips, not absolutely.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    # Trim to loudest 60% by sliding RMS (~20 ms windows).
    win = max(1, int(0.02 * sr))
    if len(audio) > 4 * win:
        rms = np.sqrt(
            np.convolve(audio ** 2, np.ones(win) / win, mode="valid")
        )
        thresh = np.quantile(rms, 0.4)
        keep = rms > thresh
        if keep.any():
            audio = audio[np.argmax(keep): len(keep) - np.argmax(keep[::-1])]

    if len(audio) < 256:
        return {"nasal_score": float("nan"), "nasal_db": float("nan"), "f1_db": float("nan")}

    # Power spectrum over the trimmed segment.
    n = len(audio)
    spec = np.abs(np.fft.rfft(audio * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)

    nasal_band = (freqs >= 200) & (freqs <= 350)
    f1_band = (freqs >= 400) & (freqs <= 1000)

    nasal_e = spec[nasal_band].sum() + 1e-12
    f1_e = spec[f1_band].sum() + 1e-12
    return {
        "nasal_score": float(nasal_e / f1_e),
        "nasal_db": float(10 * np.log10(nasal_e)),
        "f1_db": float(10 * np.log10(f1_e)),
    }


def play(path: Path):
    """Play via afplay (macOS). Returns silently if unavailable."""
    try:
        subprocess.run(["afplay", str(path)], check=False, capture_output=True)
    except FileNotFoundError:
        print("  (afplay not found — use --proxy-only or install Mac audio)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("failures_file", type=Path)
    p.add_argument("--output", type=Path, default=Path("audit_results.csv"))
    p.add_argument("--resume", action="store_true",
                   help="Skip entries already in the output CSV.")
    p.add_argument("--proxy-only", action="store_true",
                   help="Print proxy scores only; no audio, no prompts.")
    args = p.parse_args()

    entries = parse_failures(args.failures_file)
    print(f"Loaded {len(entries)} entries.\n")

    # Resume support
    seen = set()
    if args.resume and args.output.exists():
        with open(args.output) as f:
            for row in csv.DictReader(f):
                seen.add(row["path"])
        print(f"Resuming — {len(seen)} already rated.\n")

    is_new = not args.output.exists()
    out = open(args.output, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=[
        "path", "exp", "pred", "nasal_score", "rating", "notes",
    ])
    if is_new:
        writer.writeheader()

    counts = {"n": 0, "o": 0, "?": 0, "s": 0}
    try:
        for i, e in enumerate(entries):
            if e["path"] in seen:
                continue

            path = Path(e["path"])
            if not path.exists():
                print(f"[{i+1}/{len(entries)}] {e['path']}  (file not found, skipping)")
                continue

            try:
                audio, sr = sf.read(str(path))
                proxy = nasal_proxy(audio, sr)
            except Exception as ex:
                print(f"[{i+1}/{len(entries)}] {e['path']}  (read error: {ex})")
                continue

            print(f"\n[{i+1}/{len(entries)}] {e['path']}")
            if e["exp"]:
                print(f"  Expected:    {' '.join(e['exp'])}")
            if e["pred"]:
                print(f"  Predicted:   {' '.join(e['pred'])}")
            print(f"  Nasal proxy: {proxy['nasal_score']:.3f}  "
                  f"(nasal-band {proxy['nasal_db']:+.1f} dB, F1-band {proxy['f1_db']:+.1f} dB)")

            if args.proxy_only:
                writer.writerow({
                    "path": e["path"],
                    "exp": " ".join(e["exp"]),
                    "pred": " ".join(e["pred"]),
                    "nasal_score": f"{proxy['nasal_score']:.4f}",
                    "rating": "",
                    "notes": "",
                })
                continue

            play(path)
            while True:
                prompt = "  Rate [Enter/r=replay, n=nasal, o=oral, ?=ambiguous, s=skip, q=quit]: "
                r = input(prompt).strip().lower()
                if r == "" or r == "r":
                    play(path)
                    continue
                if r in ("n", "o", "?"):
                    notes = input("  Notes (optional, Enter to skip): ").strip()
                    writer.writerow({
                        "path": e["path"],
                        "exp": " ".join(e["exp"]),
                        "pred": " ".join(e["pred"]),
                        "nasal_score": f"{proxy['nasal_score']:.4f}",
                        "rating": r,
                        "notes": notes,
                    })
                    out.flush()
                    counts[r] += 1
                    break
                if r == "s":
                    counts["s"] += 1
                    break
                if r == "q":
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\nStopping. Progress saved.")
    finally:
        out.close()

    print(f"\nDone — {args.output}")
    if not args.proxy_only:
        print(f"  Tallies: n={counts['n']}  o={counts['o']}  ?={counts['?']}  s={counts['s']}")


if __name__ == "__main__":
    main()

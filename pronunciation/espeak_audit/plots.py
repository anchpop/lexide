"""Figures for the espeak audit. Saves PNGs under out/figs/."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def vowel_space(tag, title):
    segs = A.load_segments(tag)
    pts = defaultdict(list)
    for s in segs:
        if s.get("kind") == "vowel" and not s.get("oov") and s.get("f1") and s.get("f2"):
            pts[s["symbol"]].append((s["f2"], s["f1"]))
    fig, ax = plt.subplots(figsize=(8, 6))
    for sym, xy in pts.items():
        if len(xy) < 2:
            continue
        f2 = [p[0] for p in xy]; f1 = [p[1] for p in xy]
        mf2 = sum(f2) / len(f2); mf1 = sum(f1) / len(f1)
        ax.scatter(f2, f1, s=8, alpha=0.25)
        ax.text(mf2, mf1, sym, fontsize=16, ha="center", va="center",
                weight="bold", color="black")
    ax.set_xlabel("F2 (Hz)  ← back   front →")
    ax.set_ylabel("F1 (Hz)  ← close   open →")
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.axvline(1500, ls=":", c="gray", alpha=0.5)
    ax.set_title(title)
    p = FIGS / f"{tag}.vowelspace.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def flap_voicing(tag, title):
    segs = A.load_segments(tag)
    by_clip = A.clip_sequences(segs)
    inter, other = [], []
    for seq in by_clip.values():
        for i, s in enumerate(seq):
            if A._base(s["symbol"]) not in ("t", "d") or s.get("oov"):
                continue
            if s.get("voiced_frac") is None:
                continue
            pv = i > 0 and A.is_vowel(seq[i - 1]["symbol"])
            nv = i + 1 < len(seq) and A.is_vowel(seq[i + 1]["symbol"])
            (inter if (pv and nv) else other).append(s["voiced_frac"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(other, bins=12, alpha=0.6, label=f"non-intervocalic /t,d/ (n={len(other)})")
    ax.hist(inter, bins=12, alpha=0.6, label=f"intervocalic /t,d/ (n={len(inter)})")
    ax.set_xlabel("voiced fraction of closure")
    ax.set_ylabel("count")
    ax.set_title(title + "\n(intervocalic shifted toward voiced ⇒ flap [ɾ]; espeak writes /t,d/)")
    ax.legend()
    p = FIGS / f"{tag}.flap_voicing.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def nasal_vs_oral(tag, title):
    segs = A.load_segments(tag)
    nas_b1, nas_a, oral_b1, oral_a = [], [], [], []
    for s in segs:
        if s.get("kind") != "vowel" or s.get("oov"):
            continue
        is_nasal = s["symbol"] in A.NASAL_VOWELS
        if s.get("b1") is not None:
            (nas_b1 if is_nasal else oral_b1).append(s["b1"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(oral_b1, bins=20, alpha=0.6, label=f"oral vowels (n={len(oral_b1)})")
    ax.hist(nas_b1, bins=15, alpha=0.7, label=f"nasal vowels (n={len(nas_b1)})")
    ax.set_xlabel("F1 bandwidth B1 (Hz)  — wider ⇒ nasal coupling")
    ax.set_ylabel("count")
    ax.set_xlim(0, 600)
    ax.set_title(title)
    ax.legend()
    p = FIGS / f"{tag}.nasal_b1.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


if __name__ == "__main__":
    made = []
    made.append(vowel_space("eng_tatoeba_CK", "CK (English, Tatoeba) vowel space — /uː/ sits at F2≈1650 (fronted)"))
    made.append(flap_voicing("eng_tatoeba_CK", "CK: closure voicing of /t,d/ by context"))
    made.append(flap_voicing("eng_tts_en-US-Chirp3-HD-Algieba", "Google TTS (Algieba): closure voicing of /t,d/ by context"))
    made.append(nasal_vs_oral("fra_tts_fr-FR-Chirp3-HD-Schedar", "French TTS (Schedar): nasal vs oral vowel F1 bandwidth"))
    for p in made:
        print(p)

"""Apply the validated flap detector to produce flap-narrowed phoneme labels.

PoC for per-token acoustic narrowing (the general pattern: espeak broad label →
forced-align → per-token acoustic measurement → confidence-gated relabel →
narrower label). Flapping is the first plugged-in rule; the structure
generalises to any phenomenon whose per-token signal is separable.

Produces a genuine drop-in: it starts from the real `phonemes.jsonl` rows (so all
fields — `source`, `stress`, `whisper_*`, `duration_sec` — are preserved) and
rewrites the `phonemes` field that the training pipeline (`StressDataset`)
actually reads, keeping the original under `phonemes_espeak`. It only relabels a
clip when the audit-measured phoneme sequence EXACTLY matches that clip's
`phonemes.jsonl` sequence, so a position can never be misattributed; clips that
don't align are skipped and counted (never silently corrupted).

Reuses the already-computed audit measurements (out/<tag>.segments.jsonl). To
scale to the full training set, run run_audit's align+measure over all eng clips
and point this at those segments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A          # noqa: E402
import flap_relabel as F     # noqa: E402

REPO = HERE.parent
OUT = HERE / "out"
FLAP = "ɾ"

# English sources with CLEAN English audio. Pimsleur is excluded: its English
# manifest contains mixed-language instructional rows (Arabic/Korean text read
# by the en-us voice), so its phoneme streams aren't English and its flap counts
# would be meaningless.
ENG_TAGS = [
    ("tatoeba (CK, human)", "eng_tatoeba_CK"),
    ("tts (Algieba)", "eng_tts_en-US-Chirp3-HD-Algieba"),
    ("fleurs (pooled)", "eng_fleurs_ALL"),
]


def load_phonemes_jsonl(lang: str) -> dict:
    rows = {}
    p = REPO / "data" / "audio" / lang / "phonemes.jsonl"
    for line in p.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            rows[d["file"]] = d
    return rows


def relabel_tag(tag: str, truth: dict, lang: str = "eng"):
    segs = A.load_segments(tag)
    by_clip = A.clip_sequences(segs)               # ordered + phone_dur annotated
    rows = F.collect(tag, lang)
    flap_pos = {(r["key"], r["idx"]) for r in rows if r["decision"] == "flap"}
    n_inter = len(rows)

    out_rows, n_flaps, n_aligned, n_unaligned, examples = [], 0, 0, 0, []
    for key, seq in by_clip.items():
        recon = [s["symbol"] for s in seq]
        base = truth.get(key)
        # only relabel when positions provably correspond to the real labels
        if base is None or base["phonemes"] != recon:
            n_unaligned += 1
            continue
        n_aligned += 1
        narrowed = list(base["phonemes"])
        changed = [p for p, s in enumerate(seq) if (key, s["idx"]) in flap_pos]
        for p in changed:
            narrowed[p] = FLAP
        n_flaps += len(changed)
        if changed and len(examples) < 6:
            examples.append((base["sentence"], base["phonemes"], narrowed, set(changed)))
        row = dict(base)                            # preserve ALL fields
        row["phonemes_espeak"] = base["phonemes"]   # keep the broad original
        row["phonemes"] = narrowed                  # the field training reads
        row["n_flaps"] = len(changed)
        out_rows.append(row)
    return out_rows, n_inter, n_flaps, n_aligned, n_unaligned, examples


def main():
    truth = load_phonemes_jsonl("eng")
    g_inter = g_flap = g_clips = g_unaligned = 0
    print("=== flap relabel → drop-in phonemes (confidence-gated) ===\n")
    for label, tag in ENG_TAGS:
        if not (OUT / f"{tag}.segments.jsonl").exists():
            print(f"{label}: (no segments — skip)"); continue
        rows, n_inter, n_flaps, n_aligned, n_unaligned, examples = relabel_tag(tag, truth)
        dest = OUT / f"{tag}.phonemes_flapped.jsonl"
        dest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        pct = 100 * n_flaps / n_inter if n_inter else 0
        skip = f" | UNALIGNED skipped={n_unaligned}" if n_unaligned else ""
        print(f"{label:22s}: {n_aligned:4d} clips | intervoc /t,d/={n_inter:4d} | "
              f"→[ɾ] {n_flaps:4d} ({pct:4.1f}%){skip} | -> {dest.name}")
        g_inter += n_inter; g_flap += n_flaps; g_clips += n_aligned; g_unaligned += n_unaligned
        if examples:
            sent, ph, fl, pos = examples[0]
            def mark(seq):
                return " ".join((f"[{t}]" if p in pos else t) for p, t in enumerate(seq))
            print(f"    e.g. {sent[:60]!r}")
            print(f"         espeak : {mark(ph)}")
            print(f"         flapped: {mark(fl)}")
    print(f"\nTOTAL: {g_clips} clips relabeled | {g_inter} intervoc /t,d/ | "
          f"{g_flap} → [ɾ] ({100*g_flap/max(1,g_inter):.1f}%) | unaligned skipped={g_unaligned}")
    print("Register: TTS > FLEURS > CK > (Pimsleur excluded — mixed-language data).\n"
          "Gate leaves ambiguous + stops + most /d/ as the broad symbol.")


if __name__ == "__main__":
    main()

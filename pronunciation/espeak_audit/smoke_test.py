"""End-to-end smoke test on ONE clip: prep -> Modal align -> measure -> print.

Run: /opt/homebrew/Caskroom/miniconda/base/bin/python3 espeak_audit/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_audit as R


def main():
    vocab = R.load_vocab()
    clips = R.select_clips("eng", "CK", "tatoeba", n=1)
    assert clips, "no clips selected"
    prep = R.prep_clip(clips[0], "eng", vocab)
    print("KEY:", prep["key"])
    print("TEXT:", prep["sentence"])
    print("ESPEAK:", " ".join(prep["phonemes"]))
    print("N phon:", len(prep["phonemes"]), "| mappable:", len(prep["keep"]),
          "| audio_sec:", round(len(prep["audio"]) / prep["sample_rate"], 2))

    aligned = R.align_on_modal([prep])
    a = aligned[prep["key"]]
    print("MODEL READING:", " ".join(a["reading"]))
    print("n_frames:", a["n_frames"], "| n_spans:", len(a["spans"]))

    rows = R.measure_clip(prep, a)
    print("\nper-phone alignment + key measurements:")
    for r in rows:
        if r.get("start") is None:
            print(f"  {r['symbol']:4s} OOV")
            continue
        line = (f"  {r['symbol']:4s} {r['start']:.3f}-{r['end']:.3f} "
                f"({r['dur']*1000:4.0f}ms) score={r['align_score']:.2f}")
        if r.get("kind") == "vowel":
            line += (f"  F1={r.get('f1')} F2={r.get('f2')} F3={r.get('f3')} "
                     f"B1={r.get('b1')} A1P0={r.get('a1_p0')}")
        elif r.get("kind") == "stop":
            line += f"  voiced_frac={r.get('voiced_frac')} burst_hi_lo={r.get('burst_hi_lo_db')}"
        print(line)


if __name__ == "__main__":
    main()

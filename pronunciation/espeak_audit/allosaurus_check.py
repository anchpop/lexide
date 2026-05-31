"""Independent narrow check via Allosaurus (CMU, ~2000-lang phone recognizer).

Allosaurus never saw espeak output or our model's labels, so where it and the
acoustics agree that the realization differs from espeak, that's independent
confirmation — not circular. We don't need alignment: a whole-utterance phone
string is enough to ask "does an independent recognizer emit [ɾ] (flap) where
espeak insists on /t,d/?" (espeak NEVER emits ɾ for English).
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_audit as R  # noqa: E402
import soundfile as sf  # noqa: E402
import modal  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def main():
    tag, lang, allo_lang, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    clips = [json.loads(l) for l in (OUT / f"{tag}.clips.jsonl").read_text().splitlines() if l.strip()][:n]
    Allo = modal.Cls.from_name("allosaurus", "Allosaurus")
    allo = Allo()

    lang_dir = R.REPO / "data" / "audio" / lang
    rows = []
    allo_inv = Counter()
    espeak_inv = Counter()
    n_clip_allo_flap = 0
    n_clip_espeak_flap = 0
    for c in clips:
        audio, sr = sf.read(str(lang_dir / c["key"]))
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        res = allo.transcribe.remote(audio.astype("float64").tolist(), int(sr), allo_lang)
        allo_phones = [p["phoneme"] for p in res["phonemes"]]
        espeak = c["espeak"]
        allo_inv.update(allo_phones)
        espeak_inv.update(espeak)
        allo_flap = any("ɾ" in p for p in allo_phones)
        espeak_flap = any("ɾ" in p for p in espeak)
        n_clip_allo_flap += int(allo_flap)
        n_clip_espeak_flap += int(espeak_flap)
        rows.append({"key": c["key"], "sentence": c["sentence"],
                     "espeak": " ".join(espeak), "allosaurus": " ".join(allo_phones)})

    (OUT / f"{tag}.allosaurus.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    print(f"=== {tag}: {len(clips)} clips, Allosaurus lang_id={allo_lang} ===")
    print(f"clips where Allosaurus emits ɾ (flap): {n_clip_allo_flap}/{len(clips)}")
    print(f"clips where espeak    emits ɾ (flap): {n_clip_espeak_flap}/{len(clips)}")
    print(f"Allosaurus ɾ total: {allo_inv.get('ɾ', 0)} | espeak ɾ total: {espeak_inv.get('ɾ', 0)}")
    # nasal vowels present?
    for nv in ("ɑ̃", "ɔ̃", "ɛ̃", "œ̃"):
        if espeak_inv.get(nv) or allo_inv.get(nv):
            print(f"  nasal {nv}: espeak={espeak_inv.get(nv,0)} allosaurus={allo_inv.get(nv,0)}")
    print("\n3 example pairs:")
    for r in rows[:3]:
        print(f"  TEXT: {r['sentence']}")
        print(f"   espeak: {r['espeak']}")
        print(f"   allo:   {r['allosaurus']}")


if __name__ == "__main__":
    main()

"""espeak audit driver: espeak(local) -> vad-clean align(Modal) -> parselmouth(local).

Run with the pinned interpreter that has modal + parselmouth + soundfile:
  /opt/homebrew/Caskroom/miniconda/base/bin/python3 espeak_audit/run_audit.py \
      --lang eng --voice CK --source tatoeba --n 60

Produces, under espeak_audit/out/:
  <tag>.clips.jsonl    one row per clip: text, espeak phonemes, model reading
  <tag>.segments.jsonl one row per espeak phone: symbol, span, dur, align_score,
                       oov, + acoustic measurements (vowels: F1-3/bw/A1P0; stops:
                       voicing/burst tilt). The arbiter data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "train" / "scripts"))
sys.path.insert(0, str(REPO / "espeak_audit"))

FORK = Path("/Users/andrepopovitch/coding/tmp/espeak-ng")
os.environ.setdefault("ESPEAK_NG_BIN", str(FORK / "build" / "src" / "espeak-ng"))
os.environ.setdefault("ESPEAK_NG_DATA_PATH", str(FORK / "build"))

import soundfile as sf  # noqa: E402
import preprocess  # noqa: E402  (phonemize); fork via env vars above
import phonetics  # noqa: E402
from modal_aligner import MODEL_ID, MODEL_REVISION  # noqa: E402  (single pinned-commit source)

# Canonical espeak voice per dataset lang (matches training labels).
LANG_TO_VOICE = dict(preprocess.LANG_TO_ESPEAK)

MASKED_IDS = {0, 1, 2, 3}
OUT = REPO / "espeak_audit" / "out"


def load_vocab() -> dict:
    """vocab.json from the SAME pinned model commit the aligner uses — so target
    ids can't drift from the weights when MODEL_REVISION is bumped (no hard-coded
    snapshot path)."""
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(MODEL_ID, "vocab.json",
                                           revision=MODEL_REVISION)).read_text())


def select_clips(lang: str, voice: str, source: str, n: int,
                 min_words: int = 3, max_words: int = 1_000_000,
                 any_voice: bool = False) -> list[dict]:
    """Deterministic clip selection: matching voice/source, word count in range,
    sorted by file for reproducibility, first n.

    any_voice=True pools ALL speakers of the source (for FLEURS/Pimsleur, which
    have voice=null and no per-clip speaker label). Within-speaker anchoring is
    then unavailable — speaker-independent contrasts (voicing-by-position,
    deletion, nasalization-vs-own-oral) stay valid; absolute vowel formants are
    pooled across vocal tracts and should be read with caution."""
    manifest = REPO / "data" / "audio" / lang / "manifest.jsonl"
    rows = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("source") != source:
            continue
        if not any_voice and d.get("voice") != voice:
            continue
        nw = len(d["sentence"].split())
        if not (min_words <= nw <= max_words):
            continue
        wav = REPO / "data" / "audio" / lang / d["file"]
        if not wav.exists():
            continue
        rows.append(d)
    rows.sort(key=lambda d: d["file"])
    return rows[:n]


def prep_clip(d: dict, lang: str, vocab: dict) -> dict:
    """espeak fork -> phonemes/ids + audio samples for one clip."""
    voice = LANG_TO_VOICE[lang]
    phonemes, stress, word_spans = preprocess.phonemize(d["sentence"], voice)
    word_of = [-1] * len(phonemes)
    for wi, (s, e) in enumerate(word_spans):
        for j in range(s, e):
            if 0 <= j < len(phonemes):
                word_of[j] = wi
    ids, ok = [], []
    for p in phonemes:
        i = vocab.get(p, 3)
        ids.append(i)
        ok.append(i not in MASKED_IDS)
    keep = [j for j in range(len(phonemes)) if ok[j]]
    wav = REPO / "data" / "audio" / lang / d["file"]
    audio, sr = sf.read(str(wav))
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = audio.mean(axis=1)
    return {
        "key": d["file"],
        "sentence": d["sentence"],
        "wav_path": str(wav),
        "phonemes": phonemes,
        "stress": stress,
        "word_of": word_of,
        "word_spans": word_spans,
        "ids": ids,
        "ok": ok,
        "keep": keep,
        "target_ids": [ids[j] for j in keep],
        "audio": audio.astype("float64").tolist(),
        "sample_rate": int(sr),
    }


def align_on_modal(preps: list[dict], batch_size: int = 8) -> dict:
    """Call the deployed Modal aligner. Returns {key: result}."""
    import modal
    Aligner = modal.Cls.from_name("espeak-audit-aligner", "VadCleanAligner")
    aligner = Aligner()
    out = {}
    for i in range(0, len(preps), batch_size):
        chunk = preps[i:i + batch_size]
        items = [
            {"key": p["key"], "audio": p["audio"],
             "sample_rate": p["sample_rate"], "target_ids": p["target_ids"]}
            for p in chunk
        ]
        res = aligner.force_align_batch.remote(items)
        for r in res:
            out[r["key"]] = r
        print(f"  aligned {min(i + batch_size, len(preps))}/{len(preps)}", flush=True)
    return out


def measure_clip(prep: dict, aligned: dict) -> list[dict]:
    """parselmouth-measure each espeak phone using the aligned spans.

    Delegates the span→measurement logic to the shared phonetics.measure_segments
    (same code the at-scale Modal path runs) and only attaches this clip's identity
    (key/sentence). One implementation, no drift."""
    rows = phonetics.measure_segments(
        prep["audio"], prep["sample_rate"], prep["phonemes"], prep["stress"],
        prep["word_of"], prep["ok"], prep["keep"], aligned["spans"])
    for r in rows:
        r["key"] = prep["key"]
        r["sentence"] = prep["sentence"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--source", required=True, choices=["tatoeba", "tts", "fleurs", "pimsleur"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=1_000_000)
    ap.add_argument("--any-voice", action="store_true",
                    help="Pool all speakers of the source (FLEURS/Pimsleur: voice=null).")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.lang}_{args.source}_{args.voice}".replace("/", "_")
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab()

    clips = select_clips(args.lang, args.voice, args.source, args.n,
                         min_words=args.min_words, max_words=args.max_words,
                         any_voice=args.any_voice)
    print(f"[{tag}] selected {len(clips)} clips")
    preps = [prep_clip(d, args.lang, vocab) for d in clips]
    print(f"[{tag}] phonemized; aligning on Modal...")
    aligned = align_on_modal(preps)

    seg_path = OUT / f"{tag}.segments.jsonl"
    clip_path = OUT / f"{tag}.clips.jsonl"
    n_seg = 0
    with open(seg_path, "w") as fseg, open(clip_path, "w") as fclip:
        for prep in preps:
            a = aligned.get(prep["key"])
            if a is None:
                continue
            fclip.write(json.dumps({
                "key": prep["key"], "sentence": prep["sentence"],
                "espeak": prep["phonemes"], "stress": prep["stress"],
                "model_reading": a["reading"], "audio_sec": a["audio_sec"],
                "n_frames": a["n_frames"],
            }, ensure_ascii=False) + "\n")
            for row in measure_clip(prep, a):
                fseg.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_seg += 1
    print(f"[{tag}] wrote {n_seg} segments -> {seg_path}")
    print(f"[{tag}] wrote clips -> {clip_path}")


if __name__ == "__main__":
    main()

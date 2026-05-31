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

# Canonical espeak voice per dataset lang (matches training labels).
LANG_TO_VOICE = dict(preprocess.LANG_TO_ESPEAK)

VOCAB_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--anchpop--lexide-pronunciation-unified-vad-clean"
        "/snapshots/2926e06f8092935f597e0018beb5d579b95b889a/vocab.json"
    )
)
MASKED_IDS = {0, 1, 2, 3}
OUT = REPO / "espeak_audit" / "out"


def load_vocab() -> dict:
    return json.loads(VOCAB_PATH.read_text())


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
    """parselmouth-measure each espeak phone using the aligned spans."""
    snd = phonetics.load_sound(prep["wav_path"])
    f0 = phonetics.clip_f0_stats(snd)
    ceiling = phonetics.formant_ceiling_for(f0["f0_median"])
    spans = aligned["spans"]  # in keep order: [start_s, end_s, score]
    # Alignment may have been skipped (target longer than frames, etc.): then
    # spans is empty / mismatched. Measure nothing rather than misattribute.
    if len(spans) != len(prep["keep"]):
        return [{
            "key": prep["key"], "sentence": prep["sentence"], "idx": j,
            "symbol": sym, "stress": prep["stress"][j], "word_idx": prep["word_of"][j],
            "oov": not prep["ok"][j], "start": None, "end": None, "dur": None,
            "align_score": None, "align_skipped": True,
        } for j, sym in enumerate(prep["phonemes"])]
    span_by_j = {j: spans[k] for k, j in enumerate(prep["keep"])}

    # CTC forced-alignment spans are "peaky" (often ~1 frame). For acoustic
    # MEASUREMENT we dilate each span into the surrounding blank frames up to
    # the midpoint of the gap to each neighbour (max 30 ms), giving fuller phone
    # coverage. The RAW span (dur/score) is kept untouched as the independent
    # over-specification signal — dilating that would mask deleted phones.
    MAXD = 0.030
    win_by_j = {}
    for k, (s, e, _sc) in enumerate(spans):
        prev_e = spans[k - 1][1] if k > 0 else s
        next_s = spans[k + 1][0] if k + 1 < len(spans) else e
        ws = s - min(MAXD, max(0.0, (s - prev_e) / 2.0))
        we = e + min(MAXD, max(0.0, (next_s - e) / 2.0))
        win_by_j[prep["keep"][k]] = (ws, we)

    rows = []
    for j, sym in enumerate(prep["phonemes"]):
        base = {
            "key": prep["key"],
            "sentence": prep["sentence"],
            "idx": j,
            "symbol": sym,
            "stress": prep["stress"][j],
            "word_idx": prep["word_of"][j],
            "oov": not prep["ok"][j],
            "f0_median_clip": f0["f0_median"],
            "ceiling": ceiling,
        }
        if not prep["ok"][j] or j not in span_by_j:
            base.update({"start": None, "end": None, "dur": None, "align_score": None})
            rows.append(base)
            continue
        s, e, score = span_by_j[j]
        ws, we = win_by_j[j]
        base.update({"start": s, "end": e, "dur": e - s, "align_score": score,
                     "win_start": ws, "win_end": we})
        try:
            if phonetics.is_vowel(sym):
                # Vowels: dilated window -> more steady-state frames.
                base["kind"] = "vowel"
                base.update(phonetics.measure_vowel(snd, ws, we, ceiling))
            elif phonetics.is_stop(sym):
                # Stops: RAW span (the CTC core), NOT the dilated window — a
                # window bleeding into flanking vowels would spuriously inflate
                # closure voicing for intervocalic stops. (Verified material.)
                base["kind"] = "stop"
                base.update(phonetics.measure_stop(snd, s, e))
            else:
                base["kind"] = "other"
                base["intensity_db"] = phonetics._mean_intensity(snd, s, e)
        except Exception as ex:  # measurement failure shouldn't kill the run
            base["measure_error"] = str(ex)
        rows.append(base)
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

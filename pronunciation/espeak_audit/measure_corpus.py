"""At-scale align+measure over the full corpus (parallel on Modal, model-keyed cache).

Generalizes run_audit.py (a per-source sampler) into a full-corpus pass for the
narrowing detectors:

  - selects every clip bearing a TARGET token — an oral vowel before a coda nasal
    (nasalization detector) or intervocalic /t,d/ (flapping detector);
  - aligns to the TRAINING phonemes (read straight from phonemes.jsonl, so each
    measured segment maps 1:1 to a training token — no raw-vs-validated drift);
  - ships compact float32 audio to Modal, where the aligner force-aligns AND
    parselmouth-measures on the container (fans out across many containers),
    returning only the tiny per-segment numbers;
  - caches each clip keyed by (MODEL, lang, file, phonemes) so re-runs and the
    eventual preprocess integration are free.

Cache: espeak_audit/.cache/measure/<MODEL>/<h[:2]>/<h>.json

Run with the pinned interpreter (modal + soundfile + numpy):
  measure_corpus.py [--langs eng ...] [--limit N] [--batch 8] [--concurrency 24]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "train" / "scripts"))
sys.path.insert(0, str(REPO / "espeak_audit"))
import run_audit as RA  # noqa: E402  (load_vocab — resolves vocab from the pinned commit — + MASKED_IDS)
from modal_aligner import MODEL_REVISION  # noqa: E402  (single source of the pinned commit)

# Cache namespace = the EXACT alignment model commit, so the cache key changes
# in lockstep with the modal aligner's pinned weights. Bumping vad-clean ->
# new SHA here (imported, can't drift) -> fresh namespace, never silent reuse.
MODEL = f"vad-clean@{MODEL_REVISION[:12]}"
CACHE = REPO / "espeak_audit" / ".cache" / "measure" / MODEL
AUDIO = REPO / "data" / "audio"
NAS_LANGS = {"eng", "deu", "ita", "spa", "rus", "por"}  # French phonemicized its pre-nasal vowels
VOW = set("aeiouɛɔøœəɐɨʊɪyɯɤʌɒæɑ")
NASAL_C = {"n", "m", "ŋ", "ɲ", "ɴ"}
TILDE = "̃"


def _oral_vowel(s): return s and s[0] in VOW and TILDE not in unicodedata.normalize("NFD", s)
def _is_vowel(s): return s and s[0] in VOW


def has_target(phonemes, lang) -> bool:
    """Clip bears a token we want to measure: oral-vowel-before-coda-nasal
    (any NAS_LANG) or intervocalic /t,d/ (eng flap)."""
    for i, t in enumerate(phonemes):
        if _oral_vowel(t) and i + 1 < len(phonemes) and phonemes[i + 1] in NASAL_C:
            nxt2 = phonemes[i + 2] if i + 2 < len(phonemes) else None
            if nxt2 is None or not _is_vowel(nxt2):       # coda nasal, not onset
                return True
        if lang == "eng" and t in ("t", "d") and 0 < i < len(phonemes) - 1 \
                and _is_vowel(phonemes[i - 1]) and _is_vowel(phonemes[i + 1]):
            return True
    return False


def cache_path(lang, file, phon_key) -> Path:
    h = hashlib.sha256(f"{MODEL}/{lang}/{file}/{phon_key}".encode()).hexdigest()
    return CACHE / h[:2] / f"{h}.json"


def load_targets(lang):
    """Yield (file, sentence, phonemes, stress, phon_key) for target-bearing clips."""
    pf = AUDIO / lang / "phonemes.jsonl"
    if not pf.exists():
        return
    for line in pf.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        ph = d.get("phonemes", [])
        if has_target(ph, lang):
            yield d["file"], d.get("sentence", ""), ph, d.get("stress", [0] * len(ph)), \
                hashlib.sha256("\x00".join(ph).encode()).hexdigest()[:16]


def prep(lang, file, phonemes, stress, vocab):
    """Build the Modal payload: target_ids/keep/ok + compact float32 audio."""
    ids = [vocab.get(p, 3) for p in phonemes]
    ok = [i not in RA.MASKED_IDS for i in ids]
    keep = [j for j in range(len(phonemes)) if ok[j]]
    audio, sr = sf.read(str(AUDIO / lang / file))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return {
        "key": f"{lang}/{file}",   # globally unique — filenames CAN repeat across langs
        "audio_b64": base64.b64encode(audio.astype("<f4").tobytes()).decode(),
        "sample_rate": int(sr),
        "target_ids": [ids[j] for j in keep],
        "phonemes": phonemes,
        "stress": stress,
        "word_of": [-1] * len(phonemes),   # word index unused by the detectors
        "ok": ok,
        "keep": keep,
    }


def _measure_and_cache(task):
    """LOCAL measurement worker (multiprocessed). Reads the clip's audio from disk,
    runs the SAME pure phonetics.measure_segments used to run on Modal, and writes
    the cache JSON. Modal did only the alignment (spans); all DSP is here, so it's
    free to re-run with changed parameters and never touches the GPU."""
    import soundfile as sf
    import phonetics
    (lang, file, phonemes, stress, ok, keep, pk,
     spans, n_frames, audio_sec, reading, align_error) = task
    audio, sr = sf.read(str(AUDIO / lang / file))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    segments = phonetics.measure_segments(
        audio, int(sr), phonemes, stress, [-1] * len(phonemes), ok, keep, spans)
    cp = cache_path(lang, file, pk)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({
        "model": MODEL, "lang": lang, "file": file,
        "n_frames": n_frames, "audio_sec": audio_sec,
        "reading": reading, "align_error": align_error,
        "segments": segments,
    }, ensure_ascii=False))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=sorted(NAS_LANGS))
    ap.add_argument("--limit", type=int, default=None, help="per-lang cap (testing)")
    ap.add_argument("--batch", type=int, default=8, help="clips per Modal align call")
    ap.add_argument("--concurrency", type=int, default=24, help="parallel Modal align calls (→ containers)")
    ap.add_argument("--measure-workers", type=int, default=8, help="local parselmouth processes")
    args = ap.parse_args()

    vocab = RA.load_vocab()
    # Build the work list (target clips not already cached).
    todo = []
    cached = 0
    for lang in args.langs:
        n = 0
        for file, _sent, ph, stress, pk in load_targets(lang):
            if args.limit and n >= args.limit:
                break
            n += 1
            if cache_path(lang, file, pk).exists():
                cached += 1
                continue
            todo.append((lang, file, ph, stress, pk))
    print(f"target clips: cached={cached} | to measure={len(todo)}")
    if not todo:
        print("nothing to do — all cached.")
        return

    import modal
    from multiprocessing import Pool
    Aligner = modal.Cls.from_name("espeak-audit-aligner", "VadCleanAligner")
    aligner = Aligner()

    def align_batch(chunk):
        """Modal align-only → list of measure-tasks (spans + the locals the
        measurement needs). No DSP here; the GPU does only forced-align."""
        items = [prep(lang, file, ph, stress, vocab) for (lang, file, ph, stress, pk) in chunk]
        send = [{"key": it["key"], "audio_b64": it["audio_b64"],
                 "sample_rate": it["sample_rate"], "target_ids": it["target_ids"]} for it in items]
        # Plain .remote() in threads is the proven path (it cached 54k clips this
        # way). NOT .spawn() — Modal's client deadlocks on concurrent .spawn() from
        # many threads. Resilience comes from the per-super-chunk incremental cache
        # below (a wedged tail batch loses only the current chunk; re-run resumes),
        # with a couple of retries for transient RPC errors.
        res = None
        for attempt in range(3):
            try:
                res = aligner.align_batch_b64.remote(send)
                break
            except Exception as e:  # noqa: BLE001 (transient gRPC/network)
                print(f"  [align retry {attempt + 1}/3] {type(e).__name__}: {e}", flush=True)
                time.sleep(3)
        if res is None:
            print(f"  [align gave up] deferring {len(chunk)} clips to next run", flush=True)
            return []
        by_key = {r["key"]: r for r in res}
        # Stale-deploy guard: container echoes its weights' revision; if it differs
        # from the pinned one this cache namespace claims, fail rather than poison.
        for r in res:
            rev = r.get("model_revision")
            if rev is not None and rev != MODEL_REVISION:
                raise SystemExit(f"deployed aligner is revision {rev} but cache namespace "
                                 f"is {MODEL_REVISION} — redeploy modal_aligner.py.")
        tasks = []
        for (lang, file, ph, stress, pk), it in zip(chunk, items):
            r = by_key.get(f"{lang}/{file}")
            if r is None:
                continue
            tasks.append((lang, file, ph, stress, it["ok"], it["keep"], pk,
                          r["spans"], r["n_frames"], r["audio_sec"], r["reading"], r["align_error"]))
        return tasks

    # Work in SUPER-CHUNKS: align a chunk on Modal (threads), then measure+cache it
    # locally (Pool), then the next. This (a) caches incrementally so a crash/hang
    # never loses prior work (re-runs skip cached), (b) bounds memory, and (c) keeps
    # the Modal gRPC client and the multiprocessing Pool from ever being live at the
    # same time (on macOS spawn that combo wedges the async client → "Deadline
    # exceeded"). Within a chunk: ThreadPool fully closes before the Pool opens.
    SUPER = max(400, args.concurrency * args.batch * 4)
    super_chunks = [todo[i:i + SUPER] for i in range(0, len(todo), SUPER)]
    t0, done = time.time(), 0
    for si, sc in enumerate(super_chunks):
        batches = [sc[i:i + args.batch] for i in range(0, len(sc), args.batch)]
        tasks = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for fut in as_completed([ex.submit(align_batch, b) for b in batches]):
                tasks.extend(fut.result())
        with Pool(args.measure_workers) as pool:
            for _ in pool.imap_unordered(_measure_and_cache, tasks, chunksize=8):
                done += 1
        rate = done / (time.time() - t0)
        eta = (len(todo) - done) / rate / 60 if rate else 0
        print(f"  super-chunk {si+1}/{len(super_chunks)}: cached {done}/{len(todo)} "
              f"({rate:.1f}/s, ETA {eta:.0f} min)", flush=True)
    print(f"done: {done} clips aligned(Modal)+measured(local)+cached in {time.time()-t0:.0f}s → {CACHE}")


if __name__ == "__main__":
    main()

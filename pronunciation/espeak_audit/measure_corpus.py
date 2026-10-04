"""At-scale production alignment + local acoustic measurement for narrowing.

Targets come directly from training phonemes. Production returns frame matrices;
local CTC Viterbi places boundaries and parselmouth measures the signal. Cache
namespaces use the full live model identity, discovered before cache filtering.
Downstream narrow.py reads the last successful namespace offline via cache_path.

Run with bash scripts/py-linux.sh (soundfile + numpy + parselmouth):
  espeak_audit/measure_corpus.py [--langs eng ...] [--limit N] [--batch 64]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "espeak_audit"))
from production_alignment import (  # noqa: E402
    MeasurementCache, align_batch, discover_identity, validate_options,
)

CACHE = REPO / "espeak_audit" / ".cache" / "measure"
AUDIO = REPO / "data" / "audio"
NAS_LANGS = {"eng", "deu", "ita", "spa", "rus", "por"}  # French phonemicized its pre-nasal vowels
VOW = set("aeiouɛɔøœəɐɨʊɪyɯɤʌɒæɑ")
NASAL_C = {"n", "m", "ŋ", "ɲ", "ɴ"}
TILDE = "̃"


def _oral_vowel(s): return s and s[0] in VOW and TILDE not in unicodedata.normalize("NFD", s)
def _is_vowel(s): return s and s[0] in VOW


def has_target(phonemes, lang) -> bool:
    """Clip bears an oral vowel before a coda nasal or intervocalic /t,d/."""
    for i, t in enumerate(phonemes):
        if _oral_vowel(t) and i + 1 < len(phonemes) and phonemes[i + 1] in NASAL_C:
            nxt2 = phonemes[i + 2] if i + 2 < len(phonemes) else None
            if nxt2 is None or not _is_vowel(nxt2):
                return True
        if lang == "eng" and t in ("t", "d") and 0 < i < len(phonemes) - 1 \
                and _is_vowel(phonemes[i - 1]) and _is_vowel(phonemes[i + 1]):
            return True
    return False


def cache_path(lang, file, phon_key) -> Path:
    """Offline lookup for narrow.py/_measure_coda; never mixes model revisions."""
    return MeasurementCache.latest(CACHE).path(lang, file, phon_key)


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


def _measure_and_cache(task):
    import phonetics

    (lang, file, phonemes, stress, pk), alignment, cache = task
    audio, sr = sf.read(str(AUDIO / lang / file))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    keep = alignment["keep"]
    ok = [i in keep for i in range(len(phonemes))]
    segments = phonetics.measure_segments(
        audio, int(sr), phonemes, stress, [-1] * len(phonemes), ok, keep, alignment["spans"])
    cp = cache.path(lang, file, pk)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({
        **cache.identity.as_dict(), "lang": lang, "file": file,
        **alignment, "segments": segments,
    }, ensure_ascii=False))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=sorted(NAS_LANGS))
    ap.add_argument("--limit", type=int, default=None, help="per-lang cap (testing)")
    ap.add_argument("--batch", type=int, default=64, help="clips per production call (1–64)")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel HTTP calls")
    ap.add_argument("--measure-workers", type=int, default=8, help="local parselmouth processes")
    args = ap.parse_args()
    validate_options(args)
    cache = MeasurementCache(CACHE, discover_identity())
    todo = []
    cached = 0
    for lang in args.langs:
        n = 0
        for file, _sent, ph, stress, pk in load_targets(lang):
            if args.limit and n >= args.limit:
                break
            n += 1
            if cache.path(lang, file, pk).exists():
                cached += 1
                continue
            todo.append((lang, file, ph, stress, pk))
    print(f"target clips: cached={cached} | to measure={len(todo)}")
    if not todo:
        cache.select()
        print("nothing to do — all cached.")
        return

    from multiprocessing import Pool

    def align_chunk(chunk):
        items = [(AUDIO / lang / file, lang, ph) for lang, file, ph, _, _ in chunk]
        return [(row, result, cache) for row, result in
                zip(chunk, align_batch(items, cache.identity))]

    # Incremental checkpoints bound memory; close threads before forking DSP.
    super_size = max(400, args.concurrency * args.batch * 4)
    super_chunks = [todo[i:i + super_size] for i in range(0, len(todo), super_size)]
    t0, done = time.time(), 0
    for si, sc in enumerate(super_chunks):
        batches = [sc[i:i + args.batch] for i in range(0, len(sc), args.batch)]
        tasks = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for fut in as_completed([ex.submit(align_chunk, b) for b in batches]):
                tasks.extend(fut.result())
        with Pool(args.measure_workers) as pool:
            for _ in pool.imap_unordered(_measure_and_cache, tasks, chunksize=8):
                done += 1
        rate = done / (time.time() - t0)
        eta = (len(todo) - done) / rate / 60 if rate else 0
        print(f"  super-chunk {si+1}/{len(super_chunks)}: cached {done}/{len(todo)} "
              f"({rate:.1f}/s, ETA {eta:.0f} min)", flush=True)
    cache.select()
    print(f"done: {done} clips aligned+measured locally in {time.time()-t0:.0f}s → {cache.directory}")


if __name__ == "__main__":
    main()

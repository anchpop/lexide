"""Derive speaker embeddings for the sources that lack speaker labels
(FLEURS + Pimsleur, voice=null), with a per-clip cache so re-runs are fast.

CACHE: each clip's 192-d embedding is stored at .cache/<h[:2]>/<h>.npy where
h = sha256("<lang>/<file>"). wavs are immutable once written, so the filename is
a stable key — a re-run loads cached embeddings and only sends NEW clips to
Modal. This is what makes the step cheap to put in the preprocess phase (same
idea as tysm's prompt cache for the LLM steps).

Usage (run with the interpreter that has modal + soundfile + numpy):
  embed.py [--langs spa ...] [--sources fleurs pimsleur] [--limit N] [--batch 32]

Embeddings land in the cache; cluster.py reads them to assign speaker_cluster.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]          # pronunciation/
AUDIO = REPO / "data" / "audio"
CACHE = Path(__file__).resolve().parent / ".cache"
DEFAULT_SOURCES = ("fleurs", "pimsleur")             # the voice=null sources


def cache_path(lang: str, file: str) -> Path:
    h = hashlib.sha256(f"{lang}/{file}".encode()).hexdigest()
    return CACHE / h[:2] / f"{h}.npy"


def is_cached(lang: str, file: str) -> bool:
    return cache_path(lang, file).exists()


def load_embedding(lang: str, file: str) -> np.ndarray:
    return np.load(cache_path(lang, file))


def save_embedding(lang: str, file: str, vec) -> None:
    p = cache_path(lang, file)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(vec, dtype=np.float32))


def clips_needing_embeddings(langs, sources, limit):
    """Yield (lang, file) for target-source clips whose wav exists."""
    out = []
    for lang_dir in sorted(AUDIO.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if langs and lang not in langs:
            continue
        mf = lang_dir / "manifest.jsonl"
        if not mf.exists():
            continue
        n = 0
        for line in mf.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("source") not in sources:
                continue
            if not (lang_dir / d["file"]).exists():
                continue
            out.append((lang, d["file"]))
            n += 1
            if limit and n >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=None)
    ap.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    ap.add_argument("--limit", type=int, default=None, help="per-lang cap (testing)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=12, help="parallel Modal calls")
    args = ap.parse_args()

    clips = clips_needing_embeddings(args.langs, set(args.sources), args.limit)
    cached = [c for c in clips if is_cached(*c)]
    misses = [c for c in clips if not is_cached(*c)]
    print(f"clips={len(clips)} | already cached={len(cached)} | to embed={len(misses)}")
    if not misses:
        print("nothing to embed — all cached (fast path).")
        return

    import modal
    Embedder = modal.Cls.from_name("speaker-embed", "SpeakerEmbedder")
    embedder = Embedder()

    def process_batch(chunk):
        # Send compact WAV bytes (base64), not float lists — ~10x smaller payload.
        items = [{"key": f"{lang}/{file}",
                  "audio_b64": base64.b64encode((AUDIO / lang / file).read_bytes()).decode()}
                 for lang, file in chunk]
        res = embedder.embed_batch.remote(items)
        by_key = {r["key"]: r["embedding"] for r in res}
        return [(lang, file, by_key[f"{lang}/{file}"]) for lang, file in chunk]

    batches = [misses[i:i + args.batch] for i in range(0, len(misses), args.batch)]
    t0, done = time.time(), 0
    # Concurrent Modal calls — the embed itself is cheap; this is round-trip
    # bound, so parallel batches are the win. Cache is written as each returns
    # (resumable: a kill mid-run just leaves more misses next time).
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_batch, b) for b in batches]
        for fut in as_completed(futs):
            for lang, file, emb in fut.result():
                save_embedding(lang, file, emb)
                done += 1
            if done % (args.batch * 10) < args.batch or done >= len(misses):
                rate = done / (time.time() - t0)
                print(f"  embedded {done}/{len(misses)} ({rate:.0f}/s)", flush=True)
    print(f"done: embedded {done} new clips in {time.time()-t0:.0f}s "
          f"({len(cached)} were already cached).")


if __name__ == "__main__":
    main()

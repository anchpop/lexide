#!/usr/bin/env python3
"""Sanity-check Pimsleur clips for English/target-language mixing.

The Pimsleur extraction pipeline:
  1. silero-vad → speech segments
  2. Whisper-via-Groq with no language hint → keep iff detected==target

The risk: if VAD merges an English-instructor utterance and a target-
language native utterance into one segment (instructor→native handoff
silence < min_silence_duration_ms), the saved WAV contains both. Whisper
detects whichever dominates and writes a transcript of only that part.
The other part's audio is along for the ride, polluting training.

This script runs Whisper a SECOND time on each kept Pimsleur clip but
forces `language=en`. If forced-English transcription returns non-trivial
text, the clip likely contains English content the original pass missed.
Flagged clips get logged; downstream code can drop them.

Cost: ~1 Groq call per clip audited. At ~$0.0001/call, auditing 100k
Pimsleur clips ≈ $10.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import re
import string
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests
import soundfile as sf
from tqdm import tqdm

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if c in string.punctuation else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def load_pimsleur_records(audio_root: Path,
                          langs: set[str] | None,
                          sample_per_lang: int | None,
                          seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for lang_dir in sorted(audio_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if langs and lang not in langs:
            continue
        manifest = lang_dir / "manifest.jsonl"
        if not manifest.exists():
            continue
        lang_records = []
        with manifest.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("source") != "pimsleur":
                    continue
                lang_records.append({
                    "file": rec["file"],
                    "path": str(lang_dir / rec["file"]),
                    "lang": lang,
                    "expected": rec.get("sentence", ""),
                })
        if sample_per_lang is not None and len(lang_records) > sample_per_lang:
            lang_records = rng.sample(lang_records, sample_per_lang)
        records.extend(lang_records)
    return records


def existing_keys(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("path"):
                done.add(rec["path"])
    return done


def force_english_transcribe(record: dict, args, api_key: str) -> dict[str, Any]:
    """Force Whisper to transcribe as English. If output has non-trivial
    English text, the clip likely contains English content."""
    path = Path(record["path"])
    fields = [
        ("model", args.model),
        ("temperature", "0"),
        ("response_format", "verbose_json"),
        ("language", "en"),
    ]
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            with path.open("rb") as audio:
                resp = requests.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=dict(fields),
                    files={"file": (path.name, audio, "audio/wav")},
                    timeout=args.timeout,
                )
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else min(10.0, 0.5 * (attempt + 1)))
                continue
            resp.raise_for_status()
            payload = resp.json()
            forced_text = (payload.get("text") or "").strip()
            normalized = normalize_text(forced_text)
            # Heuristic: more than 8 chars of forced-English output suggests
            # the clip actually has English content. Single-word echoes
            # ("yes", "no") often appear in Whisper's English output for
            # any audio, so we want at least a small phrase.
            looks_english = (
                len(normalized) >= 8
                and any(ch.isalpha() for ch in normalized)
            )
            return {
                **record,
                "ok": True,
                "forced_english_text": forced_text,
                "forced_english_chars": len(normalized),
                "mixed_likely": bool(looks_english),
            }
        except Exception as e:
            last_error = repr(e)
            time.sleep(min(15.0, 2 ** attempt))
    return {**record, "ok": False, "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, default=Path("data/audio"))
    parser.add_argument("--out", type=Path, default=Path("train/pimsleur_mixing_audit.jsonl"))
    parser.add_argument("--model", default="whisper-large-v3-turbo")
    parser.add_argument("--lang", action="append")
    parser.add_argument("--sample-per-lang", type=int, default=200,
                        help="Random sample size per language. Set 0 to audit ALL.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    load_env_file(args.env_file)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set")

    sample = args.sample_per_lang if args.sample_per_lang > 0 else None
    records = load_pimsleur_records(
        args.audio_root, set(args.lang) if args.lang else None,
        sample, args.seed,
    )
    print(f"Loaded {len(records)} Pimsleur clips to audit "
          f"(sample_per_lang={args.sample_per_lang})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(args.out)
    todo = [r for r in records if r["path"] not in done]
    print(f"Already done: {len(done)}; remaining: {len(todo)}")

    flagged = 0
    with args.out.open("a") as out:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(force_english_transcribe, r, args, api_key) for r in todo]
            for fut in tqdm(futures.as_completed(futs), total=len(futs),
                            desc="Force-English audit"):
                result = fut.result()
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                if result.get("mixed_likely"):
                    flagged += 1
    print(f"\nDone. Flagged {flagged}/{len(todo)} clips as likely English-mixed.")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()

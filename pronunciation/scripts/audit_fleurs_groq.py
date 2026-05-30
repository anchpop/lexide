#!/usr/bin/env python3
"""Audit local FLEURS audio with Groq Whisper transcription.

Writes one JSONL row per FLEURS clip with the expected manifest sentence,
Whisper transcript, simple CER/WER scores, duration, RMS, and API metadata.
The output is append/resume-friendly: existing successful rows are skipped.
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
LANG_TO_ISO639_1 = {
    "deu": "de",
    "eng": "en",
    "fra": "fr",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "spa": "es",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if c in string.punctuation else c for c in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def cer(expected: str, actual: str) -> float:
    e = normalize_text(expected).replace(" ", "")
    a = normalize_text(actual).replace(" ", "")
    if not e:
        return 0.0 if not a else 1.0
    return levenshtein(e, a) / len(e)


def wer(expected: str, actual: str) -> float:
    e = normalize_text(expected).split()
    a = normalize_text(actual).split()
    if not e:
        return 0.0 if not a else 1.0
    return levenshtein(e, a) / len(e)


def audio_stats(path: Path) -> tuple[float, float]:
    info = sf.info(path)
    duration = info.frames / info.samplerate
    audio, _ = sf.read(path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    rms = float((audio.astype("float64") ** 2).mean() ** 0.5) if len(audio) else 0.0
    return duration, rms


def load_fleurs_records(root: Path, langs: set[str] | None) -> list[dict[str, Any]]:
    records = []
    for manifest in sorted(root.glob("*/manifest.jsonl")):
        lang = manifest.parent.name
        if langs and lang not in langs:
            continue
        with manifest.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("source") != "fleurs":
                    continue
                wav_path = manifest.parent / rec["file"]
                records.append({
                    "file": rec["file"],
                    "path": str(wav_path),
                    "lang": lang,
                    "source": rec.get("source"),
                    "voice": rec.get("voice"),
                    "expected": rec["sentence"],
                })
    return records


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def existing_successes(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ok") and rec.get("path"):
                done.add(rec["path"])
    return done


def transcribe(record: dict[str, Any], args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    path = Path(record["path"])
    duration, rms = audio_stats(path)
    fields = [
        ("model", args.model),
        ("temperature", "0"),
        ("response_format", "verbose_json"),
    ]
    if args.force_language:
        lang = LANG_TO_ISO639_1.get(record["lang"])
        if lang:
            fields.append(("language", lang))

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
                last_error = resp.text[:1000]
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    time.sleep(float(retry_after))
                else:
                    time.sleep(min(10.0, 0.25 + attempt + random.random()))
                continue
            if 500 <= resp.status_code < 600:
                last_error = f"HTTP {resp.status_code}: {resp.text[:1000]}"
                raise RuntimeError(last_error)
            resp.raise_for_status()
            payload = resp.json()
            text = payload.get("text", "")
            return {
                **record,
                "ok": True,
                "duration_sec": duration,
                "rms": rms,
                "model": args.model,
                "whisper_text": text,
                "whisper_language": payload.get("language"),
                "cer": cer(record["expected"], text),
                "wer": wer(record["expected"], text),
                "raw": payload,
            }
        except requests.HTTPError as e:
            last_error = (
                f"HTTP {e.response.status_code}: {e.response.text[:1000]}"
                if e.response is not None else repr(e)
            )
        except Exception as e:  # noqa: BLE001 - record API/network failures.
            last_error = repr(e)
        time.sleep(min(30.0, (2 ** attempt) + random.random()))

    return {
        **record,
        "ok": False,
        "duration_sec": duration,
        "rms": rms,
        "model": args.model,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, default=Path("data/audio"))
    parser.add_argument("--out", type=Path,
                        default=Path("train/fleurs_groq_whisper_audit.jsonl"),
                        help="Append-only raw audit cache. Lives under train/ so "
                             "it's tracked in git and a fresh clone inherits the "
                             "cache instead of re-paying Groq.")
    parser.add_argument("--model", default="whisper-large-v3-turbo")
    parser.add_argument("--lang", action="append", choices=sorted(LANG_TO_ISO639_1),
                        help="Limit to one or more repo language codes, e.g. --lang spa.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--force-language", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env_file(args.env_file)

    records = load_fleurs_records(args.audio_root, set(args.lang) if args.lang else None)
    if args.limit is not None:
        records = records[:args.limit]

    total_seconds = 0.0
    for rec in records:
        try:
            info = sf.info(rec["path"])
            total_seconds += info.frames / info.samplerate
        except Exception:
            pass
    print(
        f"FLEURS records: {len(records)}; audio hours: {total_seconds / 3600:.3f}; "
        f"est. cost at $0.04/h: ${total_seconds / 3600 * 0.04:.2f}"
    )
    if args.dry_run:
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_successes(args.out)
    todo = [r for r in records if r["path"] not in done]
    print(f"Already done: {len(done)}; remaining: {len(todo)}; output: {args.out}")

    with args.out.open("a") as out:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(transcribe, rec, args, api_key) for rec in todo]
            for fut in tqdm(futures.as_completed(futs), total=len(futs), desc="Groq Whisper"):
                out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                out.flush()


if __name__ == "__main__":
    main()

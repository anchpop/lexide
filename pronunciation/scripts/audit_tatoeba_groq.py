#!/usr/bin/env python3
"""Audit local Tatoeba audio with Groq Whisper transcription, phoneme-level CER.

The FLEURS audit (audit_fleurs_groq.py) compares Whisper output to the
expected sentence as raw text. For Tatoeba we use **phoneme-level CER**
instead, which is closer to what we actually care about for training a
pronunciation model — orthographic differences (apostrophes, capitalization,
contractions like "tu as" vs "t'as") get normalized away by espeak, so the
CER reflects whether the audio actually pronounces the expected phonemes.

Output schema is compatible with `load_asr_audit_exclusions` in
train_unified.py: per-clip JSONL with `ok`, `lang`, `file`, `expected_sha256`,
`cer`, `wer`, `whisper_text`. The downstream filter reads any audit file
matching that schema, so the existing CLI flags pick up Tatoeba exclusions
the same way they pick up FLEURS exclusions.

Resumeable: existing successful rows skipped. Set GROQ_API_KEY in env.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import random
import re
import string
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests
import soundfile as sf
from tqdm import tqdm

# Reuse the same espeak phonemizer the training pipeline uses, so the
# audit comparison is in the EXACT phoneme space the model is trained
# against. Import from train/scripts/preprocess.py.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "train" / "scripts"))
from preprocess import phonemize, LANG_TO_ESPEAK  # noqa: E402

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
LANG_TO_ISO639_1 = {
    "deu": "de", "eng": "en", "fra": "fr", "ita": "it",
    "por": "pt", "rus": "ru", "spa": "es",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if c in string.punctuation else c for c in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_cer(expected: str, actual: str) -> float:
    e = normalize_text(expected).replace(" ", "")
    a = normalize_text(actual).replace(" ", "")
    if not e:
        return 0.0 if not a else 1.0
    return levenshtein(e, a) / len(e)


def text_wer(expected: str, actual: str) -> float:
    e = normalize_text(expected).split()
    a = normalize_text(actual).split()
    if not e:
        return 0.0 if not a else 1.0
    return levenshtein(e, a) / len(e)


def phoneme_cer(expected_phonemes: list[str], actual_phonemes: list[str]) -> float:
    """Phoneme-level edit distance ratio. Treats each phoneme token as one symbol."""
    if not expected_phonemes:
        return 0.0 if not actual_phonemes else 1.0
    return levenshtein(expected_phonemes, actual_phonemes) / len(expected_phonemes)


def audio_stats(path: Path) -> tuple[float, float]:
    info = sf.info(path)
    duration = info.frames / info.samplerate
    audio, _ = sf.read(path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    rms = float((audio.astype("float64") ** 2).mean() ** 0.5) if len(audio) else 0.0
    return duration, rms


def load_tatoeba_records(root: Path, langs: set[str] | None) -> list[dict[str, Any]]:
    records = []
    for manifest in sorted(root.glob("*/manifest.jsonl")):
        lang = manifest.parent.name
        if langs and lang not in langs:
            continue
        with manifest.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("source") != "tatoeba":
                    continue
                wav_path = manifest.parent / rec["file"]
                records.append({
                    "file": rec["file"],
                    "path": str(wav_path),
                    "lang": lang,
                    "source": "tatoeba",
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
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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
        iso = LANG_TO_ISO639_1.get(record["lang"])
        if iso:
            fields.append(("language", iso))

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
                time.sleep(float(retry_after) if retry_after else min(10.0, 0.25 + attempt + random.random()))
                continue
            if 500 <= resp.status_code < 600:
                last_error = f"HTTP {resp.status_code}: {resp.text[:1000]}"
                raise RuntimeError(last_error)
            resp.raise_for_status()
            payload = resp.json()
            text = payload.get("text", "")

            espeak_lang = LANG_TO_ESPEAK.get(record["lang"])
            expected_phonemes, _, _ = phonemize(record["expected"], espeak_lang) if espeak_lang else ([], [], [])
            try:
                actual_phonemes, _, _ = phonemize(text, espeak_lang) if espeak_lang else ([], [], [])
            except Exception:
                # If espeak chokes on the Whisper output (rare; usually
                # non-target-language transliterations), the audit just falls
                # back to maximum CER for this clip.
                actual_phonemes = []

            expected_sha = hashlib.sha256(record["expected"].encode()).hexdigest()

            # Aggregate per-segment Whisper signals so they're easy to
            # filter at training time without parsing the full payload.
            segs = payload.get("segments") or []
            avg_logprob = None
            no_speech_prob = None
            compression_ratio = None
            if segs:
                lps = [s.get("avg_logprob") for s in segs if s.get("avg_logprob") is not None]
                nps = [s.get("no_speech_prob") for s in segs if s.get("no_speech_prob") is not None]
                cps = [s.get("compression_ratio") for s in segs if s.get("compression_ratio") is not None]
                avg_logprob = sum(lps) / len(lps) if lps else None
                no_speech_prob = max(nps) if nps else None
                compression_ratio = max(cps) if cps else None

            return {
                **record,
                "ok": True,
                "duration_sec": duration,
                "rms": rms,
                "model": args.model,
                "whisper_text": text,
                "whisper_language": payload.get("language"),
                "whisper_avg_logprob": avg_logprob,
                "whisper_no_speech_prob": no_speech_prob,
                "whisper_compression_ratio": compression_ratio,
                "expected_sha256": expected_sha,
                "expected_phonemes": expected_phonemes,
                "actual_phonemes": actual_phonemes,
                # Field name matches load_asr_audit_exclusions in train_unified.py
                # so this exclusion file is picked up by the audit-min-per threshold.
                "per": phoneme_cer(expected_phonemes, actual_phonemes),
                "cer": text_cer(record["expected"], text),
                "wer": text_wer(record["expected"], text),
                # Full verbose_json — keep for future re-analysis without
                # paying Groq again.
                "whisper": payload,
            }
        except requests.HTTPError as e:
            last_error = (
                f"HTTP {e.response.status_code}: {e.response.text[:1000]}"
                if e.response is not None else repr(e)
            )
        except Exception as e:
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
                        default=Path("train/tatoeba_asr_exclusions.jsonl"))
    parser.add_argument("--model", default="whisper-large-v3-turbo")
    parser.add_argument("--lang", action="append", choices=sorted(LANG_TO_ISO639_1),
                        help="Limit to one or more repo language codes.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--force-language", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env_file(args.env_file)

    records = load_tatoeba_records(args.audio_root, set(args.lang) if args.lang else None)
    if args.limit is not None:
        records = records[: args.limit]

    total_seconds = 0.0
    for rec in records:
        try:
            info = sf.info(rec["path"])
            total_seconds += info.frames / info.samplerate
        except Exception:
            pass
    print(
        f"Tatoeba records: {len(records)}; audio hours: {total_seconds / 3600:.3f}; "
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
            for fut in tqdm(futures.as_completed(futs), total=len(futs), desc="Groq Whisper (Tatoeba)"):
                out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                out.flush()


if __name__ == "__main__":
    main()

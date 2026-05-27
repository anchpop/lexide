#!/usr/bin/env python3
"""Extract clean native-language clips from Pimsleur audio archives.

Each Pimsleur lesson is a ~30-min MP3 containing alternating English
instructor speech and target-language native-speaker speech, separated
by silences. We want only the native-language clips for training.

Pipeline per MP3:
  1. Decode to 16 kHz mono via ffmpeg.
  2. silero-vad → list of (start, end) speech timestamps.
  3. For each speech segment of reasonable duration:
     - Send to Groq Whisper (no `language` param → it detects).
     - If detected_language == target, phonemize via espeak-ng, save
       a WAV with filename `pimsleur_<lesson_id>_<seg_idx>.wav`, and
       append to manifest with source="pimsleur".

Whisper is the source of truth for the transcript here — there's no
external ground-truth to audit against, so we trust its output.

Requires:
  - silero-vad via torch.hub (lazy-loaded, no separate pip install)
  - GROQ_API_KEY in env or .env file
  - ffmpeg on PATH
  - espeak-ng on PATH
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import re
import string
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf
import torch
from tqdm import tqdm

# Reuse the same espeak phonemizer the training pipeline uses.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "train" / "scripts"))
from preprocess import phonemize, LANG_TO_ESPEAK  # noqa: E402

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

LANG_TO_ISO639_1 = {
    "deu": "de", "eng": "en", "fra": "fr", "ita": "it",
    "por": "pt", "rus": "ru", "spa": "es",
}
# Whisper's verbose_json returns the detected language as a lowercase full
# English name ("english", "french") rather than an ISO code. We accept either
# form on input and normalize for the equality check.
LANG_TO_WHISPER_NAME = {
    "deu": "german", "eng": "english", "fra": "french", "ita": "italian",
    "por": "portuguese", "rus": "russian", "spa": "spanish",
}


def whisper_lang_matches(detected: str, target_lang: str) -> bool:
    """Whisper may return ISO codes or full names depending on version/format."""
    if not detected:
        return False
    d = detected.strip().lower()
    return d == LANG_TO_ISO639_1.get(target_lang) or d == LANG_TO_WHISPER_NAME.get(target_lang)

# Map Pimsleur folder names to our 3-letter target language codes.
# Only languages we train on are listed; anything else is skipped.
PIMSLEUR_FOLDER_TO_LANG = {
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Brazilian Portuguese": "por",
    "European Portuguese": "por",
    "Portuguese": "por",
    "Russian": "rus",
    "Spanish": "spa",
    "Castilian Spanish": "spa",
    "Latin American Spanish": "spa",
    "English": "eng",
    "ESL": "eng",
}

# Tatoeba defaults: short, clear native-language utterances. Pimsleur native
# clips look similar — 0.5 to 8 seconds typically.
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 16.0


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


def decode_mp3(mp3_path: Path) -> np.ndarray:
    """Decode an MP3 to 16 kHz mono float32 numpy array via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp3_path),
            "-ac", "1", "-ar", "16000",
            "-f", "wav", "pipe:1",
        ],
        capture_output=True, check=True,
    )
    import io
    audio, sr = sf.read(io.BytesIO(result.stdout), dtype="float32")
    assert sr == 16000
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def vad_segments(audio: np.ndarray, vad_model, get_speech_timestamps) -> list[dict]:
    """Return list of {'start': int, 'end': int} sample-index pairs."""
    t = torch.from_numpy(audio)
    return get_speech_timestamps(t, vad_model, sampling_rate=16000,
                                 min_speech_duration_ms=400,
                                 max_speech_duration_s=MAX_DURATION_SEC,
                                 min_silence_duration_ms=300)


def transcribe_clip(wav_bytes: bytes, api_key: str, model: str,
                    retries: int = 3, timeout: float = 60.0) -> dict[str, Any]:
    """Send a single WAV clip to Groq Whisper with no language hint.

    Returns {"ok": bool, "text": str, "language": str, "error": str|None}.
    """
    fields = [
        ("model", model),
        ("temperature", "0"),
        ("response_format", "verbose_json"),
    ]
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data=dict(fields),
                files={"file": ("clip.wav", wav_bytes, "audio/wav")},
                timeout=timeout,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else min(10.0, 0.5 * (attempt + 1)))
                last_error = "429"
                continue
            resp.raise_for_status()
            payload = resp.json()
            return {
                "ok": True,
                "text": (payload.get("text") or "").strip(),
                "language": payload.get("language") or "",
                "error": None,
            }
        except Exception as e:
            last_error = repr(e)
            time.sleep(min(15.0, 2 ** attempt))
    return {"ok": False, "text": "", "language": "", "error": last_error}


def process_lesson(mp3_path: Path, target_lang: str, vad_model,
                   get_speech_timestamps, save_audio,
                   audio_dir: Path, api_key: str, model: str) -> list[dict]:
    """Process one Pimsleur lesson, return list of saved manifest entries."""
    try:
        audio = decode_mp3(mp3_path)
    except Exception as e:
        print(f"  decode failed: {mp3_path.name}: {e}")
        return []

    segments = vad_segments(audio, vad_model, get_speech_timestamps)
    iso_target = LANG_TO_ISO639_1[target_lang]
    espeak_lang = LANG_TO_ESPEAK[target_lang]
    lesson_id = mp3_path.stem

    entries: list[dict] = []
    for seg_idx, seg in enumerate(segments):
        start, end = int(seg["start"]), int(seg["end"])
        clip = audio[start:end]
        dur = clip.shape[0] / 16000.0
        if dur < MIN_DURATION_SEC or dur > MAX_DURATION_SEC:
            continue

        # Skip the lesson's leading 60 sec — that's almost always the
        # English program intro. Same with the trailing 30 sec.
        if start < 60 * 16000 or end > (audio.shape[0] - 30 * 16000):
            continue

        # Serialize clip to WAV bytes (in-memory).
        import io
        buf = io.BytesIO()
        sf.write(buf, clip, 16000, format="WAV")
        wav_bytes = buf.getvalue()

        whisper = transcribe_clip(wav_bytes, api_key, model)
        if not whisper["ok"]:
            continue
        if not whisper_lang_matches(whisper["language"], target_lang):
            continue
        text = whisper["text"]
        if not text or len(text) < 2:
            continue

        try:
            phonemes, stress = phonemize(text, espeak_lang)
        except Exception:
            continue
        if not phonemes:
            continue

        fname = f"pimsleur_{lesson_id}_{seg_idx:04d}.wav"
        dest = audio_dir / fname
        sf.write(dest, clip, 16000)
        entries.append({
            "file": fname,
            "sentence": text,
            "source": "pimsleur",
            "voice": None,
            "lang": target_lang,
        })
    return entries


def discover_lessons(pimsleur_root: Path,
                     wanted_langs: set[str] | None) -> list[tuple[Path, str]]:
    """Walk Pimsleur folders, return [(mp3_path, target_lang_code), ...]."""
    out: list[tuple[Path, str]] = []
    for folder in sorted(pimsleur_root.iterdir()):
        if not folder.is_dir():
            continue
        lang = PIMSLEUR_FOLDER_TO_LANG.get(folder.name)
        if lang is None:
            continue
        if wanted_langs and lang not in wanted_langs:
            continue
        # MP3s may be nested under Level N/ subdirs or directly under the lang folder.
        for mp3 in sorted(folder.rglob("*.mp3")):
            # Skip non-lesson audio: user guides, reading booklets / readings
            # directories, anything with "Reading" or "Booklet" in the name or
            # parent path. Pimsleur readings are written-pronunciation drills
            # not the speaking-and-listening cycles we want.
            name = mp3.name
            path_parts = {p.lower() for p in mp3.parts}
            if "users_guide" in name.lower() or "user_guide" in name.lower():
                continue
            if "reading" in name.lower() or "booklet" in name.lower():
                continue
            if "readings" in path_parts or "reading" in path_parts:
                continue
            out.append((mp3, lang))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pimsleur-root", type=Path,
                        default=Path("/Volumes/T7/p_rty/Pimsleur Complete Collection"))
    parser.add_argument("--audio-root", type=Path,
                        default=Path(__file__).resolve().parent / "audio")
    parser.add_argument("--lang", action="append", choices=sorted(LANG_TO_ISO639_1),
                        help="Limit to one or more target language codes.")
    parser.add_argument("--max-lessons-per-lang", type=int, default=None)
    parser.add_argument("--model", default="whisper-large-v3-turbo")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent Whisper requests per lesson.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env_file(args.env_file)

    if not args.pimsleur_root.exists():
        sys.exit(f"Pimsleur root not found: {args.pimsleur_root}")

    wanted = set(args.lang) if args.lang else None
    lessons = discover_lessons(args.pimsleur_root, wanted)
    print(f"Discovered {len(lessons)} Pimsleur lessons across "
          f"{len({lang for _, lang in lessons})} language(s).")
    if args.dry_run:
        for path, lang in lessons[:20]:
            print(f"  [{lang}] {path}")
        if len(lessons) > 20:
            print(f"  ... and {len(lessons) - 20} more")
        return

    # Group by lang so we respect --max-lessons-per-lang.
    by_lang: dict[str, list[Path]] = {}
    for path, lang in lessons:
        by_lang.setdefault(lang, []).append(path)
    if args.max_lessons_per_lang is not None:
        for lang in by_lang:
            by_lang[lang] = by_lang[lang][: args.max_lessons_per_lang]

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set")

    print("Loading silero-vad...")
    vad_model, vad_utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", verbose=False, trust_repo=True,
    )
    get_speech_timestamps, save_audio, *_ = vad_utils

    for lang, mp3_paths in sorted(by_lang.items()):
        audio_dir = args.audio_root / lang
        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = audio_dir / "manifest.jsonl"

        existing_files: set[str] = set()
        if manifest_path.exists():
            with manifest_path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    existing_files.add(rec["file"])

        print(f"\n=== {lang}: {len(mp3_paths)} lessons ===")
        for i, mp3 in enumerate(mp3_paths):
            print(f"  [{lang}] {i+1}/{len(mp3_paths)}: {mp3.name}")
            try:
                entries = process_lesson(
                    mp3, lang, vad_model, get_speech_timestamps, save_audio,
                    audio_dir, api_key, args.model,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"    SKIP — {e}")
                continue

            kept = [e for e in entries if e["file"] not in existing_files]
            with manifest_path.open("a") as out:
                for e in kept:
                    out.write(json.dumps(e, ensure_ascii=False) + "\n")
                    existing_files.add(e["file"])
            print(f"    kept {len(kept)} target-language clips")


if __name__ == "__main__":
    main()

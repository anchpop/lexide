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


class DiskWriteError(Exception):
    """Raised when sf.write fails for what looks like a filesystem-level
    problem (read-only mount, disk full, missing path). These have to abort
    the whole run — silently continuing turns every Whisper success into a
    lost segment AND burns Groq quota.
    """

# Whisper's verbose_json returns the detected language as a lowercase full
# English name ("english", "french") rather than an ISO code.
LANG_TO_WHISPER_NAME = {
    "deu": "german", "eng": "english", "fra": "french", "ita": "italian",
    "por": "portuguese", "rus": "russian", "spa": "spanish",
    "sqi": "albanian", "ara": "arabic", "hye": "armenian", "yue": "cantonese",
    "hrv": "croatian", "ces": "czech", "dan": "danish", "fas": "persian",
    "nld": "dutch", "fin": "finnish", "hat": "haitian creole", "heb": "hebrew",
    "hin": "hindi", "hun": "hungarian", "isl": "icelandic", "ind": "indonesian",
    "gle": "irish", "jpn": "japanese", "kor": "korean", "ell": "greek",
    "nor": "norwegian", "oji": "ojibwe", "pus": "pashto", "pol": "polish",
    "pan": "punjabi", "ron": "romanian", "swa": "swahili", "swe": "swedish",
    "tgl": "tagalog", "tha": "thai", "tur": "turkish", "twi": "twi",
    "ukr": "ukrainian", "urd": "urdu", "vie": "vietnamese",
}

LANG_TO_ISO639_1 = {
    "deu": "de", "eng": "en", "fra": "fr", "ita": "it",
    "por": "pt", "rus": "ru", "spa": "es",
    "sqi": "sq", "ara": "ar", "hye": "hy", "yue": "yue", "hrv": "hr",
    "ces": "cs", "dan": "da", "fas": "fa", "nld": "nl", "fin": "fi",
    "hat": "ht", "heb": "he", "hin": "hi", "hun": "hu", "isl": "is",
    "ind": "id", "gle": "ga", "jpn": "ja", "kor": "ko", "ell": "el",
    "nor": "no", "pus": "ps", "pol": "pl", "pan": "pa", "ron": "ro",
    "swa": "sw", "swe": "sv", "tgl": "tl", "tha": "th", "tur": "tr",
    "ukr": "uk", "urd": "ur", "vie": "vi",
}


def whisper_lang_matches(detected: str, target_lang: str) -> bool:
    """Whisper may return ISO codes or full names depending on version/format."""
    if not detected:
        return False
    d = detected.strip().lower()
    return d == LANG_TO_ISO639_1.get(target_lang) or d == LANG_TO_WHISPER_NAME.get(target_lang)


def lesson_id_from_path(mp3_path: Path, pimsleur_root: Path) -> str:
    """Stable, filesystem-safe identifier derived from an mp3's location.

    Pimsleur ships multiple courses per training language: Brazilian + European
    Portuguese (both lang="por"), Castilian + Latin American + plain Spanish
    (all lang="spa"), Arabic Eastern + Egyptian + MSA, Eastern + Western
    Armenian, Dari + Farsi Persian, German + Swiss German. Their lesson files
    are often named identically (e.g. "Unit 01.mp3"). Using bare `mp3_path.stem`
    as the lesson id collides across courses: the second course overwrites the
    first course's WAVs but the manifest still points at the first course's
    transcripts, corrupting audio/label alignment.

    Including the course folder makes the identifier unique per source MP3.
    """
    rel = mp3_path.relative_to(pimsleur_root).with_suffix("")
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "-", str(p)).strip("-") for p in rel.parts]
    return "__".join(p for p in parts if p)

# Languages we currently train on. We process these FIRST so the user
# can start a training run as soon as their lessons are extracted,
# without waiting for the non-target languages to finish.
TARGET_LANGS = ("deu", "eng", "fra", "ita", "por", "rus", "spa")

# Pimsleur folder name → (3-letter lang code, espeak voice code).
# Comprehensive coverage of the Pimsleur Complete Collection — process
# every language the archive has. Languages without an espeak voice are
# included with espeak=None: audio + transcript still get saved to
# manifest.jsonl so downstream code can decide what to do with them,
# but they won't get a phonemes.jsonl entry (preprocess.py will skip).
PIMSLEUR_FOLDER_TO_LANG = {
    "Albanian": ("sqi", "sq"),
    "Arabic Eastern": ("ara", "ar"),
    "Armenian Eastern": ("hye", "hy"),
    "Brazilian Portuguese": ("por", "pt-br"),
    "Cantonese Chinese": ("yue", "yue"),
    "Castilian Spanish": ("spa", "es"),
    "Croatian": ("hrv", "hr"),
    "Czech": ("ces", "cs"),
    "Danish": ("dan", "da"),
    "Dari Persian": ("fas", "fa"),
    "Dutch": ("nld", "nl"),
    "Egyptian Arabic": ("ara", "ar"),
    "ESL (English as a Second Language)": ("eng", "en-us"),
    "European Portuguese": ("por", "pt"),
    "Farsi Persian": ("fas", "fa"),
    "Finnish": ("fin", "fi"),
    "French": ("fra", "fr-fr"),
    "German": ("deu", "de"),
    "Haitian Creole": ("hat", "ht"),
    "Hebrew": ("heb", "he"),
    "Hindi": ("hin", "hi"),
    "Hungarian": ("hun", "hu"),
    "Icelandic": ("isl", "is"),
    "Indonesian": ("ind", "id"),
    "Irish": ("gle", "ga"),
    "Italian": ("ita", "it"),
    "Japanese": ("jpn", "ja"),
    "Korean": ("kor", "ko"),
    "Modern Greek": ("ell", "el"),
    "Modern Standard Arabic": ("ara", "ar"),
    "Norwegian": ("nor", "nb"),
    "Ojibwe": ("oji", None),         # no espeak support
    "Pashto": ("pus", None),          # no espeak support
    "Polsih": ("pol", "pl"),          # typo in archive
    "Polish": ("pol", "pl"),          # in case archive ever fixes the typo
    "Punjabi": ("pan", "pa"),
    "Romanian": ("ron", "ro"),
    "Russian": ("rus", "ru"),
    "Spanish": ("spa", "es-419"),     # Latin American Spanish
    "Latin American Spanish": ("spa", "es-419"),
    "Swahili": ("swa", "sw"),
    "Swedish": ("swe", "sv"),
    "Swiss German": ("deu", "de"),    # Whisper has no dedicated Swiss variant
    "Tagalog": ("tgl", None),         # no espeak support
    "Thai": ("tha", "th"),
    "Turkish": ("tur", "tr"),
    "Twi": ("twi", None),             # no espeak support
    "Ukranian": ("ukr", "uk"),        # archive typo
    "Ukrainian": ("ukr", "uk"),       # in case archive fixes it
    "Urdu": ("urd", "ur"),
    "Vietnamese": ("vie", "vi"),
    "Western Armenian": ("hye", "hyw"),
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
    """Return list of {'start': int, 'end': int} sample-index pairs.

    `min_silence_duration_ms=800` is conservative for Pimsleur audio
    specifically: lessons alternate between English instructor and target-
    language native, and short handoff pauses (~300-500ms) would merge
    those into one mixed-language VAD segment. The audit script
    (audit_pimsleur_mixing.py) catches survivors.
    """
    t = torch.from_numpy(audio)
    return get_speech_timestamps(t, vad_model, sampling_rate=16000,
                                 min_speech_duration_ms=400,
                                 max_speech_duration_s=MAX_DURATION_SEC,
                                 min_silence_duration_ms=800)


def transcribe_clip(wav_bytes: bytes, api_key: str, model: str,
                    retries: int = 1, timeout: float = 60.0) -> dict[str, Any]:
    """Send a single WAV clip to Groq Whisper with no language hint.

    Returns the full verbose_json payload alongside an "ok" flag so the
    caller can store every Whisper signal in the manifest (segments,
    avg_logprob, no_speech_prob, compression_ratio, etc). Throwing away
    those signals at extraction time means later filtering needs another
    Groq call per clip; storing them is free.

    retries=1 is intentional: we'd rather give up early on a failing call
    and let the segment retry on the next orchestrator pass (manifest-
    based skip-set + partial-success preservation make that cheap) than
    burn 3-4x the daily Groq quota on retry-then-fail loops. The sleeps
    on 429 / transient errors are generous so the one retry actually
    gets some breathing room.
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
                time.sleep(float(retry_after) if retry_after else min(20.0, 2.0 * (attempt + 1)))
                last_error = "429"
                continue
            resp.raise_for_status()
            payload = resp.json()
            return {
                "ok": True,
                "text": (payload.get("text") or "").strip(),
                "language": payload.get("language") or "",
                "payload": payload,  # Full verbose_json — keep everything
                "error": None,
            }
        except Exception as e:
            last_error = repr(e)
            time.sleep(min(20.0, 4.0 * (attempt + 1)))
    return {"ok": False, "text": "", "language": "", "payload": None,
            "error": last_error}


def process_lesson(mp3_path: Path, lesson_id: str,
                   target_lang: str, espeak_lang: str | None,
                   vad_fn, save_audio,
                   audio_dir: Path, api_key: str, model: str,
                   workers: int = 8,
                   skip_seg_idxs: set[int] | None = None) -> tuple[list[dict], bool]:
    """Process one Pimsleur lesson.

    Returns (entries, ok). `ok=False` means the lesson hit a transient
    failure (decode error, or every Whisper call exhausted its retries)
    and the caller should NOT mark it as processed — the next run should
    retry. `ok=True` with `entries=[]` is the normal "lesson processed
    cleanly, just had no target-language clips worth keeping" case.

    `vad_fn(audio_np) -> list[{'start': int, 'end': int}]` is injected so
    the caller can wrap it with a lock for thread-safety across lessons.

    Whisper calls within the lesson are dispatched in parallel — each
    segment is independent so we don't need ordering. With ~80 segments
    per 30-min lesson and ~1 s per Whisper call, sequential would take
    ~80 s/lesson; parallelizing across 8 workers gets it to ~10 s/lesson.

    If espeak_lang is None, the lesson is still extracted (audio +
    Whisper text saved to manifest), but no phonemization happens.
    preprocess.py also skips those languages.
    """
    try:
        audio = decode_mp3(mp3_path)
    except Exception as e:
        print(f"  decode failed: {mp3_path.name}: {e}")
        return [], False

    segments = vad_fn(audio)
    audio_total = audio.shape[0]
    skip_seg_idxs = skip_seg_idxs or set()

    # First pass: filter to viable segments and pre-encode WAV bytes. Also
    # skip seg_idxs the caller has marked already-done (recovery mode:
    # existing pimsleur_<lid>_<seg:04d>.wav files on disk). silero-vad is
    # deterministic, so seg_idx numbering matches across runs.
    candidates: list[tuple[int, int, int, "np.ndarray", bytes]] = []
    for seg_idx, seg in enumerate(segments):
        if seg_idx in skip_seg_idxs:
            continue
        start, end = int(seg["start"]), int(seg["end"])
        clip = audio[start:end]
        dur = clip.shape[0] / 16000.0
        if dur < MIN_DURATION_SEC or dur > MAX_DURATION_SEC:
            continue
        # Skip the lesson's leading 60 s and trailing 30 s — typically the
        # English program intro/outro.
        if start < 60 * 16000 or end > (audio_total - 30 * 16000):
            continue
        import io
        buf = io.BytesIO()
        sf.write(buf, clip, 16000, format="WAV")
        candidates.append((seg_idx, start, end, clip, buf.getvalue()))

    if not candidates:
        # VAD ran and found nothing usable — this is a clean outcome, not a
        # transient failure. Mark processed so we don't re-VAD the same lesson.
        return [], True

    # Second pass: parallel Whisper. Each call is one segment.
    def whisper_one(item):
        seg_idx, start, end, clip, wav_bytes = item
        w = transcribe_clip(wav_bytes, api_key, model)
        return seg_idx, start, end, clip, w

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results: list[tuple[int, int, int, "np.ndarray", dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(whisper_one, c) for c in candidates]
        for fut in as_completed(futs):
            results.append(fut.result())

    # Track whether any Whisper call failed after retries. When some did,
    # we still persist the successes (manifest write + WAVs); we just
    # don't mark the lesson processed, so the next run can retry the
    # missing segments. existing_files dedupe in main() prevents duplicate
    # manifest rows when those successes show up again.
    n_failed = sum(1 for r in results if not r[4]["ok"])
    n_ok = len(results) - n_failed
    lesson_ok = (n_failed == 0)
    if n_failed > 0:
        print(f"  {n_failed}/{len(results)} Whisper calls failed: "
              f"{mp3_path.name} — keeping {n_ok} successes, will retry "
              f"failed segments on next run")

    # Third pass: filter by language match, optionally phonemize, save WAV,
    # build manifest entries. Sorted by seg_idx so filenames are stable.
    results.sort(key=lambda r: r[0])
    entries: list[dict] = []
    for seg_idx, start, end, clip, whisper in results:
        if not whisper["ok"]:
            continue
        if not whisper_lang_matches(whisper["language"], target_lang):
            continue
        text = whisper["text"]
        if not text or len(text) < 2:
            continue
        if espeak_lang is not None:
            try:
                phonemes, _, _ = phonemize(text, espeak_lang)
            except Exception:
                continue
            if not phonemes:
                continue

        fname = f"pimsleur_{lesson_id}_{seg_idx:04d}.wav"
        dest = audio_dir / fname
        try:
            sf.write(dest, clip, 16000)
        except (OSError, sf.LibsndfileError) as e:
            # Filesystem failure (read-only mount, disk full, vanished path).
            # Abort the whole run — continuing would burn Groq quota on
            # successes that can't be persisted.
            raise DiskWriteError(f"sf.write failed at {dest}: {e}") from e

        # Extract the highest-signal Whisper fields into the manifest
        # alongside the full payload. The flat fields (avg_logprob,
        # no_speech_prob, etc.) are convenient for filtering at load
        # time; `whisper` carries the rest for future re-analysis.
        payload = whisper.get("payload") or {}
        segs = payload.get("segments") or []
        # Aggregate per-segment metrics; for short clips Whisper often
        # returns a single segment, in which case avg == min == max.
        avg_logprob = None
        no_speech_prob = None
        compression_ratio = None
        if segs:
            logprobs = [s.get("avg_logprob") for s in segs if s.get("avg_logprob") is not None]
            no_speeches = [s.get("no_speech_prob") for s in segs if s.get("no_speech_prob") is not None]
            comps = [s.get("compression_ratio") for s in segs if s.get("compression_ratio") is not None]
            avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None
            # Worst-case (most likely "no speech") across the clip
            no_speech_prob = max(no_speeches) if no_speeches else None
            compression_ratio = max(comps) if comps else None

        entries.append({
            "file": fname,
            "sentence": text,
            "source": "pimsleur",
            # The espeak voice this clip's text was phonemized with at
            # extraction time. Preserved per-row so preprocess.py can
            # match phoneme labels to dialect (e.g. Brazilian vs European
            # Portuguese, both stored under lang="por" but with different
            # espeak voices "pt-br" vs "pt"). Separate field from `voice`
            # because download_tatoeba.py stores the human uploader name
            # in `voice`, which is not an espeak voice code.
            "espeak_voice": espeak_lang,
            "lang": target_lang,
            "duration_sec": clip.shape[0] / 16000.0,
            "whisper_language": whisper.get("language"),
            "whisper_avg_logprob": avg_logprob,
            "whisper_no_speech_prob": no_speech_prob,
            "whisper_compression_ratio": compression_ratio,
            # Full Whisper verbose_json — segments, tokens, etc.
            # Re-analyzing later (confidence filters, fragment detection,
            # token-level inspection) is free if we keep this.
            "whisper": payload,
        })
    return entries, lesson_ok


def discover_lessons(pimsleur_root: Path,
                     wanted_langs: set[str] | None) -> list[tuple[Path, str, str | None]]:
    """Walk Pimsleur folders, return [(mp3_path, lang_code, espeak_lang), ...].

    espeak_lang is None for Pimsleur languages without espeak voice
    support (Ojibwe, Pashto, Tagalog, Twi). Audio + transcript still get
    extracted; phonemization is skipped.
    """
    out: list[tuple[Path, str, str | None]] = []
    for folder in sorted(pimsleur_root.iterdir()):
        if not folder.is_dir():
            continue
        mapping = PIMSLEUR_FOLDER_TO_LANG.get(folder.name)
        if mapping is None:
            continue
        lang, espeak_lang = mapping
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
            out.append((mp3, lang, espeak_lang))
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
    parser.add_argument("--lesson-workers", type=int, default=4,
                        help="How many lessons to process concurrently. Total "
                             "Whisper calls in flight = workers * lesson-workers. "
                             "4*8=32 is a comfortable load for Groq.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recover-partials", action="store_true",
                        help="Reprocess already-marker'd lessons to fill in "
                             "segments missing from earlier transient Whisper "
                             "failures. Re-decodes + re-VADs each lesson and "
                             "only Whispers seg_idxs whose WAVs aren't on disk. "
                             "silero-vad is deterministic so seg_idx numbering "
                             "matches across runs. Existing manifest rows are "
                             "preserved via filename dedupe.")
    args = parser.parse_args()
    load_env_file(args.env_file)

    if not args.pimsleur_root.exists():
        sys.exit(f"Pimsleur root not found: {args.pimsleur_root}")

    wanted = set(args.lang) if args.lang else None
    lessons = discover_lessons(args.pimsleur_root, wanted)
    print(f"Discovered {len(lessons)} Pimsleur lessons across "
          f"{len({lang for _, lang, _ in lessons})} language(s).")
    if args.dry_run:
        for path, lang, espeak_lang in lessons[:20]:
            print(f"  [{lang}, espeak={espeak_lang}] {path}")
        if len(lessons) > 20:
            print(f"  ... and {len(lessons) - 20} more")
        return

    # Group by lang so we respect --max-lessons-per-lang. Each lesson
    # carries (path, lesson_id, espeak_lang); lesson_id is computed once
    # from the mp3's path-relative-to-archive-root so it stays unique
    # across courses that share the same target lang (e.g. Brazilian +
    # European Portuguese both map to "por", and their Unit-01.mp3 files
    # would otherwise collide on disk).
    by_lang: dict[str, list[tuple[Path, str, str | None]]] = {}
    for path, lang, espeak_lang in lessons:
        lid = lesson_id_from_path(path, args.pimsleur_root)
        by_lang.setdefault(lang, []).append((path, lid, espeak_lang))
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

    # Sort: target training languages first (so the user can start
    # training as soon as those are extracted), then everything else
    # alphabetical. TARGET_LANGS order itself is alphabetical, doesn't
    # matter — all targets get processed before any non-target.
    def _order(lang: str) -> tuple[int, str]:
        return (0 if lang in TARGET_LANGS else 1, lang)
    for lang, lessons_for_lang in sorted(by_lang.items(), key=lambda kv: _order(kv[0])):
        audio_dir = args.audio_root / lang
        audio_dir.mkdir(parents=True, exist_ok=True)
        # Pre-flight write probe. If audio_dir is on a read-only or
        # otherwise wedged volume, fail loudly NOW instead of burning
        # Groq quota on lessons whose WAVs can never persist.
        probe = audio_dir / ".write_probe"
        try:
            probe.write_bytes(b"ok")
            probe.unlink()
        except OSError as e:
            sys.exit(f"FATAL: cannot write to {audio_dir}: {e}. "
                     f"Check the disk (e.g. T7 mount) and re-run.")
        manifest_path = audio_dir / "manifest.jsonl"

        existing_files: set[str] = set()
        if manifest_path.exists():
            with manifest_path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    existing_files.add(rec["file"])

        # Per-lesson marker file. Each completed lesson (even if it
        # yielded zero target-language segments) gets its lesson_id
        # appended here, so re-running the orchestrator after more data
        # arrives skips lessons we've already paid Whisper for.
        processed_path = audio_dir / "pimsleur_processed.txt"
        processed_lessons: set[str] = set()
        if processed_path.exists():
            processed_lessons = {
                line.strip() for line in processed_path.read_text().splitlines()
                if line.strip()
            }

        # Recovery mode: process EVERY lesson, regardless of marker state.
        # process_lesson will see existing pimsleur_<lid>_<seg:04d>.wav files
        # on disk and skip Whisper for those seg_idxs. Use this to backfill
        # segments that were lost to transient Whisper failures in older
        # runs (when failures used to silently mark the lesson done).
        #
        # Normal mode: filter out already-processed lessons up-front.
        # Lesson_id is the path-derived stable identifier from
        # lesson_id_from_path. Also honor `mp3.stem` to stay compatible with
        # markers written by the earlier bare-stem id scheme.
        if args.recover_partials:
            todo = list(lessons_for_lang)
        else:
            todo = [(mp3, lid, e) for (mp3, lid, e) in lessons_for_lang
                    if lid not in processed_lessons
                    and mp3.stem not in processed_lessons]
        # Voices may differ across courses for the same lang (Brazilian
        # Portuguese pt-br vs European Portuguese pt, etc.) — log the set.
        voices = sorted({e for (_, _, e) in lessons_for_lang if e})
        mode_note = " [RECOVERY]" if args.recover_partials else ""
        print(f"\n=== {lang} (voices={voices or [None]}){mode_note}: "
              f"{len(lessons_for_lang)} lessons total, "
              f"{len(processed_lessons)} already done, "
              f"{len(todo)} to do ===")
        if not todo:
            continue

        # Pre-compute the set of existing seg_idxs per lesson from
        # MANIFEST filenames (not on-disk WAVs). Manifest is the source of
        # truth: a row only lands there under state_lock after the WAV is
        # written and the segment is committed. An on-disk WAV without a
        # manifest row is an orphan (process killed between sf.write and
        # the manifest append), and we WANT to re-Whisper its seg_idx
        # rather than silently skip it.
        #
        # Applied in both modes:
        #   - Recovery mode (--recover-partials): backfills gap segments
        #     in marker'd lessons.
        #   - Normal mode: when a lesson is unmarked from a previous
        #     partial-failure run, this skips the segments that already
        #     succeeded so re-runs only Whisper the actual failed calls.
        existing_segs_by_lid: dict[str, set[int]] = {}
        seg_pat = re.compile(r"^pimsleur_(.+)_(\d{4})\.wav$")
        for fname in existing_files:
            m = seg_pat.match(fname)
            if not m:
                continue
            wav_lid, seg = m.group(1), int(m.group(2))
            existing_segs_by_lid.setdefault(wav_lid, set()).add(seg)

        # Legacy bare-stem skip is only safe when the stem is unique within
        # the lang's todo. Pimsleur's per-course ISBN prefixes usually make
        # stems unique, but two courses can in principle share a stem (e.g.
        # both having a `Unit 01.mp3`). If they do, an old WAV's bare-stem
        # key matches both lessons, and using the skip set would falsely
        # skip the other course's seg_idx — exactly the collision case that
        # lesson_id_from_path was introduced to prevent.
        from collections import Counter as _Counter
        stem_counts = _Counter(mp3.stem for (mp3, _, _) in lessons_for_lang)
        unique_stems = {stem for stem, n in stem_counts.items() if n == 1}

        # Lesson-level parallelism: process N lessons concurrently. Each
        # lesson internally still runs Whisper in 8-way parallel, so
        # total in-flight Groq calls = lesson_workers * args.workers.
        # Shared mutable state (manifest write, processed-marker write,
        # existing_files / processed_lessons sets) is serialized with a lock.
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        state_lock = threading.Lock()
        # silero-vad is a small TorchScript model; sharing across threads
        # is fine for eval-mode CPU inference, but we use a lock to avoid
        # contention on the model's internal hidden state buffers.
        vad_lock = threading.Lock()

        def vad_fn(audio):
            with vad_lock:
                return vad_segments(audio, vad_model, get_speech_timestamps)

        def run_one(idx_mp3_lid_espeak):
            idx, (mp3, lid, lesson_espeak) = idx_mp3_lid_espeak
            # Skip-set from the manifest: don't re-Whisper segments already
            # committed. Honor the legacy bare-stem id too — manifests written
            # before lesson_id_from_path() landed use the old filename scheme —
            # but only when the stem is unique within this lang to avoid
            # cross-lesson collisions.
            skip = set(existing_segs_by_lid.get(lid, set()))
            if mp3.stem in unique_stems:
                skip |= existing_segs_by_lid.get(mp3.stem, set())
            try:
                entries, ok = process_lesson(
                    mp3, lid, lang, lesson_espeak,
                    vad_fn, save_audio,
                    audio_dir, api_key, args.model,
                    workers=args.workers,
                    skip_seg_idxs=skip,
                )
            except DiskWriteError:
                # Fatal: propagate so the main as_completed loop can abort.
                # Continuing would burn quota on lessons we can't persist.
                raise
            except Exception as e:
                return idx, mp3, lid, None, False, repr(e)
            return idx, mp3, lid, entries, ok, None

        kept_total = 0
        empty_count = 0
        fail_count = 0
        with ThreadPoolExecutor(max_workers=args.lesson_workers) as pool:
            futs = [pool.submit(run_one, (i, x)) for i, x in enumerate(todo)]
            try:
                completed_iter = as_completed(futs)
            except KeyboardInterrupt:
                raise
            for fut in completed_iter:
                try:
                    idx, mp3, lid, entries, ok, err = fut.result()
                except DiskWriteError as e:
                    # Filesystem went sideways mid-run (read-only mount,
                    # disk full, etc). Cancel pending lessons and exit.
                    print(f"\nFATAL: {e}", flush=True)
                    print("Cancelling outstanding lessons and exiting. "
                          "Fix the disk, then re-run — the manifest-based "
                          "skip-set will pick up where we left off.",
                          flush=True)
                    for f in futs:
                        f.cancel()
                    raise SystemExit(2)
                if err is not None:
                    fail_count += 1
                    print(f"  [{lang}] {idx+1}/{len(todo)} FAIL {mp3.name}: {err}")
                    continue
                if entries is None:
                    entries = []
                with state_lock:
                    kept = [e for e in entries if e["file"] not in existing_files]
                    with manifest_path.open("a") as out:
                        for e in kept:
                            out.write(json.dumps(e, ensure_ascii=False) + "\n")
                            existing_files.add(e["file"])
                    # Only mark processed when the lesson completed cleanly.
                    # Transient failures (decode error, all Whisper calls
                    # exhausted retries) return ok=False so the next run
                    # retries them instead of permanently skipping.
                    if ok:
                        with processed_path.open("a") as out:
                            out.write(f"{lid}\n")
                        processed_lessons.add(lid)
                    else:
                        fail_count += 1
                kept_total += len(kept)
                if ok and not kept:
                    empty_count += 1
                status = "ok" if ok else "transient-fail"
                print(f"  [{lang}] {idx+1}/{len(todo)} {mp3.name}: "
                      f"kept {len(kept)} [{status}]")
        print(f"  [{lang}] DONE. kept_total={kept_total} "
              f"empty_lessons={empty_count} failed={fail_count}")


if __name__ == "__main__":
    main()

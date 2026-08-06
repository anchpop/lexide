#!/usr/bin/env python3
"""Extract clean native-language clips from Pimsleur audio archives.

Each Pimsleur lesson is a ~30-min MP3 containing alternating English
instructor speech and target-language native-speaker speech, separated
by silences. We want only the native-language clips for training.

Pipeline per MP3:
  1. Decode to 16 kHz mono via ffmpeg.
  2. silero-vad → list of (start, end) speech timestamps.
  3. For each speech segment of reasonable duration:
     - Send to Cloudflare Workers AI Whisper (no `language` param →
       it detects).
     - If detected_language == target, phonemize via espeak-ng, save
       a WAV with filename `pimsleur_<lesson_id>_<seg_idx>.wav`, and
       append to manifest with source="pimsleur".

Whisper is the source of truth for the transcript here — there's no
external ground-truth to audit against, so we trust its output.

Requires:
  - silero-vad via torch.hub (lazy-loaded, no separate pip install)
  - CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN in env or .env file
    (token needs the account-scoped Workers AI permission)
  - ffmpeg on PATH
  - espeak-ng on PATH
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import json
import os
import random
import re
import string
import subprocess
import sys
import tempfile
import threading
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
from preprocess import phonemize, LANG_TO_ESPEAK, BACKEND_REQUIRED_LANGS  # noqa: E402

CF_AI_BASE = "https://api.cloudflare.com/client/v4/accounts"


class DiskWriteError(Exception):
    """Raised when sf.write fails for what looks like a filesystem-level
    problem (read-only mount, disk full, missing path). These have to abort
    the whole run — silently continuing turns every Whisper success into a
    lost segment AND spends transcription budget on work we can't keep.
    """

# Groq's verbose_json returned the detected language as a lowercase full
# English name ("english", "french"); Cloudflare returns an ISO code ("en",
# "fr"). whisper_lang_matches() accepts either, so manifests written by both
# backends compare correctly and this table stays useful for the old rows.
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
    "zho-hans": "chinese",
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
    "zho-hans": "zh",
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
TARGET_LANGS = (
    "deu", "eng", "fra", "ita", "por", "rus", "spa",
    "tha", "zho-hans", "hin", "jpn",
)

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
    "Mandarin Chinese": ("zho-hans", "cmn"),
    "Chinese Mandarin": ("zho-hans", "cmn"),
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


def transcribe_clip(wav_bytes: bytes, creds: tuple[str, str], model: str,
                    retries: int = 2, timeout: float = 180.0) -> dict[str, Any]:
    """Send a single WAV clip to Cloudflare Workers AI Whisper, no language hint.

    Returns the full result payload alongside an "ok" flag so the caller can
    store every Whisper signal in the manifest (segments, avg_logprob,
    no_speech_prob, compression_ratio, word timestamps). Throwing those away
    at extraction time means later filtering needs another API call per clip;
    storing them is free.

    Cloudflare serves the same @cf/openai/whisper-large-v3-turbo weights Groq
    did, and its `segments` entries carry avg_logprob / no_speech_prob /
    compression_ratio under exactly the names the manifest builder already
    reads — so the payload drops straight in. It also returns word-level
    timestamps, which Groq's verbose_json did not. Language moves: Groq put it
    at the top level, Cloudflare puts it in `transcription_info` (as an ISO
    code, "ja", where Groq said "japanese").

    Audio goes as base64 inside JSON, which inflates the body ~33%.

    Why this replaced Groq: the free on_demand tier capped at 200k requests/day
    and, once spent, returned 429 on every call for a full day — indistinguish-
    able from a transient failure, so the retry passes silently burned VAD work
    transcribing nothing. Cloudflare bills ~$0.00051/audio-minute (about $0.80
    for the entire Pimsleur corpus) with no daily wall, and measured 64/64
    successes at both 8 and 24 concurrent requests where Groq failed ~60% at 32.
    Retries are cheap now that a failure means a real error rather than an
    exhausted quota, so this retries twice instead of once.
    """
    account_id, api_token = creds
    url = f"{CF_AI_BASE}/{account_id}/ai/run/@cf/openai/{model}"
    body = json.dumps({"audio": base64.b64encode(wav_bytes).decode()})
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_token}",
                         "Content-Type": "application/json"},
                data=body,
                timeout=timeout,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else min(20.0, 2.0 * (attempt + 1)))
                last_error = "429"
                continue
            resp.raise_for_status()
            result = (resp.json() or {}).get("result") or {}
            info = result.get("transcription_info") or {}
            return {
                "ok": True,
                "text": (result.get("text") or "").strip(),
                "language": info.get("language") or "",
                "payload": result,  # segments + word timestamps + usage
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
                   audio_dir: Path, creds: tuple[str, str], model: str,
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
        w = transcribe_clip(wav_bytes, creds, model)
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
        # The espeak gate only screens for text espeak can label, so it does
        # not apply to backend languages (eSpeak is never their label source;
        # their G2P sidecars do their own explicit excluding). It would also
        # wrongly drop every segment on machines without the espeak fork.
        if espeak_lang is not None and target_lang not in BACKEND_REQUIRED_LANGS:
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
            # Abort the whole run — continuing would keep paying to
            # transcribe successes that can't be persisted.
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
    parser.add_argument("--workers", type=int, default=6,
                        help="Concurrent Whisper requests per lesson.")
    parser.add_argument("--lesson-workers", type=int, default=4,
                        help="How many lessons to process concurrently. Total "
                             "Whisper calls in flight = workers * lesson-workers. "
                             "6*4=24 is the concurrency measured clean against "
                             "Cloudflare Workers AI: 64/64 successes at both 8 "
                             "and 24 in flight, ~10.9 req/s and a ~1.35s median. "
                             "No client-side rate limiter — the Groq tier needed "
                             "one (400 RPM, and 32 in flight failed ~60% of "
                             "calls); Cloudflare did not.")
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

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    missing = [n for n, v in (("CLOUDFLARE_ACCOUNT_ID", account_id),
                              ("CLOUDFLARE_API_TOKEN", api_token)) if not v]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} not set (needs an "
                         f"account-scoped Workers AI token)")
    creds = (account_id, api_token)

    print("Loading silero-vad...")
    vad_model, vad_utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", verbose=False, trust_repo=True,
    )
    get_speech_timestamps, save_audio, *_ = vad_utils

    # Per-lang setup: pre-flight write probe, manifest load, marker load,
    # todo compute, skip-set build. Hoisted out of the processing loop so
    # we can interleave lessons across langs in the executor (round-robin
    # treats the 7 target langs equally — none has to wait for another to
    # fully complete).
    import threading
    from collections import Counter as _Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    seg_pat = re.compile(r"^pimsleur_(.+)_(\d{4})\.wav$")
    lang_states: dict[str, dict[str, Any]] = {}

    # Iterate in target-first alphabetical order so write probes fail fast
    # on the cores if anything's wrong.
    def _setup_order(lang: str) -> tuple[int, str]:
        return (0 if lang in TARGET_LANGS else 1, lang)
    for lang in sorted(by_lang, key=_setup_order):
        lessons_for_lang = by_lang[lang]
        audio_dir = args.audio_root / lang
        audio_dir.mkdir(parents=True, exist_ok=True)
        # Pre-flight write probe — fail loud if the volume is read-only.
        probe = audio_dir / ".write_probe"
        try:
            probe.write_bytes(b"ok")
            probe.unlink()
        except OSError as e:
            sys.exit(f"FATAL: cannot write to {audio_dir}: {e}. "
                     f"Check the disk (e.g. T7 mount) and re-run.")

        manifest_path = audio_dir / "manifest.jsonl"
        processed_path = audio_dir / "pimsleur_processed.txt"

        existing_files: set[str] = set()
        if manifest_path.exists():
            with manifest_path.open() as f:
                for line in f:
                    existing_files.add(json.loads(line)["file"])

        processed_lessons: set[str] = set()
        if processed_path.exists():
            processed_lessons = {
                line.strip() for line in processed_path.read_text().splitlines()
                if line.strip()
            }

        # Recovery mode processes every lesson (skip-set still gates Whisper
        # at the segment level). Normal mode skips marker'd lessons up-front.
        if args.recover_partials:
            todo = list(lessons_for_lang)
        else:
            todo = [(mp3, lid, e) for (mp3, lid, e) in lessons_for_lang
                    if lid not in processed_lessons
                    and mp3.stem not in processed_lessons]

        # Build the per-lesson skip-set from manifest filenames (source of
        # truth). Orphan WAVs without a manifest row will be re-Whispered.
        existing_segs_by_lid: dict[str, set[int]] = {}
        for fname in existing_files:
            m = seg_pat.match(fname)
            if m:
                existing_segs_by_lid.setdefault(m.group(1), set()).add(int(m.group(2)))

        # Legacy bare-stem skip only when the stem is unique within the lang.
        stem_counts = _Counter(mp3.stem for (mp3, _, _) in lessons_for_lang)
        unique_stems = {stem for stem, n in stem_counts.items() if n == 1}

        voices = sorted({e for (_, _, e) in lessons_for_lang if e})
        mode_note = " [RECOVERY]" if args.recover_partials else ""
        print(f"=== {lang} (voices={voices or [None]}){mode_note}: "
              f"{len(lessons_for_lang)} lessons total, "
              f"{len(processed_lessons)} already done, "
              f"{len(todo)} to do ===")

        lang_states[lang] = {
            "lang": lang,
            "audio_dir": audio_dir,
            "manifest_path": manifest_path,
            "processed_path": processed_path,
            "existing_files": existing_files,
            "processed_lessons": processed_lessons,
            "existing_segs_by_lid": existing_segs_by_lid,
            "unique_stems": unique_stems,
            "todo": todo,
            "lock": threading.Lock(),
            "kept_total": 0,
            "empty_count": 0,
            "fail_count": 0,
        }

    # silero-vad is a small TorchScript model; sharing across threads is
    # fine for eval-mode CPU inference, but we serialize via a lock to
    # avoid contention on the model's internal hidden state buffers.
    vad_lock = threading.Lock()
    def vad_fn(audio):
        with vad_lock:
            return vad_segments(audio, vad_model, get_speech_timestamps)

    # Build a flat round-robin task list across the target langs first
    # (so all 7 cores progress at equal rates and no lang is starved by
    # waiting for another to finish), then non-target langs.
    def _round_robin(lang_order: list[str]) -> list[tuple[str, int, tuple]]:
        max_len = max((len(lang_states[l]["todo"]) for l in lang_order), default=0)
        tasks: list[tuple[str, int, tuple]] = []
        for i in range(max_len):
            for lang in lang_order:
                td = lang_states[lang]["todo"]
                if i < len(td):
                    tasks.append((lang, i, td[i]))
        return tasks

    target_order = sorted(l for l in lang_states if l in TARGET_LANGS)
    nontarget_order = sorted(l for l in lang_states if l not in TARGET_LANGS)
    all_tasks = _round_robin(target_order) + _round_robin(nontarget_order)
    print(f"\nTotal tasks queued: {len(all_tasks)} "
          f"(target round-robin: {sum(len(lang_states[l]['todo']) for l in target_order)}; "
          f"non-target round-robin: {sum(len(lang_states[l]['todo']) for l in nontarget_order)})")
    if not all_tasks:
        return

    def run_one(task):
        lang, idx_within_lang, (mp3, lid, lesson_espeak) = task
        st = lang_states[lang]
        skip = set(st["existing_segs_by_lid"].get(lid, set()))
        if mp3.stem in st["unique_stems"]:
            skip |= st["existing_segs_by_lid"].get(mp3.stem, set())
        try:
            entries, ok = process_lesson(
                mp3, lid, lang, lesson_espeak,
                vad_fn, save_audio,
                st["audio_dir"], creds, args.model,
                workers=args.workers,
                skip_seg_idxs=skip,
            )
        except DiskWriteError:
            raise
        except Exception as e:
            return lang, idx_within_lang, mp3, lid, None, False, repr(e)
        return lang, idx_within_lang, mp3, lid, entries, ok, None

    # Single executor processes all tasks; round-robin ordering means each
    # lang's lesson N is queued before any lang's lesson N+1, so completion
    # progress walks across langs roughly in lockstep.
    with ThreadPoolExecutor(max_workers=args.lesson_workers) as pool:
        futs = [pool.submit(run_one, t) for t in all_tasks]
        for fut in as_completed(futs):
            try:
                lang, idx_within_lang, mp3, lid, entries, ok, err = fut.result()
            except DiskWriteError as e:
                print(f"\nFATAL: {e}", flush=True)
                print("Cancelling outstanding lessons and exiting. "
                      "Fix the disk, then re-run — the manifest-based "
                      "skip-set will pick up where we left off.",
                      flush=True)
                for f in futs:
                    f.cancel()
                raise SystemExit(2)
            st = lang_states[lang]
            todo_n = len(st["todo"])
            if err is not None:
                st["fail_count"] += 1
                print(f"  [{lang}] {idx_within_lang+1}/{todo_n} FAIL {mp3.name}: {err}")
                continue
            if entries is None:
                entries = []
            with st["lock"]:
                kept = [e for e in entries if e["file"] not in st["existing_files"]]
                with st["manifest_path"].open("a") as out:
                    for e in kept:
                        out.write(json.dumps(e, ensure_ascii=False) + "\n")
                        st["existing_files"].add(e["file"])
                if ok:
                    with st["processed_path"].open("a") as out:
                        out.write(f"{lid}\n")
                    st["processed_lessons"].add(lid)
                else:
                    st["fail_count"] += 1
            st["kept_total"] += len(kept)
            if ok and not kept:
                st["empty_count"] += 1
            status = "ok" if ok else "transient-fail"
            print(f"  [{lang}] {idx_within_lang+1}/{todo_n} {mp3.name}: "
                  f"kept {len(kept)} [{status}]")

    print()
    for lang in sorted(lang_states):
        st = lang_states[lang]
        print(f"  [{lang}] DONE. kept_total={st['kept_total']} "
              f"empty_lessons={st['empty_count']} failed={st['fail_count']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download Tatoeba audio for a sample of sentences per language.

Tatoeba publishes:
  - sentences_with_audio.csv: row_id, sentence_id, audio_uploader, ...
  - sentences.csv: sentence_id, lang_code, text

This script joins them, samples N sentences per language (default 10_000),
fetches the audio from https://tatoeba.org/audio/download/<id>, converts
each mp3 to 16 kHz mono WAV via ffmpeg, and appends entries to the
existing per-language `manifest.jsonl`. Then `train/scripts/preprocess.py`
regenerates `phonemes.jsonl` from the combined manifest.

File naming uses `tatoeba_<sentence_id>.wav`; that namespace doesn't
collide with FLEURS's content-hash filenames, so the two sources coexist
in the same lang directory.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LANGS = ["deu", "eng", "fra", "ita", "por", "rus", "spa"]

# Each thread gets its own requests.Session for HTTP/1.1 keep-alive
# (Tatoeba responds in ~4 sec with a warm connection vs ~40 sec cold).
MAX_WORKERS = 32

TIMEOUT = 30
RETRIES = 3
USER_AGENT = "lexide-pronunciation-trainer/0.1 (research, contact: anchpop)"

# Thread-local sessions keep connection-pool state per worker.
_thread_local = threading.local()


def get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers["User-Agent"] = USER_AGENT
        adapter = HTTPAdapter(
            pool_connections=4,
            pool_maxsize=4,
            max_retries=Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        sess.mount("https://", adapter)
        _thread_local.session = sess
    return sess


def load_join(audio_csv: Path, sentences_csv: Path):
    """Return per-lang list of records describing audio-bearing sentences.

    Each record is a dict with `sentence_id`, `audio_id`, `uploader`,
    `license`, `attribution_url`, `text`. The audio download URL uses
    `audio_id`, not `sentence_id` — those two are distinct and got
    mixed up in an earlier version of this script. The filename also
    uses `audio_id` so traceability holds.

    Tatoeba's license policy: rows whose license field is empty cannot
    be reused outside Tatoeba per the export's README, so we filter
    those out here. Same for "no license" / "All rights reserved".
    Acceptable licenses are the CC family.
    """
    # 1) audio entries: sentence_id -> [audio_id, uploader, license, attribution]
    # Schema: sentence_id \t audio_id \t uploader \t license \t attribution_url
    audio_by_sid: dict[int, dict] = {}
    skipped_no_license = 0
    with open(audio_csv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                sid = int(parts[0])
                aid = int(parts[1])
            except ValueError:
                continue
            uploader = parts[2]
            lic = parts[3].strip()
            attribution = parts[4].strip()

            # Filter unreusable licenses. Tatoeba's exporter leaves the
            # license field empty for clips that cannot be redistributed.
            if not lic or "all rights reserved" in lic.lower() or lic == "(unknown)":
                skipped_no_license += 1
                continue

            audio_by_sid[sid] = {
                "audio_id": aid,
                "uploader": uploader,
                "license": lic,
                "attribution_url": attribution,
            }

    print(f"Loaded {len(audio_by_sid):,} audio-bearing sentences "
          f"(skipped {skipped_no_license:,} with empty/proprietary license)")

    # 2) Join sentences.csv → group records by target language.
    # Schema: sentence_id \t lang \t text
    by_lang: dict[str, list[dict]] = {lang: [] for lang in LANGS}
    with open(sentences_csv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                sid = int(parts[0])
            except ValueError:
                continue
            meta = audio_by_sid.get(sid)
            if meta is None:
                continue
            lang = parts[1]
            if lang not in by_lang:
                continue
            by_lang[lang].append({
                "sentence_id": sid,
                "text": parts[2],
                **meta,
            })

    for lang, items in by_lang.items():
        print(f"  {lang}: {len(items):,} candidates")
    return by_lang


def fetch_one(audio_id: int, dest_wav: Path) -> tuple[bool, str]:
    """Download audio + convert to 16 kHz mono WAV. Returns (ok, error_msg).

    The numeric argument is the Tatoeba *audio_id* (not sentence_id). The
    `/audio/download/<id>` endpoint expects audio IDs.
    """
    if dest_wav.exists() and dest_wav.stat().st_size > 0:
        return True, "already-exists"

    url = f"https://tatoeba.org/audio/download/{audio_id}"
    sess = get_session()

    last_err = ""
    mp3_bytes = b""
    for attempt in range(RETRIES):
        try:
            resp = sess.get(url, timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code != 200:
                last_err = f"http {resp.status_code}"
                # Adapter already retried on 429/5xx; if we get here, it's a hard fail.
                if resp.status_code == 429:
                    time.sleep(1.0 + attempt)
                continue
            mp3_bytes = resp.content
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5 * (attempt + 1))
    else:
        return False, f"download: {last_err}"

    if not mp3_bytes:
        return False, "empty body"

    # ffmpeg: stdin mp3 -> stdout WAV at 16 kHz mono
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", "pipe:0",
                "-ac", "1", "-ar", "16000",
                "-f", "wav", "pipe:1",
            ],
            input=mp3_bytes, capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            return False, f"ffmpeg: {result.stderr.decode()[:200]}"
        dest_wav.write_bytes(result.stdout)
        return True, "ok"
    except Exception as e:
        return False, f"convert: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-csv", type=Path,
                        default=Path("/tmp/tat_audio/sentences_with_audio.csv"))
    parser.add_argument("--sentences-csv", type=Path,
                        default=Path("/tmp/tat_audio/sentences.csv"))
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent / "audio")
    parser.add_argument("--samples-per-lang", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.audio_csv.exists() or not args.sentences_csv.exists():
        sys.exit(
            f"Missing CSV inputs. Expected:\n"
            f"  {args.audio_csv}\n  {args.sentences_csv}\n"
            "Fetch from https://downloads.tatoeba.org/exports/ first."
        )

    by_lang = load_join(args.audio_csv, args.sentences_csv)

    rng = random.Random(args.seed)
    for lang in LANGS:
        items = by_lang[lang]
        # Stable sample: shuffle then take first N.
        rng.shuffle(items)
        sample = items[: args.samples_per_lang]
        print(f"\n=== {lang}: downloading {len(sample):,} clips ===")

        lang_dir = args.output_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = lang_dir / "manifest.jsonl"

        # Existing entries (so we don't duplicate Tatoeba on re-runs).
        existing_files: set[str] = set()
        if manifest_path.exists():
            with open(manifest_path) as f:
                for line in f:
                    rec = json.loads(line)
                    existing_files.add(rec["file"])

        new_entries = []
        ok = 0
        skipped_existing = 0
        failed: list[str] = []

        def work(rec):
            audio_id = rec["audio_id"]
            fname = f"tatoeba_{audio_id}.wav"
            if fname in existing_files:
                return (rec, fname, True, "already-in-manifest")
            wav_path = lang_dir / fname
            success, msg = fetch_one(audio_id, wav_path)
            return (rec, fname, success, msg)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(work, rec) for rec in sample]
            for i, fut in enumerate(as_completed(futures)):
                rec, fname, success, msg = fut.result()
                if success:
                    if msg == "already-in-manifest":
                        skipped_existing += 1
                    else:
                        ok += 1
                        new_entries.append({
                            "file": fname,
                            "sentence": rec["text"],
                            "source": "tatoeba",
                            "voice": rec.get("uploader") or None,
                            "lang": lang,
                            "license": rec.get("license"),
                            "attribution_url": rec.get("attribution_url"),
                            "tatoeba_sentence_id": rec["sentence_id"],
                        })
                else:
                    failed.append(f"audio={rec['audio_id']}: {msg}")
                if (i + 1) % 250 == 0:
                    print(f"  [{lang}] {i + 1}/{len(sample)} "
                          f"ok={ok} skipped={skipped_existing} fail={len(failed)}")

        # Append to manifest (preserve existing FLEURS entries).
        with open(manifest_path, "a") as out:
            for entry in new_entries:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  [{lang}] DONE. ok={ok} skipped={skipped_existing} "
              f"fail={len(failed)} → {manifest_path}")
        if failed[:5]:
            print(f"  first failures: {failed[:5]}")

    print("\nAll languages done. Next: re-run train/scripts/preprocess.py "
          "to regenerate phonemes.jsonl from the combined manifests.")


if __name__ == "__main__":
    main()

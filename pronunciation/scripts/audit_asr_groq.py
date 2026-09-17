#!/usr/bin/env python3
"""Audit local audio (FLEURS, Tatoeba, TTS, ...) with Groq Whisper, phoneme-level PER.

One source-parameterized auditor for both corpora — they differ only in which
manifest `source` to read and which exclusions file to write. We use
**phoneme-level** error, not raw text: both the expected sentence and Whisper's
transcript use the production label provider: g2p-tha (vachana-thai), g2p-hin
(unified labels), g2p-zho, g2p-jpn, g2p-kor, or preprocess.phonemize (eSpeak)
for the remaining languages. Orthographic differences that don't change sound
(apostrophes, capitalization, "15" vs "fifteen", name transliteration) normalize
away when accepted by the provider, while genuine pronunciation/content mismatch
shows up as phoneme distance. An unlabelable reference omits PER and uses the
loader's CER/WER fallback; an unlabelable Whisper transcript remains a mismatch.

Whisper is called with the target language forced (--force-language, default on):
we already know the intended language from the label, so forcing it removes
dialect/auto-detect confounds; wrong-language or garbled audio still gets caught
because it phonemizes to gibberish → large phoneme distance.

Output schema is consumed by `load_asr_audit_exclusions` in train_unified.py:
per-clip JSONL with `ok`, `lang`, `file`, `expected_sha256`, `per`, `cer`, `wer`
(+ full Whisper payload, so re-analysis is free). The loader hard-excludes any
row with per >= min_per (default ~0) after re-checking the sentence hash, so a
repaired label is never punished by a stale audit row.

Resumeable: existing successful rows (by path) are skipped — rerun to fill new
clips after a re-download without re-paying Groq for clips already done. Set
GROQ_API_KEY in .env; phonemization needs the `g2p` binary on PATH (or G2P_BIN).

  audit_asr_groq.py --source fleurs    # -> train/fleurs_asr_exclusions.jsonl
  audit_asr_groq.py --source tatoeba   # -> train/tatoeba_asr_exclusions.jsonl
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

# Reuse the production provider registry and sidecar converters rather than
# maintaining a second language/backend mapping for the audit.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "train" / "scripts"))
from g2p_client import phonemize, Unlabelable  # noqa: E402
from corpus_labels import variety_for_record  # noqa: E402


def label_phonemes(text: str, lang: str, variety: str = "default") -> list[str]:
    """Use the same g2p contract as corpus preprocessing."""
    return phonemize(text, lang, variety=variety)["phonemes"]

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
LANG_TO_ISO639_1 = {
    "deu": "de", "eng": "en", "fra": "fr", "ita": "it",
    "por": "pt", "rus": "ru", "spa": "es",
    "tha": "th", "zho-hans": "zh", "hin": "hi", "jpn": "ja",
    "kor": "ko", "ara": "ar", "ces": "cs", "dan": "da",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    # `string.punctuation` covers ASCII only. CJK full stops/commas and Thai
    # punctuation must normalize the same way or CER measures typography.
    text = "".join(
        " " if c in string.punctuation or unicodedata.category(c).startswith("P") else c
        for c in text
    )
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


def load_records(root: Path, langs: set[str] | None, source: str) -> list[dict[str, Any]]:
    records = []
    for manifest in sorted(root.glob("*/manifest.jsonl")):
        lang = manifest.parent.name
        if langs and lang not in langs:
            continue
        with manifest.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("source") != source:
                    continue
                wav_path = manifest.parent / rec["file"]
                records.append({
                    "file": rec["file"],
                    "path": str(wav_path),
                    "lang": lang,
                    "source": source,
                    "voice": rec.get("voice"),
                    # Keep dataset variety when comparing reference and ASR text.
                    "variety": variety_for_record(rec, lang),
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


def compact_audit(path: Path) -> None:
    """Make retries an idempotent last-result-wins update by audio path."""
    if not path.exists():
        return
    by_path: dict[str, dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("ok") and "expected" in rec and "whisper_text" in rec:
                rec["cer"] = text_cer(rec["expected"], rec["whisper_text"])
                rec["wer"] = text_wer(rec["expected"], rec["whisper_text"])
                rec["text_metric_version"] = 2
            key = rec.get("path")
            if key:
                by_path[key] = rec
            else:
                unkeyed.append(rec)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as out:
        for rec in unkeyed + list(by_path.values()):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


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

            score_phonemes = not args.text_only
            if score_phonemes:
                try:
                    expected_phonemes = label_phonemes(
                        record["expected"], record["lang"], variety_for_record(record, record["lang"]),
                    )
                except Unlabelable:
                    # No valid reference: omit PER so the loader uses CER/WER.
                    score_phonemes = False
                else:
                    try:
                        actual_phonemes = label_phonemes(
                            text, record["lang"], variety_for_record(record, record["lang"]),
                        )
                    except Exception:
                        # An unlabelable Whisper output against a valid reference
                        # remains a mismatch (empty actual phones gives PER 1).
                        actual_phonemes = []

            expected_sha = hashlib.sha256(record["expected"].encode()).hexdigest()

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

            result = {
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
                "cer": text_cer(record["expected"], text),
                "wer": text_wer(record["expected"], text),
                # Full verbose_json — keep for future re-analysis without
                # paying Groq again.
                "whisper": payload,
            }
            # Text-only and unlabelable references deliberately omit phoneme
            # fields: the loader applies its CER/WER thresholds instead.
            for key in ("per", "expected_phonemes", "actual_phonemes"):
                result.pop(key, None)
            if score_phonemes:
                result["expected_phonemes"] = expected_phonemes
                result["actual_phonemes"] = actual_phonemes
                result["per"] = phoneme_cer(expected_phonemes, actual_phonemes)
            return result
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


def rescore(path: Path, langs: set[str] | None) -> None:
    """Recompute expected/actual phonemes and PER from stored transcripts."""
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    records = [json.loads(line) for line in path.open() if line.strip()]
    todo = [r for r in records
            if r.get("ok") and "whisper_text" in r
            and (langs is None or r.get("lang") in langs)]
    print(f"Rescoring {len(todo)} of {len(records)} records in {path}")
    changed = 0
    for rec in tqdm(todo, desc="rescore"):
        try:
            expected_phonemes = label_phonemes(
                rec["expected"], rec["lang"], variety_for_record(rec, rec["lang"]),
            )
        except Unlabelable:
            changed += any(key in rec for key in ("per", "expected_phonemes", "actual_phonemes"))
            for key in ("per", "expected_phonemes", "actual_phonemes"):
                rec.pop(key, None)
            continue
        try:
            actual_phonemes = label_phonemes(
                rec["whisper_text"], rec["lang"], variety_for_record(rec, rec["lang"]),
            )
        except Exception:
            actual_phonemes = []
        per = phoneme_cer(expected_phonemes, actual_phonemes)
        if (expected_phonemes, actual_phonemes) != (rec.get("expected_phonemes"), rec.get("actual_phonemes")):
            changed += 1
        rec["expected_phonemes"] = expected_phonemes
        rec["actual_phonemes"] = actual_phonemes
        rec["per"] = per
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"Rescored {len(todo)}; phonemes changed on {changed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        choices=["fleurs", "tatoeba", "kathbath", "tts"],
                        help="Which manifest source to audit (selects default --out too). "
                             "`tts` matters for the Gemini backend in particular: it is "
                             "an LLM reading text, so it can in principle paraphrase or "
                             "decline rather than read, and the audit is what catches a "
                             "clip whose audio stopped matching its label.")
    parser.add_argument("--audio-root", type=Path, default=Path("data/audio"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Defaults to train/<source>_asr_exclusions.jsonl")
    parser.add_argument("--model", default="whisper-large-v3-turbo")
    parser.add_argument("--lang", action="append", choices=sorted(LANG_TO_ISO639_1),
                        help="Limit to one or more repo language codes.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--force-language", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--text-only", action="store_true",
        help="Skip phonemization/PER and audit transcript CER/WER only (useful while a G2P is being qualified).",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rescore", action="store_true",
        help="No Groq calls: re-phonemize the stored expected/whisper_text of every "
             "successful record (for --lang, if given) and recompute PER. For when the "
             "G2P side of an audit changed — a new label backend, or a phonemizer bug.",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = Path("train") / f"{args.source}_asr_exclusions.jsonl"
    load_env_file(args.env_file)

    if args.rescore:
        rescore(args.out, set(args.lang) if args.lang else None)
        return

    records = load_records(args.audio_root, set(args.lang) if args.lang else None, args.source)
    if args.limit is not None:
        random.Random(args.seed).shuffle(records)
        records = records[: args.limit]

    total_seconds = 0.0
    for rec in records:
        try:
            info = sf.info(rec["path"])
            total_seconds += info.frames / info.samplerate
        except Exception:
            pass
    print(
        f"{args.source} records: {len(records)}; audio hours: {total_seconds / 3600:.3f}; "
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
            for fut in tqdm(futures.as_completed(futs), total=len(futs),
                            desc=f"Groq Whisper ({args.source})"):
                out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                out.flush()
    compact_audit(args.out)


if __name__ == "__main__":
    main()

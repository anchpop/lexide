#!/usr/bin/env python3
"""Acquire train-only Multilingual LibriSpeech with strict speaker budgets.

MLS supplies normalized transcripts, not restored book punctuation. Text is
preserved verbatim. The second component of an MLS utterance ID is a BOOK ID,
not a chapter ID (some Hugging Face conversions misleadingly call it chapter_id).
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LANG_CONFIG = {
    "deu": "german", "eng": "english", "fra": "french",
    "ita": "italian", "por": "portuguese", "spa": "spanish",
}
SAMPLE_RATE = 16000


def filename(lang: str, utterance_id: str) -> str:
    if not re.fullmatch(r"\d+_\d+_\d+", utterance_id):
        raise ValueError(f"Invalid MLS utterance ID: {utterance_id!r}")
    return f"mls_{lang}_{utterance_id}.wav"


def fingerprint(path: Path, length: int | None = None) -> dict:
    digest = hashlib.sha256()
    size = 0
    if path.exists():
        with path.open("rb") as stream:
            while length is None or size < length:
                block = stream.read(min(1024 * 1024, length - size) if length is not None else 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


@contextmanager
def manifest_lock(path: Path):
    """Serialize this importer without locking/replacing the user's manifest."""
    with path.with_suffix(".mls.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def existing_rows(path: Path) -> tuple[set[str], dict[str, dict]]:
    files = set()
    mls = {}
    if path.exists():
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    raise ValueError(f"Unterminated manifest row in {path}; refusing to modify existing bytes")
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["file"] in files:
                    raise ValueError(f"Duplicate manifest filename: {row['file']}")
                files.add(row["file"])
                if row.get("source") == "mls":
                    mls[row["file"]] = row
    return files, mls


def budgets(rows: dict[str, dict], root: Path) -> tuple[float, dict[str, float]]:
    """Charge existing clips' actual WAV duration, never trust rounded metadata."""
    speakers = defaultdict(float)
    for row in rows.values():
        info = sf.info(root / row["file"])
        if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
            raise ValueError(f"Invalid existing MLS audio: {row['file']}")
        speakers[row["voice"]] += info.frames / SAMPLE_RATE
    return sum(speakers.values()), dict(speakers)


def balanced_order(candidates: list[dict], seed: int):
    """Seeded speaker round-robin, with a book round-robin within each speaker."""
    rng = random.Random(seed)
    grouped = defaultdict(lambda: defaultdict(list))
    for item in sorted(candidates, key=lambda row: row["id"]):
        grouped[item["speaker_id"]][item["book_id"]].append(item)
    speakers = sorted(grouped)
    rng.shuffle(speakers)
    queues = {}
    for speaker in speakers:
        books = sorted(grouped[speaker])
        rng.shuffle(books)
        queue = deque()
        for book in books:
            items = grouped[speaker][book]
            rng.shuffle(items)
            queue.append(deque(items))
        queues[speaker] = queue
    active = deque(speakers)
    while active:
        speaker = active.popleft()
        books = queues[speaker]
        clips = books.popleft()
        yield clips.popleft()
        if clips:
            books.append(clips)
        if books:
            active.append(speaker)


def select(candidates: list[dict], seed: int, max_seconds: float,
           speaker_seconds: float, existing_total: float = 0,
           existing_speakers: dict[str, float] | None = None,
           existing_ids: set[str] | None = None) -> list[dict]:
    total = existing_total
    used = defaultdict(float, existing_speakers or {})
    existing_ids = existing_ids or set()
    result = []
    seen = set()
    for item in balanced_order(candidates, seed):
        if item["id"] in seen:
            raise ValueError(f"Duplicate source ID: {item['id']}")
        seen.add(item["id"])
        if item["id"] in existing_ids:
            continue
        duration = item["duration"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration for {item['id']}")
        voice = f"mls:{item['speaker_id']}"
        if total + duration > max_seconds or used[voice] + duration > speaker_seconds:
            continue
        total += duration
        used[voice] += duration
        result.append(item)
    return result


def decode_audio(encoded: bytes) -> np.ndarray:
    audio, rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty/non-finite source audio")
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    if not np.any(audio):
        raise ValueError("Silent source audio")
    return audio


def write_wav_atomic(path: Path, audio: np.ndarray) -> None:
    """Finalize then rename atomically; process-resumable, not power-loss durable."""
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".part.wav", dir=path.parent)
    os.close(fd)
    try:
        sf.write(temporary, audio, SAMPLE_RATE, subtype="PCM_16")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def append_row(stream, row: dict) -> None:
    # Flush each record for process-interruption resume; the caller fsyncs once
    # per row group, avoiding two synchronous disk flushes for every tiny WAV.
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()

# Verified dataset revisions: the official conversion does not include English.
DATASETS = {
    "official": ("facebook/multilingual_librispeech", "2e83e61823b4c47dcbcb1980bb88601274127609"),
    "english": ("parler-tts/mls_eng", "faf6604dcf0bb9ec0d9c280a140ad74da9c931b0"),
}


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


_resolved_urls = {}
_resolve_lock = threading.Lock()
_last_resolve = 0.0


def range_response(client, url: str, start: int, end: int):
    """Resolve each HF shard once, then range-read its signed CDN URL directly.

    Repeating /resolve for every row group exhausts HF's resolver quota. Signed
    URLs stay in memory only; authentication is only sent to huggingface.co.
    """
    global _last_resolve
    headers = {"Range": f"bytes={start}-{end - 1}"}
    with _resolve_lock:
        cached = _resolved_urls.get(url)
        if cached and time.monotonic() - cached[1] < 900:
            target = cached[0]
        else:
            target = None
        if target is None:
            time.sleep(max(0, .4 - (time.monotonic() - _last_resolve)))
            _last_resolve = time.monotonic()
    if target is None:
        if urlsplit(url).hostname == "huggingface.co" and os.environ.get("HF_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        # Pace resolver starts, but don't serialize unrelated shards' network
        # latency behind the lock during the all-English footer census.
        response = client.get(url, headers=headers, stream=True, timeout=(30, 120))
        # requests strips Authorization on cross-origin redirects.
        if response.status_code == 206:
            with _resolve_lock:
                _resolved_urls[url] = (response.url, time.monotonic())
        return response
    response = client.get(target, headers=headers, stream=True, timeout=(30, 120))
    if response.status_code in (401, 403):
        response.close()
        with _resolve_lock:
            _resolved_urls.pop(url, None)
        return range_response(client, url, start, end)
    return response


class RangeFile(io.RawIOBase):
    """Seekable, bounded HTTP ranges with explicit Parquet-column prefetch.

    Never accepts a server's full-file response to a range request. In particular
    this cannot accidentally download all 705 GB of English audio while reading
    its metadata. Cached footer + one coalesced row-group range suffice for Arrow.
    """

    def __init__(self, url: str, size: int, session, footer: bytes | None = None):
        self.url, self.size, self.session = url, size, session
        self.position = 0
        self.blocks = []
        self.bytes_fetched = 0
        if footer is not None:
            self.blocks.append((size - len(footer), footer))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def fetch(self, start: int, end: int) -> bytes:
        for begin, block in self.blocks:
            if begin <= start and end <= begin + len(block):
                return block[start - begin:end - begin]
        with range_response(self.session, self.url, start, end) as response:
            response.raise_for_status()
            expected = f"bytes {start}-{end - 1}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError(f"Server did not honor byte range: {response.status_code}, {response.headers.get('Content-Range')}")
            data = response.content
        if len(data) != end - start:
            raise IOError("Truncated HTTP range")
        self.bytes_fetched += len(data)
        return data

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        data = self.fetch(self.position, end)
        self.position = end
        return data

    def prefetch(self, start, end):
        # Keep the footer, release the preceding row group's (possibly large) audio.
        footer = self.blocks[:1]
        block = self.fetch(start, end)
        self.blocks = footer + [(start, block)]


def session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    client = requests.Session()
    client.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return client


def dataset(lang: str) -> tuple[str, str]:
    return DATASETS["english" if lang == "eng" else "official"]


def parquet_url(lang: str, path: str) -> str:
    repo, revision = dataset(lang)
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def shard_inventory(lang: str, cache: Path) -> list[dict]:
    from huggingface_hub import HfApi
    path = cache / "shards.json"
    if path.exists():
        return json.loads(path.read_text())
    repo, revision = dataset(lang)
    prefix = "data" if lang == "eng" else LANG_CONFIG[lang]
    shards = [{"path": item.path, "size": item.size} for item in HfApi().list_repo_tree(
        repo, path_in_repo=prefix, recursive=True, repo_type="dataset", revision=revision)
        if item.path.startswith(prefix + "/train-") and item.path.endswith(".parquet")]
    if not shards:
        raise ValueError(f"No train shards found: {repo}/{prefix}")
    shards.sort(key=lambda row: row["path"])
    atomic_json(path, shards)
    return shards


def footer_path(cache: Path, shard: dict) -> Path:
    return cache / (Path(shard["path"]).name + ".footer")


def open_parquet(lang: str, shard: dict, cache: Path, client):
    import pyarrow.parquet as pq
    path = footer_path(cache, shard)
    remote = RangeFile(parquet_url(lang, shard["path"]), shard["size"], client)
    if path.exists():
        tail = path.read_bytes()
    else:
        # Typical MLS footers are 100–150 KB, one request rather than seek chatter.
        tail = remote.fetch(max(0, shard["size"] - 256 * 1024), shard["size"])
        if tail[-4:] != b"PAR1":
            raise ValueError("Missing Parquet trailer")
        needed = int.from_bytes(tail[-8:-4], "little") + 8
        if needed > len(tail):
            tail = remote.fetch(shard["size"] - needed, shard["size"])
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(tail)
        temporary.replace(path)
    remote.blocks = [(shard["size"] - len(tail), tail)]
    return remote, pq.ParquetFile(remote, pre_buffer=False)


def column_bounds(column) -> tuple[int, int]:
    start = min(x for x in [column.dictionary_page_offset, column.data_page_offset]
                if x is not None and x >= 0)
    return start, start + column.total_compressed_size


def prefetch_columns(remote, group, include_audio=False):
    columns = [group.column(index) for index in range(group.num_columns)
               if include_audio or group.column(index).path_in_schema != "audio.bytes"]
    bounds = [column_bounds(column) for column in columns]
    start, end = min(start for start, _ in bounds), max(end for _, end in bounds)
    if not include_audio:
        audio = next(group.column(index) for index in range(group.num_columns)
                     if group.column(index).path_in_schema == "audio.bytes")
        audio_start, audio_end = column_bounds(audio)
        if start < audio_end and end > audio_start:
            raise ValueError("Metadata prefetch would overlap embedded audio")
    remote.prefetch(start, end)


def footer_groups(lang: str, shards: list[dict], cache: Path, workers: int) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    def work(shard):
        with session() as client:
            remote, parquet = open_parquet(lang, shard, cache, client)
            groups = []
            for index in range(parquet.num_row_groups):
                group = parquet.metadata.row_group(index)
                columns = {group.column(i).path_in_schema: group.column(i) for i in range(group.num_columns)}
                speaker = columns["speaker_id"].statistics
                book = columns["book_id" if lang == "eng" else "chapter_id"].statistics
                groups.append({"shard": shard["path"], "group": index, "rows": group.num_rows,
                               "speaker": str(speaker.min) if speaker.min == speaker.max else None,
                               "book": str(book.min) if book.min == book.max else None})
            return groups
    groups = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(work, shards), 1):
            groups.extend(result)
            if index % 50 == 0 or index == len(shards):
                print(f"{lang}: footer census {index}/{len(shards)}", flush=True)
    return groups


def english_groups(groups: list[dict], seed: int, speakers: int = 512, per_speaker: int = 4) -> list[dict]:
    """Bound English transfer via seeded speakers and whole book-diverse groups.

    We sample from homogeneous groups so that a group has a genuine speaker/book
    identity; mixed groups are not silently assigned the minimum statistic.
    The census spans ALL train shards, not a first-shard convenience sample.
    """
    candidates = [{**group, "id": f"{group['shard']}:{group['group']:05}",
                   "speaker_id": group["speaker"], "book_id": group["book"]}
                  for group in groups if group["speaker"] is not None and group["book"] is not None]
    available = sorted({item["speaker_id"] for item in candidates})
    random.Random(seed).shuffle(available)
    keep = set(available[:speakers])
    counts = defaultdict(int)
    selected = []
    for item in balanced_order([item for item in candidates if item["speaker_id"] in keep], seed):
        if counts[item["speaker_id"]] < per_speaker:
            selected.append(item)
            counts[item["speaker_id"]] += 1
    return selected


def candidate_from_row(lang: str, row: dict, shard: str, group: int, index: int) -> dict:
    source_id = str(row.get("id") or Path(row["audio"]["path"]).stem)
    filename(lang, source_id)
    speaker = str(row["speaker_id"])
    book = str(row["book_id"] if lang == "eng" else row["chapter_id"])
    if source_id.split("_")[:2] != [speaker, book]:
        raise ValueError(f"MLS ID and speaker/book disagree: {source_id}, {speaker}, {book}")
    return {"id": source_id, "speaker_id": speaker, "book_id": book,
            "text": row["transcript"], "duration": float(row["audio_duration"]),
            "original_path": row["original_path"], "begin_time": row["begin_time"],
            "end_time": row["end_time"], "shard": shard, "group": group, "row": index}


def metadata_candidates(lang: str, shards: list[dict], groups: list[dict], cache: Path,
                        workers: int) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    wanted = defaultdict(list)
    for group in groups:
        wanted[group["shard"]].append(group["group"])
    def work(shard):
        result = []
        with session() as client:
            remote, parquet = open_parquet(lang, shard, cache, client)
            columns = ["audio.path", "original_path", "begin_time", "end_time", "transcript",
                       "audio_duration", "speaker_id"] + (["book_id"] if lang == "eng" else ["chapter_id", "id"])
            for index in sorted(wanted[shard["path"]]):
                saved = cache / (Path(shard["path"]).stem + f".rg{index}.json")
                if saved.exists():
                    rows = json.loads(saved.read_text())
                else:
                    prefetch_columns(remote, parquet.metadata.row_group(index))
                    rows = [candidate_from_row(lang, row, shard["path"], index, i)
                            for i, row in enumerate(parquet.read_row_group(index, columns=columns).to_pylist())]
                    atomic_json(saved, rows)
                result.extend(rows)
        return result
    candidates = []
    chosen = [shard for shard in shards if shard["path"] in wanted]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(work, chosen), 1):
            candidates.extend(result)
            if index % 10 == 0 or index == len(chosen):
                print(f"{lang}: projected metadata {index}/{len(chosen)} shards, {len(candidates)} clips", flush=True)
    return candidates


def eligible(item: dict, max_clip_seconds: float) -> bool:
    text = item["text"]
    return (0 < item["duration"] <= max_clip_seconds and math.isfinite(item["duration"])
            and bool(text.strip()) and any(char.isalpha() for char in text)
            and not any(ord(char) < 32 for char in text))


def capacity(candidates: list[dict], speaker_seconds: float) -> dict:
    speakers = defaultdict(float)
    for item in candidates:
        speakers[item["speaker_id"]] += item["duration"]
    return {"clips": len(candidates), "hours": sum(speakers.values()) / 3600,
            "speakers": len(speakers), "books": len({item["book_id"] for item in candidates}),
            "speaker_capped_hours": sum(min(value, speaker_seconds) for value in speakers.values()) / 3600}


def manifest_row(lang: str, item: dict, duration: float) -> dict:
    repo, revision = dataset(lang)
    return {"file": filename(lang, item["id"]), "sentence": item["text"],
            "source": "mls", "voice": f"mls:{item['speaker_id']}", "lang": lang,
            "duration_sec": duration, "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Multilingual LibriSpeech (Pratap et al., 2020), Meta/Facebook AI; LibriVox readers",
            "attribution_url": "https://www.openslr.org/94/",
            "mls_id": item["id"], "mls_book_id": item["book_id"],
            # This is a genuinely published chapter recording URL, not an ID
            # invented from the last (utterance sequence) ID component.
            "original_audio_url": item["original_path"],
            "original_begin_time": item["begin_time"], "original_end_time": item["end_time"],
            "hf_dataset": repo, "hf_revision": revision, "hf_split": "train",
            "hf_shard": item["shard"], "hf_row_group": item["group"], "hf_row": item["row"],
            "transcript_form": "published MLS normalized transcript",
            "audio_conversion": "16 kHz mono PCM16; no trimming"}


def download_selected(lang: str, selected: list[dict], shards: list[dict], cache: Path,
                      out_dir: Path, files: set[str], old: dict[str, dict],
                      max_seconds: float, speaker_seconds: float, max_clip_seconds: float,
                      workers: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    by_group = defaultdict(list)
    for item in selected:
        name = filename(lang, item["id"])
        if name in files:
            if name not in old or old[name].get("mls_id") != item["id"]:
                raise ValueError(f"Filename collision: {name}")
            continue
        by_group[(item["shard"], item["group"])].append(item)
    shard_by_path = {shard["path"]: shard for shard in shards}

    def work(task):
        (path, group), items = task
        with session() as client:
            remote, parquet = open_parquet(lang, shard_by_path[path], cache, client)
            prefetch_columns(remote, parquet.metadata.row_group(group), include_audio=True)
            rows = parquet.read_row_group(group, columns=["audio"]).to_pylist()
            result = []
            for item in items:
                source = rows[item["row"]]["audio"]
                if Path(source["path"]).stem != item["id"]:
                    raise ValueError(f"Audio ID changed at {path}:{group}:{item['row']}")
                audio = decode_audio(source["bytes"])
                duration = len(audio) / SAMPLE_RATE
                if abs(duration - item["duration"]) > 0.05:
                    raise ValueError(f"Audio/metadata duration mismatch for {item['id']}: {duration} vs {item['duration']}")
                result.append((item, audio))
            return result, remote.bytes_fetched

    total, used = budgets(old, out_dir)
    used = defaultdict(float, used)
    tasks = iter(sorted(by_group.items()))
    added, skipped, transferred = 0, 0, 0
    # Bound in-flight decoded audio, not just active workers. An unbounded map
    # can otherwise queue hundreds of decoded row groups behind one slow shard.
    with ThreadPoolExecutor(max_workers=workers) as pool, (out_dir / "manifest.jsonl").open("a") as manifest:
        pending = set()
        for _ in range(workers):
            task = next(tasks, None)
            if task is not None:
                pending.add(pool.submit(work, task))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                result, byte_count = future.result()
                transferred += byte_count
                for item, audio in result:
                    duration = len(audio) / SAMPLE_RATE
                    voice = f"mls:{item['speaker_id']}"
                    if (duration > max_clip_seconds or total + duration > max_seconds
                            or used[voice] + duration > speaker_seconds):
                        skipped += 1
                        continue
                    row = manifest_row(lang, item, duration)
                    # Orphan files from a killed run are safely replaced only
                    # after re-decoding the pinned source, never trusted by name.
                    write_wav_atomic(out_dir / row["file"], audio)
                    append_row(manifest, row)
                    files.add(row["file"])
                    used[voice] += duration
                    total += duration
                    added += 1
                os.fsync(manifest.fileno())
                if added and (added // 1000 != (added - len(result)) // 1000):
                    print(f"{lang}: appended {added} clips, {total / 3600:.3f}h, range audio {transferred / 1e9:.3f}GB", flush=True)
                task = next(tasks, None)
                if task is not None:
                    pending.add(pool.submit(work, task))
    return {"appended": added, "skipped_decoded_cap": skipped, "audio_range_bytes": transferred}


def summarize(lang: str, root: Path, prefix: dict, seed: int) -> dict:
    path = root / "manifest.jsonl"
    if fingerprint(path, prefix["bytes"]) != prefix:
        raise ValueError(f"Existing manifest prefix changed: {path}")
    _, rows = existing_rows(path)
    total, used = budgets(rows, root)
    books = {row["mls_book_id"] for row in rows.values()}
    rng = random.Random(seed)
    samples = rng.sample(sorted(rows), min(10, len(rows)))
    spotchecks = []
    for name in samples:
        row = rows[name]
        audio, rate = sf.read(root / name, dtype="float32")
        info = sf.info(root / name)
        assert rate == SAMPLE_RATE and audio.ndim == 1 and np.isfinite(audio).all()
        assert row["sentence"].strip() and np.any(audio)
        spotchecks.append({"file": name, "sentence": row["sentence"], "duration_sec": len(audio) / rate,
                           "sample_rate": rate, "channels": info.channels, "subtype": info.subtype,
                           "peak": float(np.max(np.abs(audio))),
                           "rms": float(np.sqrt(np.mean(audio ** 2))),
                           "chars_per_second": len(row["sentence"]) / (len(audio) / rate),
                           "words_per_second": len(row["sentence"].split()) / (len(audio) / rate),
                           "book": row["mls_book_id"], "voice": row["voice"],
                           "structural_check": "passed; not listened to"})
    return {"lang": lang, "hours": total / 3600, "clips": len(rows), "speakers": len(used),
            "books": len(books), "max_speaker_hours": max(used.values(), default=0) / 3600,
            "top1_speaker_share": max(used.values(), default=0) / total if total else 0,
            "top5_speaker_share": sum(sorted(used.values(), reverse=True)[:5]) / total if total else 0,
            "wav_bytes": sum((root / name).stat().st_size for name in rows),
            "original_prefix": prefix, "prefix_unchanged": True,
            "manifest": fingerprint(path), "spotchecks": spotchecks}


def acquire(lang: str, args) -> dict:
    repo, revision = dataset(lang)
    cache = args.cache_dir / lang / revision
    cache.mkdir(parents=True, exist_ok=True)
    out_dir = args.output_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"dataset": repo, "revision": revision, "split": "train", "seed": args.seed,
              "max_hours": args.max_hours, "speaker_hours": args.speaker_hours,
              "max_clip_seconds": args.max_clip_seconds, "english_speakers": 512,
              "english_groups_per_speaker": 4, "output_root": str(args.output_root.resolve())}
    state_path = cache / (hashlib.sha256(str(out_dir.resolve()).encode()).hexdigest()[:16] + ".run.json")
    with manifest_lock(out_dir / "manifest.jsonl"):
        files, old = existing_rows(out_dir / "manifest.jsonl")
        total, used = budgets(old, out_dir)
        if total > args.max_hours * 3600 or any(value > args.speaker_hours * 3600 for value in used.values()):
            raise ValueError("Existing MLS already exceeds requested limits; refusing to remove/modify rows")
        plan_path = state_path.with_suffix(".plan.json")
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["config"] != config:
                changed = {key for key in config if state["config"].get(key) != config[key]}
                if changed != {"speaker_hours"} or config["speaker_hours"] <= state["config"]["speaker_hours"]:
                    raise ValueError("Acquisition configuration changed; only a monotonic speaker-cap increase is supported")
                # Invalidate the old plan BEFORE publishing the raised cap. A
                # crash on either side remains safe to retry. Keep the original
                # pre-acquisition prefix, provenance and all durable rows.
                plan_path.unlink(missing_ok=True)
                state.setdefault("previous_speaker_caps", []).append(state["config"]["speaker_hours"])
                state["config"] = config
                state.pop("complete", None)
                atomic_json(state_path, state)
        else:
            state = {"config": config, "prefix": fingerprint(out_dir / "manifest.jsonl")}
            atomic_json(state_path, state)
        if fingerprint(out_dir / "manifest.jsonl", state["prefix"]["bytes"]) != state["prefix"]:
            raise ValueError("Manifest prefix changed since acquisition began")
        if state.get("complete"):
            result = summarize(lang, out_dir, state["prefix"], args.seed)
            result.update({"appended": 0, "already_complete": True, "capacity": state["capacity"],
                           "speaker_cap_hours": args.speaker_hours,
                           "shortfall_hours": args.max_hours - result["hours"]})
            return result
        shards = shard_inventory(lang, cache)
        if plan_path.exists():
            selected = json.loads(plan_path.read_text())
        else:
            groups_path = cache / "groups.json"
            if groups_path.exists():
                groups = json.loads(groups_path.read_text())
            else:
                groups = footer_groups(lang, shards, cache, args.workers)
                atomic_json(groups_path, groups)
            chosen = english_groups(groups, args.seed) if lang == "eng" else groups
            candidates = metadata_candidates(lang, shards, chosen, cache, args.workers)
            raw_capacity = capacity(candidates, args.speaker_hours * 3600)
            candidates = [item for item in candidates if eligible(item, args.max_clip_seconds)]
            eligible_capacity = capacity(candidates, args.speaker_hours * 3600)
            # Fixed English sampling pool is deliberately wider than the desired
            # 300h. Do not call a thin sampled pool a corpus-capacity shortfall.
            if lang == "eng" and eligible_capacity["speaker_capped_hours"] < args.max_hours:
                chosen = english_groups(groups, args.seed, speakers=1024, per_speaker=8)
                candidates = metadata_candidates(lang, shards, chosen, cache, args.workers)
                raw_capacity = capacity(candidates, args.speaker_hours * 3600)
                candidates = [item for item in candidates if eligible(item, args.max_clip_seconds)]
                eligible_capacity = capacity(candidates, args.speaker_hours * 3600)
                if eligible_capacity["speaker_capped_hours"] < args.max_hours:
                    raise ValueError("English expanded sample pool still below target; refusing misleading corpus shortfall")
            state["capacity"] = {"before_clip_gate": raw_capacity, "eligible": eligible_capacity,
                                 "english_sampled_pool": lang == "eng",
                                 "eligible_row_groups": len(chosen), "all_train_row_groups": len(groups)}
            print(f"{lang}: CAPACITY {json.dumps(state['capacity'])}", flush=True)
            selected = select(candidates, args.seed, args.max_hours * 3600, args.speaker_hours * 3600,
                              total, used, {row["mls_id"] for row in old.values()})
            # Save state first: an interruption cannot leave a plan with missing
            # capacity/provenance state. Selection itself is deterministic.
            atomic_json(state_path, state)
            atomic_json(plan_path, selected)
        print(f"{lang}: plan {len(selected)} clips, {sum(item['duration'] for item in selected) / 3600:.3f}h", flush=True)
        if args.plan_only:
            return {"lang": lang, "plan_only": True, "capacity": state["capacity"], "selected_clips": len(selected)}
        progress = download_selected(lang, selected, shards, cache, out_dir, files, old,
                                     args.max_hours * 3600, args.speaker_hours * 3600,
                                     args.max_clip_seconds, args.workers)
        result = summarize(lang, out_dir, state["prefix"], args.seed)
        result.update(progress)
        result["capacity"] = state["capacity"]
        result["speaker_cap_hours"] = args.speaker_hours
        result["shortfall_hours"] = args.max_hours - result["hours"]
        state["complete"] = True
        atomic_json(state_path, state)
        return result


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--langs", nargs="+", choices=LANG_CONFIG, default=list(LANG_CONFIG))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "audio")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/lexide-pronunciation/mls")
    parser.add_argument("--max-hours", type=float, default=300)
    parser.add_argument("--speaker-hours", type=float, default=2,
                        help="Per-speaker hours (default: 2; explicit additive topups may raise this to 6)")
    parser.add_argument("--max-clip-seconds", type=float, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true", help="Fetch metadata and freeze selection, but no audio")
    parser.add_argument("--stats", type=Path, help="Write machine-readable per-language statistics and 10 structural spotchecks")
    args = parser.parse_args()
    if not (0 < args.max_hours <= 300 and 0 < args.speaker_hours <= 6
            and 0 < args.max_clip_seconds <= 16 and args.workers > 0):
        parser.error("Require 0<hours<=300, 0<speaker-hours<=6, 0<max-clip-seconds<=16, workers>0")
    results = []
    for lang in args.langs:
        result = acquire(lang, args)
        results.append(result)
        if args.stats:
            args.stats.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.stats, results)
        print(json.dumps({key: value for key, value in result.items() if key != "spotchecks"}), flush=True)


if __name__ == "__main__":
    main()

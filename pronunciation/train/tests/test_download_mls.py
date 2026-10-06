"""MLS acquisition never spends speaker budgets or mutates old corpus bytes."""

import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
import download_mls as mls


def candidate(speaker, book, clip, duration=10):
    return {"id": f"{speaker}_{book}_{clip:06}", "speaker_id": str(speaker),
            "book_id": str(book), "duration": duration, "text": "published text",
            "original_path": "http://archive.org/chapter.mp3", "begin_time": 0,
            "end_time": duration, "shard": "german/train-00000-of-00001.parquet",
            "group": 0, "row": clip}


def test_seeded_balance_not_first_rows():
    rows = [candidate(speaker, book, clip) for speaker in range(4)
            for book in range(3) for clip in range(50)]
    chosen = mls.select(rows, 42, 120, 40)
    assert len(chosen) == 12
    assert {row["speaker_id"] for row in chosen} == {"0", "1", "2", "3"}
    for speaker in range(4):
        assert {row["book_id"] for row in chosen if row["speaker_id"] == str(speaker)} == {"0", "1", "2"}
    assert chosen == mls.select(list(reversed(rows)), 42, 120, 40)
    assert chosen != mls.select(rows, 43, 120, 40)


def test_resume_accounts_both_budgets_and_existing_ids():
    rows = [candidate(speaker, 1, clip) for speaker in range(2) for clip in range(10)]
    old = {rows[0]["id"]}
    selected = mls.select(rows, 42, 40, 20, 25, {"mls:0": 15, "mls:1": 10}, old)
    assert len(selected) == 1 and selected[0]["speaker_id"] == "1"
    assert selected[0]["id"] not in old
    assert mls.select(rows, 42, 40, 20, 40, {"mls:0": 20, "mls:1": 20}) == []


def test_scarce_speakers_underfill_without_relaxing():
    rows = [candidate(1, 2, clip, 11) for clip in range(20)]
    selected = mls.select(rows, 42, 300, 20)
    assert len(selected) == 1
    assert sum(row["duration"] for row in selected) == 11


def test_duplicate_source_ids_fail():
    row = candidate(1, 2, 3)
    with pytest.raises(ValueError, match="Duplicate source ID"):
        mls.select([row, row], 42, 100, 100)


@pytest.mark.parametrize("identifier", ["../escape", "1_2", "1_2_3/4", "chapter_1_2"])
def test_filename_rejects_unsafe_ids(identifier):
    with pytest.raises(ValueError):
        mls.filename("eng", identifier)


def test_namespaced_identity_and_honest_provenance():
    item = candidate(1, 2, 3)
    row = mls.manifest_row("eng", item, 10)
    assert row["file"] == "mls_eng_1_2_000003.wav"
    assert row["voice"] == "mls:1"
    assert row["mls_book_id"] == "2"
    assert "chapter_id" not in row and "variety" not in row
    assert row["original_audio_url"] == item["original_path"]
    assert row["sentence"] == item["text"]
    assert row["hf_split"] == "train" and row["license"] == "CC BY 4.0"


def test_candidate_reads_actual_source_fields():
    row = {"audio": {"path": "1_2_000003.opus"}, "speaker_id": "1", "chapter_id": "2",
           "id": "1_2_000003", "transcript": "lower case no restored punctuation",
           "audio_duration": 10, "original_path": "chapter.mp3", "begin_time": 1, "end_time": 11}
    item = mls.candidate_from_row("deu", row, "train.parquet", 2, 3)
    assert item["book_id"] == "2" and item["text"] == row["transcript"]
    row["book_id"] = row.pop("chapter_id")
    row.pop("id")
    assert mls.candidate_from_row("eng", row, "train.parquet", 2, 3) == item
    row["speaker_id"] = "4"
    with pytest.raises(ValueError, match="disagree"):
        mls.candidate_from_row("eng", row, "train.parquet", 2, 3)


@pytest.mark.parametrize("duration,text,accepted", [(16, "text", True), (16.01, "text", False),
    (0, "text", False), (float("nan"), "text", False), (10, "  ", False), (10, "a\nb", False)])
def test_eligibility_precedes_audio_fetch(duration, text, accepted):
    assert mls.eligible({"duration": duration, "text": text}, 16) is accepted


def test_stereo_resample_and_atomic_wav(tmp_path):
    encoded = io.BytesIO()
    sf.write(encoded, np.full((4800, 2), .25), 48000, format="WAV")
    audio = mls.decode_audio(encoded.getvalue())
    assert audio.shape == (1600,) and np.isfinite(audio).all()
    path = tmp_path / "mls_eng_1_2_000003.wav"
    mls.write_wav_atomic(path, audio)
    info = sf.info(path)
    assert (info.samplerate, info.channels, info.frames, info.subtype) == (16000, 1, 1600, "PCM_16")
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_failed_atomic_write_leaves_final_file_untouched(tmp_path, monkeypatch):
    path = tmp_path / "mls_eng_1_2_000003.wav"
    path.write_bytes(b"previous")
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(mls.sf, "write", fail)
    with pytest.raises(OSError, match="disk full"):
        mls.write_wav_atomic(path, np.ones(10))
    assert path.read_bytes() == b"previous"
    assert len(list(tmp_path.iterdir())) == 1


def test_append_preserves_prefix_and_budget_uses_actual_wav(tmp_path):
    path = tmp_path / "manifest.jsonl"
    prefix = b'{"file":"unrelated.wav", "sentence":"keep whitespace", "source":"tts", "custom":123}\n'
    path.write_bytes(prefix)
    row = mls.manifest_row("eng", candidate(1, 2, 3), 999)
    mls.write_wav_atomic(tmp_path / row["file"], np.full(1600, .25))
    with path.open("a") as stream:
        mls.append_row(stream, row)
    assert path.read_bytes().startswith(prefix)
    files, rows = mls.existing_rows(path)
    assert len(files) == 2
    assert mls.budgets(rows, tmp_path) == (.1, {"mls:1": .1})


def test_truncated_existing_manifest_refused(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_bytes(b'{"file":"old.wav"}')
    with pytest.raises(ValueError, match="Unterminated"):
        mls.existing_rows(path)
    assert path.read_bytes() == b'{"file":"old.wav"}'


def test_range_reader_rejects_entire_shard_response():
    class Response:
        status_code = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        @property
        def content(self): raise AssertionError("Must not read full shard body")
    client = SimpleNamespace(get=lambda *a, **kw: Response())
    remote = mls.RangeFile("https://example.invalid/shard", 1_000_000, client)
    with pytest.raises(RuntimeError, match="did not honor byte range"):
        remote.fetch(100, 200)


def test_english_pool_is_seeded_and_book_diverse():
    groups = [{"shard": f"data/train-{s}.parquet", "group": b * 4 + i,
               "speaker": str(s), "book": str(b), "rows": 100}
              for s in range(20) for b in range(3) for i in range(4)]
    groups.append({"shard": "mixed", "group": 0, "speaker": None, "book": None, "rows": 100})
    chosen = mls.english_groups(groups, 42, speakers=5, per_speaker=3)
    assert len(chosen) == 15
    assert len({g["speaker"] for g in chosen}) == 5
    assert len({g["book"] for g in chosen}) == 3
    assert chosen == mls.english_groups(groups, 42, speakers=5, per_speaker=3)
    assert chosen != mls.english_groups(groups, 43, speakers=5, per_speaker=3)


def test_end_to_end_append_resume_and_changed_config(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    lang = "deu"
    root = tmp_path / "audio"
    out = root / lang
    out.mkdir(parents=True)
    prefix = b'{"file":"unrelated.wav", "source":"tts", "keep":true}\n'
    (out / "manifest.jsonl").write_bytes(prefix)
    audio = io.BytesIO()
    sf.write(audio, np.full(16000, .25), 16000, format="WAV")
    rows = []
    for index in range(3):
        rows.append({"audio": {"bytes": audio.getvalue(), "path": f"1_2_{index:06}.opus"},
                     "original_path": "http://archive.org/chapter.mp3", "begin_time": float(index),
                     "end_time": float(index + 1), "transcript": f"published text {index}",
                     "audio_duration": 1., "speaker_id": "1", "chapter_id": "2",
                     "id": f"1_2_{index:06}"})
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=2)
    shard = {"path": "german/train-00000-of-00001.parquet", "size": source.stat().st_size}
    monkeypatch.setattr(mls, "shard_inventory", lambda *args: [shard])
    class LocalRemote:
        bytes_fetched = 0
        def prefetch(self, *args): pass
    monkeypatch.setattr(mls, "open_parquet", lambda *args: (LocalRemote(), pq.ParquetFile(source)))
    args = SimpleNamespace(cache_dir=tmp_path / "cache", output_root=root, max_hours=1,
                           speaker_hours=2 / 3600, max_clip_seconds=16, seed=42, workers=2, plan_only=False)
    first = mls.acquire(lang, args)
    assert first["appended"] == 2 and first["clips"] == 2
    after = (out / "manifest.jsonl").read_bytes()
    assert after.startswith(prefix)
    second = mls.acquire(lang, args)
    assert second["appended"] == 0 and second["clips"] == 2
    assert (out / "manifest.jsonl").read_bytes() == after
    args.speaker_hours = 3 / 3600
    topup = mls.acquire(lang, args)
    assert topup["appended"] == 1 and topup["clips"] == 3
    assert topup["original_prefix"]["bytes"] == len(prefix)
    assert topup["top5_speaker_share"] == 1
    topped_up = (out / "manifest.jsonl").read_bytes()
    assert topped_up.startswith(after)
    assert mls.acquire(lang, args)["appended"] == 0
    assert (out / "manifest.jsonl").read_bytes() == topped_up
    args.seed = 43
    with pytest.raises(ValueError, match="configuration changed"):
        mls.acquire(lang, args)


def test_metadata_prefetch_refuses_audio_between_metadata_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = tmp_path / "bad-layout.parquet"
    pq.write_table(pa.Table.from_pylist([{"audio": {"path": "clip.opus", "bytes": b"audio"},
                                        "transcript": "text"}]), source)
    group = pq.ParquetFile(source).metadata.row_group(0)
    remote = SimpleNamespace(prefetch=lambda *args: pytest.fail("Must reject before fetching audio"))
    with pytest.raises(ValueError, match="overlap embedded audio"):
        mls.prefetch_columns(remote, group)


def test_resolver_cache_avoids_repeated_hf_calls_and_cross_origin_auth(monkeypatch):
    url = "https://huggingface.co/datasets/test/resolve/revision/train.parquet"
    cdn = "https://cdn.example.invalid/signed"
    calls = []
    monkeypatch.setenv("HF_TOKEN", "test-private-token")
    monkeypatch.setattr(mls, "_resolved_urls", {})
    monkeypatch.setattr(mls.time, "sleep", lambda seconds: None)
    def get(target, **kwargs):
        calls.append((target, kwargs["headers"]))
        return SimpleNamespace(status_code=206, url=cdn)
    client = SimpleNamespace(get=get)
    mls.range_response(client, url, 0, 100)
    mls.range_response(client, url, 100, 200)
    assert calls[0][0] == url and calls[0][1]["Authorization"] == "Bearer test-private-token"
    assert calls[1][0] == cdn and "Authorization" not in calls[1][1]
    assert calls[1][1]["Range"] == "bytes=100-199"

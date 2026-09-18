"""Header-only eligibility gates and provenance-aware stress supervision."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.dataset import StressDataset


class Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"a": 1, "b": 2}.get(token, -1)


def make_dataset(tmp_path, monkeypatch, rows, frames=16000, sr=16000, **kwargs):
    records = [dict(file=f"{i}.wav", lang="fra", sentence="bonjour",
                    phonemes=["a", "b"], stress=[1, 0]) | row
               for i, row in enumerate(rows)]
    path = tmp_path / "phonemes.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    for row in records:
        (tmp_path / row["file"]).touch()
    monkeypatch.setattr("src.dataset.sf.info", lambda _: SimpleNamespace(
        frames=frames, samplerate=sr))
    reads = []

    def read(path, **kwargs):
        reads.append(path)
        return np.full(frames, 0.1, dtype=np.float32), sr

    monkeypatch.setattr("src.dataset.sf.read", read)
    return StressDataset(path, Tokenizer(), **kwargs), reads


@pytest.mark.parametrize("frames,sr", [(16001, 16000), (32001, 32000)])
def test_overlong_rejected_without_decoding(tmp_path, monkeypatch, capsys, frames, sr):
    ds, reads = make_dataset(tmp_path, monkeypatch, [{}], frames, sr, max_audio_sec=1)
    assert not ds.samples and not reads
    assert "'too_long': 1" in capsys.readouterr().out


def test_exact_duration_keeps_full_audio_and_labels(tmp_path, monkeypatch):
    ds, reads = make_dataset(tmp_path, monkeypatch, [{}], max_audio_sec=1)
    assert ds.samples[0]["n_audio_samples"] == 16000
    sample = ds[0]
    assert len(sample["audio"]) == 16000
    assert sample["phoneme_ids"].tolist() == [1, 2]
    assert sample["stress_seq"].tolist() == [1, 0]
    assert len(reads) == 2


@pytest.mark.parametrize("frames,sr", [(719, 16000), (1439, 32000), (399, 16000)])
def test_ctc_infeasible_rejected_without_decoding(tmp_path, monkeypatch, capsys, frames, sr):
    ds, reads = make_dataset(tmp_path, monkeypatch, [{}], frames, sr, min_duration_sec=0)
    assert not ds.samples and not reads
    assert "'ctc_infeasible': 1" in capsys.readouterr().out


def test_ctc_exact_encoder_boundary(tmp_path, monkeypatch):
    ds, reads = make_dataset(tmp_path, monkeypatch, [{}], frames=720, min_duration_sec=0)
    assert len(ds) == 1 and len(reads) == 1


def test_french_provenance_masks_only_explicit_non_override(tmp_path, monkeypatch, capsys):
    ds, _ = make_dataset(tmp_path, monkeypatch, [
        {"stress_source": "override"}, {"stress_source": "espeak"},
        {"stress_source": None}, {},
    ])
    assert [s["stress_available"] for s in ds.samples] == [True, False, False, True]
    assert all(s["stress_seq"] == [1, 0] for s in ds.samples)
    assert "1 French rows missing stress_source provenance" in capsys.readouterr().out


def test_boilerplate_sidecar_schema_and_sentence_hash():
    path = Path(__file__).resolve().parents[1] / "boilerplate_exclusions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {hashlib.sha256(text.encode()).hexdigest() for text in [
        "Sous-titrage Société Radio-Canada",
        "Редактор субтитров А.Семкин Корректор А.Егорова",
        "Субтитры создавал DimaTorzok",
        "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
    ]}
    assert rows
    assert len({(r["lang"], r["file"]) for r in rows}) == len(rows)
    for row in rows:
        assert set(row) == {"lang", "file", "expected_sha256", "ok", "reason"}
        assert row["ok"] is False and row["reason"]
        assert row["expected_sha256"] in expected


def test_non_french_provenance_does_not_mask_stress(tmp_path, monkeypatch):
    ds, _ = make_dataset(tmp_path, monkeypatch, [
        {"lang": "eng", "stress_source": "espeak"},
    ])
    assert ds[0]["stress_available"] is True


def test_legacy_unstressed_language_stays_masked(tmp_path, monkeypatch):
    ds, _ = make_dataset(tmp_path, monkeypatch, [{"stress": [0, 0]}])
    assert ds[0]["stress_available"] is False


def test_stale_boilerplate_hash_does_not_exclude_repaired_sentence(tmp_path, monkeypatch, capsys):
    ds, _ = make_dataset(tmp_path, monkeypatch, [{}], excluded_target_hashes={
        "0.wav": hashlib.sha256(b"old boilerplate").hexdigest(),
    })
    assert len(ds) == 1
    assert "'stale_asr_audit': 1" in capsys.readouterr().out


def test_matching_boilerplate_hash_excludes_before_audio(tmp_path, monkeypatch, capsys):
    ds, reads = make_dataset(tmp_path, monkeypatch, [{}], excluded_target_hashes={
        "0.wav": hashlib.sha256(b"bonjour").hexdigest(),
    })
    assert not ds.samples and not reads
    assert "'asr_audit': 1" in capsys.readouterr().out


def test_loader_uses_cer_and_wer_only_without_per(tmp_path):
    from src.train_unified import load_asr_audit_exclusions

    rows = [dict(ok=True, lang="hin", file=f"{i}.wav", expected="text", **metrics)
            for i, metrics in enumerate([
                {"per": 0, "cer": 1, "wer": 1},
                {"per": 1, "cer": 0, "wer": 0},
                {"cer": 0.5, "wer": 0.5}, {"cer": 0, "wer": 1},
                {"cer": 1, "wer": 0},
            ])]
    path = tmp_path / "audit.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = load_asr_audit_exclusions(path, min_per=1e-12, min_cer=1e-12, min_wer=1e-12)
    assert set(result["hin"]) == {"1.wav", "2.wav"}

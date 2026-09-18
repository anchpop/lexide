"""Preparation applies the trainer's transcript-keyed exclusions before g2p."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import preprocess_support as preprocess


@pytest.mark.parametrize("sidecar", [
    "fleurs_asr_exclusions", "tatoeba_asr_exclusions", "tts_asr_exclusions",
    "lang_exclusions", "mixed_script_exclusions", "boilerplate_exclusions",
])
@pytest.mark.parametrize("hash_field", ["expected_sha256", "expected"])
def test_prepare_exclusions(tmp_path, capsys, sidecar, hash_field):
    data_dir = tmp_path / "audio"
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    rows = [
        {"file": "matching.wav", "sentence": "cinco"},
        {"file": "stale.wav", "sentence": "seis"},
        {"file": "failed.wav", "sentence": "siete"},
    ]
    audits = []
    for row in rows:
        expected = "old transcript" if row["file"] == "stale.wav" else row["sentence"]
        audits.append({
            "lang": "spa", "file": row["file"], "per": 1,
            "ok": row["file"] != "failed.wav",
            hash_field: (hashlib.sha256(expected.encode()).hexdigest()
                         if hash_field == "expected_sha256" else expected),
        })
    (train_dir / f"{sidecar}.jsonl").write_text(
        "\n".join(json.dumps(row) for row in audits) + "\n"
    )
    for lang in ("spa", "ita"):
        lang_dir = data_dir / lang
        lang_dir.mkdir(parents=True)
        original = "\n".join(json.dumps(row) for row in rows) + "\n"
        (lang_dir / "manifest.jsonl").write_text(original)
        for row in rows:
            sf.write(lang_dir / row["file"], np.full(1600, 0.1, dtype=np.float32), 16000)
        output = tmp_path / f"{lang}.jsonl"
        preprocess.prepare(data_dir, lang, output, False, train_dir)
        kept = [json.loads(line) for line in output.read_text().splitlines()]
        assert kept == (rows[1:] if lang == "spa" else rows)
        assert (lang_dir / "manifest.jsonl").read_text() == original
    summary = capsys.readouterr().out
    assert "1 by exclusions" in summary
    assert "0 by exclusions" in summary


@pytest.mark.parametrize("metrics, excluded", [
    ({"per": 0, "cer": 1, "wer": 1}, False),
    ({"per": 1e-12}, True),
    ({"per": 1e-13}, False),
    ({"cer": 1, "wer": 1}, True),
    ({"cer": 0, "wer": 1}, False),
    ({"cer": 1, "wer": 0}, False),
    ({"cer": 1}, False),
    ({}, False),
])
def test_exclusion_metric_thresholds(tmp_path, metrics, excluded):
    row = {"lang": "spa", "file": "a.wav", "expected": "cinco", **metrics}
    (tmp_path / "fleurs_asr_exclusions.jsonl").write_text(json.dumps(row) + "\n")
    exclusions = preprocess.load_training_exclusions(tmp_path)
    assert bool(exclusions) is excluded

"""Offline regressions for label provenance, narrowing and warm-node staging."""

import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

TRAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN / "scripts"))
import preprocess_support as preprocess


@pytest.mark.parametrize("token", sorted(preprocess.MERGED_TOKEN_BASES) + ["tʃː", "tʃʲ", "dzː"])
def test_merged_narrowing_refuses_before_import_or_writes(tmp_path, monkeypatch, token):
    lang_dir = tmp_path / "eng"
    lang_dir.mkdir()
    (lang_dir / "phonemes.jsonl").write_text(json.dumps({"phonemes": [token]}) + "\n")
    narrowed = lang_dir / "phonemes_narrowed.jsonl"
    narrowed.write_text("carried acoustic decisions\n")
    old_path = list(sys.path)
    original_import = builtins.__import__

    def no_cloud(name, *args, **kwargs):
        assert name not in {"narrow", "modal", "modal_aligner"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cloud)
    with pytest.raises(ValueError, match="--skip-narrowing.*pronunciation/CLAUDE.md"):
        preprocess.run_narrowing("eng", tmp_path)
    assert narrowed.read_text() == "carried acoustic decisions\n"
    assert sys.path == old_path


def test_split_labels_still_run_narrowing(tmp_path, monkeypatch):
    (tmp_path / "eng").mkdir()
    (tmp_path / "eng" / "phonemes.jsonl").write_text(
        json.dumps({"phonemes": ["a", "ɪ", "t", "ʃ"]}) + "\n"
    )
    calls = []
    narrow = SimpleNamespace(AUDIO=None, run=lambda langs: calls.append(langs))
    monkeypatch.setitem(sys.modules, "narrow", narrow)
    preprocess.run_narrowing("eng", tmp_path)
    assert calls == [["eng"]]
    assert narrow.AUDIO == tmp_path


@pytest.mark.parametrize("model, revision", [
    ("anchpop/lexide-pronunciation-unified-vad-clean", "new-merged-model-commit"),
    ("anchpop/lexide-pronunciation-merged", "2926e06f8092935f597e0018beb5d579b95b889a"),
])
def test_new_aligner_pin_not_blocked(tmp_path, model, revision):
    aligner = tmp_path / "aligner.py"
    aligner.write_text(
        f"raise AssertionError('must not execute')\nMODEL_ID = {model!r}\n"
        f"MODEL_REVISION = {revision!r}\n"
    )
    preprocess.guard_narrowing_labels("eng", tmp_path, aligner)


@pytest.mark.parametrize("sentence, targets, expected_source, expected_stress", [
    ("bonjour", None, "g2p", [2]),
    ("bonjour", ["bonjour"], "override", [1]),
    ("bonjour ami", ["ami"], "g2p", [2]),
    ("bonjour", ["absent"], "g2p", [2]),
    ("bonjour", ["bonjour", "absent"], "g2p", [2]),
    ("bonjour", ["bonjour", "bonjour"], "g2p", [2]),
    ("bonjour", [""], "g2p", [2]),
    ("bonjour", [], "override", [0]),
])
def test_written_stress_provenance(tmp_path, monkeypatch, sentence, targets,
                                   expected_source, expected_stress):
    lang_dir = tmp_path / "fra"
    lang_dir.mkdir()
    (lang_dir / "manifest.jsonl").write_text(json.dumps({
        "file": "a.wav", "sentence": sentence, "source": "tts",
    }) + "\n")
    if targets is not None:
        (lang_dir / "stress_overrides.jsonl").write_text(json.dumps({
            "file": "a.wav", "stressed_words": targets,
        }) + "\n")
    sf.write(lang_dir / "a.wav", np.full(1600, 0.1, dtype=np.float32), 16000)
    exchange = tmp_path / "labels.jsonl"
    exchange.write_text(json.dumps({
        "record": json.loads((lang_dir / "manifest.jsonl").read_text()),
        "language": "fra", "labels": {
            "phonemes": ["u"], "stress": [2], "word_spans": [[0, 1]],
        },
    }) + "\n")
    preprocess.finalize(tmp_path, "fra", exchange, "test-build")
    row = json.loads((lang_dir / "phonemes.jsonl").read_text())
    assert row["stress_source"] == expected_source
    assert row["stress"] == expected_stress


def write_tar(home, files):
    with tarfile.open(home / "data.tar", "w") as archive:
        for name, content in files.items():
            payload = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return hashlib.sha256((home / "data.tar").read_bytes()).hexdigest()


@pytest.fixture(params=["sky_train.yaml", "sky_train_merged.yaml", "sky_smoke.yaml"])
def stage(request, tmp_path):
    yaml = (TRAIN / request.param).read_text()
    assert "preprocess/Cargo.toml -- --skip-narrowing" in yaml
    # Execute the actual YAML staging shell, stopping before any training.
    shell = textwrap.dedent(yaml.split("run: |\n", 1)[1].split('  echo "data langs:', 1)[0])

    def run():
        return subprocess.run(["bash", "-c", shell], env={**os.environ, "HOME": str(tmp_path)},
                              text=True, capture_output=True)

    return run


def test_staging_cold_matching_and_changed_tar(tmp_path, stage):
    first = write_tar(tmp_path, {"eng/phonemes.jsonl": "old", "removed": "stale"})
    result = stage()
    assert result.returncode == 0, result.stderr
    data = tmp_path / "data"
    assert (data / ".tar_id").read_text().strip() == first
    (data / "local-proof-of-reuse").touch()
    assert stage().returncode == 0
    assert (data / "local-proof-of-reuse").exists()
    second = write_tar(tmp_path, {"eng/phonemes.jsonl": "new", "fra/phonemes.jsonl": "new"})
    assert stage().returncode == 0
    assert (data / "eng/phonemes.jsonl").read_text() == "new"
    assert (data / ".tar_id").read_text().strip() == second
    assert not (data / "removed").exists()
    assert not (data / "local-proof-of-reuse").exists()
    assert (tmp_path / "data.tar").exists()
    assert not list(tmp_path.glob("data.stage.*"))
    assert not list(tmp_path.glob("data.previous.*"))


def test_unmarked_warm_data_is_replaced(tmp_path, stage):
    data = tmp_path / "data" / "eng"
    data.mkdir(parents=True)
    (data / "phonemes.jsonl").write_text("old")
    write_tar(tmp_path, {"eng/phonemes.jsonl": "new"})
    assert stage().returncode == 0
    assert (data / "phonemes.jsonl").read_text() == "new"


def test_failed_extraction_keeps_old_data(tmp_path, stage):
    first = write_tar(tmp_path, {"eng/phonemes.jsonl": "old"})
    assert stage().returncode == 0
    write_tar(tmp_path, {"eng/phonemes.jsonl": "replacement" * 1000})
    archive = tmp_path / "data.tar"
    archive.write_bytes(archive.read_bytes()[:1024])  # valid header, truncated payload
    result = stage()
    assert result.returncode != 0
    assert (tmp_path / "data/eng/phonemes.jsonl").read_text() == "old"
    assert (tmp_path / "data/.tar_id").read_text().strip() == first
    assert archive.exists()
    assert not list(tmp_path.glob("data.stage.*"))


def test_absent_tar_refuses_even_with_marker(tmp_path, stage):
    write_tar(tmp_path, {"eng/phonemes.jsonl": "old"})
    assert stage().returncode == 0
    (tmp_path / "data.tar").unlink()
    result = stage()
    assert result.returncode != 0
    assert "restage the dataset" in result.stderr
    assert (tmp_path / "data/eng/phonemes.jsonl").read_text() == "old"


@pytest.mark.parametrize("token", ["tːs", "t͡ʃʲ", "t͜s"])
def test_short_old_pin_and_decorated_merges_refuse(tmp_path, token):
    aligner = tmp_path / "aligner.py"
    aligner.write_text(
        "raise AssertionError('must not execute')\n"
        "MODEL_ID = 'anchpop/lexide-pronunciation-unified-vad-clean'\n"
        "MODEL_REVISION = '2926e06'\n"
    )
    (tmp_path / "eng").mkdir()
    (tmp_path / "eng/phonemes.jsonl").write_text(json.dumps({"phonemes": [token]}))
    with pytest.raises(ValueError, match="--skip-narrowing"):
        preprocess.guard_narrowing_labels("eng", tmp_path, aligner)


def test_hindi_annotations_survive_preprocess(tmp_path, monkeypatch):
    cases = [json.loads(line) for line in
             (TRAIN / "tests/fixtures/hindi_flat/cases.jsonl").read_text().splitlines()]
    case = next(c for c in cases if c["record"]["file"] == "synthetic-multiword.wav")
    rec, response = case["record"], case["response"]
    lang_dir = tmp_path / "hin"
    lang_dir.mkdir()
    (lang_dir / "manifest.jsonl").write_text(json.dumps(rec) + "\n")
    sf.write(lang_dir / rec["file"], np.full(1600, 0.1, dtype=np.float32), 16000)
    exchange = tmp_path / "labels.jsonl"
    exchange.write_text(json.dumps({"record": rec, "language": "hin", "labels": response}) + "\n")
    preprocess.finalize(tmp_path, "hin", exchange, "test-build")
    written = json.loads((lang_dir / "phonemes.jsonl").read_text())
    assert written["phonemes"] == response["phonemes"]
    assert written["stress"] == response["stress"]
    assert len(written["syllables"]) == len(response["syllables"])
    assert written["stress_source"] == "g2p"


def test_failed_promotion_restores_old_dataset(tmp_path, stage, monkeypatch):
    write_tar(tmp_path, {"eng/phonemes.jsonl": "old"})
    assert stage().returncode == 0
    write_tar(tmp_path, {"eng/phonemes.jsonl": "new"})
    import shutil
    real_mv = shutil.which("mv")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mv = bin_dir / "mv"
    mv.write_text(f'#!/bin/bash\ncase "$1" in *.stage.*) exit 1;; esac\nexec {real_mv} "$@"\n')
    mv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    assert stage().returncode != 0
    assert (tmp_path / "data/eng/phonemes.jsonl").read_text() == "old"
    assert not list(tmp_path.glob("data.stage.*"))

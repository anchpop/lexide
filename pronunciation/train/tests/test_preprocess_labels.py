"""Label validation and dialect provenance regressions; no corpus/network writes."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import preprocess_support as preprocess


def test_vocabulary_check_does_not_rewrite_pronunciation():
    phones = ["ə", "ɪ", "hʲ", "."]
    assert preprocess.unknown_phonemes(phones) == {"hʲ", "."}
    assert phones == ["ə", "ɪ", "hʲ", "."]


@pytest.mark.parametrize("valid", [True, False])
def test_preprocess_records_language_only_with_valid_labels(tmp_path, valid):
    lang_dir = tmp_path / "spa"
    lang_dir.mkdir()
    manifest = lang_dir / "manifest.jsonl"
    row = {"file": "a.wav", "sentence": "cinco", "source": "tts",
           "voice": "es-US-Chirp3-HD-Kore", "speaker_cluster": "keep"}
    original = json.dumps(row) + "\n"
    manifest.write_text(original)
    sf.write(lang_dir / "a.wav", np.full(1600, 0.1, dtype=np.float32), 16000)
    prepared = tmp_path / "prepared.jsonl"
    preprocess.prepare(tmp_path, "spa", prepared, False)
    item = json.loads(prepared.read_text())
    assert item["language"] == "spa-419"
    item["labels"] = {"phonemes": ["s"] if valid else ["INVALID"],
                      "stress": [0], "word_spans": [[0, 1]]}
    prepared.write_text(json.dumps(item) + "\n")
    if valid:
        preprocess.finalize(tmp_path, "spa", prepared, "test-build")
        assert manifest.read_text() == original
        label = json.loads((lang_dir / "phonemes.jsonl").read_text())
        assert label["g2p_language"] == "spa-419"
        assert label["phonemes"] == ["s"]
    else:
        with pytest.raises(ValueError, match="unsupported model labels"):
            preprocess.finalize(tmp_path, "spa", prepared, "test-build")
        assert manifest.read_text() == original
        assert not (lang_dir / "phonemes.jsonl").exists()

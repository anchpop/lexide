"""Label validation and dialect provenance regressions; no corpus/network writes."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import preprocess
import g2p_client


@pytest.fixture(autouse=True)
def label_build(monkeypatch):
    monkeypatch.setattr(g2p_client, "identity", lambda: "test-build")


def test_vocabulary_check_does_not_rewrite_pronunciation():
    phones = ["ə", "ɪ", "hʲ", "."]
    assert preprocess.unknown_phonemes(phones) == {"hʲ", "."}
    assert phones == ["ə", "ɪ", "hʲ", "."]


@pytest.mark.parametrize("rec, lang, expected", [
    ({"source": "pimsleur", "espeak_voice": "es-419"}, "spa", "es-419"),
    ({"source": "pimsleur", "espeak_voice": "es"}, "spa", "es"),
    ({"source": "fleurs", "espeak_voice": "es"}, "spa", "es"),
    ({"source": "fleurs", "espeak_voice": None}, "spa", "es-419"),
    ({"source": "tts", "voice": "es-US-Chirp3-HD-Kore"}, "spa", "es-419"),
    ({"source": "tts", "voice": "es-ES-Chirp3-HD-Kore", "tts_backend": "chirp3"}, "spa", "es"),
    ({"source": "tts", "voice": "es-US-Chirp3-HD-Kore", "espeak_voice": "es"}, "spa", "es"),
    ({"source": "tts", "voice": "gemini:spa:Kore", "tts_backend": "gemini"}, "spa", "es"),
    ({"source": "tts", "voice": "es-US-Chirp3-HD-Kore", "tts_backend": "gemini"}, "spa", "es"),
    ({"source": "tatoeba", "voice": "es-US-Chirp3-HD-Kore"}, "spa", "es"),
    ({"source": "pimsleur"}, "spa", "es"),
    ({"source": "film", "espeak_voice": "es-419"}, "spa", "es-419"),
    ({"source": "film", "espeak_voice": None}, "spa", "es"),
    ({"source": "tts", "voice": "es-US-Chirp3-HD-Kore"}, "eng", "en-us"),
    ({"source": "pimsleur", "espeak_voice": "pt"}, "por", "pt"),
])
def test_dialect_resolution(rec, lang, expected):
    before = dict(rec)
    assert preprocess.resolve_espeak_voice(rec, lang) == expected
    assert rec == before


@pytest.mark.parametrize("valid", [True, False])
def test_preprocess_records_variety_only_with_valid_labels(tmp_path, monkeypatch, valid):
    lang_dir = tmp_path / "spa"
    lang_dir.mkdir()
    manifest = lang_dir / "manifest.jsonl"
    row = {"file": "a.wav", "sentence": "cinco", "source": "tts",
           "voice": "es-US-Chirp3-HD-Kore", "speaker_cluster": "keep"}
    original = json.dumps(row) + "\n"
    manifest.write_text(original)
    sf.write(lang_dir / "a.wav", np.full(1600, 0.1, dtype=np.float32), 16000)
    calls = []

    def phonemize(text, lang, *, variety):
        calls.append((text, lang, variety))
        return {"phonemes": ["s"] if valid else ["INVALID"],
                "stress": [0], "word_spans": [[0, 1]]}

    monkeypatch.setattr(g2p_client, "phonemize", phonemize)
    monkeypatch.setattr(preprocess, "_tokenizer_vocab", lambda: {"s"})
    monkeypatch.setattr(preprocess, "run_narrowing", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["preprocess.py", "--data-dir", str(tmp_path),
                                     "--langs", "spa", "--skip-vad",
                                     "--skip-speaker-cluster", "--no-pack"])
    if valid:
        preprocess.main()
        assert manifest.read_text() == original
        label = json.loads((lang_dir / "phonemes.jsonl").read_text())
        assert label["variety"] == "latin_american"
        assert label["phonemes"] == ["s"]
    else:
        with pytest.raises(SystemExit, match="1"):
            preprocess.main()
        assert manifest.read_text() == original
        assert not (lang_dir / "phonemes.jsonl").exists()
    assert calls == [("cinco", "spa", "latin_american")]


@pytest.mark.parametrize("skip", [False, True])
def test_skip_narrowing_preserves_existing_labels(tmp_path, monkeypatch, skip):
    lang_dir = tmp_path / "eng"
    lang_dir.mkdir()
    (lang_dir / "manifest.jsonl").write_text(json.dumps({
        "file": "a.wav", "sentence": "hello", "source": "tts",
    }) + "\n")
    sf.write(lang_dir / "a.wav", np.full(1600, 0.1, dtype=np.float32), 16000)
    narrowed = lang_dir / "phonemes_narrowed.jsonl"
    narrowed.write_text("preserve existing narrowed labels\n")
    calls = []
    monkeypatch.setattr(g2p_client, "phonemize", lambda *args, **kwargs: {
        "phonemes": ["h"], "stress": [0], "word_spans": [[0, 1]],
    })
    monkeypatch.setattr(preprocess, "_tokenizer_vocab", lambda: {"h"})
    monkeypatch.setattr(preprocess, "run_narrowing", lambda *args: calls.append(args))
    argv = ["preprocess.py", "--data-dir", str(tmp_path), "--skip-vad",
            "--skip-speaker-cluster", "--no-pack"]
    if skip:
        argv.append("--skip-narrowing")
    monkeypatch.setattr(sys, "argv", argv)
    preprocess.main()
    assert (lang_dir / "phonemes.jsonl").exists()
    assert calls == ([] if skip else [("eng", tmp_path)])
    assert narrowed.read_text() == "preserve existing narrowed labels\n"


@pytest.mark.parametrize("skip", [False, True])
def test_parallel_children_propagate_skip_narrowing(tmp_path, monkeypatch, skip):
    from argparse import Namespace

    commands = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            return 0

    monkeypatch.setattr(preprocess, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(preprocess.subprocess, "Popen", Process)
    args = Namespace(jobs=2, data_dir=tmp_path, espeak_batch_size=8,
                     skip_vad=True, skip_speaker_cluster=True,
                     skip_narrowing=skip, allow_noncommercial=False)
    preprocess._run_parallel_languages(args, ["eng", "deu"])
    assert len(commands) == 2
    assert all(("--skip-narrowing" in command) == skip for command in commands)
    assert all("--no-pack" in command for command in commands)

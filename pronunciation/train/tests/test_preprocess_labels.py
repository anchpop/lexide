"""Label-canon and dialect provenance regressions; no corpus/network writes."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import preprocess


@pytest.mark.parametrize("lang, expected", [
    ("eng", ["ə", "ə", "ɪ", "ɐ̯", "ɐː", "ᵻː"]),
    ("deu", ["ɐ", "ᵻ", "ɪ", "ɐ̯", "ɐː", "ᵻː"]),
    (None, ["ɐ", "ᵻ", "ɪ", "ɐ̯", "ɐː", "ᵻː"]),
])
def test_english_remap_is_exact_and_language_scoped(monkeypatch, lang, expected):
    phones = ["ɐ", "ᵻ", "ɪ", "ɐ̯", "ɐː", "ᵻː"]
    stress = [0, 1, 2, 0, 1, 0]
    monkeypatch.setattr(preprocess, "_tokenizer_vocab", lambda: set(phones) | {"ə"})
    assert preprocess.validate_phonemes(phones, stress, lang) == (expected, stress, set())
    assert phones[:3] == ["ɐ", "ᵻ", "ɪ"]


def test_remap_precedes_vocab_check(monkeypatch):
    monkeypatch.setattr(preprocess, "_tokenizer_vocab", lambda: {"ə", "ɪ"})
    assert preprocess.validate_phonemes(["ɐ", "ᵻ", "ɪ"], [0, 1, 2], "eng") == (
        ["ə", "ə", "ɪ"], [0, 1, 2], set(),
    )


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


def test_persist_only_emitted_espeak_rows(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rows = [
        {"file": "a.wav", "voice": "es-US-Chirp3-HD-Kore", "speaker_cluster": "keep"},
        {"file": "b.wav", "espeak_voice": "es"},
        {"file": "excluded.wav", "espeak_voice": None},
        {"file": "external.wav"},
    ]
    entries = [
        {"file": "a.wav", "phoneme_backend": "espeak", "espeak_voice": "es-419"},
        {"file": "b.wav", "phoneme_backend": "espeak", "espeak_voice": "es"},
        {"file": "external.wav", "phoneme_backend": "external"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert preprocess.persist_espeak_voices(path, rows, entries) == 1
    written = [json.loads(s) for s in path.read_text().splitlines()]
    assert written == rows
    assert written[0]["espeak_voice"] == "es-419"
    assert written[0]["speaker_cluster"] == "keep"
    assert written[2]["espeak_voice"] is None
    assert "espeak_voice" not in written[3]
    mtime = path.stat().st_mtime_ns
    assert preprocess.persist_espeak_voices(path, rows, entries) == 0
    assert path.stat().st_mtime_ns == mtime
    assert not path.with_suffix(".jsonl.tmp").exists()


@pytest.mark.parametrize("valid", [True, False])
def test_preprocess_persists_voice_only_after_valid_labels(tmp_path, monkeypatch, valid):
    lang_dir = tmp_path / "spa"
    lang_dir.mkdir()
    manifest = lang_dir / "manifest.jsonl"
    row = {"file": "a.wav", "sentence": "cinco", "source": "tts",
           "voice": "es-US-Chirp3-HD-Kore", "speaker_cluster": "keep"}
    original = json.dumps(row) + "\n"
    manifest.write_text(original)
    sf.write(lang_dir / "a.wav", np.full(1600, 0.1, dtype=np.float32), 16000)
    calls = []

    def phonemize(text, lang, *, voice):
        calls.append((text, lang, voice))
        return {"phonemes": ["s"] if valid else ["INVALID"],
                "stress": [0], "word_spans": [[0, 1]]}

    monkeypatch.setattr(preprocess, "phonemize", phonemize)
    monkeypatch.setattr(preprocess, "_tokenizer_vocab", lambda: {"s"})
    monkeypatch.setattr(preprocess, "run_narrowing", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["preprocess.py", "--data-dir", str(tmp_path),
                                     "--langs", "spa", "--skip-vad",
                                     "--skip-speaker-cluster", "--no-pack"])
    if valid:
        preprocess.main()
        assert json.loads(manifest.read_text()) == {**row, "espeak_voice": "es-419"}
        label = json.loads((lang_dir / "phonemes.jsonl").read_text())
        assert label["espeak_voice"] == "es-419"
        assert label["phonemes"] == ["s"]
    else:
        with pytest.raises(SystemExit, match="1"):
            preprocess.main()
        assert manifest.read_text() == original
        assert not (lang_dir / "phonemes.jsonl").exists()
    assert calls == [("cinco", "spa", "es-419")]


def test_verifier_prefers_label_voice_and_keeps_legacy_fallback(tmp_path, monkeypatch):
    path = SCRIPTS.parents[1] / "scripts" / "verify_espeak_build.py"
    spec = importlib.util.spec_from_file_location("verify_espeak_build", path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    lang_dir = tmp_path / "data" / "audio" / "spa"
    lang_dir.mkdir(parents=True)
    rows = [
        {"file": "a.wav", "espeak_voice": "es-419"},
        {"file": "b.wav"},
        {"file": "c.wav"},
    ]
    labels = [{**r, "sentence": "cinco", "phonemes": ["s"], "stress": [0]} for r in rows]
    (lang_dir / "phonemes.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
    (lang_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"file": "a.wav", "espeak_voice": "es"},
        {"file": "b.wav", "espeak_voice": "es-419"},
        {"file": "c.wav", "source": "tts", "voice": "es-US-Chirp3-HD-Kore"},
    ]))
    voices = []

    def phonemize(text, lang, *, voice):
        assert lang == "spa"
        voices.append(voice)
        return {"phonemes": ["s"], "stress": [0], "word_spans": [[0, 1]]}

    monkeypatch.setattr(verifier, "REPO", tmp_path)
    monkeypatch.setattr(verifier, "phonemize", phonemize)
    monkeypatch.setattr(preprocess, "_tokenizer_vocab", lambda: {"s"})
    monkeypatch.setattr(sys, "argv", ["verify_espeak_build.py", "--langs", "spa"])
    assert verifier.main() == 0
    assert voices == ["es-419", "es-419", "es"]


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
    monkeypatch.setattr(preprocess, "phonemize", lambda *args, **kwargs: {
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
    preprocess._run_parallel_languages(args, ["eng", "deu"], {})
    assert len(commands) == 2
    assert all(("--skip-narrowing" in command) == skip for command in commands)
    assert all("--no-pack" in command for command in commands)

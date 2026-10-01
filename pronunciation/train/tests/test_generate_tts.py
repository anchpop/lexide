"""Cloud TTS dialect additions must not replace existing recordings."""

import hashlib
import io
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
import generate_tts as tts


@pytest.mark.parametrize("lang,locale", tts.LANG_CONFIG.items())
def test_default_cloud_stems_unchanged(lang, locale):
    assert tts.cloud_clip_hash("Você chegou.", locale, lang) == tts.sentence_hash("Você chegou.")


def test_dialect_stems_are_namespaced():
    text = "Você chegou."
    assert tts.cloud_clip_hash(text, "pt-PT", "por") == hashlib.sha256(f"pt-PT:{text}".encode()).hexdigest()[:16]
    assert tts.cloud_clip_hash(text, "pt-PT", "por") != tts.cloud_clip_hash(text, "pt-BR", "por")
    assert tts.cloud_clip_hash(text, "es-ES", "por") == tts.sentence_hash(f"es-ES:{text}")


@pytest.mark.parametrize("text", ["Ônibus", "celular", "CELULARES", "trem", "trens", "banheiro", "banheiros", "Antônio", "GÊMEO", "fenômeno"])
def test_pt_pt_filter_rejects_brazilian_forms(text):
    assert tts.PT_PT_EXCLUSIONS.search(text)


def test_filter_precedes_selection_and_extremes(monkeypatch):
    keep = ["Você chegou.", "Vocês chegaram.", "Tremeu.", "Um telefone."]
    monkeypatch.setattr(tts, "load_sentences", lambda lang: keep + ["celular", "Um banheiro muito grande."])
    chosen, _ = tts.select_sentences("por", 2, 0, 10, 42, "pt-pt")
    assert set(chosen) == set(keep)
    assert chosen == tts.select_sentences("por", 2, 0, 10, 42, "pt-pt")[0]


@pytest.mark.parametrize("locale,espeak", [("pt-PT", "pt"), ("pt-BR", "pt-br"), ("es-ES", "es"), ("es-US", "es-419"), ("fr-FR", None)])
def test_cloud_metadata_and_resume(tmp_path, monkeypatch, locale, espeak):
    # Stub the Cloud module rather than making the training test venv depend on its SDK.
    from unittest.mock import MagicMock
    sdk = MagicMock()
    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(texttospeech=sdk))
    monkeypatch.setitem(sys.modules, "google.cloud.texttospeech", sdk)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 160)
    client = MagicMock()
    client.list_voices.return_value.voices = [SimpleNamespace(name=f"{locale}-{family}-{sex}") for family in ("Standard", "Wavenet") for sex in ("E", "F")]
    client.synthesize_speech.return_value.audio_content = buffer.getvalue()
    monkeypatch.setattr(tts, "make_client", lambda: client)
    monkeypatch.setattr(tts, "load_sentences", lambda lang: ["Você chegou.", "Você chegou."])
    lang = "por"
    root = tmp_path / lang
    root.mkdir()
    manifest = root / "manifest.jsonl"
    previous = json.dumps({"file": "unrelated.wav", "sentence": "Old sentence"}) + "\n"
    manifest.write_text(previous)
    tts.generate_chirp3(lang, 5000, tmp_path, 42, 1, 10000, 0, 0, locale, "Wavenet")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 2
    assert manifest.read_text().startswith(previous)
    record = rows[1]
    assert record["file"] == tts.cloud_clip_hash("Você chegou.", locale, lang) + ".wav"
    assert record.get("espeak_voice") == espeak
    assert "license" not in record and "tts_backend" not in record
    assert "Wavenet" in record["voice"]
    assert record["lang"] == "por"
    sdk.AudioConfig.assert_called_once_with(audio_encoding=sdk.AudioEncoding.LINEAR16, sample_rate_hertz=16000)
    with wave.open(str(root / record["file"])) as wav:
        assert wav.getframerate() == 16000
    tts.generate_chirp3(lang, 5000, tmp_path, 42, 1, 10000, 0, 0, locale, "Wavenet")
    assert client.synthesize_speech.call_count == 1
    assert len(manifest.read_text().splitlines()) == 2

"""Combined language selection at the Python request boundary."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import g2p_client
from corpus_labels import language_for_record


@pytest.mark.parametrize("rec, lang, expected", [
    ({"source": "fleurs"}, "spa", "spa-419"),
    ({"source": "tatoeba"}, "spa", "spa-ES"),
    ({"espeak_voice": "pt"}, "por", "por-PT"),
    ({"espeak_voice": "pt-br"}, "por", "por-BR"),
    ({"variety": "latin_american"}, "spa", "spa-419"),
    ({"g2p_language": "spa-419"}, "spa", "spa-419"),
    ({}, "eng", "eng"),
])
def test_record_language(rec, lang, expected):
    assert language_for_record(rec, lang) == expected


@pytest.mark.parametrize("lang", ["spa-ES", "spa-419", "por-BR", "por-PT"])
def test_structured_call_forwards_all_fields(monkeypatch, lang):
    response = {"phonemes": [], "stress": [], "word_spans": [],
                "tone": [None], "pitch": [], "syllables": []}
    calls = []

    def request(**req):
        calls.append(req)
        return response

    monkeypatch.setattr(g2p_client, "request", request)
    assert g2p_client.phonemize("text", lang) is response
    assert calls == [{"text": "text", "lang": lang}]


def test_refusal_propagates_unchanged(monkeypatch):
    error = g2p_client.Unlabelable("refused:test", "refused")

    def request(**req):
        raise error

    monkeypatch.setattr(g2p_client, "request", request)
    with pytest.raises(g2p_client.Unlabelable) as caught:
        g2p_client.phonemize("text", "spa-ES")
    assert caught.value is error

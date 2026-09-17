"""Language + voice shadows the unchanged voice-only transport oracle.

Synthetic cases run when g2p is installed. LEXIDE_G2P_ESPEAK_SHADOW=1 also
samples real manifests, read-only, stratified by source/backend/resolved voice.
"""

import json
import os
from pathlib import Path
import random
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "train" / "scripts"))
import g2p_client
import preprocess
import audit_asr_groq

LANGS = ("spa", "eng", "fra", "deu")


@pytest.fixture
def live_g2p():
    if not (os.environ.get("G2P_BIN") or shutil.which("g2p")):
        pytest.skip("g2p binary unavailable; set G2P_BIN")


def wire(value):
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def shadow(rec, lang):
    voice = preprocess.resolve_espeak_voice(rec, lang)
    before = g2p_client.request(text=rec["sentence"], voice=voice)
    after = g2p_client.phonemize(
        rec["sentence"], lang, variety=preprocess.variety_for_record(rec, lang),
    )
    fields = ("phonemes", "stress", "word_spans")
    assert {k: after[k] for k in fields} == {k: before[k] for k in fields}
    # The old wrapper and the live preprocess loop must hand downstream
    # exactly the same tuple, including tuple-valued (not list-valued) spans.
    old_tuple, new_tuple = [
        (r["phonemes"], r["stress"], [tuple(s) for s in r["word_spans"]])
        for r in (before, after)
    ]
    assert wire(new_tuple) == wire(old_tuple)
    return after


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("text", ["", "cinco zapatos", "Hello, world!\nAnother sentence."])
def test_synthetic_voice_only_shadow(live_g2p, lang, text):
    result = shadow({"sentence": text, "source": "tatoeba"}, lang)
    if not text:
        assert result["phonemes"] == result["stress"] == result["word_spans"] == []
    else:
        assert len(result["word_spans"]) >= 2


def test_fleurs_seseo_not_default_castilian(live_g2p):
    fleurs = {"sentence": "cinco zapatos", "source": "fleurs", "voice": None}
    default = {**fleurs, "source": "tatoeba"}
    assert preprocess.resolve_espeak_voice(fleurs, "spa") == "es-419"
    assert preprocess.resolve_espeak_voice(default, "spa") == "es"
    latin = shadow(fleurs, "spa")["phonemes"]
    castilian = shadow(default, "spa")["phonemes"]
    assert "s" in latin and "θ" not in latin
    assert "θ" in castilian
    assert latin != castilian


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.skipif(not os.environ.get("LEXIDE_G2P_ESPEAK_SHADOW"),
                    reason="opt-in real-manifest voice-only shadow")
def test_real_manifest_voice_only_shadow(live_g2p, lang):
    manifest = ROOT / "data" / "audio" / lang / "manifest.jsonl"
    if not manifest.exists():
        pytest.skip(f"real manifest unavailable: {manifest}")
    rng = random.Random(0)
    strata = {}
    counts = {}
    with manifest.open() as rows:
        for line in rows:
            rec = json.loads(line)
            key = (rec.get("source"), rec.get("tts_backend"),
                   preprocess.resolve_espeak_voice(rec, lang))
            picked = strata.setdefault(key, [])
            counts[key] = counts.get(key, 0) + 1
            if len(picked) < 20:
                picked.append(rec)
            else:
                index = rng.randrange(counts[key])
                if index < 20:
                    picked[index] = rec
    assert strata, f"empty manifest: {manifest}"
    assert len({key[0] for key in strata}) > 1
    if lang == "spa":
        assert any(key[0] == "fleurs" and key[2] == "es-419" for key in strata)
    for key, picked in strata.items():
        for rec in picked:
            shadow(rec, lang)
        print(f"{lang} {key}: {len(picked)}/{counts[key]} rows byte-identical")


@pytest.mark.parametrize("kwargs", [{}, {"variety": "latin_american"}])
def test_structured_call_forwards_all_fields(monkeypatch, kwargs):
    response = {"phonemes": [], "stress": [], "word_spans": [],
                "tone": [None], "pitch": [], "syllables": []}
    calls = []

    def request(**req):
        calls.append(req)
        return response

    monkeypatch.setattr(g2p_client, "request", request)
    assert g2p_client.phonemize("text", "spa", **kwargs) is response
    assert calls == [{"text": "text", "lang": "spa", "variety": "default", **kwargs}]


def test_refusal_propagates_unchanged(monkeypatch):
    error = g2p_client.Unlabelable("refused:test", "refused")

    def request(**req):
        raise error

    monkeypatch.setattr(g2p_client, "request", request)
    with pytest.raises(g2p_client.Unlabelable) as caught:
        g2p_client.phonemize("text", "spa", variety="european")
    assert caught.value is error

"""Shadow audit request wrappers against their pre-refactor implementations."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_g2p_backends as providers
import build_external_phoneme_sidecars as sidecars
import g2p_client

ORACLE = Path(__file__).parent / "fixtures" / "g2p_superset" / "original_providers.py"
spec = importlib.util.spec_from_file_location("original_g2p_providers", ORACLE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
LANGS = ("zho-hans", "jpn", "tha", "kor")
BASE = {
    "raw": "  音\t a\n b  ", "phonemes": ["k", "a"], "stress": [0, 1],
    "word_spans": [[0, 2]], "tone": [None, {"value": 2}],
    "pitch": [None, {"source": "backend", "level": "H"}],
    "accent_withheld": None, "ignored": "not an audit field",
}


def functions(lang):
    provider = sidecars.CONFIG[lang][0]
    current = providers.PROVIDERS[provider][1]
    return getattr(original, current.__name__), current


def wire(row):
    return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")


def observe(fn, response, monkeypatch, *, request_error=None, identity_error=None):
    events = []
    native = copy.deepcopy(response)

    def request(**kwargs):
        events.append(("request", wire(kwargs)))
        if request_error is not None:
            raise request_error
        return native

    def identity():
        events.append(("identity",))
        if identity_error is not None:
            raise identity_error
        return "shadow-build"

    with monkeypatch.context() as patch:
        patch.setattr(g2p_client, "request", request)
        patch.setattr(g2p_client, "identity", identity)
        try:
            output = fn("字幕\ntext")
        except Exception as exc:
            result = ("error", type(exc), str(exc), getattr(exc, "reason", None))
        else:
            result = ("output", wire(output))
            for field in ("phonemes", "stress"):
                if field in output:
                    assert output[field] is native[field]
    assert native == response
    return result, events


RESPONSES = [
    BASE,
    {"raw": "", "phonemes": [], "stress": [], "tone": []},
    {},
    *[{key: value for key, value in BASE.items() if key != absent}
      for absent in ("raw", "phonemes", "stress", "tone", "pitch", "accent_withheld")],
    *[{**BASE, field: value} for field, value in (
        ("tone", None), ("tone", []), ("tone", [1]), ("pitch", None),
        ("pitch", []), ("pitch", ["bad"]), ("raw", None),
        ("stress", []), ("phonemes", None), ("accent_withheld", "withheld"),
        ("accent_withheld", ""),
    )],
]


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("response", RESPONSES)
def test_serialized_projection_and_errors_shadow_original(lang, response, monkeypatch):
    before, after = [observe(fn, response, monkeypatch) for fn in functions(lang)]
    assert after == before
    assert after[1][0] == ("request", wire({"text": "字幕\ntext", "lang": lang}))
    if after[0][0] == "error":
        assert len(after[1]) == 1  # Malformed responses fail before identity.
    else:
        assert after[1][-1] == ("identity",)
        if lang == "kor":
            assert json.loads(after[0][1])["tone"] == [None] * len(response["phonemes"])


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("reason", ["digits:12", "foreign_script:abc", ""])
def test_refusal_bytes_and_identity_shadow_original(lang, reason, monkeypatch):
    error = g2p_client.Unlabelable(reason, "backend refused")
    before, after = [observe(fn, {}, monkeypatch, request_error=error)
                     for fn in functions(lang)]
    assert after == before
    assert after[0] == ("output", wire({"exclude_reason": reason, "g2p": "shadow-build"}))
    assert len(after[1]) == 2


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("stage", ["request", "identity", "refusal-identity"])
@pytest.mark.parametrize("error", [RuntimeError("dead server"), ModuleNotFoundError("uv"),
                                   g2p_client.Unlabelable("identity:bad", "bad identity")])
def test_failure_boundaries_shadow_original(lang, stage, error, monkeypatch):
    kwargs = {"request_error": error} if stage == "request" else {"identity_error": error}
    if stage == "refusal-identity":
        kwargs["request_error"] = g2p_client.Unlabelable("refused", "no labels")
    before, after = [observe(fn, BASE, monkeypatch, **kwargs) for fn in functions(lang)]
    assert after == before
    if stage != "request" or not isinstance(error, g2p_client.Unlabelable):
        assert after[0][0] == "error"


@pytest.mark.parametrize("lang", LANGS)
def test_projection_unlabelable_is_not_a_backend_refusal(lang, monkeypatch):
    class BrokenResponse(dict):
        def __getitem__(self, key):
            raise g2p_client.Unlabelable("projection:bad", "projection failed")

    before, after = [observe(fn, BrokenResponse(BASE), monkeypatch) for fn in functions(lang)]
    assert after == before
    assert after[0] == ("error", g2p_client.Unlabelable, "projection failed", "projection:bad")
    assert len(after[1]) == 1


@pytest.mark.parametrize("lang", LANGS)
def test_audit_and_sidecar_bytes_and_cache_reuse_shadow_original(lang, tmp_path, monkeypatch):
    provider = sidecars.CONFIG[lang][0]
    records = [{"file": f"{i}.wav", "sentence": text, "source": "tts"}
               for i, text in enumerate(("音", "refused", "empty"))]
    generated = []
    for name, fn in zip(("original", "shared"), functions(lang), strict=True):
        folder = tmp_path / name / lang
        folder.mkdir(parents=True)
        manifest = folder / "manifest.jsonl"
        manifest.write_bytes(b"".join(wire(rec) for rec in records))
        audit = folder / f"g2p_audit_{provider}.jsonl"
        events = []

        def request(**kwargs):
            events.append(kwargs)
            assert kwargs["lang"] == lang and set(kwargs) == {"text", "lang"}
            if kwargs["text"] == "refused":
                raise g2p_client.Unlabelable("foreign_script:abc", "refused")
            if kwargs["text"] == "empty":
                return {"raw": "", "phonemes": [], "stress": [], "tone": []}
            return copy.deepcopy(BASE)

        with monkeypatch.context() as patch:
            patch.setattr(g2p_client, "request", request)
            patch.setattr(g2p_client, "identity", lambda: "shadow-build")
            patch.setitem(providers.PROVIDERS, provider, (lang, fn))
            providers.run_audit(lang, provider, manifest=manifest, output=audit)
            assert len(events) == 4  # Probe, then the three dispositions.
            audit_bytes = audit.read_bytes()
            events.clear()
            providers.run_audit(lang, provider, manifest=manifest, output=audit)
            assert events == [] and audit.read_bytes() == audit_bytes
            labels = sidecars.build_sidecar(lang, data_root=folder.parent)
        generated.append((audit_bytes, labels.read_bytes()))
    assert generated[0] == generated[1]
    rows = [json.loads(line) for line in generated[1][0].splitlines()]
    assert rows[1]["sentence_sha256"] == hashlib.sha256(b"refused").hexdigest()
    assert rows[1]["output"] == {"exclude_reason": "foreign_script:abc", "g2p": "shadow-build"}

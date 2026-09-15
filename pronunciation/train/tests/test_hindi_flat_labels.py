"""Byte-level Hindi migration gates; default tests use frozen offline fixtures.

The goldens were captured through the unmodified production provider and
build_sidecar, with the pinned binary/current canon (see provenance.json).
Set LEXIDE_HINDI_LIVE_SHADOW=1 to replay the fixtures through that binary, or
LEXIDE_HINDI_FULL_SHADOW=1 to shadow every current Hindi manifest disposition.
Neither mode writes to the corpus or updates the goldens.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_g2p_backends as providers
import build_external_phoneme_sidecars as sidecars
import g2p_client

FIXTURES = Path(__file__).parent / "fixtures" / "hindi_flat"
CASES = sidecars.read_jsonl(FIXTURES / "cases.jsonl")
PROVENANCE = json.loads((FIXTURES / "provenance.json").read_text())


def legacy_output(response):
    """Frozen pre-refactor hindi_words reshaping, solely for shadow comparison."""
    if response.get("exclude_reason"):
        return {"exclude_reason": response["exclude_reason"]}
    words = []
    for start, end in response["word_spans"]:
        syllables = [
            {"start": s["start"] - start, "end": s["end"] - start,
             "nucleus": s["nucleus"] - start, "moras": s["moras"],
             "stress": int(s["stressed"])}
            for s in response.get("syllables", []) if start <= s["start"] < end
        ]
        words.append({
            "phonemes": response["phonemes"][start:end],
            "stress": response["stress"][start:end],
            "syllables": syllables,
        })
    return {"words": words}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def build_bytes(root, records, outputs, monkeypatch, *, legacy=False):
    lang_dir = root / "hin"
    lang_dir.mkdir(parents=True)
    write_jsonl(lang_dir / "manifest.jsonl", records)
    write_jsonl(lang_dir / "g2p_audit_g2p-hin.jsonl", [
        {"file": rec["file"],
         "sentence_sha256": hashlib.sha256(rec["sentence"].encode()).hexdigest(),
         "provider_schema": providers.PROVIDER_SCHEMA["g2p-hin"],
         "output": legacy_output(out) if legacy else out}
        for rec, out in zip(records, outputs, strict=True)
    ])
    with monkeypatch.context() as patch:
        if legacy:
            patch.setitem(sidecars.CONFIG, "hin", ("g2p-hin", sidecars.hindi_labels))
        path = sidecars.build_sidecar("hin", data_root=root)
    return path.read_bytes()


def test_frozen_goldens_are_byte_identical(tmp_path, monkeypatch):
    for name, key in [("cases.jsonl", "cases_sha256"),
                      ("expected_sidecar.jsonl", "expected_sidecar_sha256")]:
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == PROVENANCE[key]
    records = [c["record"] for c in CASES]
    outputs = [c["response"] for c in CASES]
    expected = (FIXTURES / "expected_sidecar.jsonl").read_bytes()
    assert build_bytes(tmp_path / "legacy", records, outputs, monkeypatch, legacy=True) == expected
    assert build_bytes(tmp_path / "direct", records, outputs, monkeypatch) == expected
    assert len(expected.splitlines()) == PROVENANCE["case_count"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["record"]["file"])
def test_adapter_preserves_annotations_and_input(case):
    response = copy.deepcopy(case["response"])
    labels = sidecars.g2p_hindi_labels(case["record"], {"output": response})
    assert response == case["response"]
    assert "word_spans" not in labels and "raw" not in labels and "tone" not in labels
    if "exclude_reason" in labels:
        return
    assert labels["phonemes"] is response["phonemes"]
    assert labels["stress"] is response["stress"]
    for native, syllable in zip(response["syllables"], labels["syllables"], strict=True):
        assert list(syllable) == ["start", "end", "nucleus", "moras", "stress", "word", "source"]
        assert type(syllable["stress"]) is int
        assert syllable["stress"] == int(native["stressed"])
        assert syllable["source"] == labels["stress_source"] == "roy-2017-rules-on-schwa-hin"
        start, end = response["word_spans"][syllable["word"]]
        assert start <= syllable["start"] <= syllable["nucleus"] < syllable["end"] <= end
        for key in ("start", "end", "nucleus", "moras"):
            assert syllable[key] == native[key]


@pytest.mark.parametrize("syllables", [None, []])
def test_empty_response_preserves_hindi_reason(syllables):
    response = {"raw": "", "phonemes": [], "stress": [], "word_spans": []}
    if syllables is not None:
        response["syllables"] = syllables
    assert sidecars.g2p_hindi_labels({"file": "empty"}, {"output": response}) == {
        "exclude_reason": "hindi_no_devanagari_phones",
    }


@pytest.mark.parametrize("field,value", [
    ("stress", []), ("word_spans", []), ("word_spans", [[1, 2], [2, 7]]),
    ("word_spans", [[0, 2], [1, 7]]), ("word_spans", [[0, 2], [3, 7]]),
    ("word_spans", [[0, 2], [2, 8]]), ("syllables", []),
])
def test_bad_alignment_fails_closed(field, value):
    case = next(c for c in CASES if c["record"]["file"] == "synthetic-multiword.wav")
    response = copy.deepcopy(case["response"])
    response[field] = value
    with pytest.raises(ValueError, match="Hindi"):
        sidecars.g2p_hindi_labels(case["record"], {"output": response})


@pytest.mark.parametrize("field,value", [
    ("start", 1), ("end", 3), ("nucleus", -1), ("nucleus", 2), ("stressed", True),
])
def test_bad_syllable_fails_closed(field, value):
    case = next(c for c in CASES if c["record"]["file"] == "synthetic-multiword.wav")
    response = copy.deepcopy(case["response"])
    response["syllables"][0][field] = value
    with pytest.raises(ValueError, match="Hindi"):
        sidecars.g2p_hindi_labels(case["record"], {"output": response})


def test_trailing_syllable_and_missing_mandatory_fields_fail():
    response = {"phonemes": [], "stress": [], "word_spans": [],
                "syllables": [{"start": 0, "end": 1, "nucleus": 0, "stressed": False, "moras": 1}]}
    with pytest.raises(ValueError, match="coverage"):
        sidecars.g2p_hindi_labels({"file": "bad"}, {"output": response})
    with pytest.raises(KeyError):
        sidecars.g2p_hindi_labels({"file": "bad"}, {"output": {"syllables": []}})


@pytest.mark.parametrize("reason", [None, "hindi_digits:१२", "hindi_latin_script:test"])
def test_provider_dispatch_canon_and_refusal(monkeypatch, reason):
    native = {"raw": "", "phonemes": [], "stress": [], "word_spans": []}
    request = Mock(return_value=native)
    if reason:
        request.side_effect = g2p_client.Unlabelable(reason, "refused")
    monkeypatch.setattr(g2p_client, "request", request)
    monkeypatch.setattr(g2p_client, "identity", lambda: "test-build")
    result = providers._g2p_hin("text")
    request.assert_called_once_with(text="text", lang="hin", canon="current")
    assert result == {**({"exclude_reason": reason} if reason else native),
                      "canon": "current", "g2p": "test-build"}
    assert "words" not in result


def test_provider_does_not_swallow_infrastructure_errors(monkeypatch):
    monkeypatch.setattr(g2p_client, "request", Mock(side_effect=RuntimeError("server died")))
    with pytest.raises(RuntimeError, match="server died"):
        providers._g2p_hin("text")


@pytest.mark.parametrize("exclusion", [False, True])
@pytest.mark.parametrize("old_schema", [None, 1, 2])
def test_cache_shape_migration(tmp_path, monkeypatch, exclusion, old_schema):
    rec = {"file": "a.wav", "sentence": "text"}
    manifest = tmp_path / "manifest.jsonl"
    path = tmp_path / "audit.jsonl"
    write_jsonl(manifest, [rec])
    stale = {"file": rec["file"], "sentence_sha256": hashlib.sha256(b"text").hexdigest(),
             "output": {"exclude_reason": "old-refusal"} if exclusion else {"words": []}}
    if old_schema is not None:
        stale["provider_schema"] = old_schema
    if old_schema == 2 and not exclusion:
        stale["output"] = {"raw": "", "phonemes": [], "stress": [], "word_spans": [],
                           "canon": "current", "g2p": "test-build"}
    write_jsonl(path, [stale])
    generate = Mock(return_value={"exclude_reason": "new-refusal"})
    monkeypatch.setitem(providers.PROVIDERS, "g2p-hin", ("hin", generate))
    providers.run_audit("hin", "g2p-hin", manifest=manifest, output=path)
    written = sidecars.read_jsonl(path)[0]
    if old_schema == 2:
        assert written == stale
        generate.assert_not_called()
    else:
        assert written["provider_schema"] == 2
        assert written["output"] == {"exclude_reason": "new-refusal"}
        assert generate.call_count == 2  # probe, then the actual row


def test_failed_schema_refresh_keeps_old_cache(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    path = tmp_path / "audit.jsonl"
    write_jsonl(manifest, [{"file": "a.wav", "sentence": "text"}])
    write_jsonl(path, [{"file": "a.wav", "sentence_sha256": hashlib.sha256(b"text").hexdigest(),
                       "provider_schema": 1, "output": {"words": []}}])
    before = path.read_bytes()
    monkeypatch.setitem(providers.PROVIDERS, "g2p-hin", (
        "hin", Mock(side_effect=RuntimeError("server died")),
    ))
    with pytest.raises(SystemExit, match="cache left untouched"):
        providers.run_audit("hin", "g2p-hin", manifest=manifest, output=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("schema", [None, 1, 3])
@pytest.mark.parametrize("exclusion", [False, True])
def test_standalone_builder_rejects_stale_schema_before_writing(tmp_path, schema, exclusion):
    lang_dir = tmp_path / "hin"
    lang_dir.mkdir()
    rec = {"file": "a.wav", "sentence": "text"}
    write_jsonl(lang_dir / "manifest.jsonl", [rec])
    audit = {"file": "a.wav", "sentence_sha256": hashlib.sha256(b"text").hexdigest(),
             "output": {"exclude_reason": "refused"} if exclusion else {"words": []}}
    if schema is not None:
        audit["provider_schema"] = schema
    write_jsonl(lang_dir / "g2p_audit_g2p-hin.jsonl", [audit])
    output = lang_dir / "phoneme_backend_g2p-hin.jsonl"
    output.write_bytes(b"existing sidecar\n")
    with pytest.raises(ValueError, match="obsolete Hindi audit schema"):
        sidecars.build_sidecar("hin", data_root=tmp_path)
    assert output.read_bytes() == b"existing sidecar\n"


def require_pinned_binary():
    assert g2p_client.identity() == PROVENANCE["g2p_identity"]
    assert hashlib.sha256(Path(g2p_client.binary()).read_bytes()).hexdigest() == PROVENANCE["binary_sha256"]


@pytest.mark.skipif(not (os.environ.get("LEXIDE_HINDI_LIVE_SHADOW") or
                        os.environ.get("LEXIDE_HINDI_FULL_SHADOW")),
                    reason="opt-in pinned-binary Hindi fixture replay")
def test_live_fixture_shadow(tmp_path, monkeypatch):
    require_pinned_binary()
    records = [c["record"] for c in CASES]
    outputs = [providers._g2p_hin(r["sentence"]) for r in records]
    assert outputs == [c["response"] for c in CASES]
    assert build_bytes(tmp_path / "direct", records, outputs, monkeypatch) == (
        FIXTURES / "expected_sidecar.jsonl"
    ).read_bytes()


@pytest.mark.skipif(not os.environ.get("LEXIDE_HINDI_FULL_SHADOW"),
                    reason="opt-in full Hindi manifest shadow")
def test_full_corpus_shadow(tmp_path, monkeypatch):
    require_pinned_binary()
    records = sidecars.read_jsonl(ROOT / "data/audio/hin/manifest.jsonl")
    assert records
    outputs = [providers._g2p_hin(r["sentence"]) for r in records]
    legacy = build_bytes(tmp_path / "legacy", records, outputs, monkeypatch, legacy=True)
    direct = build_bytes(tmp_path / "direct", records, outputs, monkeypatch)
    assert direct == legacy
    print(f"Hindi full shadow: {len(records)} byte-identical dispositions; "
          f"sha256={hashlib.sha256(direct).hexdigest()}")

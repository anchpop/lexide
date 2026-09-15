"""Byte-level superset-adapter gates against frozen original converters.

Default: offline goldens. LEXIDE_G2P_LIVE_SHADOW=1 replays the pinned binary;
LEXIDE_G2P_FULL_SHADOW=1 freshly labels every manifest with audio (slow).
LEXIDE_G2P_CACHED_SHADOW=1 compares converters over existing full audits, whose
build identities may predate the pinned binary. All writes are under tmp_path.
Korean has no audio: only text fixtures and synthetic adapter cases are covered.
"""

import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_g2p_backends as providers
import build_external_phoneme_sidecars as sidecars
import g2p_client

FIXTURES = Path(__file__).parent / "fixtures"
ORACLE_PATH = FIXTURES / "g2p_superset" / "original_converters.py"
module_spec = importlib.util.spec_from_file_location("original_g2p_converters", ORACLE_PATH)
original = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(original)
LANGS = ("hin", "tha", "zho-hans", "jpn", "kor")
AUDIO_LANGS = LANGS[:-1]
ORACLES = {lang: (original.g2p_hindi_labels if lang == "hin" else
                  original.g2p_japanese_labels if lang == "jpn" else
                  original.g2p_flat_labels) for lang in LANGS}


def fixture_dir(lang):
    return FIXTURES / "hindi_flat" if lang == "hin" else FIXTURES / "g2p_superset" / lang


def provenance(lang):
    return json.loads((fixture_dir(lang) / "provenance.json").read_text())


def cases(lang):
    return sidecars.read_jsonl(fixture_dir(lang) / "cases.jsonl")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def build_bytes(root, lang, records, outputs, monkeypatch, *, legacy=False):
    provider = sidecars.CONFIG[lang][0]
    folder = root / lang
    folder.mkdir(parents=True)
    write_jsonl(folder / "manifest.jsonl", records)
    write_jsonl(folder / f"g2p_audit_{provider}.jsonl", [
        {"file": rec["file"], "sentence_sha256": sha(rec["sentence"].encode()),
         "provider_schema": providers.PROVIDER_SCHEMA[provider], "output": out}
        for rec, out in zip(records, outputs, strict=True)
    ])
    with monkeypatch.context() as patch:
        if legacy:
            patch.setitem(sidecars.CONFIG, lang, (provider, ORACLES[lang]))
        result = sidecars.build_sidecar(lang, data_root=root)
    return result.read_bytes()


@pytest.mark.parametrize("lang", LANGS)
def test_frozen_sidecars_are_byte_identical(lang, tmp_path, monkeypatch):
    meta = provenance(lang)
    for name, key in [("cases.jsonl", "cases_sha256"),
                      ("expected_sidecar.jsonl", "expected_sidecar_sha256")]:
        assert sha((fixture_dir(lang) / name).read_bytes()) == meta[key]
    if lang != "hin":
        assert sha(ORACLE_PATH.read_bytes()) == meta["oracle_sha256"]
    frozen = cases(lang)
    assert len(frozen) == meta["case_count"]
    records = [c["record"] for c in frozen]
    outputs = [c["response"] for c in frozen]
    expected = (fixture_dir(lang) / "expected_sidecar.jsonl").read_bytes()
    assert build_bytes(tmp_path / "original", lang, records, outputs, monkeypatch, legacy=True) == expected
    assert build_bytes(tmp_path / "unified", lang, records, outputs, monkeypatch) == expected
    print(f"{lang} fixture shadow: {len(records)} byte-identical dispositions; sha256={sha(expected)}")


@pytest.mark.parametrize("lang", LANGS)
def test_input_arrays_field_order_and_omission(lang):
    convert = sidecars.CONFIG[lang][1]
    assert convert.func is sidecars.g2p_labels
    spec = convert.keywords["spec"]
    assert isinstance(spec.factors, tuple)
    with pytest.raises(FrozenInstanceError):
        spec.empty_reason = "changed"
    for case in cases(lang):
        rec, out = copy.deepcopy((case["record"], case["response"]))
        labels = convert(rec, {"output": out})
        assert rec == case["record"] and out == case["response"]
        assert json.dumps(labels, ensure_ascii=False) == json.dumps(
            ORACLES[lang](rec, {"output": out}), ensure_ascii=False,
        )
        if "exclude_reason" not in labels:
            assert labels["phonemes"] is out["phonemes"]
            assert labels["stress"] is out["stress"]
            for field in ("tone", "pitch_accent"):
                if field in labels:
                    assert labels[field] is out[field]
            assert not {"raw", "word_spans", "canon", "g2p"} & labels.keys()


@pytest.mark.parametrize("lang", ("tha", "zho-hans", "jpn"))
def test_real_samples_cover_every_source_stratum(lang):
    frozen = cases(lang)
    actual = {}
    for case in frozen:
        if case["kind"] == "manifest":
            rec = case["record"]
            key = (rec.get("source"), rec.get("tts_backend") if rec.get("source") == "tts" else None)
            actual[key] = actual.get(key, 0) + 1
    expected = {(s["source"], s["tts_backend"]): s["sampled"]
                for s in provenance(lang)["manifest_strata"]}
    assert actual == expected
    assert {key[0] for key in actual} >= {"fleurs", "pimsleur", "tatoeba", "tts", "film"}


@pytest.mark.parametrize("lang", LANGS)
def test_exclusion_precedes_all_validation(lang):
    assert sidecars.CONFIG[lang][1]({"file": "bad"}, {"output": {
        "exclude_reason": "refused", "phonemes": None, "stress": [1],
        "tone": None, "syllables": None, "pitch_accent": None,
    }}) == {"exclude_reason": "refused"}


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("phones", [[], ["a"]])
def test_stress_alignment_is_required_even_when_empty(lang, phones):
    with pytest.raises(ValueError, match="misalignment"):
        sidecars.CONFIG[lang][1]({"file": "bad"}, {"output": {
            "phonemes": phones, "stress": [0] * (len(phones) + 1),
        }})


@pytest.mark.parametrize("lang", ("tha", "zho-hans", "kor"))
@pytest.mark.parametrize("phones", [[], ["a"]])
def test_tone_required_and_aligned_before_empty(lang, phones):
    out = {"phonemes": phones, "stress": [0] * len(phones)}
    convert = sidecars.CONFIG[lang][1]
    with pytest.raises(KeyError, match="tone"):
        convert({"file": "bad"}, {"output": out})
    out["tone"] = [None] * (len(phones) + 1)
    with pytest.raises(ValueError, match="misalignment"):
        convert({"file": "bad"}, {"output": out})
    out["tone"] = [None] * len(phones)
    expected = out if phones else {"exclude_reason": "g2p_no_phonemes"}
    assert convert({"file": "valid"}, {"output": out}) == expected


@pytest.mark.parametrize("logprob", [None, -0.350001, -0.35, -0.349999])
@pytest.mark.parametrize("reason", [None, "provider-withheld", ""])
@pytest.mark.parametrize("present", [False, True])
def test_japanese_pitch_presence_and_withholding_priority(logprob, reason, present):
    out = {"phonemes": ["a"], "stress": [0], "pitch_accent_exclude_reason": reason}
    if present:
        out["pitch_accent"] = [None]
    rec = {"file": "pitch", "whisper_avg_logprob": logprob}
    withhold = reason if reason is not None else (
        "japanese_accent_low_asr_confidence" if logprob is not None and logprob < -0.35 else None
    )
    convert = sidecars.CONFIG["jpn"][1]
    if withhold is None and not present:
        with pytest.raises(KeyError, match="pitch_accent"):
            convert(rec, {"output": out})
    else:
        expected = {"phonemes": ["a"], "stress": [0]}
        expected.update({"pitch_accent_exclude_reason": withhold} if withhold is not None
                        else {"pitch_accent": [None]})
        assert convert(rec, {"output": out}) == expected


@pytest.mark.parametrize("phones,pitch", [([], [None]), (["a"], [None, None]), (["a"], None)])
@pytest.mark.parametrize("withhold", [None, "provider-withheld", ""])
@pytest.mark.parametrize("logprob", [None, -1])
def test_japanese_malformed_pitch_raises_even_when_withheld(phones, pitch, withhold, logprob):
    out = {"phonemes": phones, "stress": [0] * len(phones), "pitch_accent": pitch,
           "pitch_accent_exclude_reason": withhold}
    with pytest.raises(ValueError, match="pitch accent misalignment"):
        sidecars.CONFIG["jpn"][1]({"file": "bad", "whisper_avg_logprob": logprob}, {"output": out})


@pytest.mark.parametrize("withhold", [None, "provider-withheld", ""])
@pytest.mark.parametrize("logprob", [None, -1])
def test_japanese_empty_pitch_is_only_a_provider_withholding_sentinel(withhold, logprob):
    rec = {"file": "empty-pitch", "whisper_avg_logprob": logprob}
    out = {"phonemes": ["a"], "stress": [0], "pitch_accent": [],
           "pitch_accent_exclude_reason": withhold}
    if withhold is None:
        with pytest.raises(ValueError, match="pitch accent misalignment"):
            sidecars.CONFIG["jpn"][1](rec, {"output": out})
    else:
        assert sidecars.CONFIG["jpn"][1](rec, {"output": out}) == {
            "phonemes": ["a"], "stress": [0], "pitch_accent_exclude_reason": withhold,
        }


@pytest.mark.parametrize("out,expected", [
    ({"phonemes": [], "stress": [], "pitch_accent": []},
     {"phonemes": [], "stress": [], "pitch_accent": []}),
    ({"phonemes": [], "stress": [], "pitch_accent_exclude_reason": "no-content"},
     {"phonemes": [], "stress": [], "pitch_accent_exclude_reason": "no-content"}),
])
def test_japanese_empty_is_not_an_exclusion(out, expected):
    assert sidecars.CONFIG["jpn"][1]({"file": "empty"}, {"output": out}) == expected


def test_hindi_empty_invalid_word_span_is_not_an_exclusion():
    with pytest.raises(ValueError, match="Hindi word span"):
        sidecars.CONFIG["hin"][1]({"file": "empty"}, {"output": {
            "phonemes": [], "stress": [], "word_spans": [[0, 0]], "syllables": [],
        }})


def test_korean_provider_still_fabricates_null_tones(monkeypatch):
    native = {"phonemes": ["k", "a"], "stress": [0, 0], "raw": "가"}
    def request(**kwargs):
        assert kwargs == {"text": "가", "lang": "kor"}
        return native
    monkeypatch.setattr(g2p_client, "request", request)
    monkeypatch.setattr(g2p_client, "identity", lambda: "test-build")
    out = providers._g2p_kor("가")
    assert "tone" not in native
    assert sidecars.CONFIG["kor"][1]({"file": "fixture"}, {"output": out}) == {
        "phonemes": ["k", "a"], "stress": [0, 0], "tone": [None, None],
    }
    assert provenance("kor")["manifest_sha256"] is None
    assert not any(c.get("kind") == "manifest" for c in cases("kor"))


def require_pinned_binary(lang):
    meta = provenance(lang)
    assert g2p_client.identity() == meta["g2p_identity"]
    assert sha(Path(g2p_client.binary()).read_bytes()) == meta["binary_sha256"]


@pytest.mark.skipif(not (os.environ.get("LEXIDE_G2P_LIVE_SHADOW") or
                        os.environ.get("LEXIDE_G2P_FULL_SHADOW")), reason="opt-in pinned-binary fixture replay")
@pytest.mark.parametrize("lang", LANGS)
def test_live_fixture_shadow(lang, tmp_path, monkeypatch):
    require_pinned_binary(lang)
    frozen = cases(lang)
    records = [c["record"] for c in frozen]
    generate = providers.PROVIDERS[sidecars.CONFIG[lang][0]][1]
    outputs = [c["response"] if c.get("kind") == "adapter" else generate(c["record"]["sentence"])
               for c in frozen]
    assert outputs == [c["response"] for c in frozen]
    assert build_bytes(tmp_path, lang, records, outputs, monkeypatch) == (
        fixture_dir(lang) / "expected_sidecar.jsonl"
    ).read_bytes()


@pytest.mark.skipif(not os.environ.get("LEXIDE_G2P_FULL_SHADOW"), reason="opt-in fresh full-corpus shadow")
@pytest.mark.parametrize("lang", AUDIO_LANGS)
def test_full_corpus_shadow(lang, tmp_path, monkeypatch):
    require_pinned_binary(lang)
    records = sidecars.read_jsonl(sidecars.DATA_ROOT / lang / "manifest.jsonl")
    assert records
    generate = providers.PROVIDERS[sidecars.CONFIG[lang][0]][1]
    outputs = [generate(rec["sentence"]) for rec in records]
    before = build_bytes(tmp_path / "original", lang, records, outputs, monkeypatch, legacy=True)
    after = build_bytes(tmp_path / "unified", lang, records, outputs, monkeypatch)
    assert after == before
    print(f"{lang} fresh full shadow: {len(records)} byte-identical dispositions; sha256={sha(after)}")


@pytest.mark.skipif(not os.environ.get("LEXIDE_G2P_CACHED_SHADOW"), reason="opt-in existing full-audit shadow")
@pytest.mark.parametrize("lang", AUDIO_LANGS)
def test_cached_full_corpus_shadow(lang, tmp_path, monkeypatch):
    provider = sidecars.CONFIG[lang][0]
    # build_sidecar checks sentence hashes, errors, coverage, and Hindi schema.
    # Do not claim these older audits were generated by today's pinned binary.
    before = tmp_path / f"{lang}-original.jsonl"
    after = tmp_path / f"{lang}-unified.jsonl"
    with monkeypatch.context() as patch:
        patch.setitem(sidecars.CONFIG, lang, (provider, ORACLES[lang]))
        sidecars.build_sidecar(lang, output=before)
    sidecars.build_sidecar(lang, output=after)
    assert after.read_bytes() == before.read_bytes()
    data = after.read_bytes()
    assert data
    print(f"{lang} cached full shadow: {len(data.splitlines())} byte-identical dispositions; sha256={sha(data)}")

"""ASR audit uses production label providers; all services are mocked."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import audit_asr_groq as audit
@pytest.mark.parametrize("lang", ["eng", "tha", "hin", "zho-hans", "jpn", "kor"])
def test_labels_use_unified_api(monkeypatch, lang):
    phonemize = Mock(return_value={"phonemes": ["a"]})
    monkeypatch.setattr(audit, "phonemize", phonemize)
    assert audit.label_phonemes("text", lang) == ["a"]
    phonemize.assert_called_once_with("text", lang)


def run_live(tmp_path, monkeypatch, label, *, text_only=False, lang="hin"):
    path = tmp_path / "audio.wav"
    path.touch()
    monkeypatch.setattr(audit, "audio_stats", lambda path: (1.0, 0.1))
    response = Mock(status_code=200)
    response.json.return_value = {"text": "actual", "language": "hindi"}
    monkeypatch.setattr(audit.requests, "post", Mock(return_value=response))
    monkeypatch.setattr(audit, "label_phonemes", label)
    args = SimpleNamespace(model="test", force_language=True, retries=0,
                           timeout=1, text_only=text_only)
    return audit.transcribe({
        "path": str(path), "file": path.name, "lang": lang,
        "expected": "expected", "variety": "default",
        "per": 99, "expected_phonemes": ["stale"], "actual_phonemes": ["stale"],
    }, args, "fake-key")


@pytest.mark.parametrize("text_only", [False, True])
def test_live_unscorable_reference_omits_all_phoneme_fields(tmp_path, monkeypatch, text_only):
    label = Mock(side_effect=audit.Unlabelable("bad", "bad"))
    result = run_live(tmp_path, monkeypatch, label, text_only=text_only)
    assert result["ok"] and result["cer"] > 0 and result["wer"] > 0
    assert not {"per", "expected_phonemes", "actual_phonemes"} & result.keys()
    assert label.call_count == (0 if text_only else 1)


@pytest.mark.parametrize("failure", [audit.Unlabelable("bad", "bad"), RuntimeError("failed")])
def test_live_whisper_failure_remains_per_one(tmp_path, monkeypatch, failure):
    result = run_live(tmp_path, monkeypatch, Mock(side_effect=[["a"], failure]))
    assert result["ok"] and result["per"] == 1
    assert result["expected_phonemes"] == ["a"] and result["actual_phonemes"] == []


def test_live_passes_language_for_both_texts(tmp_path, monkeypatch):
    label = Mock(return_value=["a"])
    result = run_live(tmp_path, monkeypatch, label, lang="eng")
    assert result["per"] == 0
    assert [call.args for call in label.call_args_list] == [
        ("expected", "eng"), ("actual", "eng"),
    ]


def test_rescore_dispositions_and_scope(tmp_path, monkeypatch):
    rows = [dict(ok=True, lang="hin", file=f"{i}.wav", expected=text,
                 whisper_text="actual", per=0.5, expected_phonemes=["stale"],
                 actual_phonemes=["stale"], cer=0.2, wer=0.3)
            for i, text in enumerate(["unlabelable", "good", "same", "other", "failed"])]
    rows[3]["lang"] = "eng"
    rows[4]["ok"] = False
    path = tmp_path / "audit.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    label = Mock(side_effect=[audit.Unlabelable("bad", "bad"), ["a"],
                             audit.Unlabelable("bad", "bad"), ["a"], ["a"]])
    monkeypatch.setattr(audit, "label_phonemes", label)
    audit.rescore(path, {"hin"})
    result = [json.loads(line) for line in path.read_text().splitlines()]
    assert not {"per", "expected_phonemes", "actual_phonemes"} & result[0].keys()
    assert result[0]["cer"] == 0.2 and result[0]["wer"] == 0.3
    assert result[1]["per"] == 1 and result[1]["actual_phonemes"] == []
    assert result[2]["per"] == 0
    assert result[3:] == rows[3:]
    assert not path.with_suffix(".jsonl.tmp").exists()



def test_loader_uses_cer_and_wer_only_without_per(tmp_path):
    from src.train_unified import load_asr_audit_exclusions

    rows = [dict(ok=True, lang="hin", file=f"{i}.wav", expected="text", **metrics)
            for i, metrics in enumerate([
                {"per": 0, "cer": 1, "wer": 1},
                {"per": 1, "cer": 0, "wer": 0},
                {"cer": 0.5, "wer": 0.5}, {"cer": 0, "wer": 1},
                {"cer": 1, "wer": 0},
            ])]
    path = tmp_path / "audit.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = load_asr_audit_exclusions(path, min_per=1e-12, min_cer=1e-12, min_wer=1e-12)
    assert set(result["hin"]) == {"1.wav", "2.wav"}

import ast
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from pronunciation.inference.eval_per import (
    Client, IDENTITY_FIELDS, align, allocate, extract_tokens, inspect_population,
    metrics, score, stratified_sample, summarize, validate_batch, validate_identity,
)

IDENTITY = dict(model_id="model", model_revision="revision", deploy_marker="unique", decoder_version="nonblank_v1")


def test_hand_calculated_edits_and_deterministic_ties():
    assert align(["a", "b"], ["b", "a"]) == [("a", "b"), ("b", "a")]
    assert align(["a", "a"], ["a"]) == [("a", None), ("a", "a")]
    assert align([], ["a"]) == [(None, "a")]
    assert align(["a"], []) == [("a", None)]
    s = score(["k͈", "a", "p"], ["k", "a", "t", "i"])
    assert (s["edits"], s["substitutions"], s["insertions"], s["deletions"]) == (3, 2, 1, 0)
    assert s["per"] == 1
    with pytest.raises(ValueError):
        score([], ["a"])


def test_largest_remainder_and_reproducible_uniform_sampling():
    assert allocate({"z": 1, "b": 1, "a": 1}, 2) == {"z": 0, "b": 1, "a": 1}
    assert allocate({"film": 71, "tts": 29}, 10) == {"film": 7, "tts": 3}
    rows = [dict(source="a" if i < 70 else "b", file=f"{i:03}.wav", label_line=i) for i in range(100)]
    a, counts, allocation = stratified_sample(rows, 20, 123, "kor")
    b, _, _ = stratified_sample(list(reversed(rows)), 20, 123, "kor")
    c, _, _ = stratified_sample(rows, 20, 124, "kor")
    assert a == b and a != c
    assert len({r["file"] for r in a}) == 20
    assert counts == {"a": 70, "b": 30} and allocation == {"a": 14, "b": 6}
    with pytest.raises(ValueError):
        allocate({"a": 2}, 3)


def test_population_exclusions_and_manifest_fallback(tmp_path):
    directory = tmp_path / "kor"
    directory.mkdir()
    for name in ["a.wav", "b.wav", "duplicate.wav", "empty.wav"]:
        (directory / name).touch()
    rows = [dict(file="a.wav", source="film", duration_sec=1, phonemes=["tɕ͈"]),
            dict(file="b.wav", phonemes=["a"]),
            dict(file="missing.wav", source="film", duration_sec=1, phonemes=["a"]),
            dict(file="duplicate.wav"), dict(file="duplicate.wav"),
            dict(file="empty.wav", phonemes=[])]
    (directory / "phonemes_narrowed.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    (directory / "manifest.jsonl").write_text(json.dumps(dict(file="b.wav", source="tts", duration_sec=2)))
    valid, audit = inspect_population(tmp_path, "kor")
    assert [r["file"] for r in valid] == ["a.wav", "b.wav"]
    assert valid[0]["reference_tokens"] == ["tɕ͈"]
    assert audit["total_rows"] == 7 and audit["eligible_rows"] == 2
    assert audit["exclusion_counts"] == dict(invalid_json_object=1, missing_audio=1, duplicate_file=2, invalid_or_empty_reference=1)
    assert audit["manifest_fallback_counts"] == dict(source=1, duration_sec=1)
    assert len(audit["labels_sha256"]) == 64


def test_stress_only_extraction_preserves_unicode_and_adjacent_tokens():
    result = {"phonemes": [{"phoneme": "ˈtɕ͈", "stress": 1}, {"phoneme": "tɕ͈", "stress": 0},
                            {"phoneme": "ˌã", "stress": 2}, {"phoneme": "ã"}]}
    assert extract_tokens(result) == ["tɕ͈", "tɕ͈", "ã", "ã"]
    assert score(["ã"], ["ã"])["edits"] == 1
    assert extract_tokens({"phonemes": []}) == []
    for invalid in [{"error": {"type": "OOM"}}, {}, {"phonemes": ["a"]}, {"phonemes": [{"phoneme": "ˈ"}]}]:
        with pytest.raises(ValueError):
            extract_tokens(invalid)


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_identity_mismatch_missing_and_load_failure(field):
    validate_identity(IDENTITY, IDENTITY)
    for value in [None, "wrong"]:
        with pytest.raises(ValueError, match="identity mismatch"):
            validate_identity({**IDENTITY, field: value}, IDENTITY)
    with pytest.raises(ValueError, match="model load failed"):
        validate_identity({**IDENTITY, "load_error": "broken"}, IDENTITY)


def test_batch_shape_errors_never_disappear():
    envelope = {**IDENTITY, "results": [{"phonemes": [{"phoneme": "a"}]}]}
    assert validate_batch(envelope, IDENTITY, 1) == [["a"]]
    with pytest.raises(ValueError, match="count"):
        validate_batch(envelope, IDENTITY, 2)
    envelope["results"].append({"error": {"message": "failed clip"}})
    with pytest.raises(ValueError, match="prediction error"):
        validate_batch(envelope, IDENTITY, 2)


def scored(ref, pred, source="film"):
    return dict(source=source, language="kor", prediction_tokens=pred, **score(ref, pred))


def test_ratio_clip_mean_median_and_stratified_delta_se():
    rows = [scored(["a", "b"], ["a", "b"]), scored(["a", "b"], ["a", "x"]),
            scored(["a"], ["x"], "tts"), scored(["a"], ["a"], "tts")]
    result = metrics(rows)
    assert result["scored"] == 4 and result["reference_tokens"] == 6 and result["edits"] == 2
    assert result["per"] == pytest.approx(1 / 3)
    assert result["mean_clip_per"] == .375 and result["median_clip_per"] == .25
    # Both strata have residual sample variance .5: sqrt(2*.5 + 2*.5)/6.
    assert result["per_se"] == pytest.approx(math.sqrt(2) / 6)
    assert metrics([rows[0]])["per_se"] is None
    assert metrics([rows[0], rows[0]])["per_se"] == 0


def test_tense_supports_confusions_and_source_metrics():
    rows = [scored(["k͈", "p͈", "t͈", "s͈", "tɕ͈"], ["k͈", "p", "t", "s", "tɕ"]),
            scored(["k͈"], []), scored(["a"], ["a", "k͈"])]
    plan = dict(languages=["kor"], stage="contaminated", populations={"kor": {"source_population": {"film": 100}}})
    result = summarize(rows, plan)
    assert result["korean_tense"]["k͈"] == dict(plain_counterpart="k", reference_support=2,
        prediction_support=2, correct_matches=1, plain_substitutions=0, deletions=1, other_substitutions=0)
    assert result["korean_tense"]["p͈"]["plain_substitutions"] == 1
    assert {"reference": None, "prediction": "k͈", "count": 1} in result["korean_top_15_confusions"]
    assert result["languages"]["kor"]["sources"]["film"]["population"] == 100


def response(status, envelope):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(envelope).encode()
    result.headers = {"content-type": "application/json"}
    return result


def test_http_retry_only_transient_and_preserve_raw(monkeypatch):
    monkeypatch.setattr("pronunciation.inference.eval_per.time.sleep", lambda _: None)
    pending = [response(503, {"detail": "temporarily unavailable"}), response(200, IDENTITY)]
    log = io.StringIO()
    client = Client(IDENTITY, log, session=SimpleNamespace(post=lambda *a, **kw: pending.pop(0)))
    assert client.post("https://eval", {"marker_only": True}, []) == IDENTITY
    assert len(log.getvalue().splitlines()) == 2
    for status, envelope in [(422, {**IDENTITY, "detail": "bad"}), (200, {**IDENTITY, "load_error": "bad"}),
                             (503, {"detail": {**IDENTITY, "load_error": "bad"}}),
                             (503, {**IDENTITY, "deploy_marker": "stale"}), (200, {})]:
        pending = [response(status, envelope)]
        with pytest.raises((ValueError, requests.HTTPError)):
            client.post("https://eval", {}, [])
        assert not pending


def test_transport_retry_exhaustion_records_every_attempt(monkeypatch):
    monkeypatch.setattr("pronunciation.inference.eval_per.time.sleep", lambda _: None)
    def timeout(*args, **kwargs):
        raise requests.Timeout("timed out")
    log = io.StringIO()
    client = Client(IDENTITY, log, attempts=2, session=SimpleNamespace(post=timeout))
    with pytest.raises(requests.Timeout):
        client.post("https://eval", {}, [])
    assert len(log.getvalue().splitlines()) == 2


@pytest.mark.parametrize("value", [None, "1", "0"])
def test_modal_xet_passthrough_is_opt_in(monkeypatch, value):
    # Evaluate the actual image.env argument without importing Modal or downloading.
    import os
    path = Path(__file__).resolve().parents[2] / "modal" / "wav2vec2_phoneme.py"
    tree = ast.parse(path.read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "env")
    if value is None:
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_DISABLE_XET", value)
    scope = dict(os=os, MODEL_ID="m", MODEL_REVISION="r", DEPLOY_MARKER="d", GPU="g", BATCH_SIZE=8,
                 MAX_PADDED_SECONDS=120, MAX_LENGTH_RATIO=1.25, BATCH_DIAGNOSTICS=False, TRAINED_AGAINST_G2P=None)
    result = eval(compile(ast.Expression(call.args[0]), str(path), "eval"), scope)
    assert result.get("HF_HUB_DISABLE_XET") == value
    assert ("HF_HUB_DISABLE_XET" in result) == (value is not None)


def test_duration_header_fallback_keeps_valid_corpus_rows(tmp_path):
    import numpy as np
    import soundfile as sf
    directory = tmp_path / "kor"
    directory.mkdir()
    sf.write(directory / "a.wav", np.zeros(1600, dtype=np.float32), 16000)
    row = dict(file="a.wav", source="fleurs", phonemes=["a"])
    (directory / "phonemes_narrowed.jsonl").write_text(json.dumps(row))
    valid, audit = inspect_population(tmp_path, "kor")
    assert len(valid) == 1 and valid[0]["duration_sec"] == .1
    assert valid[0]["duration_origin"] == "audio_header"
    assert audit["manifest_fallback_counts"] == {"audio_header_duration": 1}
    assert audit["exclusions"] == []


def test_prepare_and_run_end_to_end_preserves_payload_and_results(tmp_path, monkeypatch):
    import base64
    import numpy as np
    import soundfile as sf
    from pronunciation.inference.eval_per import prepare, run, digest, file_digest

    directory = tmp_path / "audio" / "kor"
    directory.mkdir(parents=True)
    labels = []
    for i in range(2):
        sf.write(directory / f"{i}.wav", np.array([[.25, .75], [.5, .5]], dtype=np.float32),
                 16000, subtype="FLOAT")
        labels.append(dict(file=f"{i}.wav", source="tts", phonemes=["tɕ͈"], stress=[0]))
    (directory / "phonemes_narrowed.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
    plan = tmp_path / "plan.json"
    prepare(directory.parent, ["kor"], 2, 1, plan)
    def post(self, url, payload, clip_ids):
        if payload.get("marker_only"):
            return IDENTITY
        assert len(payload["requests"]) == 2
        for request in payload["requests"]:
            pcm = base64.b64decode(request["audio_f32_b64"])
            assert np.frombuffer(pcm, dtype="<f4").tolist() == [.5, .5]
            assert request["sample_rate"] == 16000 and request["language"] == "kor"
        return {**IDENTITY, "results": [{"phonemes": [{"phoneme": "ˈtɕ͈", "stress": 1}]}] * 2}
    monkeypatch.setattr(Client, "post", post)
    out = tmp_path / "output"
    run(plan, out, "https://single", "https://batch", IDENTITY, 32, 4, 240)
    rows = [json.loads(line) for line in (out / "stage1-per-clip.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(r["edits"] == 0 and r["identity"] == IDENTITY for r in rows)
    assert rows[0]["pcm_f32le_sha256"] == digest(np.array([.5, .5], dtype="<f4").tobytes())
    summary = json.loads((out / "stage1-summary.json").read_text())
    assert summary["languages"]["kor"]["reference_tokens"] == 2
    assert summary["plan_sha256"] == file_digest(plan)
    with pytest.raises(FileExistsError):
        run(plan, out, "https://single", "https://batch", IDENTITY, 32, 4, 240)


def test_singleton_stratum_uses_explicit_unstratified_fallback():
    rows = [scored(["a"], ["a"]), scored(["a"], ["b"]), scored(["a"], ["a"], "tiny")]
    result = metrics(rows)
    # Overall residual variance=1/3: sqrt(3 * 1/3)/3.
    assert result["per_se"] == pytest.approx(1 / 3)
    assert result["se_singleton_sources"] == ["tiny"]
    assert result["se_method"] == "unstratified_clip_delta_singleton_fallback"

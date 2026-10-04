"""Offline production-frame alignment, identity, batching and audit regressions."""
import base64
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zlib

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "espeak_audit"))
import production_alignment as alignment
import pitch_accent_audit as pitch
import measure_corpus as corpus

IDENTITY = alignment.ModelIdentity("owner/model", "revision-full-123")


def tensor(values, semantics):
    values = np.asarray(values, dtype="<f2")
    return dict(shape=list(values.shape), dtype="float16", encoding="zlib+base64",
                value_semantics=semantics,
                data=base64.b64encode(zlib.compress(values.tobytes())).decode())


def matrix(scores, labels=("a", "<pad>", "b"), blank=1, nonblank=None, legacy=False):
    phone = dict(tensor(scores, "joint_log_probability"), blank_id=blank)
    if legacy:
        return dict(phone, vocab=list(labels))
    if nonblank is None:
        nonblank = 1 - np.exp(np.asarray(scores)[:, blank])
    return dict(schema_version=1, producer=IDENTITY.as_dict(), heads={
        "phone": dict(phone, labels=list(labels)),
        "nonblank": tensor(nonblank, "sigmoid_probability"),
    })


def collapse(path, blank):
    return [x for i, x in enumerate(path) if x != blank and (i == 0 or x != path[i - 1])]


@pytest.mark.parametrize("targets", [[0, 2], [0, 0], [2], [2, 0, 2]])
def test_viterbi_matches_exhaustive_optimum(targets):
    scores = np.random.default_rng(15).normal(-3, 1, (6, 3))
    paths = [p for p in itertools.product(range(3), repeat=6) if collapse(p, 1) == targets]
    best = max(paths, key=lambda p: sum(scores[t, c] for t, c in enumerate(p)))
    expected = []
    for c, group in itertools.groupby(enumerate(best), key=lambda x: x[1]):
        indices = [t for t, _ in group]
        if c != 1:
            expected.append([indices[0] / 10, (indices[-1] + 1) / 10,
                             scores[indices, c].mean()])
    spans, error = alignment.viterbi_spans(scores, targets, 1, .6)
    assert error is None
    np.testing.assert_allclose(spans, expected)


def test_repeats_require_blank_and_empty_target_is_ok():
    scores = np.array([[-.1, -5], [-5, -.2], [-.3, -5]])
    spans, error = alignment.viterbi_spans(scores, [0, 0], 1, .3)
    assert error is None
    np.testing.assert_allclose(spans, [[0, .1, -.1], [.2, .3, -.3]])
    assert alignment.viterbi_spans(scores[:2], [0, 0], 1, .2) == (
        [], "target_too_long: L=2+rep1 > T=2")
    assert alignment.viterbi_spans(scores[:0], [], 1, 0) == ([], None)
    assert alignment.viterbi_spans(np.full((3, 2), -np.inf), [0], 1, .3) == (
        [], "no_alignment_path")


@pytest.mark.parametrize("legacy", [False, True])
def test_mapping_reading_and_spans_do_not_use_fixed_ids(legacy):
    # Blank joint probability wins, but the >.5 nonblank gate still emits a.
    scores = np.log([[.3, .25, .45, .9, 1], [.7, .1, .2, .9, 1],
                     [.05, .05, .9, .9, 1], [.3, .25, .45, .9, 1]])
    scores[:, 4] = -np.inf
    payload = matrix(scores, ["a", "b", "<pad>", "<unk>", "masked"], 2, legacy=legacy)
    result = alignment.align_matrix(payload, ["unknown", "a", "<unk>", "masked", "a"], .4)
    assert result["keep"] == [1, 4]
    assert result["reading"] == ["a", "a"]
    assert result["align_error"] is None
    assert len(result["spans"]) == 2
    np.testing.assert_allclose(np.array(result["spans"])[:, :2], [[.1, .2], [.3, .4]])
    assert alignment.align_matrix(payload, ["unknown"], .4)["spans"] == []


def test_schema_gate_half_is_blank_and_no_double_softmax():
    scores = np.log([[.2, .4, .1], [.2, .4, .1]])
    result = alignment.align_matrix(matrix(scores, nonblank=[.5, .6]), ["a"], .2)
    assert result["reading"] == ["a"]
    # Scores are already joint log-probs, not logits to normalize again.
    assert result["spans"][0][2] == pytest.approx(np.float16(np.log(.2)), abs=1e-6)


@pytest.mark.parametrize("version", [0, 2, None, True])
def test_unknown_schema_is_not_legacy(version):
    with pytest.raises(ValueError, match="schema"):
        alignment.decode_matrix({"schema_version": version})


def test_cache_uses_full_model_and_revision_offline(tmp_path, monkeypatch):
    cache = alignment.MeasurementCache(tmp_path, IDENTITY)
    cache.select()
    monkeypatch.setattr(alignment, "post", lambda *a: pytest.fail("offline cache made HTTP call"))
    assert alignment.MeasurementCache.latest(tmp_path) == cache
    for identity in [alignment.ModelIdentity("other/model", IDENTITY.model_revision),
                     alignment.ModelIdentity(IDENTITY.model_id, IDENTITY.model_revision + "changed")]:
        assert alignment.MeasurementCache(tmp_path, identity).path("jpn", "a", "p") != cache.path("jpn", "a", "p")
    assert list(tmp_path.rglob("*.json")) == []
    monkeypatch.setattr(corpus, "CACHE", tmp_path)
    assert corpus.cache_path("eng", "a", "p") == cache.path("eng", "a", "p")


def test_probe_uses_marker_only_and_rejects_load_error(monkeypatch):
    calls = []
    monkeypatch.setattr(alignment, "post", lambda url, body: calls.append((url, body)) or IDENTITY.as_dict())
    assert alignment.discover_identity() == IDENTITY
    assert calls == [(alignment.PREDICT_URL, {"marker_only": True})]
    with pytest.raises(RuntimeError, match="model load failed"):
        alignment.ModelIdentity.from_response(dict(IDENTITY.as_dict(), load_error="bad checkpoint"))
    with pytest.raises(ValueError, match="missing"):
        alignment.ModelIdentity.from_response({})


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros((1600, 2)), 16000)
    return path


@pytest.mark.parametrize("count", [1, 64])
def test_batch_contract_order_and_compact_audio(monkeypatch, audio, count):
    payload = matrix(np.log([[.7, .2, .1], [.1, .2, .7]]))
    def post(url, body):
        assert url == alignment.BATCH_URL
        assert len(body["requests"]) == count
        for request in body["requests"]:
            assert request["language"] == "jpn"
            assert request["sample_rate"] == 16000
            assert request["return_frame_matrix"] is True
            assert len(base64.b64decode(request["audio_f32_b64"])) == 1600 * 4
        return dict(IDENTITY.as_dict(), results=[{"frame_matrix": payload}] * count)
    monkeypatch.setattr(alignment, "post", post)
    targets = [["a"] if i % 2 == 0 else ["b", "a"] for i in range(count)]
    results = alignment.align_batch([(audio, "jpn", p) for p in targets], IDENTITY)
    assert [len(r["spans"]) for r in results] == list(map(len, targets))
    assert all(r["audio_sec"] == .1 for r in results)


@pytest.mark.parametrize("count", [0, 65])
def test_batch_limits_before_network(count, monkeypatch):
    monkeypatch.setattr(alignment, "post", lambda *a: pytest.fail("HTTP called"))
    with pytest.raises(ValueError, match="between 1 and 64"):
        alignment.align_batch([None] * count, IDENTITY)


@pytest.mark.parametrize("failure", ["envelope", "producer", "item", "missing", "count", "error"])
def test_batch_rejects_drift_and_errors(monkeypatch, audio, failure):
    payload = matrix(np.log([[.7, .2, .1]]))
    result = {"frame_matrix": payload}
    response = dict(IDENTITY.as_dict(), results=[result])
    if failure == "envelope":
        response["model_revision"] = "changed"
    elif failure == "producer":
        payload["producer"]["model_id"] = "other/model"
    elif failure == "item":
        result.update(IDENTITY.as_dict(), model_revision="changed")
    elif failure == "missing":
        del response["model_revision"]
    elif failure == "count":
        response["results"] = []
    else:
        result["error"] = {"message": "invalid audio"}
    monkeypatch.setattr(alignment, "post", lambda *a: response)
    with pytest.raises((RuntimeError, ValueError)):
        alignment.align_batch([(audio, "jpn", ["a"])], IDENTITY)


class InlinePool:
    def __init__(self, *a): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def imap_unordered(self, function, tasks, **kwargs): return map(function, tasks)


def test_measure_cache_resume_revision_drift_and_offline_verdict(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("multiprocessing.Pool", InlinePool)
    monkeypatch.setattr(pitch, "measure_clip", lambda *a: [])
    rows = [{"file": f"{i}.wav", "phonemes": ["a"], "pitch_accent": [dict(phrase=0, mora=0, level=1)]}
            for i in range(5)]
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(map(json.dumps, rows)))
    args = SimpleNamespace(labels=labels, limit=None, batch=2, concurrency=2,
                           measure_workers=1, cache_dir=tmp_path / "cache")
    calls = []
    live = [IDENTITY]
    drift = [False]
    monkeypatch.setattr(pitch, "discover_identity", lambda: calls.append("probe") or live[0])
    def batch(items, identity):
        calls.append(len(items))
        if drift[0]:
            raise RuntimeError("deployment drift")
        return [dict(spans=[[0, .1, -.1]], keep=[0], align_error=None,
                     reading=["a"], n_frames=5, audio_sec=.1) for _ in items]
    monkeypatch.setattr(pitch, "align_batch", batch)
    pitch.cmd_measure(args)
    assert calls[0] == "probe" and sorted(calls[1:]) == [1, 2, 2]
    cache = alignment.MeasurementCache.latest(args.cache_dir)
    records = list(cache.directory.rglob("*.json"))
    assert len(records) == 5
    assert json.loads(records[0].read_text())["keep"] == [0]
    calls.clear()
    pitch.cmd_measure(args)
    assert calls == ["probe"]
    live[0] = alignment.ModelIdentity(IDENTITY.model_id, "new-revision")
    drift[0] = True
    with pytest.raises(RuntimeError, match="drift"):
        pitch.cmd_measure(args)
    assert alignment.MeasurementCache.latest(args.cache_dir) == cache
    assert not alignment.MeasurementCache(args.cache_dir, live[0]).directory.exists()
    drift[0] = False
    pitch.cmd_measure(args)
    assert alignment.MeasurementCache.latest(args.cache_dir).identity == live[0]
    monkeypatch.setattr(pitch, "discover_identity", lambda: pytest.fail("verdict made HTTP call"))
    pitch.cmd_verdict(SimpleNamespace(cache_dir=args.cache_dir, agree_st=.75,
                      contra_st=1.5, max_contra_frac=.2, report=False, dry_run=True))
    assert "measured clips: 5" in capsys.readouterr().out


def test_measure_clip_keeps_original_indices_and_verdict(monkeypatch):
    monkeypatch.setattr(pitch.sf, "read", lambda *a: (np.zeros(1600), 16000))
    monkeypatch.setattr(pitch, "clip_f0", lambda *a: (
        np.array([.01, .02, .03, .21, .22, .23]), np.array([80, 80, 80, 84, 84, 84])))
    row = dict(file="unused.wav", pitch_accent=[dict(phrase=0, mora=1, level=0), None,
                                               dict(phrase=0, mora=2, level=1)])
    moras = pitch.measure_clip(row, [[0, .1, -.2], [.2, .3, -.1]], [0, 2])
    assert [m["index"] for m in moras] == [0, 2]
    assert [m["st"] for m in moras] == [80, 84]
    assert pitch.clip_verdict(moras) == (1, 0, 0)


def test_corpus_measurement_and_downstream_cache_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr("multiprocessing.Pool", InlinePool)
    monkeypatch.setattr(corpus, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(corpus, "AUDIO", tmp_path / "audio")
    monkeypatch.setattr(corpus, "discover_identity", lambda: IDENTITY)
    monkeypatch.setattr(corpus.sf, "read", lambda *a: (np.zeros(1600), 16000))
    monkeypatch.setattr(corpus, "load_targets", lambda lang: iter([
        ("a.wav", "", ["a", "unknown", "n"], [0, 0, 0], "phones-key")]))
    spans = [[0, .02, -.1], [.05, .07, -.2]]
    monkeypatch.setattr(corpus, "align_batch", lambda items, identity: [dict(
        spans=spans, keep=[0, 2], align_error=None, reading=["a", "n"],
        n_frames=5, audio_sec=.1)])
    def measure(audio, sr, phones, stress, word_of, ok, keep, actual_spans):
        assert ok == [True, False, True]
        assert keep == [0, 2] and actual_spans == spans
        return [{"idx": 0, "symbol": "a", "start": 0, "end": .02}]
    monkeypatch.setitem(sys.modules, "phonetics", SimpleNamespace(measure_segments=measure))
    monkeypatch.setattr(sys, "argv", ["measure_corpus.py", "--langs", "eng", "--limit", "1"])
    corpus.main()
    path = corpus.cache_path("eng", "a.wav", "phones-key")
    assert json.loads(path.read_text())["segments"][0]["idx"] == 0
    monkeypatch.setattr(corpus, "align_batch", lambda *a: pytest.fail("cached clip inferred"))
    corpus.main()

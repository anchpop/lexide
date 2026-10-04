"""Production frame matrices -> local CTC boundaries, never acoustic verdicts.

HTTP URLs and the unauthenticated JSON contract match pronunciation/remote.rs.
No Modal SDK, private checkpoint download, or GPU alignment deployment is needed.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import urllib.request
import zlib

import numpy as np
import soundfile as sf

PREDICT_URL = "https://anchpop--wav2vec2-phoneme-wav2vec2phoneme-predict.modal.run"
BATCH_URL = PREDICT_URL.replace("-predict.modal.run", "-predict-batch.modal.run")


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    model_revision: str

    @classmethod
    def from_response(cls, response):
        if "load_error" in response:
            raise RuntimeError(f"model load failed: {response['load_error']}")
        fields = [response.get(k) for k in ("model_id", "model_revision")]
        if not all(isinstance(v, str) and v for v in fields):
            raise ValueError("response is missing model_id/model_revision")
        return cls(*fields)

    def as_dict(self):
        return dict(model_id=self.model_id, model_revision=self.model_revision)

    @property
    def namespace(self):
        # Hash BOTH full fields, not a shortened revision or a model basename.
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    def check(self, response):
        actual = self.from_response(response)
        if actual != self:
            raise RuntimeError(f"deployment drift: expected {self}, received {actual}")


@dataclass(frozen=True)
class MeasurementCache:
    root: Path
    identity: ModelIdentity

    @property
    def directory(self):
        return self.root / self.identity.namespace

    def path(self, lang, file, phon_key):
        key = hashlib.sha256(json.dumps([lang, file, phon_key]).encode()).hexdigest()
        return self.directory / key[:2] / f"{key}.json"

    def select(self):
        """Publish only after a successful measure; offline consumers need no probe.

        Metadata is outside the namespace record tree and not a *.json record.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.root / "latest.tmp"
        temp.write_text(json.dumps(self.identity.as_dict()))
        temp.replace(self.root / "latest")

    @classmethod
    def latest(cls, root):
        root = Path(root)
        try:
            identity = ModelIdentity.from_response(json.loads((root / "latest").read_text()))
        except FileNotFoundError:
            raise RuntimeError(f"no successful measurement under {root}; run measure first") from None
        return cls(root, identity)


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def discover_identity():
    return ModelIdentity.from_response(post(PREDICT_URL, {"marker_only": True}))


def audio_request(path, language):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return {
        "audio_f32_b64": base64.b64encode(audio.astype("<f4").tobytes()).decode(),
        "sample_rate": int(sr), "language": language, "return_frame_matrix": True,
    }, len(audio) / sr


def _tensor(head, semantics=None):
    if head["dtype"] != "float16" or head["encoding"] != "zlib+base64":
        raise ValueError("unsupported frame matrix encoding")
    if semantics is not None and head["value_semantics"] != semantics:
        raise ValueError(f"expected {semantics} frame matrix")
    raw = zlib.decompress(base64.b64decode(head["data"], validate=True))
    return np.frombuffer(raw, dtype="<f2").reshape(head["shape"]).astype(np.float32)


def decode_matrix(matrix):
    """Read schema 1 and the legacy phone-only payload still served by old deploys."""
    if "schema_version" not in matrix:
        phone = matrix
        labels = phone["vocab"]
        scores = _tensor(phone)
        nonblank = None
    elif type(matrix["schema_version"]) is int and matrix["schema_version"] == 1:
        phone = matrix["heads"]["phone"]
        labels = phone["labels"]
        scores = _tensor(phone, "joint_log_probability")
        nonblank = _tensor(matrix["heads"]["nonblank"], "sigmoid_probability")
    else:
        raise ValueError(f"unsupported frame matrix schema {matrix['schema_version']!r}")
    blank = phone["blank_id"]
    if (scores.ndim != 2 or scores.shape[1] != len(labels)
            or not 0 <= blank < len(labels) or len(set(labels)) != len(labels)):
        raise ValueError("invalid phone matrix dimensions/labels/blank")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("invalid phone log probabilities")
    if nonblank is not None and (nonblank.shape != (len(scores),)
            or not np.all((nonblank >= 0) & (nonblank <= 1))):
        raise ValueError("invalid nonblank probabilities")
    return scores, labels, blank, nonblank


def viterbi_spans(scores, targets, blank, duration):
    """Standard CTC max-sum path; spans exclude all surrounding blank frames."""
    frames = len(scores)
    repeats = sum(a == b for a, b in zip(targets, targets[1:]))
    if not targets:
        return [], None
    if len(targets) + repeats > frames:
        return [], f"target_too_long: L={len(targets)}+rep{repeats} > T={frames}"
    labels = np.full(2 * len(targets) + 1, blank, dtype=np.intp)
    labels[1::2] = targets
    # A +2 jump skips a blank only between DISTINCT target labels.
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    previous = np.full(len(labels), -np.inf)
    previous[:2] = scores[0, labels[:2]]
    back = np.zeros((frames, len(labels)), dtype=np.int8)
    states = np.arange(len(labels))
    for t in range(1, frames):
        advance = np.r_[-np.inf, previous[:-1]]
        jump = np.r_[-np.inf, -np.inf, previous[:-2]]
        jump[~skip] = -np.inf
        candidates = np.stack([previous, advance, jump])
        back[t] = candidates.argmax(axis=0)
        previous = candidates[back[t], states] + scores[t, labels]
    state = len(labels) - 2 + int(previous[-1] > previous[-2])
    if not np.isfinite(previous[state]):
        return [], "no_alignment_path"
    path = np.empty(frames, dtype=np.intp)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    step = duration / frames
    spans = []
    for i, target in enumerate(targets):
        emitted = np.flatnonzero(path == 2 * i + 1)
        spans.append([float(emitted[0] * step), float((emitted[-1] + 1) * step),
                      float(scores[emitted, target].mean())])
    return spans, None


def align_matrix(matrix, phonemes, duration):
    scores, labels, blank, nonblank = decode_matrix(matrix)
    phone_ids = [i for i, label in enumerate(labels)
                 if i != blank and not (label.startswith("<") and label.endswith(">"))
                 and np.isfinite(scores[:, i]).any()]
    vocab = {labels[i]: i for i in phone_ids}
    keep = [i for i, phone in enumerate(phonemes) if phone in vocab]
    spans, error = viterbi_spans(scores, [vocab[phonemes[i]] for i in keep], blank, duration)
    reading = []
    if phone_ids and len(scores):
        chosen = np.asarray(phone_ids)[scores[:, phone_ids].argmax(axis=1)]
        # Legacy has no separate gate. P(blank) >= .5 means blank.
        gate = nonblank > .5 if nonblank is not None else scores[:, blank] < np.log(.5)
        chosen[~gate] = blank
        previous = None
        for index in chosen:
            if index != blank and index != previous:
                reading.append(labels[index])
            previous = index
    return dict(spans=spans, keep=keep, align_error=error, reading=reading,
                n_frames=len(scores), audio_sec=duration)


def align_batch(items, identity):
    """Items are (audio path, language, target phones); results preserve order.

    Validate every identity before returning any cacheable work. Network/item
    errors fail the run (not cached as acoustic exclusions); restart resumes.
    """
    if not 1 <= len(items) <= 64:
        raise ValueError("batch must contain between 1 and 64 clips")
    prepared = [audio_request(path, lang) for path, lang, _ in items]
    response = post(BATCH_URL, {"requests": [request for request, _ in prepared]})
    identity.check(response)
    results = response["results"]
    if len(results) != len(items):
        raise ValueError(f"batch returned {len(results)} results for {len(items)} clips")
    aligned = []
    for result, (_, _, phones), (_, duration) in zip(results, items, prepared):
        if "error" in result:
            raise RuntimeError(f"prediction failed: {result['error']}")
        if "model_id" in result or "model_revision" in result:
            identity.check(result)
        matrix = result["frame_matrix"]
        if "schema_version" in matrix:
            identity.check(matrix["producer"])
        aligned.append(align_matrix(matrix, phones, duration))
    return aligned


def validate_options(args):
    if not 1 <= args.batch <= 64:
        raise ValueError("--batch must be between 1 and 64")
    if args.concurrency < 1 or args.measure_workers < 1:
        raise ValueError("--concurrency and --measure-workers must be positive")

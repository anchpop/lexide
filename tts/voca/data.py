"""Training data: acoustic feature files joined with the annotation layer.

Tokens, stress and durations come ONLY from an annotation manifest written by
``tiny_voca.annotate``. The ``tokens`` tensor inside a feature file is a stale
byte encoding from the old front end and is never read here.
"""

import hashlib
import json
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence

from .text import PHONE_IDS, SYMBOL_TO_ID, VOWELS, frontend_fingerprint

VOWEL_IDS = frozenset(SYMBOL_TO_ID[v] for v in VOWELS)


def continuous_pitch(pitch, voiced, default=0.0, window=5, tolerance=0.45):
    """Clean a pitch track and make it continuous.

    ``pitch`` is log(F0/reference) on voiced frames. Values further than
    ``tolerance`` (about 0.65 octave) from the local median are tracker errors
    (octave jumps); they stay voiced but their value is not trusted. Untrusted
    and unvoiced frames are filled by linear interpolation so the oscillator and
    the pitch predictor see one smooth curve. Returns ``(curve, trusted)``.
    """
    voiced = voiced > 0.5
    trusted = voiced.clone()
    if int(voiced.sum()) >= 3:
        values = pitch[voiced]
        padded = torch.nn.functional.pad(values[None, None], (window // 2, window // 2), mode="replicate")[0, 0]
        median = padded.unfold(0, window, 1).median(dim=1).values
        trusted[voiced.nonzero().squeeze(1)[(values - median).abs() > tolerance]] = False
    index = trusted.nonzero().squeeze(1)
    if index.numel() == 0:
        return torch.full_like(pitch, default), trusted.float()
    position = torch.arange(pitch.numel())
    right = torch.searchsorted(index, position).clamp(max=index.numel() - 1)
    left = (torch.searchsorted(index, position, right=True) - 1).clamp(min=0)
    a, b = index[left], index[right]
    span = (b - a).clamp(min=1).float()
    weight = ((position - a).float() / span).clamp(0, 1)
    curve = pitch[a] * (1 - weight) + pitch[b] * weight
    return curve, trusted.float()


class AnnotatedDataset:
    def __init__(self, manifest, config, acoustic_root=None, cache=True, voiced_vowels=False):
        self.manifest = Path(manifest)
        metadata_path = self.manifest.parent / "annotations.json"
        if not metadata_path.exists():
            raise ValueError("not an annotation manifest (legacy byte-token manifests are stale): run tiny_voca.annotate")
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get("complete"):
            raise ValueError("annotation run is incomplete; finish it with --resume before training")
        self.annotation = metadata["fingerprint"]
        self.root = self.manifest.parent / (acoustic_root or metadata["acoustic_root"])   # relative to the manifest
        self.fingerprint = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        # Physical LF-delimited records: Unicode separators inside text are data.
        self.rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").split("\n") if line.strip()]
        self.config = config
        if not self.rows:
            raise ValueError("empty dataset")
        frontend = frontend_fingerprint()
        for row in self.rows:
            if "durations" not in row or row.get("frontend") != frontend:
                raise ValueError("tokens were produced by a different front end; re-run annotation")
            if row.get("acoustic") != config.acoustic_fingerprint:
                raise ValueError("acoustic feature settings differ from this configuration")
            if row.get("annotation") != self.annotation:
                raise ValueError("manifest rows come from a different annotation run")
            if len(row["tokens"]) != len(row["durations"]) or len(row["tokens"]) != len(row["stress"]):
                raise ValueError(f"token/duration length mismatch: {row['id']}")
            if sum(row["durations"]) != row["frames"] or min(row["durations"]) < 0:
                raise ValueError(f"durations do not cover the frames exactly: {row['id']}")
            if any(d < 1 for t, d in zip(row["tokens"], row["durations"]) if t in PHONE_IDS):
                raise ValueError(f"phoneme without frames: {row['id']}")
            if not 0 < len(row["tokens"]) <= config.max_tokens or not 0 < row["frames"] <= config.max_frames:
                raise ValueError(f"utterance exceeds model limits: {row['id']}")
            if max(row["tokens"]) >= config.vocab_size or min(row["tokens"]) < 1 or row["speaker"] not in config.speakers:
                raise ValueError(f"invalid token or speaker: {row['id']}")
        ids = [r["id"] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate IDs in manifest")
        self.lengths = [r["frames"] for r in self.rows]
        self._cache = {} if cache else None
        self.voiced_vowels = voiced_vowels

    def __len__(self):
        return len(self.rows)

    def preload(self, indices=None, workers=8):
        """Read feature files into memory up front (about 10 ms each otherwise
        lands on the first training steps). Needs ``cache=True``."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(self.__getitem__, range(len(self)) if indices is None else indices))

    def __getitem__(self, index):
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        row, c = self.rows[index], self.config
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("feature file escapes the acoustic root")
        stored = torch.load(path, map_location="cpu", weights_only=True)
        t = row["frames"]
        if stored["mel"].shape != (t, c.n_mels) or stored["prosody"].shape != (t, 3) or stored["wave"].shape != (t * c.hop,):
            raise ValueError(f"feature dimensions differ from the manifest: {path.name}")
        if any(not torch.isfinite(stored[k]).all() for k in ("mel", "prosody", "wave")):
            raise ValueError("non-finite features")
        pitch, trusted = continuous_pitch(stored["prosody"][:, 0], stored["prosody"][:, 1])
        voiced = stored["prosody"][:, 1]
        if self.voiced_vowels:
            # The pitch tracker misses low, creaky vowels (12% of M2's vowel frames),
            # and a model trained on those labels drops whole vowels. A vowel is voiced
            # by definition: mark its frames voiced. Their pitch stays interpolated and
            # untrusted, so the pitch loss still ignores what the tracker did not measure.
            vowel = torch.repeat_interleave(torch.tensor([t in VOWEL_IDS for t in row["tokens"]]), torch.tensor(row["durations"]))
            voiced = torch.maximum(voiced, vowel.to(voiced.dtype))
        item = {
            "tokens": torch.tensor(row["tokens"], dtype=torch.long), "stress": torch.tensor(row["stress"], dtype=torch.long),
            "durations": torch.tensor(row["durations"], dtype=torch.long), "speaker": c.speakers.index(row["speaker"]),
            "wave": stored["wave"], "mel": stored["mel"],
            # Continuous log pitch, voiced flag, normalized log energy.
            "prosody": torch.stack((pitch, voiced, (stored["prosody"][:, 2] - c.energy_mean) / c.energy_scale), dim=-1),
            "pitch_trusted": trusted,
        }
        if self._cache is not None:
            self._cache[index] = item
        return item


def collate(items, device="cpu"):
    """A padded batch on ``device``; ``batch["host"]`` keeps the lengths on the CPU,
    so shapes and LSTM packing never wait for the GPU."""
    batch = {key: pad_sequence([x[key] for x in items], batch_first=True)
             for key in ("tokens", "stress", "durations", "mel", "prosody", "pitch_trusted", "wave")}
    batch["token_lengths"] = torch.tensor([x["tokens"].numel() for x in items])
    batch["frame_lengths"] = torch.tensor([x["mel"].shape[0] for x in items])
    batch["speakers"] = torch.tensor([x["speaker"] for x in items])
    host = {"token_lengths": batch["token_lengths"], "frame_lengths": batch["frame_lengths"]}
    pin = torch.device(device).type == "cuda"
    batch = {key: (value.pin_memory() if pin else value).to(device, non_blocking=True) for key, value in batch.items()}
    return {**batch, "host": host}


def prefetch(make, depth=2):
    """``make()`` batches in a background thread, ``depth`` ahead of the consumer.
    Order is preserved, so a seeded sampler gives the same data as without it."""
    import queue
    import threading
    ready = queue.Queue(depth)

    def work():
        try:
            while True:
                ready.put(make())
        except BaseException as error:
            ready.put(error)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = ready.get()
        if isinstance(item, BaseException):
            raise item
        yield item


class BucketSampler:
    """Random batches of similar length under a padded-frame budget.

    Sampling is with replacement from pools; the generator state and the
    not-yet-used batches are checkpointed, so a resumed run sees the same data.
    """

    def __init__(self, lengths, frame_budget, max_batch, seed, pool=256):
        self.lengths, self.frame_budget, self.max_batch, self.pool = lengths, frame_budget, max_batch, pool
        self.generator = torch.Generator().manual_seed(seed)
        self.pending = []

    def next(self):
        if not self.pending:
            indices = sorted(torch.randint(len(self.lengths), (self.pool,), generator=self.generator).tolist(), key=lambda i: self.lengths[i])
            batches, current = [], []
            for index in indices:
                if current and (len(current) == self.max_batch or (len(current) + 1) * self.lengths[index] > self.frame_budget):
                    batches.append(current)
                    current = []
                current.append(index)
            batches.append(current)
            self.pending = [batches[i] for i in torch.randperm(len(batches), generator=self.generator).tolist()]
        return self.pending.pop()

    def state(self):
        return {"generator": self.generator.get_state(), "pending": [list(b) for b in self.pending]}

    def load_state(self, state):
        self.generator.set_state(state["generator"].cpu())
        self.pending = [list(b) for b in state["pending"]]

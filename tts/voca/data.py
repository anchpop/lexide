"""Training data: acoustic feature files joined with the annotation layer.

Tokens, their features (``voca.text.LEXICAL``), word occurrences and durations
come ONLY from an annotation manifest written by ``voca.build``. Word-level
prosody targets (``word_prosody``) are measured from the recording when an
item is first loaded.
"""

import collections
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

from .model import word_layout
from .text import ALWAYS_VOICED, LANGUAGE, LEXICAL, PHONE_IDS, SONORANTS, SYMBOL_TO_ID, VOWELS, frontend_fingerprint

ALWAYS_VOICED_IDS = frozenset(SYMBOL_TO_ID[v] for v in ALWAYS_VOICED)
SONORANT_IDS = frozenset(SYMBOL_TO_ID[v] for v in SONORANTS)
VOWEL_IDS = frozenset(SYMBOL_TO_ID[v] for v in VOWELS)
SEMITONES = 12 / math.log(2)
# Contour shapes of a word's trusted pitch, from the medians of its first, middle
# and last third: a middle this far above both ends is a rise-fall, below both a
# fall-rise; otherwise the ends decide.
SHAPES = ("level", "rise", "fall", "rise-fall", "fall-rise")
SHAPE_SEMITONES = 1.0
# Word-level prosody targets (``word_prosody``), the columns of item["word_prosody"].
WORD_TARGETS = ("prominence", "shape", "excursion", "final")
# Frames where the second tracker (voca.prepare.praat_pitch) is voiced too but
# this far from pYIN: one of them made an error, usually an octave. Measured on
# the 10 h Gemini set, 99% of such frames agree within 1.2 semitones and the
# rest mostly differ by an octave, so the threshold barely matters.
DISPUTED_SEMITONES = 2.0


def despiked(pitch, voiced, window=5, tolerance=0.45):
    """The voiced frames whose log pitch is within ``tolerance`` (about 0.65
    octave) of the median of the ``window`` voiced values around it; the others
    are tracker errors (octave jumps)."""
    trusted = voiced.clone()
    if int(voiced.sum()) >= 3:
        values = pitch[voiced]
        padded = torch.nn.functional.pad(values[None, None], (window // 2, window // 2), mode="replicate")[0, 0]
        median = padded.unfold(0, window, 1).median(dim=1).values
        trusted[voiced.nonzero().squeeze(1)[(values - median).abs() > tolerance]] = False
    return trusted


def continuous_pitch(pitch, voiced, check, default=0.0):
    """Clean a pitch track and make it continuous.

    ``pitch`` is log(F0/reference) on voiced frames. Where pYIN found no voicing
    but the second tracker ``check`` ([T, 2] log pitch, voiced: Praat) did, its
    pitch is taken and the frame is voiced: pYIN drops half of all
    sentence-final vowel frames (falling, creaky pitch; Gemini Neon's audit), and
    Praat tracks a third of those. Values the despiker rejects stay voiced but are
    not trusted, and neither are values the second tracker disputes. Untrusted and
    unvoiced frames are filled by linear interpolation so the oscillator and
    the pitch predictor see one smooth curve. Returns ``(curve, trusted, voiced)``.
    """
    adopted = (voiced <= 0.5) & (check[:, 1] > 0.5)
    pitch, voiced = torch.where(adopted, check[:, 0], pitch), (voiced > 0.5) | adopted
    trusted = despiked(pitch, voiced) & ~((check[:, 1] > 0.5) & ((pitch - check[:, 0]).abs() > DISPUTED_SEMITONES * math.log(2) / 12))
    index = trusted.nonzero().squeeze(1)
    if index.numel() == 0:
        return torch.full_like(pitch, default), trusted.float(), voiced.float()
    position = torch.arange(pitch.numel())
    right = torch.searchsorted(index, position).clamp(max=index.numel() - 1)
    left = (torch.searchsorted(index, position, right=True) - 1).clamp(min=0)
    a, b = index[left], index[right]
    span = (b - a).clamp(min=1).float()
    weight = ((position - a).float() / span).clamp(0, 1)
    curve = pitch[a] * (1 - weight) + pitch[b] * weight
    return curve, trusted.float(), voiced.float()


def voiced_like(energy, mel):
    """Frames whose audio could be voicing [T]: louder than ``energy`` -0.3 (normalized; about
    -40 dBFS) and with the low mel bands 2 above the high ones (log units). Vowel and
    sonorant frames pass 94%, voiceless frication 26% (its median tilt is -1.4 against
    vowels' +6), stop closures fail on energy."""
    return (energy > -0.3) & (mel[:, :25].mean(-1) > mel[:, 55:].mean(-1) + 2.0)


def duration_reference(rows, vocabulary):
    """[vocabulary, stress levels]: the mean log frames of each phone at each
    stress level over the manifest ``rows`` (the overall mean where unseen)."""
    total, count = np.zeros((vocabulary, LEXICAL["stress"])), np.zeros((vocabulary, LEXICAL["stress"]))
    for row in rows:
        durations = np.array(row["durations"])
        spoken = durations > 0
        cell = (np.array(row["tokens"])[spoken], np.array(row["lexical"])[spoken, 0])
        np.add.at(total, cell, np.log(durations[spoken]))
        np.add.at(count, cell, 1)
    return np.where(count > 0, total / np.maximum(count, 1), total.sum() / count.sum())


def standardized(x):
    """Columns z-scored; a missing value counts as its column's lowest (0 when all are missing)."""
    x = np.nan_to_num(np.where(np.isnan(x), np.fmin.reduce(x, axis=0), x))
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


def contour_shape(early, middle, late):
    """The ``SHAPES`` index of a contour from three pitch landmarks (semitones)."""
    if middle - max(early, late) > SHAPE_SEMITONES:
        return 3
    if min(early, late) - middle > SHAPE_SEMITONES:
        return 4
    return 1 if late - early > SHAPE_SEMITONES else 2 if early - late > SHAPE_SEMITONES else 0


def word_prosody(tokens, stress, durations, pitch, trusted, energy, reference):
    """Per word occurrence of a recording (``voca.model.word_layout``), [W, 4]
    (``WORD_TARGETS``), NaN where it cannot be measured. Pitch is the trusted frames only.

    * prominence: within its phrase, the z-scored sum of the z-scored peak pitch,
      peak vowel energy and longest vowel (log frames minus ``reference``, the
      mean for that phone and stress, so a naturally long vowel has no head
      start): an acoustic proxy for where the accent is. 0 alone in a phrase;
    * shape: the ``SHAPES`` class of its pitch (6+ frames);
    * excursion: P90 - P10 of its pitch in semitones (4+ frames);
    * final: on the last word of a phrase, the median of its last third of pitch
      minus that of its first third (``voca.evaluate``'s final movement), semitones.
    """
    layout = word_layout(tokens[None], torch.ones(1, len(tokens), dtype=torch.bool))
    count = int(layout.mask.sum())
    index, phrase, final = layout.index[0].numpy(), layout.phrase[0, :count].numpy(), layout.final[0, :count].numpy()
    tokens, stress, durations = tokens.numpy(), stress.numpy(), durations.numpy()
    bounds = np.concatenate(([0], np.cumsum(durations)))
    semitones, trusted, energy = pitch.numpy() * SEMITONES, trusted.numpy() > 0.5, energy.numpy()
    out, cues = np.full((count, len(WORD_TARGETS)), np.nan), np.full((count, 3), np.nan)
    for w in range(count):
        members = np.flatnonzero(index == w)
        span = slice(bounds[members[0]], bounds[members[-1] + 1])
        p = semitones[span][trusted[span]]
        vowels = [k for k in members if tokens[k] in VOWEL_IDS]
        vowel_frames = np.concatenate([np.arange(bounds[k], bounds[k + 1]) for k in vowels]) if vowels else np.arange(span.start, span.stop)
        cues[w] = (p.max() if len(p) else np.nan, energy[vowel_frames].max(),
                   max(math.log(durations[k]) - reference[tokens[k], stress[k]] for k in vowels) if vowels else np.nan)
        if len(p) >= 4:
            out[w, 2] = np.subtract(*np.quantile(p, [0.9, 0.1]))
        if len(p) >= 6:
            out[w, 1] = contour_shape(*(np.median(third) for third in np.array_split(p, 3)))
        third = len(p) // 3
        if final[w] and third >= 2:
            out[w, 3] = np.median(p[-third:]) - np.median(p[:third])
    for key in np.unique(phrase):
        members = phrase == key
        out[members, 0] = standardized(standardized(cues[members]).sum(1))
    return torch.from_numpy(out).float()


class AnnotatedDataset:
    """``reference``: the phone durations prominence is measured against
    (``duration_reference``); a held-out set takes its training set's."""

    def __init__(self, manifest, config, acoustic_root=None, cache=True, voiced_vowels=False, reference=None):
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
                raise ValueError(f"tokens were produced by a different front end (this one is {LANGUAGE}'s, VOCA_LANGUAGE); re-run annotation")
            if row.get("acoustic") != config.acoustic_fingerprint:
                raise ValueError("acoustic feature settings differ from this configuration")
            if row.get("annotation") != self.annotation:
                raise ValueError("manifest rows come from a different annotation run")
            if not len(row["tokens"]) == len(row["durations"]) == len(row["lexical"]):
                raise ValueError(f"token/duration length mismatch: {row['id']}")
            if any(not 0 <= v < size for features in row["lexical"] for v, size in zip(features, LEXICAL.values(), strict=True)):
                raise ValueError(f"lexical feature out of range: {row['id']}")
            if sum(row["durations"]) != row["frames"] or min(row["durations"]) < 0:
                raise ValueError(f"durations do not cover the frames exactly: {row['id']}")
            if any(d < 1 for t, d in zip(row["tokens"], row["durations"]) if t in PHONE_IDS):
                raise ValueError(f"phoneme without frames: {row['id']}")
            if not 0 < len(row["tokens"]) <= config.max_tokens or not 0 < row["frames"] <= config.max_frames:
                raise ValueError(f"utterance exceeds model limits: {row['id']}")
            if max(row["tokens"]) >= config.vocab_size or min(row["tokens"]) < 1 or row["speaker"] not in config.speakers:
                raise ValueError(f"invalid token or speaker {row['speaker']!r} (the configuration's: {config.speakers}): {row['id']}")
        ids = [r["id"] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate IDs in manifest")
        self.lengths = [r["frames"] for r in self.rows]
        self.token_counts = [len(r["tokens"]) for r in self.rows]
        self.speakers = [r["speaker"] for r in self.rows]
        self._cache = {} if cache else None
        self.voiced_vowels = voiced_vowels
        self.reference = duration_reference(self.rows, config.vocab_size) if reference is None else reference

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
        if (stored["mel"].shape != (t, c.n_mels) or stored["prosody"].shape != (t, 3) or stored["pitch_check"].shape != (t, 2)
                or stored["wave"].shape != (t * c.hop,)):
            raise ValueError(f"feature dimensions differ from the manifest: {path.name}")
        if any(not torch.isfinite(stored[k]).all() for k in ("mel", "prosody", "pitch_check", "wave")):
            raise ValueError("non-finite features")
        pitch, trusted, voiced = continuous_pitch(stored["prosody"][:, 0], stored["prosody"][:, 1], stored["pitch_check"])
        energy = (stored["prosody"][:, 2] - c.energy_mean) / c.energy_scale
        if self.voiced_vowels:
            # Both trackers miss low, creaky voicing, and a model trained on those labels drops
            # whole vowels and voiced consonants, which the source then renders as noise bursts
            # inside the word. Vowels and sonorants are voiced by definition (the language's
            # inventory says which): mark their frames voiced where the audio is voiced-like
            # (``voiced_like``), so a stop closure or frication the alignment gave them is not.
            # Their pitch stays interpolated and untrusted, so the pitch loss ignores what no tracker measured.
            phones = torch.tensor([t in ALWAYS_VOICED_IDS or t in SONORANT_IDS for t in row["tokens"]])
            forced = torch.repeat_interleave(phones, torch.tensor(row["durations"])) & voiced_like(energy, stored["mel"])
            voiced = torch.maximum(voiced, forced.to(voiced.dtype))
        tokens, lexical, durations = (torch.tensor(row[key], dtype=torch.long) for key in ("tokens", "lexical", "durations"))
        item = {
            "tokens": tokens, "lexical": lexical, "durations": durations, "speaker": c.speakers.index(row["speaker"]),
            "wave": stored["wave"], "mel": stored["mel"],
            # Continuous log pitch, voiced flag, normalized log energy.
            "prosody": torch.stack((pitch, voiced, energy), dim=-1),
            "pitch_trusted": trusted,
            "word_prosody": word_prosody(tokens, lexical[:, 0], durations, pitch, trusted, energy, self.reference),
        }
        if self._cache is not None:
            self._cache[index] = item
        return item


def collate(items, shape=None):
    """A padded batch on the CPU: to the longest item, or to ``shape`` (a
    sampler's fixed tokens and frames). ``host`` repeats the lengths and the
    longest window every item fits, for shapes decided without the GPU."""
    tokens = shape["tokens"] if shape else max(x["tokens"].numel() for x in items)
    frames = shape["frames"] if shape else max(x["mel"].shape[0] for x in items)
    hop = items[0]["wave"].numel() // items[0]["mel"].shape[0]

    def pad(key, size):
        out = items[0][key].new_zeros((len(items), size, *items[0][key].shape[1:]))
        for row, item in zip(out, items):
            row[:len(item[key])] = item[key]
        return out
    batch = {key: pad(key, tokens) for key in ("tokens", "lexical", "durations", "word_prosody")}   # words: room for one per token
    batch.update({key: pad(key, frames) for key in ("mel", "prosody", "pitch_trusted")}, wave=pad("wave", frames * hop))
    batch["token_lengths"] = torch.tensor([x["tokens"].numel() for x in items])
    batch["frame_lengths"] = torch.tensor([x["mel"].shape[0] for x in items])
    batch["speakers"] = torch.tensor([x["speaker"] for x in items])
    shortest = shape["shortest"] if shape else int(batch["frame_lengths"].min())
    return {**batch, "host": {"token_lengths": batch["token_lengths"], "frame_lengths": batch["frame_lengths"], "shortest": shortest}}


def pin(batch):
    return {key: value if key == "host" else value.pin_memory() for key, value in batch.items()}


def to_device(batch, device):
    """Pinned (``pin``) batches copy without making the CPU wait."""
    return {key: value if key == "host" else value.to(device, non_blocking=True) for key, value in batch.items()}


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


class TemperedSampler:
    """Items drawn with replacement: group g (a voice) gets a share of the draws
    proportional to its total ``sizes`` to the power ``alpha``, spread evenly
    over its items. alpha 1 is uniform by size, 0 gives every group the same
    share; between, as multilingual ASR balances languages, a small group is
    seen more often than its size alone would give without being repeated as
    often as a large one is seen. ``shares``: each group's expected share of the draws."""

    def __init__(self, groups, sizes, alpha, seed):
        totals, counts = collections.Counter(), collections.Counter(groups)
        for group, size in zip(groups, sizes, strict=True):
            totals[group] += size
        weight = {g: totals[g] ** alpha for g in totals}
        self.shares = {g: weight[g] / sum(weight.values()) for g in sorted(totals)}
        cdf = torch.tensor([self.shares[g] / counts[g] for g in groups], dtype=torch.float64).cumsum(0)
        self.cdf = cdf / cdf[-1]
        self.generator = torch.Generator().manual_seed(seed)

    def draw(self, count):
        """``count`` item indices, a LongTensor."""
        u = torch.rand(count, generator=self.generator, dtype=torch.float64)
        return torch.searchsorted(self.cdf, u).clamp(max=len(self.cdf) - 1)


class BucketSampler:
    """Random batches of fixed shapes, drawn with replacement (``TemperedSampler``
    over ``groups``, each utterance's speaker; by frames).

    Utterances fall into ``buckets`` length ranges of about equal counts. Each
    bucket has one shape: frames and tokens padded to its longest utterance,
    ``max_batch`` utterances (fewer when that would exceed ``frame_budget``
    padded frames), and ``shortest``, its shortest utterance (the longest window
    every batch from it fits). A drawn utterance waits in its bucket until that
    bucket has a full batch. A handful of shapes is what lets the training step
    be replayed from CUDA graphs; the weights only change which utterances are
    drawn, never a shape. The generator state and the waiting utterances are
    checkpointed, so a resumed run sees the same data.
    """

    def __init__(self, lengths, tokens, groups, frame_budget, max_batch, seed, alpha, buckets=8):
        order = sorted(range(len(lengths)), key=lambda i: (lengths[i], tokens[i]))
        buckets = min(buckets, len(order))
        self.bucket, self.shapes = [0] * len(order), []
        for k in range(buckets):
            members = order[k * len(order) // buckets:(k + 1) * len(order) // buckets]
            for i in members:
                self.bucket[i] = k
            frames = max(lengths[i] for i in members)
            self.shapes.append({"size": max(1, min(max_batch, frame_budget // frames)), "frames": frames,
                                "tokens": max(tokens[i] for i in members), "shortest": min(lengths[i] for i in members)})
        self.items = TemperedSampler(groups, lengths, alpha, seed)
        self.generator = self.items.generator
        self.waiting = [[] for _ in self.shapes]

    def next(self):
        """(indices, shape) of the next batch."""
        while True:
            index = int(self.items.draw(1))
            k = self.bucket[index]
            self.waiting[k].append(index)
            if len(self.waiting[k]) == self.shapes[k]["size"]:
                batch, self.waiting[k] = self.waiting[k], []
                return batch, self.shapes[k]

    def state(self):
        return {"generator": self.generator.get_state(), "waiting": [list(w) for w in self.waiting]}

    def load_state(self, state):
        self.generator.set_state(state["generator"].cpu())
        self.waiting = [list(w) for w in state["waiting"]]

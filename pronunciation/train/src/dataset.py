"""Dataset, collation, and sampling for pronunciation CTC training."""

import json
import math
import random

import torch
from torch.utils.data import Dataset, Sampler

from .audio import AudioProcessor
from .vocab import IPAVocab


class PronunciationDataset(Dataset):
    """Loads JSONL manifest entries and returns mel spectrograms + IPA labels."""

    def __init__(
        self,
        manifest_paths: list[str],
        vocab: IPAVocab,
        audio_processor: AudioProcessor,
        max_duration_s: float = 30.0,
        min_duration_s: float = 0.5,
    ):
        self.vocab = vocab
        self.audio_processor = audio_processor
        self.entries = []
        for path in manifest_paths:
            with open(path) as f:
                for line in f:
                    entry = json.loads(line)
                    dur = entry["duration_s"]
                    if min_duration_s <= dur <= max_duration_s:
                        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        mel = self.audio_processor(entry["audio_path"])
        labels = self.vocab.encode(entry["ipa"])
        return {
            "mel": mel,
            "mel_length": mel.shape[0],
            "labels": torch.tensor(labels, dtype=torch.long),
            "label_length": len(labels),
            "lang": entry["lang"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad mels and labels to max length in batch."""
    max_mel_len = max(b["mel_length"] for b in batch)
    max_label_len = max(b["label_length"] for b in batch)
    n_mels = batch[0]["mel"].shape[1]
    bs = len(batch)

    mels = torch.zeros(bs, max_mel_len, n_mels)
    mel_lengths = torch.tensor([b["mel_length"] for b in batch], dtype=torch.long)
    labels = torch.zeros(bs, max_label_len, dtype=torch.long)
    label_lengths = torch.tensor([b["label_length"] for b in batch], dtype=torch.long)
    langs = [b["lang"] for b in batch]

    for i, b in enumerate(batch):
        mels[i, : b["mel_length"]] = b["mel"]
        labels[i, : b["label_length"]] = b["labels"]

    return {
        "mels": mels,
        "mel_lengths": mel_lengths,
        "labels": labels,
        "label_lengths": label_lengths,
        "langs": langs,
    }


class BucketBatchSampler(Sampler):
    """Groups utterances by similar audio duration to minimize padding waste.

    Sorts all indices by duration, chunks into buckets, shuffles within
    each bucket, then yields batches of batch_size.
    """

    def __init__(
        self,
        durations: list[float],
        batch_size: int,
        bucket_multiplier: int = 10,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Sort indices by duration
        sorted_indices = sorted(range(len(durations)), key=lambda i: durations[i])
        # Chunk into buckets
        bucket_size = batch_size * bucket_multiplier
        self.buckets = [
            sorted_indices[i : i + bucket_size]
            for i in range(0, len(sorted_indices), bucket_size)
        ]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches = []
        for bucket in self.buckets:
            indices = list(bucket)
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)
        if self.shuffle:
            rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        total = sum(len(b) for b in self.buckets)
        return total // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def compute_language_weights(
    entries: list[dict], temperature: float = 2.0
) -> list[float]:
    """Compute per-sample weights for temperature-based language upsampling.

    P(lang) proportional to count(lang)^(1/T). Higher T = more uniform.
    """
    lang_counts: dict[str, int] = {}
    for e in entries:
        lang = e["lang"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    lang_probs = {
        lang: count ** (1.0 / temperature) for lang, count in lang_counts.items()
    }
    total = sum(lang_probs.values())
    lang_probs = {lang: p / total for lang, p in lang_probs.items()}

    # Per-sample weight = target_prob / actual_prob
    total_samples = len(entries)
    weights = []
    for e in entries:
        lang = e["lang"]
        actual_prob = lang_counts[lang] / total_samples
        weights.append(lang_probs[lang] / actual_prob)
    return weights

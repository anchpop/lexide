"""Padding is drawn once in the sampler and budgeted before worker collation."""
import pickle
import random
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from src.dataset import (
    augmented_audio_lengths, make_train_collate, PlannedPaddingBatchSampler,
    PlannedPaddingDataset, TokenBudgetBatchSampler,
)
from test_speed_perturb import sample


class Samples(Dataset):
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return sample(n=self.lengths[index])


@pytest.mark.parametrize('draws,frames', [([0, 0, .999999], (0, 62)), ([0, .5, .25], (15, 3))])
def test_sampler_owns_independent_margins(monkeypatch, draws, frames):
    values = iter(draws)
    monkeypatch.setattr('src.dataset.random.Random', lambda _: SimpleNamespace(random=lambda: next(values)))
    sampler = PlannedPaddingBatchSampler([16000], 100000, padding_prob=1)
    assert sampler.padding_frames == [frames]
    item = PlannedPaddingDataset(Samples([16000]))[next(iter(sampler))[0]]
    # Collate must not redraw margins, even under worker-local RNG state.
    monkeypatch.setattr('src.dataset.random.random', lambda: pytest.fail('redrew padding'))
    result = make_train_collate(keep_clean=True)([item])
    head, tail = frames
    assert result['audio_lens'].item() == 16000 + (head + tail) * 256
    clean = result['audio_clean'][0]
    torch.testing.assert_close(clean[head * 256:head * 256 + 16000], item['audio'])
    assert not result['vad_probs'][0, :head].any()
    assert not result['vad_probs'][0, head + 16000 // 256:].any()
    assert (result['audio'][0] - clean).abs().max() < .001


def test_skipped_padding_has_no_legacy_margins_or_reserve():
    sampler = PlannedPaddingBatchSampler([16000] * 20, 80000, padding_prob=0)
    assert sampler.lengths == [16000] * 20
    assert sampler.padding_frames == [(0, 0)] * 20
    assert len(sampler) == len(TokenBudgetBatchSampler([16000] * 20, 80000))
    item = PlannedPaddingDataset(Samples([16000] * 20))[next(iter(sampler))[0]]
    result = make_train_collate(keep_clean=True)([item])
    torch.testing.assert_close(result['audio_clean'][0], item['audio'])


def test_default_collate_retains_legacy_padding():
    random.seed(13)
    item = sample()
    result = make_train_collate(keep_clean=True)([item])
    assert item['audio'].numel() + 25 * 256 <= result['audio_lens'].item() <= item['audio'].numel() + 43 * 256


def test_sampler_bounds_speed_padding_rounding_and_degradation(monkeypatch):
    lengths = [16001, 32003, 64007, 1001] * 4
    sampler = PlannedPaddingBatchSampler(lengths, 400000, speed_min=.85,
                                         padding_prob=1, pad_audio_multiple=1280)
    dataset = PlannedPaddingDataset(Samples(lengths))
    monkeypatch.setattr('src.dataset.degrade_waveform', lambda audio, *a, **kw: audio + .001)
    collate = make_train_collate(speed_prob=1, speed_min=.85, speed_max=.85,
                                 pad_audio_multiple=1280, keep_clean=True, degrade_prob=1)
    for indices in sampler:
        result = collate([dataset[index] for index in indices])
        assert result['audio'].numel() <= sampler.token_budget
        for j, (index, _) in enumerate(indices):
            assert result['audio_lens'][j] <= sampler.lengths[index]
        assert result['audio_clean'].shape == result['audio'].shape


def test_epoch_plans_are_sparse_reproducible_and_picklable():
    sampler = PlannedPaddingBatchSampler([16000] * 10000, 3200000, seed=42)
    first = list(sampler)
    assert sorted(index for batch in first for index, _ in batch) == list(range(10000))
    active = sum(padding != (0, 0) for padding in sampler.padding_frames)
    assert 2700 < active < 3300
    assert all(0 <= frame <= 62 for padding in sampler.padding_frames for frame in padding)
    assert list(pickle.loads(pickle.dumps(sampler))) == first
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first
    assert sampler.stats()['padded_audio_samples'] == sum(
        max(sampler.lengths[index] for index, _ in batch) * len(batch) for batch in sampler)


def test_plans_cross_nested_subsets_and_spawn_workers():
    base = ConcatDataset([Samples([8000, 16000]), Samples([24000, 32000])])
    nested = Subset(Subset(base, [3, 1, 0]), [1, 2])
    sampler = PlannedPaddingBatchSampler([16000, 8000], 100000, padding_prob=1, seed=4)
    loader = DataLoader(PlannedPaddingDataset(nested), batch_sampler=sampler,
                        collate_fn=make_train_collate(keep_clean=True), num_workers=1,
                        multiprocessing_context='spawn')
    batch = next(iter(loader))
    expected = [sampler.raw_lengths[index] + sum(padding) * 256
                for index, padding in next(iter(sampler))]
    assert batch['audio_lens'].tolist() == expected


def test_missing_vad_and_padding_rounding_bound():
    result = make_train_collate()([{**sample(vad=False), 'padding_frames': (0, 3)}])
    assert result['vad_probs'] is None
    assert augmented_audio_lengths([123], padding_frames=[(0, 0)]) == [123]
    assert augmented_audio_lengths([123], padding_frames=[(1, 2)], pad_audio_multiple=256) == [1024]

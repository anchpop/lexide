"""Playback edits on known contours; held-out v11 rarely produces a within-vowel spike."""
import math
from types import SimpleNamespace

import torch

from voca.model import Voca
from voca.styletts import StyleVoca


def test_spike_threshold_depends_on_phone_ownership():
    model = SimpleNamespace(voiced_tokens=torch.tensor([False, True]))
    tokens = torch.tensor([[1, 1, 0]])
    owner = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2]])
    p = torch.zeros(1, 9, 3)
    p[0, [1, 3, 7], 0] = torch.tensor([4., 4., 10.]) * math.log(2) / 12
    expected = p.clone()
    expected[0, [1, 7], 0] = 0
    torch.testing.assert_close(Voca.remove_pitch_spikes(model, p, tokens, owner), expected)


def test_rising_final_word_uses_its_own_typical_phone_positions():
    # The trailing punctuation and an earlier word have their own frames.
    word_index = torch.tensor([[0, -1, 1, -1]])
    durations = torch.tensor([[2, 0, 6, 1]])
    owner = torch.arange(4).repeat_interleave(durations[0])[None]
    p = torch.zeros(1, 9, 3)
    p[..., 1] = 1
    p[0, 6:8, 0] = 0.5
    final, rising = StyleVoca.ending(p, word_index, owner)
    assert rising.item()
    expected = p.clone()
    typical_duration = torch.tensor([[1, 0, 3, 2]])
    typical = torch.zeros(1, 6, 3)
    typical[0, 1:4, 0] = torch.tensor([0., 0.2, 0.4])
    target = torch.tensor([0., 0.05, 0.15, 0.25, 0.35, 0.4])
    weight = 0.75 * torch.tensor([0.25, 0.5, 0.75, 1., 1., 1.])
    expected[0, 2:8, 0] += (target - expected[0, 2:8, 0]) * weight
    torch.testing.assert_close(StyleVoca.blend_ending(p, durations, owner, final, typical, typical_duration), expected)
    p[..., 1] = -1
    assert not StyleVoca.ending(p, word_index, owner)[1].item()

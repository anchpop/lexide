"""Word-level prosody targets (voca.data.word_prosody) and phonebert's masking of yap's features.

    PYTHONPATH=. python -m pytest tests
"""

import numpy as np
import pytest
import torch

from voca.data import SEMITONES, SHAPES, contour_shape, word_prosody
from voca.phonebert import IGNORE, POS, RARITY, mask_words
from voca.text import LEXICAL, POS_ID, SYMBOLS, rarity_bucket, tokenize


def test_contour_shapes():
    assert [SHAPES[contour_shape(*marks)] for marks in ((0, 0.5, 0.2), (0, 1, 3), (3, 2, 0), (0, 3, 0.5), (2, 0, 2.5))] == [
        "level", "rise", "fall", "rise-fall", "fall-rise"]


def test_word_targets():
    """ma (rising, loud) na (level) . -- one phrase of two words."""
    p = tokenize([{"words": ["ma"], "pos": ["NOUN"], "count": 3, "phonemes": "m ˈɑː"},
                  {"words": ["na"], "pos": ["NOUN"], "count": 3, "phonemes": "n ˈɑː"},
                  {"words": ["."], "pos": ["PUNCT"], "count": 0, "phonemes": ""}])
    assert " ".join(p.symbols) == "<bos> m ɑː <sp> n ɑː . <eos>"
    durations = torch.tensor([2, 5, 10, 0, 5, 10, 3, 2])
    frames = int(durations.sum())
    semitones = torch.zeros(frames)
    semitones[2:17] = torch.linspace(0, 4, 15)
    energy = torch.zeros(frames)
    energy[7:17] = 1.0
    lexical = torch.tensor(p.lexical())
    targets = word_prosody(torch.tensor(p.ids), lexical[:, 0], durations, semitones / SEMITONES, torch.ones(frames), energy,
                           np.zeros((len(SYMBOLS), LEXICAL["stress"])))
    assert targets.shape == (2, 4)
    assert targets[:, 0].tolist() == pytest.approx([1.0, -1.0], abs=1e-3)       # prominence, standardized in the phrase
    assert [SHAPES[int(s)] for s in targets[:, 1]] == ["rise", "level"]
    assert targets[0, 2] == pytest.approx(3.2, abs=0.2) and targets[1, 2] == 0
    assert targets[0, 3].isnan() and targets[1, 3] == 0                       # final movement: phrase-final words only


def batch(*grams):
    p = tokenize([{"words": w, "pos": pos, "count": count, "phonemes": ph} for w, pos, count, ph in grams])
    return (torch.tensor([p.ids]), torch.tensor([p.lexical()]), torch.tensor([p.word]), torch.tensor([p.gram]))


def test_masking_hides_a_masked_words_gram_rarity():
    tokens, lexical, occurrence, gram = batch((["the", "cat"], ["DET", "NOUN"], 174, "ð ə # k ˈæ t"),
                                              (["sat"], ["VERB"], 900, "s ˈæ t"))
    first, second, other = occurrence[0] == 0, occurrence[0] == 1, occurrence[0] == 2
    for seed in range(200):
        _, features, phones, pos, rarity = mask_words(tokens, lexical, occurrence, gram, 0.5, torch.Generator().manual_seed(seed))
        chosen = phones[0] != IGNORE
        if chosen[first].all() and not chosen[second | other].any():
            break
    else:
        pytest.fail("no draw masked 'the' alone")
    assert (features[0, first, POS] == 0).all() and (features[0, second, POS] == POS_ID["NOUN"]).all()   # its word's tag only
    assert (features[0, first | second, RARITY] == 0).all()                                              # its gram's rarity
    assert (features[0, other, RARITY] == rarity_bucket(900)).all()
    assert (pos[0, first] == POS_ID["DET"]).all() and (pos[0, ~first] == IGNORE).all()
    assert (rarity[0, first] == rarity_bucket(174)).all() and (rarity[0, ~first] == IGNORE).all()


def test_praat_voicing_fills_pyin_dropouts():
    """Where pYIN found no voicing but Praat did, Praat's pitch is taken and the frame is voiced and trusted."""
    from voca.data import continuous_pitch
    pitch, voiced = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.0, 0.0]), torch.tensor([1.0, 1, 1, 1, 0, 0])
    check = torch.tensor([[0.1, 1], [0.1, 1], [0.1, 1], [0.1, 1], [-0.2, 1], [0.0, 0]])
    curve, trusted, voiced = continuous_pitch(pitch, voiced, check)
    assert voiced.tolist() == [1, 1, 1, 1, 1, 0] and trusted[4] == 1 and abs(float(curve[4]) + 0.2) < 1e-6


def test_forced_voicing_only_where_it_sounds_voiced():
    """Vowel and sonorant frames are forced voiced only where the audio is loud enough and low-band heavy."""
    from voca.data import voiced_like
    mel = torch.zeros(3, 80)
    mel[0, :25], mel[0, 55:] = 0.0, -6.0      # a vowel: energy in the low bands
    mel[1, :25], mel[1, 55:] = -6.0, 0.0      # frication: energy in the high bands
    mel[2, :25], mel[2, 55:] = 0.0, -6.0      # spectrally vowel-like but silent (a closure)
    assert voiced_like(torch.tensor([0.5, 0.5, -1.5]), mel).tolist() == [True, False, False]

"""Phone durations from CTC spans (voca.align.assemble)."""
import numpy as np
from voca.align import AlignmentSettings, assemble
from voca.text import BOS, EOS, PHONE_SET

HOP, RATE, STRIDE = 256, 24000, 0.02
V, T = "æ", "t"
assert {V, T} <= set(PHONE_SET)


def durations(symbols, spans, loud, frames=100):
    silent = np.ones(frames, dtype=bool)
    for a, b in loud:
        silent[a:b] = False
    return assemble(symbols, spans, frames, silent, HOP, RATE, AlignmentSettings())


def test_a_final_stop_keeps_its_release():
    """Closure quiet, release loud, emission held through the release: only the silence after it is the end."""
    symbols = [BOS, V, T, EOS]
    # a emits at 20-40 ms; t from 100 ms (its closure) to 220 ms (its release); audio: loud until 130 ms, quiet, loud 190-240 ms
    frame = lambda s: int(s * RATE / HOP)
    loud = [(0, frame(0.13)), (frame(0.19), frame(0.24))]
    d = durations(symbols, [(1, 2), (5, 11)], loud, frames=80)
    assert d.sum() == 80
    end = d[:3].sum()
    assert frame(0.24) <= end <= frame(0.24) + 3


def test_a_final_vowel_ends_where_its_sound_does():
    frame = lambda s: int(s * RATE / HOP)
    d = durations([BOS, V, EOS], [(5, 5)], [(frame(0.05), frame(0.3))], frames=80)
    assert abs(d[:2].sum() - frame(0.3)) <= 1


def test_a_stop_before_a_comma_keeps_its_release_too():
    frame = lambda s: int(s * RATE / HOP)
    symbols = [BOS, V, T, ",", V, EOS]
    loud = [(0, frame(0.13)), (frame(0.19), frame(0.24)), (frame(0.55), frame(0.8))]
    d = durations(symbols, [(1, 2), (5, 11), (28, 28)], loud, frames=100)
    assert frame(0.24) <= d[:3].sum() <= frame(0.24) + 3 and d[3] >= 10

"""The generator's excitation (voca.styletts.SourceModule)."""

import torch

from voca.styletts import SourceModule

RATE, HOP, FRAMES = 24000, 256, 40
F0 = torch.full((1, FRAMES), 200.0)
X = torch.zeros(1, 4, FRAMES)
PERIOD = RATE // 200


def module(noise=True):
    torch.manual_seed(0)
    source = SourceModule(4, RATE, HOP)
    if not noise:
        with torch.no_grad():
            source.gains.weight[18:] = 0
            source.gains.bias[18:] = -30
    return source


def run(source, voicing):
    return source(F0, torch.full((1, FRAMES), voicing), X)[0, 4 * HOP:-4 * HOP].detach()


def test_noise_bands_add_up_to_the_noise():
    bands = module().bands
    impulse = torch.zeros(bands.shape[-1])
    impulse[len(impulse) // 2] = 1
    assert torch.allclose(bands.sum((0, 1)), impulse, atol=1e-6)


def test_each_noise_band_stays_in_its_band():
    source = module()
    response = torch.fft.rfft(source.bands[:, 0], 4096).abs()
    frequency = torch.fft.rfftfreq(4096, 1 / RATE)
    edges = (-1e6, *source.BAND_EDGES, 1e6)
    for k, (low, high) in enumerate(zip(edges, edges[1:])):
        assert response[k][(frequency > low + 700) & (frequency < high - 700)].min() > 0.97
        assert response[k][(frequency < low - 700) | (frequency > high + 700)].max() < 0.03


def test_starts_as_equal_sines_that_follow_the_voicing():
    source = module(noise=False)
    wave = run(source, 1.0)
    spectrum = torch.fft.rfft(wave[:60 * PERIOD] * torch.hann_window(60 * PERIOD)).abs() / (60 * PERIOD / 4)
    levels = spectrum[[60 * k for k in range(1, 10)]]
    assert torch.allclose(levels, torch.full_like(levels, 0.02), rtol=0.1)
    assert torch.allclose(run(source, 0.5), wave / 2, atol=1e-3)
    assert run(source, 0.0).abs().max() < 1e-6


def test_starts_with_noise_at_its_unvoiced_and_voiced_levels():
    source = module()
    assert abs(run(source, 0.0).std().item() / 0.02 - 1) < 0.1
    difference = run(source, 1.0) - run(source, 1.0)          # two draws over the same sines
    assert abs(difference.std().item() / (0.002 * 2 ** 0.5) - 1) < 0.15


def test_pitch_glides_between_frames():
    """A glide's frequency moves every sample, not in steps at frame edges (held per frame: 1.6 Hz off a smooth curve)."""
    source = module(noise=False)
    with torch.no_grad():
        source.gains.weight[1:9] = 0                                              # the fundamental alone
    wave = source(torch.linspace(100.0, 300.0, FRAMES)[None], torch.ones(1, FRAMES), X)[0].detach().double()
    up = torch.nonzero((wave[:-1] < 0) & (wave[1:] >= 0))[:, 0]
    crossings = up + wave[up] / (wave[up] - wave[up + 1])                        # upward zero crossings, interpolated
    t, frequency = crossings[1:], RATE / crossings.diff()
    inner = (t > 8 * HOP) & (t < (FRAMES - 8) * HOP)
    t, frequency = t[inner], frequency[inner]
    basis = torch.stack((t ** 0, t, t ** 2), 1)
    smooth = basis @ torch.linalg.lstsq(basis, frequency[:, None]).solution[:, 0]
    assert (frequency - smooth).abs().max() < 0.1
    held = run(module(noise=False), 1.0)
    assert torch.allclose(held[:-PERIOD], held[PERIOD:], atol=1e-4)                 # a steady vowel is still periodic


def test_the_model_chooses_a_sines_phase():
    source = module(noise=False)
    before = run(source, 1.0)
    with torch.no_grad():
        source.gains.weight[:9] = 0
        source.gains.weight[9:18, -1, 1] = 1
    after = run(source, 1.0)
    assert torch.allclose(after[:-PERIOD], after[PERIOD:], atol=1e-4)      # still periodic
    spectra = [torch.fft.rfft(w[:60 * PERIOD])[[60 * k for k in range(1, 10)]] for w in (before, after)]
    turned = torch.angle(spectra[1] / spectra[0])
    assert torch.allclose(turned, torch.full_like(turned, torch.pi / 2), atol=0.05)

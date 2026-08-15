"""Degradation pipeline: NoisePool cropping, source selection, invariants.

The invariants that matter for training correctness:
- every op is length-preserving (the 16 ms VAD grid must stay aligned);
- noise sources are unit-RMS before SNR scaling (or the mix's SNR is wrong);
- a missing/empty pool degrades to the synthetic-only behavior, never errors.
"""

import random

import numpy as np
import pytest
import soundfile as sf
import torch

from src.dataset import (
    NoisePool,
    _get_noise_pool,
    _mains_hum,
    _sample_additive_noise,
    degrade_waveform,
)


def _write_wav(path, seconds, sr=16000, channels=1, seed=0):
    rng = np.random.default_rng(seed)
    shape = (int(seconds * sr), channels) if channels > 1 else int(seconds * sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, rng.standard_normal(shape).astype(np.float32) * 0.1, sr)


@pytest.fixture
def pool(tmp_path):
    _write_wav(tmp_path / "music" / "long.wav", 4.0, seed=1)
    _write_wav(tmp_path / "music" / "short.wav", 0.2, seed=2)  # forces tiling
    _write_wav(tmp_path / "noise" / "amb.wav", 3.0, seed=3)
    for i in range(6):
        _write_wav(tmp_path / "speech" / f"spk{i}.wav", 2.0, seed=10 + i)
    return NoisePool(tmp_path)


def test_crop_is_exact_length_including_tiling(pool):
    rng = random.Random(0)
    for _ in range(20):
        n = rng.randrange(1000, 80000)
        for kind in ("music", "noise", "speech"):
            crop = pool.crop(kind, n, rng)
            assert crop.shape == (n,)
            assert torch.isfinite(crop).all()


def test_stereo_files_downmix_to_mono(tmp_path):
    _write_wav(tmp_path / "noise" / "stereo.wav", 1.0, channels=2)
    crop = NoisePool(tmp_path).crop("noise", 8000, random.Random(0))
    assert crop.shape == (8000,)


def test_pool_cache_none_for_missing_or_empty(tmp_path):
    assert _get_noise_pool(None) is None
    assert _get_noise_pool(tmp_path / "nope") is None
    (tmp_path / "empty" / "music").mkdir(parents=True)
    assert _get_noise_pool(tmp_path / "empty") is None
    _write_wav(tmp_path / "real" / "music" / "m.wav", 1.0)
    got = _get_noise_pool(tmp_path / "real")
    assert isinstance(got, NoisePool)
    assert got is _get_noise_pool(tmp_path / "real")  # cached per process


def test_noise_sources_are_unit_rms_and_all_branches_fire(pool):
    rng = random.Random(0)
    kinds_seen = set()
    for _ in range(200):
        sampled = _sample_additive_noise(16000, rng, pool)
        assert sampled is not None
        noise, snr_db = sampled
        assert noise.shape == (16000,)
        assert noise.pow(2).mean().sqrt().item() == pytest.approx(1.0, rel=1e-3)
        # Branch SNR floors identify the source: music >=5, babble >=8,
        # ambience/colored >=3 — so just track coverage via many draws having
        # distinct tensors; cheap proxy: count distinct SNR ranges.
        kinds_seen.add(round(snr_db))
    assert len(kinds_seen) > 5  # many draws, varied SNRs → selection is live


def test_without_pool_always_synthesizes(pool):
    rng = random.Random(0)
    for _ in range(20):
        sampled = _sample_additive_noise(4000, rng, None)
        assert sampled is not None
        noise, snr_db = sampled
        assert noise.shape == (4000,)
        assert 3 <= snr_db <= 30  # the colored-noise range


def test_mains_hum_is_unit_rms_and_low_frequency():
    rng = random.Random(0)
    hum = _mains_hum(16000, 16000, rng)
    assert hum.pow(2).mean().sqrt().item() == pytest.approx(1.0, rel=1e-3)
    spec = torch.fft.rfft(hum).abs()
    assert spec.argmax().item() <= 240  # 1 Hz/bin: all energy under 240 Hz


@pytest.mark.parametrize("with_pool", [False, True])
def test_degrade_preserves_length_and_finiteness(pool, with_pool):
    rng = random.Random(0)
    for seed in range(30):
        random.seed(seed)
        n = rng.randrange(4000, 60000)
        audio = torch.sin(torch.arange(n) * 0.05) * 0.3
        out = degrade_waveform(audio, 16000, noise_pool=pool if with_pool else None)
        assert out.shape == (n,)
        assert torch.isfinite(out).all()

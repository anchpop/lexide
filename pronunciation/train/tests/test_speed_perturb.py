"""Speed perturbation preserves CTC feasibility and the moving VAD grid."""

import math
import random

import pytest
import torch
import torchaudio.functional as AF

from src.dataset import _collate, make_train_collate


def sample(n=16001, phones=2, source="tts", vad=True):
    return dict(
        audio=torch.full((n,), 0.1),
        vad_probs=torch.cat([torch.zeros(n // 512),
                             torch.ones(n // 256 - n // 512)]) if vad else torch.empty(0),
        phoneme_ids=torch.arange(1, phones + 1, dtype=torch.long),
        stress_seq=torch.zeros(phones, dtype=torch.long),
        tone_seq=torch.zeros(phones, dtype=torch.long),
        pitch_accent_seq=torch.zeros(phones, dtype=torch.long),
        stress_available=True, tone_available=False, pitch_accent_available=False,
        lang="eng", source=source,
    )


@pytest.fixture
def fixed_padding(monkeypatch):
    monkeypatch.setattr("src.dataset.random.randint", lambda lo, hi: lo)


@pytest.mark.parametrize("speed", [0.85, 1.3, 1.12345])
@pytest.mark.parametrize("source", ["tts", "film"])
def test_length_vad_step_and_clean_snapshot(fixed_padding, monkeypatch, speed, source):
    item = sample(source=source)
    # A deterministic length-preserving corruption must happen AFTER the snapshot.
    monkeypatch.setattr("src.dataset.degrade_waveform", lambda audio, *a, **kw: audio + 1)
    result = make_train_collate(
        degrade_prob=1, keep_clean=True, speed_prob=1, speed_min=speed, speed_max=speed,
    )([item])
    rate = math.ceil(speed * 100)
    n = (16001 * 100 + rate - 1) // rate
    head, tail = 6 * 256, 19 * 256
    assert result["audio_lens"].item() == n + head + tail
    assert result["audio_mask"].sum().item() == n + head + tail
    assert result["vad_lens"].item() == (n + head + tail) // 256
    clean = result["audio_clean"][0]
    torch.testing.assert_close(clean[head:head + n], AF.resample(item["audio"], rate, 100))
    assert not clean[:head].any() and not clean[head + n:].any()
    expected_vad = torch.nn.functional.interpolate(
        item["vad_probs"][None, None], size=n // 256, mode="linear", align_corners=False,
    )[0, 0]
    torch.testing.assert_close(result["vad_probs"][0, 6:6 + n // 256], expected_vad)
    transition = (result["vad_probs"][0] >= 0.5).nonzero()[0].item()
    assert abs(transition - (6 + n / 512)) <= 1
    if source == "tts":
        torch.testing.assert_close(result["audio"][0], clean + 1)
    else:
        assert (result["audio"][0] - clean).abs().max() < 0.001
    assert item["audio"].numel() == 16001  # no mutation of the dataset item


@pytest.mark.parametrize("phones,repeated,accepted", [
    (24, False, True), (25, False, False), (12, True, True), (13, True, False),
])
def test_ctc_guard_before_padding(fixed_padding, phones, repeated, accepted):
    # 10400 / 1.3 = 8000 samples => exactly 24 encoder frames.
    item = sample(n=10400, phones=phones)
    if repeated:
        item["phoneme_ids"].fill_(1)
    result = make_train_collate(keep_clean=True, speed_prob=1, speed_min=1.3,
                                speed_max=1.3)([item])
    n = 8000 if accepted else 10400
    assert result["audio_lens"].item() == n + 25 * 256
    if not accepted:
        torch.testing.assert_close(result["audio_clean"][0, 6 * 256:6 * 256 + n], item["audio"])
        torch.testing.assert_close(result["vad_probs"][0, 6:6 + n // 256], item["vad_probs"])


@pytest.mark.parametrize("augment,prob,speed", [(True, 0, 1.3), (True, 1, 1), (False, 1, 1.3)])
def test_noop(augment, prob, speed):
    item = sample()
    random.seed(42)
    torch.manual_seed(42)
    baseline = _collate([item], augment=augment, keep_clean=True)
    random.seed(42)
    torch.manual_seed(42)
    result = _collate([item], augment=augment, keep_clean=True,
                      speed_prob=prob, speed_min=speed, speed_max=speed)
    for key in ("audio", "audio_clean", "audio_lens", "audio_mask", "vad_probs", "vad_lens"):
        torch.testing.assert_close(result[key], baseline[key])


def test_missing_vad_stays_missing(fixed_padding):
    result = make_train_collate(speed_prob=1, speed_min=0.85, speed_max=0.85)([sample(vad=False)])
    assert result["vad_probs"] is None
    assert result["vad_lens"].item() == 0


@pytest.mark.parametrize("args", [
    ["--speed-min", "0"], ["--speed-min", "-1"],
    ["--speed-min", "1.4", "--speed-max", "1.3"],
    ["--speed-min", "nan"], ["--speed-max", "inf"],
    ["--speed-perturb-prob", "-0.1"], ["--speed-perturb-prob", "1.1"],
    ["--speed-perturb-prob", "nan"],
])
def test_invalid_cli_speed_settings(monkeypatch, args, capsys):
    from src.train_unified import main

    monkeypatch.setattr("sys.argv", ["train_unified", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "speed" in capsys.readouterr().err

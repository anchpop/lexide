"""AcousticSidechannel: geometry, frame alignment, and checkpoint compat.

The side-channel's contract has three load-bearing pieces:

1. Legacy equivalence — the published champion was trained with the old
   inline pipeline (n_fft=400 mel, no low band, no pre-pad). The refactored
   module must reproduce it bit-for-bit or the champion's heads read shifted
   features.
2. Frame alignment — wider windows (1024/2048) must produce the SAME frame
   count as the 400-point bank AND keep window centers on wav2vec2's conv
   receptive-field centers (sample 320t+200). The factor heads are read at
   peaky CTC spike frames; a systematic offset lands pitch evidence on the
   wrong mora.
3. Checkpoint round-trip — new payloads carry one "sidechannel" state dict +
   geometry keys; legacy payloads carry separate "mel_norm"/"mel_proj" keys
   and no geometry keys. Both must load.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.factorized_ctc import (
    LOWBAND_BIN_HI,
    LOWBAND_BIN_LO,
    W2V2_RECEPTIVE_FIELD,
    W2V2_STRIDE,
    AcousticSidechannel,
    FactorizedCTCModel,
)


def w2v2_frames(num_samples: int) -> int:
    return (num_samples - W2V2_RECEPTIVE_FIELD) // W2V2_STRIDE + 1


def test_legacy_geometry_reproduces_old_pipeline():
    torch.manual_seed(0)
    sc = AcousticSidechannel(n_mels=80, mel_proj_dim=64, mel_n_fft=400, lowband_dim=0)
    sc.eval()
    wave = torch.randn(2, 16000)

    for T_target in (w2v2_frames(16000), 45, 60):  # exact, truncate, pad
        out = sc(wave, T_target, torch.float32)

        # The pre-refactor inline pipeline, verbatim.
        mel = sc.mel_spec(wave.float()).transpose(1, 2)
        if mel.shape[1] > T_target:
            mel = mel[:, :T_target, :]
        elif mel.shape[1] < T_target:
            mel = F.pad(mel, (0, 0, 0, T_target - mel.shape[1]))
        mel = torch.log(mel + 1e-6)
        expected = sc.mel_proj(sc.mel_norm(mel))

        assert out.shape == (2, T_target, 64)
        assert torch.equal(out, expected)


@pytest.mark.parametrize("n_fft", [400, 1024, 2048])
def test_prepadding_preserves_frame_count(n_fft):
    import torchaudio

    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, win_length=n_fft, hop_length=W2V2_STRIDE, center=False,
    )
    pad = (n_fft - W2V2_RECEPTIVE_FIELD) // 2
    for length in (4000, 15999, 16000, 16321):
        frames = spec(F.pad(torch.randn(1, length), (pad, pad))).shape[-1]
        assert frames == w2v2_frames(length)


@pytest.mark.parametrize("bank", ["mel", "low"])
def test_window_centers_align_with_w2v2_frames(bank):
    # An impulse has a flat spectrum, so a frame's total energy is just the
    # analysis window's value at the impulse — maximal when the impulse sits
    # at the window center. Placing it at sample 320k+200 (the center of
    # wav2vec2 frame k) must make frame k the peak for every bank.
    sc = AcousticSidechannel(n_mels=128, mel_proj_dim=128, mel_n_fft=1024, lowband_dim=64)
    spec = sc.mel_spec if bank == "mel" else sc.low_spec
    k = 20
    wave = torch.zeros(1, 16000)
    wave[0, k * W2V2_STRIDE + W2V2_RECEPTIVE_FIELD // 2] = 1.0

    pad = (spec.n_fft - W2V2_RECEPTIVE_FIELD) // 2
    energy = spec(F.pad(wave, (pad, pad))).sum(dim=1).squeeze(0)
    assert energy.argmax().item() == k


def test_lowband_covers_f0_and_concatenates():
    sc = AcousticSidechannel(n_mels=128, mel_proj_dim=128, mel_n_fft=1024, lowband_dim=64)
    assert sc.out_dim == 192
    # 62.5–601.6 Hz at 7.8125 Hz/bin: brackets adult F0, resolves ~3 bins
    # per 2-semitone move at 200 Hz.
    bin_hz = 16000 / 2048
    assert LOWBAND_BIN_LO * bin_hz == 62.5
    assert LOWBAND_BIN_HI * bin_hz == pytest.approx(609.4, abs=1.0)

    out = sc(torch.randn(2, 16000), w2v2_frames(16000), torch.float32)
    assert out.shape == (2, w2v2_frames(16000), 192)

    # The low band must actually contribute: a pure 200 Hz tone and a pure
    # 3 kHz tone are both out-of-band identical to the low channel only if
    # the slice were wrong.
    t = torch.arange(16000) / 16000.0
    low_tone = torch.sin(2 * torch.pi * 200 * t).unsqueeze(0)
    pad = (sc.low_spec.n_fft - W2V2_RECEPTIVE_FIELD) // 2
    band = sc.low_spec(F.pad(low_tone, (pad, pad)))[:, LOWBAND_BIN_LO:LOWBAND_BIN_HI, :]
    full = sc.low_spec(F.pad(low_tone, (pad, pad)))
    assert band.sum() / full.sum() > 0.99  # 200 Hz energy lives in-slice


def test_overfine_mel_bank_fails_at_construction():
    with pytest.raises(ValueError, match="empty filters"):
        AcousticSidechannel(n_mels=256, mel_proj_dim=8, mel_n_fft=400)


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8, final_dropout=0.0, num_hidden_layers=1,
        )

    def forward(self, input_values, attention_mask=None, **kwargs):
        hidden = input_values.unsqueeze(-1).expand(-1, -1, 8)
        return SimpleNamespace(last_hidden_state=hidden)

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def dummy_backbone(monkeypatch):
    monkeypatch.setattr(
        "src.factorized_ctc.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: DummyBackbone(),
    )


def _mk_model(**kwargs):
    return FactorizedCTCModel(
        model_name="dummy", vocab_size=5, mel_sidechannel=True,
        mlp_heads=False, **kwargs,
    )


def test_sidechannel_model_roundtrips_geometry(tmp_path, dummy_backbone):
    model = _mk_model(n_mels=128, acoustic_dim=128, mel_n_fft=1024, lowband_dim=64)
    model.eval()
    wave = torch.randn(2, 4000)
    out = model(wave)
    assert out["log_probs"].shape == (2, 4000, 5)

    model.save_to_dir(tmp_path)
    restored = FactorizedCTCModel.load_from_dir(tmp_path)
    restored.eval()
    assert restored.mel_n_fft == 1024
    assert restored.lowband_dim == 64
    assert restored.sidechannel.out_dim == 192
    assert torch.equal(restored(wave)["log_probs"], out["log_probs"])


def test_legacy_checkpoint_payload_loads(tmp_path, dummy_backbone):
    # The published champion's payload predates the low band: separate
    # "mel_norm"/"mel_proj" keys, no "sidechannel"/"mel_n_fft"/"lowband_dim".
    model = _mk_model(n_mels=80, acoustic_dim=64, mel_n_fft=400, lowband_dim=0)
    model.eval()
    model.save_to_dir(tmp_path)

    path = tmp_path / "factorized_heads.pt"
    heads = torch.load(path, map_location="cpu", weights_only=False)
    sc_state = heads.pop("sidechannel")
    del heads["mel_n_fft"], heads["lowband_dim"]
    for module in ("mel_norm", "mel_proj"):
        heads[module] = {
            key[len(module) + 1:]: value
            for key, value in sc_state.items() if key.startswith(module + ".")
        }
    torch.save(heads, path)

    restored = FactorizedCTCModel.load_from_dir(tmp_path)
    restored.eval()
    assert restored.mel_n_fft == 400
    assert restored.lowband_dim == 0
    wave = torch.randn(1, 4000)
    assert torch.equal(restored(wave)["log_probs"], model(wave)["log_probs"])

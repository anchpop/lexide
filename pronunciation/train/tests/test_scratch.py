import torch
import pytest

from src.scratch_backbone import ScratchConformer
from src.factorized_ctc import FactorizedCTCModel
from src.consistency import consistency_loss, symmetric_kl, warmup_cosine_multiplier


CONFIG = dict(hidden_size=32, num_hidden_layers=2, heads=2, ff_size=64, kernel_size=7)


@pytest.mark.parametrize("length", [400, 559, 560, 719, 720, 4000, 15999, 16000, 16321])
def test_lengths(length):
    model = ScratchConformer(CONFIG).eval()
    with torch.no_grad():
        out = model(torch.randn(1, length)).last_hidden_state
    assert out.shape == (1, (length - 400) // 320 + 1, 32)
    assert out.shape[1] == model._get_feat_extract_output_lengths(torch.tensor([length])).item()


def test_padding_and_save(tmp_path):
    torch.manual_seed(8)
    model = FactorizedCTCModel(model_name="scratch", vocab_size=12, scratch_config=CONFIG).eval()
    assert not model.mel_sidechannel
    x = torch.randn(1, 5101)
    with torch.no_grad():
        expected = model(x)["log_probs"]
        padded = torch.cat([x, torch.randn(1, 2100)], -1)
        mask = torch.arange(7201)[None] < 5101
        actual = model(padded, attention_mask=mask)["log_probs"][:, :expected.shape[1]]
        torch.testing.assert_close(expected, actual, atol=2e-5, rtol=1e-5)
        model.save_to_dir(tmp_path)
        loaded = FactorizedCTCModel.load_from_dir(tmp_path).eval()
        torch.testing.assert_close(loaded(x)["log_probs"], expected, atol=0, rtol=0)


def test_augmentation_train_only():
    model = ScratchConformer({**CONFIG, "dropout": 0.0})
    x = torch.randn(2, 8000)
    assert not torch.equal(model(x).last_hidden_state, model(x).last_hidden_state)
    model.eval()
    torch.testing.assert_close(model(x).last_hidden_state, model(x).last_hidden_state, atol=0, rtol=0)


def test_real_dft_matches_stft():
    model = ScratchConformer(CONFIG)
    x = torch.randn(2, 6400)
    conv = torch.nn.functional.conv1d(x[:, None], model.frontend.dft, stride=160)
    stft = torch.stft(x, 400, 160, window=torch.hann_window(400), center=False, return_complex=True)
    torch.testing.assert_close(conv[:, :201], stft.real, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(conv[:, 201:], stft.imag, atol=1e-3, rtol=1e-3)


def test_frame_centers_match_vad_grid():
    model = ScratchConformer({**CONFIG, "num_hidden_layers": 0}).eval()
    with torch.no_grad():
        model.subsample.weight.zero_()
        model.subsample.bias.zero_()
        model.subsample.weight[0, :, 1] = 1
        for frame in (1, 10, 30):
            waveform = torch.zeros(1, 16000)
            waveform[0, 200 + 320 * frame] = 1
            out = model(waveform).last_hidden_state[0, :, 0]
            assert out.argmax().item() == frame


def test_cr_masks_and_gradients():
    model = FactorizedCTCModel(model_name="scratch", vocab_size=8, scratch_config=CONFIG)
    batch = dict(stress_available=torch.tensor([True, False]), tone_available=torch.tensor([False, False]),
                 pitch_accent_available=torch.tensor([False, False]), langs=["eng", "jpn"])
    x = torch.randn(2, 2000)
    a, b = model(x), model(x)
    lengths = torch.tensor([3, 4])
    loss = consistency_loss(a, b, lengths, batch, model, stress_active=True, stress_weight=.3)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(model.backbone.subsample.weight.grad).all()
    assert all(p.grad is not None for p in model.stress_head.parameters())
    identical = consistency_loss(a, a, lengths, batch, model, stress_active=True, stress_weight=.3)
    torch.testing.assert_close(identical, torch.tensor(0.))
    copied = {**b, "stress_logits": b["stress_logits"].detach().clone()}
    copied["stress_logits"][1] += torch.randn_like(copied["stress_logits"][1]) * 10
    torch.testing.assert_close(loss, consistency_loss(a, copied, lengths, batch, model,
                               stress_active=True, stress_weight=.3))


def test_symmetric_kl_detaches_targets():
    a = torch.tensor([[.2, .8]]).log().requires_grad_()
    b = torch.tensor([[.6, .4]]).log().requires_grad_()
    symmetric_kl(a, b).sum().backward()
    torch.testing.assert_close(a.grad, -.5 * b.detach().exp())
    torch.testing.assert_close(b.grad, -.5 * a.detach().exp())


def test_cr_matches_enumerated_joint_and_masks_padding():
    from types import SimpleNamespace
    def outputs():
        nb = torch.randn(2, 5)
        ph = torch.randn(2, 5, 2).log_softmax(-1)
        return dict(nonblank_logit=nb,
                    log_probs=torch.cat([torch.nn.functional.logsigmoid(-nb)[..., None],
                        torch.nn.functional.logsigmoid(nb)[..., None] + ph], -1),
                    stress_logits=torch.randn(2, 5, 3),
                    language_head_logits={"tone": torch.randn(2, 5, 2)})
    a, b = outputs(), outputs()
    batch = dict(stress_available=torch.tensor([True, False]),
                 tone_available=torch.tensor([True, True]), langs=["tha", "eng"])
    model = SimpleNamespace(language_head_specs={"tone": dict(lang="tha", target="tone", weight=.3)})
    lengths = torch.tensor([3, 4])
    got = consistency_loss(a, b, lengths, batch, model, stress_active=True, stress_weight=.3)
    expected = 0.
    for i, n in enumerate(lengths):
        joints = []
        for out in (a, b):
            phone = out["log_probs"][i, :n]
            if i == 0:
                stress = (out["stress_logits"][i, :n] * .3).log_softmax(-1)
                tone = (out["language_head_logits"]["tone"][i, :n] * .3).log_softmax(-1)
                emission = (phone[:, 1:, None, None] + stress[:, None, :, None] + tone[:, None, None, :]).flatten(1)
            else:
                emission = phone[:, 1:]
            joints.append(torch.cat([phone[:, :1], emission], -1))
        expected = expected + symmetric_kl(*joints).sum()
    torch.testing.assert_close(got, expected / lengths.sum())
    b["log_probs"][0, 3:] = torch.randn_like(b["log_probs"][0, 3:])
    b["stress_logits"][0, 3:] += 100
    b["language_head_logits"]["tone"][1] = torch.randn_like(b["language_head_logits"]["tone"][1]) * 100
    torch.testing.assert_close(got, consistency_loss(a, b, lengths, batch, model,
                               stress_active=True, stress_weight=.3))


def test_warmup_schedule():
    p = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([p], lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: warmup_cosine_multiplier(n, 4, 12))
    assert optimizer.param_groups[0]["lr"] == .00025
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == 4
    assert optimizer.param_groups[0]["lr"] == .001
    assert warmup_cosine_multiplier(12, 4, 12) == 0
    # Restoring both optimizer and scheduler preserves the LR used by the
    # very next update, not just the displayed completed-step count.
    saved_optimizer, saved_scheduler = optimizer.state_dict(), scheduler.state_dict()
    other = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=.001)
    resumed = torch.optim.lr_scheduler.LambdaLR(other, lambda n: warmup_cosine_multiplier(n, 4, 12))
    other.load_state_dict(saved_optimizer)
    resumed.load_state_dict(saved_scheduler)
    for _ in range(8):
        assert other.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        other.step()
        resumed.step()
    assert resumed.last_epoch == 12

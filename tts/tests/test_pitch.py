import pytest
import torch

from voca.train import delta_percentiles, prosody_terms, smoothed_spread


def test_percentiles_use_only_supported_adjacent_frames():
    pitch = torch.tensor([[0., 1., 3., 30., 100., 104.]])
    pairs = torch.tensor([[True, True, False, False, True]])
    expected = torch.quantile(torch.tensor([1., 2., 4.]), torch.tensor([.25, .5, .9, .99, .999]))
    torch.testing.assert_close(delta_percentiles(pitch, pairs), expected)
    torch.testing.assert_close(delta_percentiles(pitch, ~torch.ones_like(pairs)), torch.zeros(5))


def test_spread_and_delta_on_spike_and_phrase_boundary():
    label = torch.zeros(1, 101, 3)
    label[..., 1] = 1
    pitch = label.clone()
    pitch[0, 45, 0] = 1
    pitch.requires_grad_()
    trusted = torch.ones(1, 101)
    phrase = torch.zeros_like(trusted)
    terms, _ = prosody_terms(pitch, label, trusted, trusted, phrase)
    assert terms["pitch_delta"] > terms["pitch_spread"] > 0
    assert smoothed_spread(pitch[..., 0], trusted) < pitch[..., 0].std()
    sum(terms.values()).backward()
    assert pitch.grad.isfinite().all()
    phrase[:, 45] = 1
    terms, _ = prosody_terms(pitch, label, trusted, trusted, phrase)
    assert terms["pitch_delta"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CREPE feature builds run on CUDA")
def test_crepe_tone_and_silence():
    pytest.importorskip("torchcrepe")
    from voca.crepe_pitch import track
    t = torch.arange(24000, device="cuda") / 24000
    # A bright harmonic tone has sufficient A-weighted loudness to pass the silence gate.
    tone = 0.3 * torch.sin(2 * torch.pi * 217 * t).sign()
    pitch, confidence = track(torch.cat((tone, torch.zeros_like(tone))))
    assert pitch.shape == confidence.shape == (188,)
    assert (pitch[8:80].median() - 217).abs() < 3
    assert confidence[8:80].median() > .3
    assert confidence[110:].max() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Viterbi runs on the extraction GPU")
def test_gpu_viterbi_matches_dense_reference():
    pytest.importorskip("torchcrepe")
    import triton
    from voca.crepe_pitch import _decode
    # Start and end in different registers; a one-frame competing peak is
    # reachable but loses on the total path score.
    scores = torch.rand(90, 83, generator=torch.Generator().manual_seed(1))
    bins = torch.arange(scores.shape[1])
    transition = (12 - (bins[:, None] - bins).abs()).clamp_min(0).float().log()
    previous = scores[0]
    back = []
    for emission in scores[1:]:
        best, predecessor = (previous[:, None] + transition).max(0)
        back.append(predecessor)
        previous = best + emission
    path = [int(previous.argmax())]
    for predecessor in reversed(back):
        path.append(int(predecessor[path[-1]]))
    scores = scores.cuda()
    result = torch.empty(len(scores), device="cuda", dtype=torch.int32)
    _decode[(1,)](scores, torch.empty_like(scores, dtype=torch.int8), result,
                  *scores.shape, triton.next_power_of_2(scores.shape[1]))
    assert result.tolist() == path[::-1]

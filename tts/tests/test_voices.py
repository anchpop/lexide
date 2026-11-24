"""Several voices in one model: balanced sampling, per-voice styles and validation sampling,
TinyVoca's GAN balance, and the StyleTTS schedules (GAN ramp, fixed styles) under CUDA graphs."""

import collections
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from voca.config import Config
from voca.data import BucketSampler, TemperedSampler, collate
from voca.text import BOS, EOS, PHONES, SYMBOL_TO_ID, LEXICAL
from voca.train import TRAINING_SYSTEMS, GraphedSteps, adamw, balanced_backward, balanced_indices

ROOT = Path(__file__).parents[1]


def test_tempered_shares():
    """p(voice) ∝ hours^alpha: 10 h against five half hours."""
    groups = ["achernar"] * 100 + [v for v in ("kore", "leda", "puck", "iapetus", "orus") for _ in range(5)]
    sizes = [1.0] * len(groups)    # achernar 100, every other voice 5
    for alpha, expected in ((1.0, 100 / 125), (0.5, 10 / (10 + 5 * 5 ** 0.5)), (0.0, 1 / 6)):
        sampler = TemperedSampler(groups, sizes, alpha, seed=0)
        assert sampler.shares["achernar"] == pytest.approx(expected)
        drawn = collections.Counter(groups[i] for i in sampler.draw(40000).tolist())
        assert drawn["achernar"] / 40000 == pytest.approx(expected, abs=0.01)
        assert drawn["kore"] / 40000 == pytest.approx((1 - expected) / 5, abs=0.01)


def test_shares_follow_audio_not_clip_count():
    """A voice's share is by its total frames: many short clips do not outweigh few long ones."""
    sampler = TemperedSampler(["a"] * 10 + ["b"] * 40, [400] * 10 + [100] * 40, 1.0, seed=0)
    assert sampler.shares == {"a": 0.5, "b": 0.5}


def test_bucket_sampler_keeps_its_shapes_and_realises_the_weights():
    torch.manual_seed(0)
    lengths = torch.randint(50, 400, (600,)).tolist()
    tokens = [n // 4 for n in lengths]
    voices = ["achernar"] * 500 + ["kore"] * 100
    sampler = BucketSampler(lengths, tokens, voices, frame_budget=2000, max_batch=8, seed=1, alpha=0.0)
    shapes = {json.dumps(s, sort_keys=True) for s in sampler.shapes}
    drawn = collections.Counter()
    for _ in range(3000):
        indices, shape = sampler.next()
        assert json.dumps(shape, sort_keys=True) in shapes and len(indices) == shape["size"]
        assert all(lengths[i] <= shape["frames"] and lengths[i] >= shape["shortest"] for i in indices)
        drawn.update(voices[i] for i in indices)
    assert drawn["kore"] / sum(drawn.values()) == pytest.approx(0.5, abs=0.02)   # alpha 0: every voice the same share


def test_balanced_indices_per_voice():
    """Each voice's held-out sentences are taken round-robin over their kinds: with 64 of 72
    (French: 60 plain then 12 quoted, in manifest order), every quoted one and question is in."""
    rows = ([{"role": "validation", "text": "Bon."}] * 50 + [{"role": "validation", "text": "Bon ?"}] * 10
            + [{"role": "quoted-validation", "text": "« Bon. »"}] * 12)
    data = SimpleNamespace(speakers=["a"] * 72, rows=rows)
    picked = [rows[i] for i in balanced_indices(data, 64, per_voice=True)]
    assert len(picked) == 64 and sum(r["role"] == "quoted-validation" for r in picked) == 12
    assert sum(r["text"] == "Bon ?" for r in picked) == 10
    data = SimpleNamespace(speakers=["a", "b", "a", "b", "a", "c"], rows=[{"role": "validation", "text": "."}] * 6)
    assert sorted(balanced_indices(data, 3)) == [0, 1, 5]


def small_config(speakers=("achernar", "kore")):
    c = Config.from_dict(json.loads((ROOT / "configs/styletts.json").read_text()))
    return dataclasses.replace(c, speakers=speakers, channels=64, feedforward=128, encoder_layers=2, style_dim=32, style_prosody_channels=64,
                               style_prosody_layers=2, decoder_channels=64, generator_channels=32, style_encoder_channels=16,
                               style_encoder_max=64, crop_frames=96, phoneme_encoder_layers=0)


def test_each_voice_keeps_its_own_style():
    from voca.styletts import StyleVoca
    model = StyleVoca(small_config(("a", "b", "c")))
    model.observe_style(torch.ones(4, model.style.shape[1]), torch.tensor([0, 0, 2, 2]), momentum=0.5)
    assert torch.allclose(model.style[0], torch.full_like(model.style[0], 0.5)) and torch.allclose(model.style[2], model.style[0])
    assert model.style[1].abs().max() == 0     # unseen in this batch: unchanged


def test_balanced_backward_is_the_scaled_gan_term():
    """One backward of loss with the GAN gradient sent in at the waveform equals backward of
    loss + scale * gan, with scale = ratio * ramp * |grad reconstruction| / |grad gan| at the waveform."""
    torch.manual_seed(0)
    weight = torch.randn(16, requires_grad=True)

    def terms():
        wave = (weight[None] * torch.linspace(0, 1, 16)[:, None]).sum(0, keepdim=True).tanh()
        reconstruction = (wave - 0.3).square().sum()
        return wave, reconstruction, reconstruction + weight.square().sum(), (wave.sin() * 3).sum()
    recipe = {"gan_ratio": 0.7, "gan_scale_max": 1e6}
    wave, reconstruction, loss, gan = terms()
    _, metrics = balanced_backward(loss, reconstruction, gan, wave, None, 0.5, recipe)
    ours, weight.grad = weight.grad.clone(), None
    wave, reconstruction, loss, gan = terms()
    g_rec = torch.autograd.grad(reconstruction, wave, retain_graph=True)[0].norm()
    g_gan = torch.autograd.grad(gan, wave, retain_graph=True)[0].norm()
    scale = 0.7 * 0.5 * g_rec / g_gan
    (loss + scale * gan).backward()
    assert torch.allclose(metrics["gan_scale"], scale) and torch.allclose(metrics["gan_gradient_ratio"], g_gan / g_rec)
    assert torch.allclose(ours, weight.grad, atol=1e-6)


def synthetic_item(speaker, phones=12, frames=120, seed=0):
    generator = torch.Generator().manual_seed(seed)
    ids = [SYMBOL_TO_ID[BOS]] + [SYMBOL_TO_ID[PHONES[k % 20]] for k in range(phones)] + [SYMBOL_TO_ID[EOS]]
    durations = [0] + [frames // phones] * phones + [0]
    durations[-2] += frames - sum(durations)
    return {"tokens": torch.tensor(ids), "lexical": torch.zeros(len(ids), len(LEXICAL), dtype=torch.long),
            "durations": torch.tensor(durations), "word_prosody": torch.full((len(ids), 4), math.nan),
            "mel": torch.randn(frames, 80, generator=generator) - 5,
            "prosody": torch.stack((torch.full((frames,), -0.3), torch.ones(frames), torch.zeros(frames)), -1),
            "pitch_trusted": torch.ones(frames), "wave": 0.1 * torch.randn(frames * 256, generator=generator), "speaker": speaker}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_schedules_under_graphs():
    """The GAN ramp and the fixed-style chance are device tensors rewritten before every replay:
    replayed steps weigh the GAN terms by ramp * gan and draw the stored styles at the scheduled
    chance, not at the values the graph was captured with."""
    torch.manual_seed(0)
    c = small_config()
    system = TRAINING_SYSTEMS["styletts"](c).cuda().train()
    critics = system.critics().cuda()
    optimizer = adamw([{"params": list(system.parameters()), "scale": 1.0}], "cuda", lr=1e-6, betas=(0.8, 0.99), weight_decay=0.01)
    critic_optimizer = adamw([{"params": list(critics.parameters())}], "cuda", lr=1e-6, betas=(0.8, 0.99), weight_decay=0.0)
    recipe = dict(weights={**system.WEIGHTS, "shift": 1.0}, learning_rate=1e-6, warmup=1, lr_half_life=1e5, grad_clip=10.0, gan_start=0, gan_ramp=10,
                  feature_matching=system.FEATURE_MATCHING, teacher_hold=0, teacher_ramp=1, teacher_final=0.0, word_teacher_hold=0,
                  word_teacher_ramp=1, fixed_style_hold=1, fixed_style_ramp=1, bf16=False)
    graphed = GraphedSteps(system, critics, optimizer, critic_optimizer, recipe)
    batch = collate([synthetic_item(0, seed=1), synthetic_item(1, seed=2)])
    for step in (1, 2, 3):      # step 1 runs eagerly and is captured, 2 and 3 replay it
        metrics = graphed(batch, step)
        assert float(metrics["gan_scale"]) == pytest.approx((step + 1) / 10 * system.WEIGHTS["gan"]), step
        assert float(metrics["fixed_style_fraction"]) == (0.0 if step == 1 else 1.0), step
        assert 0 < float(metrics["shift"]) < 10, step        # the shift-consistency loss, its random delay drawn in the graph
    assert len(graphed.graphs) == 1


def test_no_aliased_source_line():
    """A held 147 Hz vowel: before the coarse (stride-4) source injection, the coefficient streams keep
    almost nothing above the decimated Nyquist, where it would fold into an inharmonic line."""
    from voca.styletts import Generator
    c = small_config()
    generator = Generator(c).eval()
    stride = math.prod(c.generator_rates[1:])
    frames = 80
    with torch.no_grad():
        source = generator.m_source(torch.full((1, frames), 147.0), torch.ones(1, frames), torch.zeros(1, c.generator_channels, frames))
        spectrum = torch.stft(source, generator.fft, generator.hop, window=generator.window, return_complex=True)
        streams = torch.cat((spectrum.real, spectrum.imag), dim=1)

        def folding(x):    # power above the decimated Nyquist over power below, at the stream rate
            power = torch.fft.rfft(x - x.mean(-1, keepdim=True)).abs().square().sum(1)[0]
            rate = c.sample_rate / generator.hop
            above = torch.fft.rfftfreq(x.shape[-1], 1 / rate) > rate / stride / 2
            return float(power[above].sum() / power[~above].sum())
        assert folding(streams) > 0.2
        assert folding(generator.anti_alias[0](streams)) < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_graphs_survive_a_full_fft_plan_cache():
    """FFTs run outside the graphs (the deployed evaluation) fill cuFFT's plan cache and evict the plans the
    captured graphs launch; replaying them then crashes. A full cache drops the graphs, which are recaptured."""
    torch.manual_seed(0)
    cache = torch.backends.cuda.cufft_plan_cache[0]
    previous = cache.max_size
    cache.clear()
    cache.max_size = 400
    try:
        system = TRAINING_SYSTEMS["styletts"](small_config()).cuda().train()
        critics = system.critics().cuda()
        optimizer = adamw([{"params": list(system.parameters()), "scale": 1.0}], "cuda", lr=1e-6, betas=(0.8, 0.99), weight_decay=0.01)
        critic_optimizer = adamw([{"params": list(critics.parameters())}], "cuda", lr=1e-6, betas=(0.8, 0.99), weight_decay=0.0)
        recipe = dict(weights=system.WEIGHTS, learning_rate=1e-6, warmup=1, lr_half_life=1e5, grad_clip=10.0, gan_start=100, gan_ramp=10,
                      feature_matching=system.FEATURE_MATCHING, teacher_hold=0, teacher_ramp=1, teacher_final=0.0, word_teacher_hold=0,
                      word_teacher_ramp=1, fixed_style_hold=1, fixed_style_ramp=1, bf16=False)
        graphed = GraphedSteps(system, critics, optimizer, critic_optimizer, recipe)
        batch = collate([synthetic_item(0, seed=1), synthetic_item(1, seed=2)])
        graphed(batch, 1)
        graphed(batch, 2)
        for n in range(100, 100 + cache.max_size):      # an evaluation's worth of new FFT sizes
            torch.fft.rfft(torch.ones(n, device="cuda"))
        assert cache.size == cache.max_size
        metrics = graphed(batch, 3)
        torch.cuda.synchronize()
        assert graphed.recaptures == 1 and torch.isfinite(metrics["loss"])
    finally:
        cache.clear()
        cache.max_size = previous


def test_render_undoes_its_delay():
    """``delayed`` there and back is the identity away from the edges, so ``Generator.render`` at a delay
    differs from the undelayed render only by what the generator ties to its own grid."""
    from voca.styletts import delayed
    wave = torch.randn(2, 300)
    d = torch.tensor([7])
    assert torch.equal(delayed(delayed(wave, d), -d)[:, 7:-7], wave[:, 7:-7])
    assert torch.equal(delayed(wave, d)[:, 7:], wave[:, :-7])

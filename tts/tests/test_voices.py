"""Several voices in one model: balanced sampling, per-voice styles and validation sampling,
TinyVoca's GAN balance, the own-timing pass, and the StyleTTS schedules (GAN ramp, fixed styles) under CUDA graphs."""

import collections
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from voca.config import Config
from voca.data import BucketSampler, OwnTexts, TemperedSampler, collate
from voca.speak import COMMA_PAUSE, STOP_PAUSE
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


def own_batch(frames=120):
    """A batch of two, with what the own-timing pass says (the recorded sentence)."""
    batch = collate([synthetic_item(0, frames=frames, seed=1), synthetic_item(1, frames=frames, seed=2)])
    return {**batch, "own_tokens": batch["tokens"][:1], "own_lexical": batch["lexical"][:1], "own_lengths": batch["token_lengths"][:1]}


def test_moving_frames_ship_the_integers_and_teach_the_durations():
    """The value is the hard repetition; following the gradient moves the boundary to where the target has it."""
    from voca.styletts import frames_of, moving_frames
    values = torch.eye(3)[None]

    def owners(durations):
        return torch.searchsorted(durations.cumsum(1), torch.arange(30)[None], right=True).clamp(max=2)
    target = frames_of(values, owners(torch.tensor([[10, 14, 6]])))
    soft = torch.tensor([[10.0, 8.0, 12.0]], requires_grad=True)
    for _ in range(300):
        durations = soft.detach().round().long()
        moved = moving_frames(values, durations, soft, owners(durations))
        assert torch.equal(moved, frames_of(values, owners(durations)))
        (gradient,) = torch.autograd.grad((moved - target).square().sum(), soft)
        soft = (soft - 0.2 * gradient).detach().requires_grad_(True)
    assert durations[0, :2].tolist() == [10, 14]      # the last token ends past the window


def test_own_texts(tmp_path):
    """Half the own-timing rows say an unrecorded sentence of as many tokens (or the closest shorter), padded alike."""
    from voca.phonebert import Corpus
    from voca.text import frontend_fingerprint
    lengths = torch.tensor([14, 14, 9])
    total = int(lengths.sum())
    torch.save({"frontend": frontend_fingerprint(), "lengths": lengths, "tokens": torch.full((total,), 7, dtype=torch.int16),
                "lexical": torch.ones(total, len(LEXICAL), dtype=torch.int8), "occurrence": torch.zeros(total, dtype=torch.int16),
                "gram": torch.zeros(total, dtype=torch.int16), "words": torch.zeros(total, dtype=torch.int32)}, tmp_path / "train.pt")
    texts = OwnTexts(tmp_path / "train.pt")
    generator = torch.Generator().manual_seed(0)
    assert {texts.rows(14, generator) for _ in range(50)} == {0, 1} and texts.rows(12, generator) == 2 and texts.rows(99, generator) < 2 and texts.rows(8, generator) is None
    batch = collate([synthetic_item(k % 2, seed=k) for k in range(8)] + [synthetic_item(0, phones=20)])
    replaced = 0
    for _ in range(40):
        own = texts.add(batch, generator)
        assert len(own["own_tokens"]) == 5
        for tokens, lexical, n, recorded in zip(own["own_tokens"], own["own_lexical"], own["own_lengths"], batch["tokens"]):
            unrecorded = not torch.equal(tokens, recorded)
            replaced += unrecorded
            assert n == 14 and (tokens[n:] == 0).all()
            if unrecorded:
                assert (tokens[:n] == 7).all() and (lexical[:n] == 1).all() and (lexical[n:] == 0).all()
    assert replaced / 200 == pytest.approx(0.5, abs=0.12)


def test_own_timing_is_what_ships_and_teaches_every_part():
    """The own-timing pass speaks as ``synthesize`` does (durations, melody, styles), and its audio's gradient reaches
    the durations, the melody, the decoder and the generator, but not the text encoder."""
    torch.manual_seed(0)
    c = small_config()     # with its dropout, which this pass is without
    assert c.dropout > 0
    system = TRAINING_SYSTEMS["styletts"](c).train()
    m = system.model
    m.style.normal_()
    batch = own_batch(frames=480)     # room for what an untrained predictor says
    batch["own_tokens"] = batch["own_tokens"].clone()
    batch["own_tokens"][0, 6] = SYMBOL_TO_ID[","]     # the app's minimum pause
    speech = {}
    original = m.prosody_condition
    m.prosody_condition = lambda p: speech.setdefault("condition", original(p))
    durations = m.predict_durations
    m.predict_durations = lambda *a, **k: speech.setdefault("durations", durations(*a, **k))
    n = int(batch["own_lengths"][0])
    m.eval()
    _, _, shipped = m.synthesize(batch["own_tokens"][:, :n], batch["own_lexical"][:, :n], batch["own_lengths"], batch["speakers"][:1],
                                 comma_pause=COMMA_PAUSE, stop_pause=STOP_PAUSE)
    assert int(shipped["durations"][0, 6]) >= 15
    size = int(shipped["durations"].sum())
    ours = dict(speech)
    speech.clear()
    m.train()
    wave, valid = system.own_timing(batch, torch.zeros(2), size)
    assert torch.equal(speech["durations"][:, :n], ours["durations"]) and valid.tolist() == [1.0]
    assert torch.allclose(speech["condition"], ours["condition"], atol=1e-4) and wave.shape == (1, size * c.hop)
    speech.clear()
    assert system.own_timing(batch, torch.zeros(2), size + 1)[1].tolist() == [0.0]
    wave.square().mean().backward()
    p = m.predictor
    for part in (p.duration_proj, p.lstm, p.F0_proj, p.N_proj, m.decoder.encode, m.decoder.generator.conv_post, m.decoder.generator.m_source):
        assert sum(float(x.grad.abs().sum()) for x in part.parameters()) > 0
    assert float(p.duration_proj.weight.grad[-1].abs().sum()) > 0     # pauses: whether there is one
    assert set(map(id, p.parameters())) <= set(map(id, system.parts()["prosody"]))
    assert sum(map(len, system.parts().values())) == len(list(system.parameters()))
    assert all(x.grad is None for x in m.encoder.parameters() if x is not m.tokens.weight)


def test_melody_does_not_depend_on_padding():
    """A batch predicts each utterance's melody as inference does alone; the waveform losses teach it."""
    torch.manual_seed(0)
    c = dataclasses.replace(small_config(), dropout=0.0)
    system = TRAINING_SYSTEMS["styletts"](c).train()
    states = torch.randn(2, 14, system.model.predictor.shared.input_size - 3)
    durations = torch.full((2, 14), 10)
    durations[0, 8:] = 0
    index, _, lengths = system.model.align(durations, torch.ones(2, 14, dtype=torch.bool))
    style = torch.randn(2, c.style_dim)
    together = system.model.melody(states.gather(1, index[..., None].expand(-1, -1, states.shape[-1])), durations, index, style, lengths)
    alone = system.model.melody(states[:1].gather(1, index[:1, :80, None].expand(-1, -1, states.shape[-1])), durations[:1], index[:1, :80], style[:1], lengths[:1])
    assert torch.allclose(together[:1, :80], alone, atol=1e-5)
    x = torch.randn(1, c.style_prosody_channels, 80)
    block = system.model.predictor.F0[0]
    assert torch.allclose(block(x, style[:1], torch.ones(1, 1, 80)), block(x, style[:1]), atol=1e-4)     # the unmasked norm's values
    batch = collate([synthetic_item(0, seed=1), synthetic_item(1, seed=2)])
    weights = {**{name: 0.0 for name in system.WEIGHTS}, "mel_convergence": 1.0}
    system.eval()     # no speech model here
    system.objective(batch, (0.0, 0.0, 0.0, 1.0), weights)[0].backward()
    assert float(system.model.predictor.F0_proj.weight.grad.abs().sum()) > 0 and float(system.model.predictor.N_proj.weight.grad.abs().sum()) > 0


def test_complex_spectrogram_critic_sees_phase():
    """The critic's scores and features depend on phase, as the recording's magnitude alone cannot."""
    from voca.losses import ComplexSpecDiscriminator
    torch.manual_seed(0)
    critic = ComplexSpecDiscriminator(512).eval()
    wave = (0.3 * torch.randn(2, 8192)).requires_grad_()
    scores, features = critic(wave)
    assert len(features) == 6 and all(f.shape[0] == 2 and f.shape[1] == (1 if f is features[-1] else 32) for f in features)
    assert scores.shape[0] == 2
    (gradient,) = torch.autograd.grad(scores.square().mean(), wave)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    with torch.no_grad():
        # A negated copy has the same magnitude spectrogram but not the same phase.
        assert not torch.allclose(critic(-wave)[0], scores, atol=1e-3)


def test_energy_distance_rewards_two_different_takes():
    """Rows with a second take count as half the energy distance: the plain distance when the takes are
    equal, and less when they differ while both stay as far from the recording."""
    from voca.losses import MelConvergenceLoss
    torch.manual_seed(0)
    loss = MelConvergenceLoss(24000)
    target, a, b = (0.2 * torch.randn(3, 12000) for _ in range(3))
    assert torch.allclose(loss(a, target, a[:2])[0], loss(a, target)[0]) and loss(a, target, a[:2])[1] == 0
    assert loss(a, target, b[:2])[0] < loss(a, target)[0] and loss(a, target, b[:2])[1] > 0


def test_contour_loss_sees_direction():
    from voca.train import contour_loss
    time = torch.arange(99)[None] / 99
    sung = torch.ones(1, 99)
    fall = -0.3 * time
    assert contour_loss(fall, fall, sung) == 0
    assert contour_loss(torch.full((1, 99), fall.mean()), fall, sung) > contour_loss(fall + 0.2, fall, sung)     # a level melody; a shifted one
    assert contour_loss(0.3 * time, fall, sung) > contour_loss(torch.full((1, 99), fall.mean()), fall, sung)     # the opposite movement


def test_prosody_critic_teaches_melody_and_durations():
    """The critic's scores reach the predicted durations (through ``soft``) and melody, on padded and real tokens apart."""
    torch.manual_seed(0)
    c = dataclasses.replace(small_config(), dropout=0.0)
    system = TRAINING_SYSTEMS["styletts"](c).train()
    system.model.style.normal_()
    critics = system.critics()
    batch = own_batch(frames=480)
    mask, speakers, aligned = system.batch_tokens(batch)
    told = system.own_prosody(batch["tokens"], batch["lexical"], batch["token_lengths"], speakers, 720)
    recorded = critics.prosody.delivery(batch["durations"], None, batch["prosody"], *aligned)
    spoken = critics.prosody.delivery(told.durations, told.soft, told.prosody, told.index, told.frame_mask)
    assert recorded.shape == spoken.shape == (*batch["tokens"].shape, 4)
    rows = torch.ones(len(speakers))
    real, fake = (critics.prosody(batch["tokens"], batch["lexical"], speakers, x, mask) for x in (recorded, spoken))
    critics.prosody.discriminator_loss(real, fake, rows).backward(retain_graph=True)
    critics.prosody.generator_loss(critics.prosody(batch["tokens"], batch["lexical"], speakers, spoken, mask), rows).backward()
    p = system.model.predictor
    assert float(p.F0_proj.weight.grad.abs().sum()) > 0 and float(p.duration_proj.weight.grad.abs().sum()) > 0
    assert float(p.words.output.weight.grad.abs().sum()) > 0      # the word head, through its features
    frozen = critics.prosody(batch["tokens"], batch["lexical"], speakers, spoken.detach(), mask)[0]
    longer = torch.nn.functional.pad(batch["tokens"], (0, 3)), torch.nn.functional.pad(batch["lexical"], (0, 0, 0, 3))
    padded = critics.prosody(*longer, speakers, torch.nn.functional.pad(spoken.detach(), (0, 0, 0, 3)), torch.nn.functional.pad(mask, (0, 3)))[0]
    assert torch.allclose(padded[:, :-4], frozen[:, :-1], atol=1e-5) and torch.allclose(padded[:, -1], frozen[:, -1], atol=1e-5)


def test_critics_one_at_a_time_give_the_joint_gradients():
    """Backward critic by critic gives the gradients of the summed losses, at the critics and at the waveforms."""
    from voca.losses import StyleCritics, relativistic_loss, masked_mean
    torch.manual_seed(0)
    critics = StyleCritics(24000, 90, 2).double()
    real, fake, own, short = (0.1 * torch.randn(n, 12000, dtype=torch.float64) for n in (2, 2, 1, 1))
    valid = torch.ones(1, dtype=torch.float64)
    critics.waveform_critics = lambda: (critics.mpd[0], critics.mrd[0], critics.complex_mrd[0])     # enough to compare
    critics.discriminator_backward(real, fake, own, valid, None, short)
    one = [p.grad.clone() for c in critics.waveform_critics() for p in c.parameters()]
    critics.zero_grad()
    total = 0
    for critic in critics.waveform_critics():
        r, f, o, e = (critic(w)[0] for w in (real, fake, own, short))
        total = total + masked_mean(torch.cat(((1 - r).square().mean(1), (1 - e).square().mean(1))), torch.ones(3, dtype=torch.float64)) \
            + masked_mean(torch.cat((f.square().mean(1), o.square().mean(1))), torch.ones(3, dtype=torch.float64)) + relativistic_loss([r], [f])
    total.backward()
    assert all(torch.allclose(a, p.grad) for a, p in zip(one, [p for c in critics.waveform_critics() for p in c.parameters()]))
    f, o = fake.clone().requires_grad_(), own.clone().requires_grad_()
    *_, at_fake, at_own = critics.generator_gradients(real, f, o, valid, 2.0, 3.0, 0.5)
    joint = 0
    for critic in critics.waveform_critics():
        real_score, real_features = critic(real)
        score, features = critic(f)
        joint = joint + 2.0 * ((1 - score).square().mean() + relativistic_loss([score], [real_score.detach()])
                               + 3.0 * sum((a.detach() - b).abs().mean() for a, b in zip(real_features, features))) \
            + 0.5 * (1 - critic(o)[0]).square().mean()
    expected = torch.autograd.grad(joint, (f, o))
    assert torch.allclose(at_fake, expected[0], atol=1e-8) and torch.allclose(at_own, expected[1], atol=1e-8)


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
    recipe = dict(weights={**system.WEIGHTS, "shift": 1.0}, learning_rate=1e-6, warmup=1, lr_half_life=1e5, gan_start=0, gan_ramp=10,
                  feature_matching=system.FEATURE_MATCHING, teacher_hold=0, teacher_ramp=1, teacher_final=0.0, word_teacher_hold=0,
                  word_teacher_ramp=1, fixed_style_hold=1, fixed_style_ramp=1, bf16=False)
    graphed = GraphedSteps(system, critics, optimizer, critic_optimizer, recipe)
    batch = own_batch(frames=480)     # room for what an untrained predictor says
    for step in (1, 2, 3):      # step 1 runs eagerly and is captured, 2 and 3 replay it
        metrics = graphed(batch, step)
        assert float(metrics["gan_scale"]) == pytest.approx((step + 1) / 10 * system.WEIGHTS["gan"]), step
        assert float(metrics["fixed_style_fraction"]) == (0.0 if step == 1 else 1.0), step
        assert 0 < float(metrics["shift"]) < 10, step        # the shift-consistency loss, its random delay drawn in the graph
        assert float(metrics["own_valid"]) == 1.0 and float(metrics["own_adversarial"]) > 0 and float(metrics["prosody_adversarial"]) > 0
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
        recipe = dict(weights=system.WEIGHTS, learning_rate=1e-6, warmup=1, lr_half_life=1e5, gan_start=100, gan_ramp=10,
                      feature_matching=system.FEATURE_MATCHING, teacher_hold=0, teacher_ramp=1, teacher_final=0.0, word_teacher_hold=0,
                      word_teacher_ramp=1, fixed_style_hold=1, fixed_style_ramp=1, bf16=False)
        graphed = GraphedSteps(system, critics, optimizer, critic_optimizer, recipe)
        batch = own_batch()
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

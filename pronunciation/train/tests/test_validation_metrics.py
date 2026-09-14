"""Holdout identity, production decoding, update warmup and VAD timing."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from src import train_unified as training
from src.validation_metrics import (
    DecodeMetrics, collapse_phones, edit_counts, identify_validation,
    normalize_sentence, sample_records, sentence_split,
)


class Records(Dataset):
    def __init__(self, records):
        self.samples = records

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_sentence_disjoint_across_languages_sources_and_nested_subsets():
    records = [{"sentence": f"sentence {i}", "wav_path": f"/eng/{i}.wav", "lang": "eng"}
               for i in range(30)]
    records += [{"sentence": " SENTENCE 1?! ", "wav_path": "/fra/other.wav", "lang": "fra"}]
    ds = ConcatDataset([Subset(Records(records), list(range(len(records))))])
    train, val = sentence_split(ds, .3)
    train_text = {normalize_sentence(r["sentence"]) for r in sample_records(train)}
    val_text = {normalize_sentence(r["sentence"]) for r in sample_records(val)}
    assert not train_text & val_text
    assert sentence_split(ds, .3)[1].indices == val.indices
    reverse = Records(list(reversed(records)))
    assert {r["wav_path"] for r in sample_records(sentence_split(reverse, .3)[1])} == {
        r["wav_path"] for r in sample_records(val)}


def test_fixed_subset_identity_survives_order_and_caps_each_language():
    records = [{"lang": lang, "wav_path": f"/{lang}/{i}.wav"}
               for lang, n in [("eng", 120), ("fra", 17)] for i in range(n)]
    ds = Records(records)
    identified, selected = identify_validation(Subset(ds, list(range(len(ds)))))
    reversed_ds, reversed_selected = identify_validation(Subset(ds, list(reversed(range(len(ds))))))
    assert selected == reversed_selected
    assert len([x for x in selected if x.startswith("eng/")]) == 100
    assert len([x for x in selected if x.startswith("fra/")]) == 17
    assert identified.__getitems__([4, 1])[0]["clip_id"] == "eng/4.wav"
    assert reversed_ds[0]["clip_id"] == "fra/16.wav"


def test_collapses_blank_separated_and_stress_split_repeats():
    assert collapse_phones([1, 1, 0, 1], 0) == [1, 1]
    assert collapse_phones([1, 1], 0, [0, 1]) == [1, 1]
    assert collapse_phones([1, 1], 0, [0, 0]) == [1]
    assert edit_counts([1, 2], [1]) == (0, 1, 0)
    assert edit_counts([1], [1, 2]) == (0, 0, 1)


def test_metrics_nonblank_first_padded_tails_empty_and_stable_selection():
    tokenizer = SimpleNamespace(convert_ids_to_tokens=lambda i: ["<pad>", "a", "b"][i])
    metrics = DecodeMetrics(tokenizer, 0, {"eng/a", "eng/b"})
    # Joint argmax picks blank, but the production gate emits phone a.
    lp = torch.tensor([.45, .3, .25]).log().expand(3, 4, 3).clone()
    outputs = {"log_probs": lp, "nonblank_logit": torch.tensor([[1., 1., 1., 1.],
               [-1., -1., 1., 1.], [1., 1., 1., 1.]]),
               "stress_logits": torch.tensor([[[2., 0.], [0., 2.], [2., 0.], [2., 0.]]]).expand(3, -1, -1)}
    batch = {"clip_ids": ["eng/a", "eng/b", "eng/skip"], "langs": ["eng"] * 3,
             "phoneme_ids": torch.tensor([[1, 1], [2, 0], [1, 0]]),
             "phoneme_lens": torch.tensor([2, 1, 1])}
    metrics.update(outputs, batch, torch.tensor([2, 2, 4]))
    actual = metrics.compute()["eng"]
    assert actual["per"] == pytest.approx(2 / 3)
    assert actual["deletions"] == pytest.approx(2 / 3)
    assert actual["insertions"] == 0
    assert actual["empty_fraction"] == .5
    assert actual["hyp_ref_length_ratio"] == pytest.approx(1 / 3)
    assert actual["factor_aware_per"] == pytest.approx(1 / 3)
    assert actual["clips"] == 2


def test_vad_centers_do_not_resize_to_clip_duration():
    ramp = torch.arange(20).float() / 20
    expected = torch.tensor([200, 520, 840]).float() / 256 / 20
    assert torch.allclose(training.sample_vad_centers(ramp, 3), expected)
    assert torch.allclose(training.sample_vad_centers(ramp[:5], 3), expected)
    assert training.sample_vad_centers(torch.tensor([.8]), 1).tolist() == pytest.approx([.8])
    loss = training.vad_loss(torch.zeros(1, 3), ramp[None], torch.tensor([20]), torch.tensor([3]))
    assert loss == pytest.approx(float(((expected - .5).abs() * 2).mean() * torch.log(torch.tensor(2.))))


class TinyModel(torch.nn.Module):
    language_head_specs = {}
    blank_id = 0
    backbone = SimpleNamespace(_get_feat_extract_output_lengths=lambda lengths: lengths)

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.calls = 0

    def forward(self, audio, **kwargs):
        self.calls += 1
        b, t = audio.shape
        logits = torch.stack([self.weight, -self.weight, self.weight * 2])
        return {"log_probs": logits.log_softmax(-1).expand(b, t, -1),
                "nonblank_logit": self.weight.expand(b, t),
                "stress_logits": logits.expand(b, t, -1), "language_head_logits": {}}


def tiny_batch(size=1):
    return {"audio": torch.ones(size, 4), "audio_mask": torch.ones(size, 4).long(),
            "phoneme_ids": torch.ones(size, 1).long(), "phoneme_lens": torch.ones(size).long(),
            "stress_seq": torch.ones(size, 1).long(), "tone_seq": torch.zeros(size, 1).long(),
            "pitch_accent_seq": torch.zeros(size, 1).long(),
            "stress_available": torch.ones(size).bool(), "tone_available": torch.zeros(size).bool(),
            "pitch_accent_available": torch.zeros(size).bool(), "langs": ["eng"] * size}


@pytest.mark.parametrize("epoch_batches", [2, 3])
def test_warmup_switches_after_exactly_400_updates_across_short_epochs(monkeypatch, epoch_batches):
    for name in ["reset_peak_memory_stats", "synchronize"]:
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    original_loss = training.joint_ctc_loss
    observed = []

    def record_loss(**kwargs):
        observed.append(kwargs["stress_logits"] is not None)
        return original_loss(**kwargs)

    monkeypatch.setattr(training, "joint_ctc_loss", record_loss)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    kwargs = dict(use_bf16=False, blank_id=0, stress_active=False, stress_weight=.3,
                  vad_weight=0., invalid_mass_weight=0., grad_clip_norm=1.,
                  debug_finite=True, max_train_batches=epoch_batches, stress_warmup_steps=400)
    first = training.train_epoch(model, [tiny_batch()] * 4, optimizer, "cpu", 1,
                                 optimizer_steps=398, **kwargs)
    second = training.train_epoch(model, [tiny_batch()] * 4, optimizer, "cpu", 2,
                                  optimizer_steps=first["optimizer_steps"], **kwargs)
    assert observed == [False, False] + [True] * (epoch_batches * 2 - 2)
    assert second["optimizer_steps"] == 398 + epoch_batches * 2


def test_validation_phone_objective_stable_and_single_forward_per_batch():
    model = TinyModel()
    kwargs = dict(use_bf16=False, blank_id=0, stress_weight=.3, debug_finite=True)
    joint = training.eval_epoch(model, [tiny_batch(2), tiny_batch()], "cpu", stress_active=True, **kwargs)
    assert model.calls == 2
    phone = training.eval_epoch(model, [tiny_batch(3)], "cpu", stress_active=False, **kwargs)
    assert joint["phone_ctc_loss"] == pytest.approx(phone["ctc_loss"], abs=1e-6)
    assert joint["per_lang_phone_ctc"]["eng"] == pytest.approx(phone["per_lang_ctc"]["eng"], abs=1e-6)
    assert joint["ctc_loss"] != pytest.approx(joint["phone_ctc_loss"])

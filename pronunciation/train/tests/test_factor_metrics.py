"""Production-onset factor scoring, on exact-phone matches of fixed probes."""
from types import SimpleNamespace

import pytest
import torch

from src.factorized_ctc import DEFAULT_LANGUAGE_HEAD_SPECS
from src.validation_metrics import DecodeMetrics, collapse_phones, edit_counts

FACTOR_METRICS = ("stress_acc", "stress_recall", "tone_acc", "tone_fp",
                  "pitch_acc", "pitch_fp", "pitch_majority_baseline")


TOKENIZER = SimpleNamespace(convert_ids_to_tokens=lambda i: ["<pad>", "a", "b", "c", "d"][i])


def example(lang="eng", ref=(1, 2), frames=(1, 1, 2, 2),
            labels=(1, 2, 0, 1), targets=(1, 0), target="stress", available=True, clip="probe"):
    phones = torch.tensor([frames])
    logits = torch.nn.functional.one_hot(phones, 5).float() * 4
    batch = {"clip_ids": [clip], "langs": [lang], "phoneme_ids": torch.tensor([ref]),
             "phoneme_lens": torch.tensor([len(ref)])}
    outputs = {"log_probs": logits.log_softmax(-1),
               "nonblank_logit": torch.where(phones == 0, -1., 1.),
               "stress_logits": torch.zeros(1, len(frames), 3), "language_head_logits": {}}
    for factor in ("stress", "tone", "pitch_accent"):
        batch[f"{factor}_seq"] = torch.tensor([targets]) if target == factor else torch.zeros(1, len(ref)).long()
        batch[f"{factor}_available"] = torch.tensor([available and target == factor])
    factor_logits = torch.nn.functional.one_hot(torch.tensor([labels]), 3 if target == "stress" else 6).float() * 4
    if target == "stress":
        outputs["stress_logits"] = factor_logits
    else:
        for name, spec in DEFAULT_LANGUAGE_HEAD_SPECS.items():
            if spec["target"] == target:
                outputs["language_head_logits"][name] = factor_logits
    return outputs, batch, torch.tensor([len(frames)])


def metrics(selected=("probe",), specs=DEFAULT_LANGUAGE_HEAD_SPECS):
    return DecodeMetrics(TOKENIZER, 0, set(selected), specs)


def factors(result):
    return {k: v for k, v in result.items() if k in FACTOR_METRICS}


def test_run_starts_and_diagonal_first_alignment():
    assert collapse_phones([0, 1, 1, 0, 1, 2], 0, return_starts=True) == ([1, 1, 2], [1, 4, 5])
    assert edit_counts([1, 1], [1], return_matches=True) == ((0, 1, 0), [(1, 0)])
    assert edit_counts([1], [1, 1], return_matches=True) == ((0, 0, 1), [(0, 1)])
    assert edit_counts([1, 2], [2, 1], return_matches=True) == ((2, 0, 0), [])
    assert edit_counts([], [1], return_matches=True) == ((0, 0, 1), [])
    assert edit_counts([1], [], return_matches=True) == ((0, 1, 0), [])


def test_onset_padding_exact_stress_and_aggregation():
    m = metrics()
    outputs, batch, lengths = example(frames=(1, 1, 2, 3))
    # Padding is a real phone and a stressed target, but neither may count.
    batch["phoneme_ids"] = torch.tensor([[1, 2, 3]])
    batch["stress_seq"] = torch.tensor([[1, 0, 2]])
    m.update(outputs, batch, torch.tensor([3]))
    assert factors(m.compute()["eng"]) == {"stress_acc": 1, "stress_recall": 1}
    m.update(*example(ref=(1,), frames=(1,), labels=(2,), targets=(1,)))
    assert factors(m.compute()["eng"]) == {"stress_acc": 2 / 3, "stress_recall": .5}
    # Secondary vs primary is wrong, not binary stressed recall.
    m.update(*example(ref=(1,), frames=(1,), labels=(1,), targets=(2,)))
    assert m.compute()["eng"]["stress_recall"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("lang", ["kor", "fra", "eng"])
def test_unavailable_stress_including_no_stress_language_and_unannotated_french(lang):
    m = metrics()
    m.update(*example(lang=lang, available=False))
    assert factors(m.compute()[lang]) == {}


@pytest.mark.parametrize("lang,target,prefix", [("tha", "tone", "tone"), ("zho-hans", "tone", "tone"),
                                               ("jpn", "pitch_accent", "pitch")])
def test_language_heads_bearers_nonbearers_and_availability(lang, target, prefix):
    m = metrics()
    m.update(*example(lang=lang, target=target, labels=(2, 1, 1, 0), targets=(2, 0)))
    actual = m.compute()[lang]
    assert actual[f"{prefix}_acc"] == 1
    assert actual[f"{prefix}_fp"] == 1
    assert "stress_acc" not in actual
    m.update(*example(lang=lang, target=target, labels=(0, 2, 0, 1), targets=(2, 0)))
    assert m.compute()[lang][f"{prefix}_acc"] == .5
    assert m.compute()[lang][f"{prefix}_fp"] == .5
    unavailable = metrics()
    unavailable.update(*example(lang=lang, target=target, available=False))
    assert factors(unavailable.compute()[lang]) == {}
    wrong_lang = metrics()
    wrong_lang.update(*example(lang="eng", target=target))
    assert factors(wrong_lang.compute()["eng"]) == {}


@pytest.mark.parametrize("ref,frames", [((1, 2), (1, 3)), ((1, 2), (1,)), ((1,), (1, 3))])
def test_substitution_deletion_insertion_excluded(ref, frames):
    m = metrics()
    m.update(*example(ref=ref, frames=frames, labels=(1,) + (0,) * (len(frames) - 1),
                      targets=(1,) * len(ref)))
    assert factors(m.compute()["eng"]) == {"stress_acc": 1, "stress_recall": 1}


@pytest.mark.parametrize("frames", [(0, 0), (3, 3)])
def test_empty_hypothesis_or_no_matches_omits_factor_metrics(frames):
    m = metrics()
    m.update(*example(frames=frames, labels=(1, 1)))
    assert factors(m.compute()["eng"]) == {}


def test_zero_bearer_and_zero_nonbearer_denominators_omitted():
    m = metrics()
    m.update(*example(targets=(0, 0), labels=(0, 1, 0, 1)))
    assert factors(m.compute()["eng"]) == {"stress_acc": 1}
    for targets, expected in [((0, 0), {"pitch_fp": 0}),
                              ((1, 1), {"pitch_acc": 0, "pitch_majority_baseline": 1})]:
        m = metrics()
        m.update(*example(lang="jpn", target="pitch_accent", targets=targets, labels=(0, 1, 0, 1)))
        assert factors(m.compute()["jpn"]) == expected


def test_baseline_full_validation_unselected_unmatched_padding_and_updates():
    m = metrics()
    # One matched low, one substituted high: both count toward baseline.
    m.update(*example(lang="jpn", target="pitch_accent", targets=(1, 2), frames=(1, 3), labels=(1, 0)))
    assert m.compute()["jpn"]["pitch_majority_baseline"] == .5
    outputs, batch, lengths = example(lang="jpn", target="pitch_accent", targets=(2, 2, 1),
                                      ref=(1, 2, 3), clip="unselected")
    batch["phoneme_lens"] = torch.tensor([2])
    m.update(outputs, batch, lengths)
    m.update(*example(lang="jpn", target="pitch_accent", targets=(1, 1), available=False, clip="unselected"))
    actual = m.compute()["jpn"]
    assert actual["pitch_majority_baseline"] == .75
    assert actual["pitch_acc"] == 1
    assert actual["clips"] == 1
    baseline_only = metrics(selected=())
    baseline_only.update(outputs, batch, lengths)
    assert baseline_only.compute() == {"jpn": {"pitch_majority_baseline": 1}}


def test_custom_head_name_and_language_from_specs():
    m = metrics(specs={"custom": {"lang": "other", "target": "tone", "num_labels": 6}})
    outputs, batch, lengths = example(lang="other", target="tone")
    outputs["language_head_logits"] = {"custom": outputs["language_head_logits"]["tha_tone"]}
    m.update(outputs, batch, lengths)
    assert factors(m.compute()["other"]) == {"tone_acc": 1, "tone_fp": 0}

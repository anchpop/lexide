import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import StressDataset, collate_fn
from src.factorized_ctc import FactorizedCTCModel
from src.joint_ctc import (
    compose_joint_log_probs,
    encode_joint_targets,
    joint_ctc_loss,
)


def test_composed_distribution_is_normalized_and_targets_share_its_encoding():
    phone_lp = F.log_softmax(torch.randn(2, 4, 5), dim=-1)
    stress_logits = torch.randn(2, 4, 3)
    tone_logits = torch.randn(2, 4, 6)

    joint_lp, blank = compose_joint_log_probs(
        phone_lp,
        phone_blank_id=0,
        factor_logits=[stress_logits, tone_logits],
    )

    assert joint_lp.shape == (2, 4, 5 * 3 * 6 + 1)
    assert blank == 5 * 3 * 6
    assert torch.allclose(
        torch.logsumexp(joint_lp, dim=-1),
        torch.zeros(2, 4),
        atol=1e-4,
    )

    phones = torch.tensor([[1, 1]])
    stress = torch.tensor([[0, 1]])
    tones = torch.tensor([[2, 2]])
    encoded = encode_joint_targets(phones, [stress, tones], [3, 6])
    assert encoded.tolist() == [[20, 26]]
    # Equal phones with different stress are different CTC symbols, so native
    # CTC collapse does not incorrectly merge the two occurrences.
    assert encoded[0, 0] != encoded[0, 1]


def test_phone_only_composition_matches_original_ctc():
    phone_lp = F.log_softmax(torch.randn(2, 6, 5), dim=-1)
    targets = torch.tensor([[1, 2, 0], [3, 0, 0]])
    input_lengths = torch.tensor([6, 5])
    target_lengths = torch.tensor([2, 1])

    expected = F.ctc_loss(
        phone_lp.transpose(0, 1), targets, input_lengths, target_lengths,
        blank=0, reduction="none", zero_infinity=True,
    )
    joint_lp, joint_blank = compose_joint_log_probs(
        phone_lp, phone_blank_id=0, factor_logits=[],
    )
    actual = F.ctc_loss(
        joint_lp.transpose(0, 1), targets, input_lengths, target_lengths,
        blank=joint_blank, reduction="none", zero_infinity=True,
    )
    assert torch.allclose(actual, expected, atol=1e-4)


def test_unrelated_and_unavailable_language_heads_receive_no_gradient():
    raw_phone = torch.randn(2, 6, 5, requires_grad=True)
    phone_lp = F.log_softmax(raw_phone, dim=-1)
    stress = torch.randn(2, 6, 3, requires_grad=True)
    tha = torch.randn(2, 6, 6, requires_grad=True)
    zho = torch.randn(2, 6, 6, requires_grad=True)
    jpn = torch.randn(2, 6, 3, requires_grad=True)

    specs = {
        "tha_tone": {"lang": "tha", "target": "tone", "num_labels": 6},
        "zho_hans_tone": {
            "lang": "zho-hans", "target": "tone", "num_labels": 6,
        },
        "jpn_pitch_accent": {
            "lang": "jpn", "target": "pitch_accent", "num_labels": 3,
        },
    }
    loss = joint_ctc_loss(
        phone_log_probs=phone_lp,
        phone_targets=torch.tensor([[1, 2], [2, 1]]),
        target_lengths=torch.tensor([2, 2]),
        input_lengths=torch.tensor([6, 6]),
        phone_blank_id=0,
        stress_logits=stress,
        stress_weight=0.3,
        stress_targets=torch.tensor([[0, 1], [1, 0]]),
        language_head_logits={
            "tha_tone": tha,
            "zho_hans_tone": zho,
            "jpn_pitch_accent": jpn,
        },
        language_head_specs=specs,
        aligned_targets={
            "tone": torch.tensor([[1, 0], [0, 0]]),
            "pitch_accent": torch.zeros(2, 2, dtype=torch.long),
        },
        factor_available={
            "tone": torch.tensor([True, False]),
            "pitch_accent": torch.tensor([False, False]),
        },
        # The second row is Thai too, but its tone label was explicitly
        # withheld. It must not update the Thai head either.
        langs=["tha", "tha"],
    )
    loss.backward()

    assert raw_phone.grad is not None and raw_phone.grad.abs().sum() > 0
    assert stress.grad is not None and stress.grad.abs().sum() > 0
    assert tha.grad is not None and tha.grad[0].abs().sum() > 0
    assert tha.grad[1].abs().sum() == 0
    assert zho.grad is None
    assert jpn.grad is None


def test_dataset_encodes_aligned_prosody_and_availability(tmp_path, monkeypatch):
    rows = [
        {
            "file": "jpn.wav", "lang": "jpn", "sentence": "箸",
            "phonemes": ["h", "a"], "stress": [0, 0],
            "pitch_accent": [
                None,
                {"phrase": 1, "mora": 1, "phrase_moras": 2, "nucleus": 1},
            ],
        },
        {
            "file": "jpn-withheld.wav", "lang": "jpn", "sentence": "外来語",
            "phonemes": ["ɡ", "a"], "stress": [0, 0],
            "pitch_accent_exclude_reason": "njd_hts_phrase_boundary_mismatch",
        },
    ]
    labels = tmp_path / "phonemes.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for row in rows:
        (tmp_path / row["file"]).touch()

    monkeypatch.setattr("src.dataset.sf.info", lambda _: SimpleNamespace(frames=8000))
    monkeypatch.setattr(
        "src.dataset.sf.read",
        lambda *args, **kwargs: (np.ones(8000, dtype=np.float32) * 0.1, 16000),
    )

    class Tokenizer:
        unk_token_id = -1
        _ids = {"h": 1, "a": 2, "ɡ": 3}

        def convert_tokens_to_ids(self, token):
            return self._ids.get(token, self.unk_token_id)

    dataset = StressDataset(labels, Tokenizer())
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["pitch_accent_seq"].tolist() == [[0, 2], [0, 0]]
    assert batch["pitch_accent_available"].tolist() == [True, False]
    assert batch["tone_available"].tolist() == [False, False]


def test_language_heads_are_selective_and_checkpointed(tmp_path, monkeypatch):
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

    monkeypatch.setattr(
        "src.factorized_ctc.AutoModel.from_pretrained", lambda *_args, **_kwargs: DummyBackbone(),
    )
    model = FactorizedCTCModel(
        model_name="dummy", vocab_size=5, mel_sidechannel=False, mlp_heads=True,
    )
    phone_only = model(torch.randn(2, 5))
    assert phone_only["language_head_logits"] == {}
    thai = model(torch.randn(2, 5), active_language_heads=["tha_tone"])
    assert set(thai["language_head_logits"]) == {"tha_tone"}
    assert thai["language_head_logits"]["tha_tone"].shape == (2, 5, 6)

    model.save_to_dir(tmp_path)
    restored = FactorizedCTCModel.load_from_dir(tmp_path)
    assert restored.language_head_specs == model.language_head_specs
    assert set(restored.language_heads) == set(model.language_heads)
    for name in model.language_heads:
        for expected, actual in zip(
            model.language_heads[name].parameters(),
            restored.language_heads[name].parameters(),
        ):
            assert torch.equal(expected, actual)


def test_reduction_none_returns_ordered_per_sample_losses():
    torch.manual_seed(0)
    phone_lp = F.log_softmax(torch.randn(3, 6, 5), dim=-1)
    stress = torch.randn(3, 6, 3)
    kwargs = dict(
        phone_targets=torch.tensor([[1, 2], [2, 1], [3, 3]]),
        target_lengths=torch.tensor([2, 2, 2]),
        input_lengths=torch.tensor([6, 6, 6]),
        phone_blank_id=0,
        stress_logits=stress,
        stress_weight=0.3,
        stress_targets=torch.tensor([[0, 1], [1, 0], [2, 2]]),
        language_head_specs={
            "tha_tone": {"lang": "tha", "target": "tone", "num_labels": 6},
        },
        language_head_logits={"tha_tone": torch.randn(3, 6, 6)},
        aligned_targets={
            "tone": torch.tensor([[1, 0], [0, 0], [0, 0]]),
            "pitch_accent": torch.zeros(3, 2, dtype=torch.long),
        },
        factor_available={
            "tone": torch.tensor([True, False, False]),
            "pitch_accent": torch.tensor([False, False, False]),
        },
        # Samples 0 and 2 are both Thai but only 0 has a tone label, so they
        # land in different groups — the order restore is genuinely exercised.
        langs=["tha", "eng", "tha"],
    )
    per_sample = joint_ctc_loss(
        phone_log_probs=phone_lp, reduction="none", **kwargs,
    )
    mean = joint_ctc_loss(phone_log_probs=phone_lp, **kwargs)
    assert per_sample.shape == (3,)
    assert torch.allclose(per_sample.mean(), mean)

    for i, lang in enumerate(kwargs["langs"]):
        single = joint_ctc_loss(
            phone_log_probs=phone_lp[i:i + 1],
            phone_targets=kwargs["phone_targets"][i:i + 1],
            target_lengths=kwargs["target_lengths"][i:i + 1],
            input_lengths=kwargs["input_lengths"][i:i + 1],
            phone_blank_id=0,
            stress_logits=stress[i:i + 1],
            stress_weight=0.3,
            stress_targets=kwargs["stress_targets"][i:i + 1],
            language_head_specs=kwargs["language_head_specs"],
            language_head_logits={
                "tha_tone": kwargs["language_head_logits"]["tha_tone"][i:i + 1],
            },
            aligned_targets={
                k: v[i:i + 1] for k, v in kwargs["aligned_targets"].items()
            },
            factor_available={
                k: v[i:i + 1] for k, v in kwargs["factor_available"].items()
            },
            langs=[lang],
        )
        assert torch.allclose(per_sample[i], single, atol=1e-5)

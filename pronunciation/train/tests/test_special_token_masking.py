"""Sentinels are never phone emissions, including with feature_mode=off."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2CTCTokenizer

from src.articulatory import detect_special_token_ids
from src.factorized_ctc import FactorizedCTCModel


TOKENS = ["a", "<s>", "</s>", "<pad>", "b", "<unk>"]
BLANK = 3
SPECIALS = [1, 2, 5]
MODES = ["off", "aux", "factorized"]


class DummyBackbone(torch.nn.Module):
    config = SimpleNamespace(hidden_size=4, final_dropout=0.0)

    def forward(self, input_values, attention_mask=None):
        return SimpleNamespace(last_hidden_state=input_values.unsqueeze(-1).expand(-1, -1, 4))

    def save_pretrained(self, path):
        pass


@pytest.fixture(autouse=True)
def dummy_backbone(monkeypatch):
    monkeypatch.setattr(
        "src.factorized_ctc.AutoModel.from_pretrained",
        lambda *args, **kwargs: DummyBackbone(),
    )


def make_model(mode, special_ids=None):
    table = torch.ones(len(TOKENS), 2, dtype=torch.long)
    table[4] = 2
    kwargs = {}
    if mode == "factorized":
        kwargs["feature_table"] = table
    elif mode == "aux":
        kwargs["aux_feature_table"] = table
        kwargs["feature_emission_weight"] = 0.3
    return FactorizedCTCModel(
        model_name="dummy", vocab_size=len(TOKENS), blank_id=BLANK,
        special_token_ids=special_ids, mel_sidechannel=False,
        mlp_heads=False, language_head_specs={}, **kwargs,
    ).eval()


def assert_masked(model):
    result = model(torch.ones(1, 5), labels=torch.tensor([[0, 4]]))
    lp = result["log_probs"]
    assert torch.equal(lp[..., SPECIALS].exp(), torch.zeros_like(lp[..., SPECIALS]))
    assert torch.allclose(lp.exp().sum(-1), torch.ones(1, 5), atol=1e-6)
    assert torch.equal(lp[..., BLANK], F.logsigmoid(-result["nonblank_logit"]))
    assert (lp[..., [0, 4]].exp() > 0).all()
    assert not set(lp.argmax(-1).flatten().tolist()) & set(SPECIALS)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return lp.detach()


@pytest.mark.parametrize("mode", MODES)
def test_detect_mask_and_round_trip_every_mode(mode, tmp_path):
    ids = detect_special_token_ids(TOKENS)
    assert ids == [1, 2, 3, 5]
    model = make_model(mode, ids + ids)
    assert model._masked_slots.tolist() == ids
    assert "_masked_slots" in dict(model.named_buffers())
    if model.phoneme_head is not None:
        with torch.no_grad():
            # Even overwhelming sentinel logits must not affect normalization.
            model.phoneme_head.bias[SPECIALS] = 1000
    expected = assert_masked(model)
    model.save_to_dir(tmp_path)
    payload = torch.load(tmp_path / "factorized_heads.pt", weights_only=False)
    assert payload["masked_slots"] == ids
    loaded = FactorizedCTCModel.load_from_dir(tmp_path).eval()
    assert torch.equal(assert_masked(loaded), expected)


def remove_saved_mask(path, old_mode=False):
    heads_path = path / "factorized_heads.pt"
    payload = torch.load(heads_path, weights_only=False)
    del payload["masked_slots"]
    if old_mode:
        del payload["feature_mode"]
    torch.save(payload, heads_path)


@pytest.mark.parametrize("mode", MODES)
def test_legacy_checkpoint_recovers_mask_from_local_tokenizer(mode, tmp_path):
    model = make_model(mode)
    model.save_to_dir(tmp_path)
    remove_saved_mask(tmp_path, old_mode=mode != "aux")
    # Deliberately nonstandard IDs; <unk> lives in added_tokens.json rather
    # than vocab.json, so recovery must use the complete tokenizer vocabulary.
    vocab = {token: idx for idx, token in enumerate(TOKENS[:-1])}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    tokenizer = Wav2Vec2CTCTokenizer(str(tmp_path / "vocab.json"))
    assert tokenizer.get_vocab()["<unk>"] == 5
    tokenizer.save_pretrained(tmp_path)
    loaded = FactorizedCTCModel.load_from_dir(tmp_path).eval()
    assert loaded._masked_slots.tolist() == [1, 2, 3, 5]
    assert_masked(loaded)
    if mode != "off":
        assert not loaded._first_occurrence_mask[SPECIALS + [BLANK]].any()
        assert loaded._first_occurrence_mask[0]
    loaded.save_to_dir(tmp_path)
    assert torch.load(tmp_path / "factorized_heads.pt", weights_only=False)["masked_slots"] == [1, 2, 3, 5]


@pytest.mark.parametrize("mode", MODES)
def test_legacy_checkpoint_without_tokenizer_still_loads(mode, tmp_path):
    model = make_model(mode)
    expected = model(torch.ones(1, 5))["log_probs"]
    model.save_to_dir(tmp_path)
    remove_saved_mask(tmp_path, old_mode=mode != "aux")
    loaded = FactorizedCTCModel.load_from_dir(tmp_path).eval()
    assert loaded._masked_slots.tolist() == [BLANK]
    assert torch.equal(loaded(torch.ones(1, 5))["log_probs"], expected)


@pytest.mark.parametrize("mode", MODES)
def test_saved_mask_takes_precedence_over_tokenizer(mode, tmp_path):
    model = make_model(mode, SPECIALS)
    model.save_to_dir(tmp_path)
    # A present but invalid tokenizer must not be read for modern checkpoints.
    (tmp_path / "vocab.json").write_text("not json")
    loaded = FactorizedCTCModel.load_from_dir(tmp_path).eval()
    assert loaded._masked_slots.tolist() == [1, 2, 3, 5]
    assert_masked(loaded)

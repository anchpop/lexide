"""Real scratch checkpoint round trips through the unchanged serving contract."""
import base64
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import huggingface_hub
import pytest
import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Processor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train" / "src"))
import wav2vec2_phoneme as service
from factorized_ctc import FactorizedCTCModel

Service = service.Wav2Vec2Phoneme._get_user_cls()


@pytest.fixture(params=[False, True])
def worker(tmp_path, request, monkeypatch):
    torch.manual_seed(42)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5, "|": 6}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    processor = Wav2Vec2Processor(
        Wav2Vec2FeatureExtractor(do_normalize=request.param, return_attention_mask=True),
        Wav2Vec2CTCTokenizer(str(tmp_path / "vocab.json")),
    )
    processor.save_pretrained(tmp_path)
    model = FactorizedCTCModel(
        model_name="scratch", vocab_size=len(vocab), blank_id=0,
        special_token_ids=[0, 1, 2, 3, 6],
        scratch_config={"hidden_size": 16, "num_hidden_layers": 1, "heads": 2,
                        "ff_size": 32, "kernel_size": 3},
        language_head_specs={
            "tha_tone": {"lang": "tha", "target": "tone", "num_labels": 6},
            "jpn_pitch_accent": {"lang": "jpn", "target": "pitch", "num_labels": 3},
        },
    )
    model.save_to_dir(tmp_path)
    load = FactorizedCTCModel.load_from_dir
    calls = []

    def tracked_load(path):
        calls.append(path)
        return load(path)

    monkeypatch.setattr(FactorizedCTCModel, "load_from_dir", tracked_load)
    obj = Service()
    obj._pool_number = 0
    obj._label_cache = {}
    obj.load_error = None
    load_scratch = obj._load_scratch_model
    monkeypatch.setattr(obj, "_load_scratch_model", lambda path: load_scratch(path, device="cpu"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(tmp_path))

    def no_wav2vec2(*args, **kwargs):
        pytest.fail("scratch loading must not construct a wav2vec2 backbone")

    monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", no_wav2vec2)
    obj._load_model_impl()
    assert calls == [tmp_path]
    assert obj.processor.feature_extractor.do_normalize == request.param
    assert obj.backbone.frontend.dft.dtype == torch.float32
    assert obj.frame_rate_ms == 20.0
    return obj


def test_shared_forward_and_aux_selection(worker):
    audio = torch.randn(3, 1040)
    lengths = torch.tensor([720, 880, 1040])
    mask = (torch.arange(1040)[None] < lengths[:, None]).long()
    with torch.inference_mode():
        lp, stress, nb, aux = worker._forward(
            audio, ["tha", "fra", "fra"], mask, [False, False, True],
        )
        expected = worker._scratch_model(audio, attention_mask=mask,
                                         active_language_heads=set(worker.language_heads))
    torch.testing.assert_close(lp, expected["log_probs"], rtol=0, atol=0)
    torch.testing.assert_close(stress, expected["stress_logits"], rtol=0, atol=0)
    torch.testing.assert_close(nb, expected["nonblank_logit"].sigmoid(), rtol=0, atol=0)
    assert [set(heads) for heads in aux] == [
        {"tha_tone"}, set(), {"tha_tone", "jpn_pitch_accent"},
    ]
    for i, heads in enumerate(aux):
        for name, values in heads.items():
            torch.testing.assert_close(values, expected["language_head_logits"][name][i])
    with torch.inference_mode():
        single = worker._forward(audio[:1], language="jpn")
    assert set(single[3]) == {"jpn_pitch_accent"}


def test_sample_masks_frame_lengths_and_wire_contract(worker):
    # Exercise the mel/subsample boundary where physical output can include an
    # extra padded frame. Only valid CTC frames may reach any wire-format head.
    lengths = [400, 559, 560, 719, 720, 880, 1040]
    requests = [{"audio": torch.randn(n).tolist(), "language": "tha",
                 "return_frame_matrix": True, "return_all_heads": True} for n in lengths]
    items = [(i, req, worker._prepare_audio(req)) for i, req in enumerate(requests)]
    seen = []
    hook = worker.backbone.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append((args, kwargs)), with_kwargs=True,
    )
    outputs = worker._forward_requests(items)
    hook.remove()
    _, kwargs = seen[0]
    assert kwargs["input_values"].dtype == torch.float32
    assert kwargs["attention_mask"].sum(-1).tolist() == lengths
    assert outputs[-1] == [(n - 400) // 320 + 1 for n in lengths]
    # Batch padding must not alter the valid scratch outputs.
    for i, item in enumerate(items):
        single = worker._forward_requests([item])
        frames = outputs[-1][i]
        for batched, alone in zip(outputs[:3], single[:3]):
            torch.testing.assert_close(batched[i, :frames], alone[0, :frames], atol=2e-5, rtol=2e-5)
    results = dict(worker._process_microbatch(items, len(items), 1))
    for i, n in enumerate(lengths):
        matrix = results[i]["frame_matrix"]
        frames = (n - 400) // 320 + 1
        assert matrix["schema_version"] == 1
        assert matrix["sample_rate"] == 16000
        assert matrix["frame_rate_ms"] == 20.0
        assert set(matrix["heads"]) == {"phone", "nonblank", "stress", "tha_tone", "jpn_pitch_accent"}
        for head in matrix["heads"].values():
            assert head["shape"][0] == frames
            decoded = np.frombuffer(zlib.decompress(base64.b64decode(head["data"])), dtype="<f2")
            assert decoded.size == np.prod(head["shape"])
        phone = matrix["heads"]["phone"]
        assert phone["value_semantics"] == "joint_log_probability"
        decoded = np.frombuffer(zlib.decompress(base64.b64decode(phone["data"])), dtype="<f2")
        np.testing.assert_array_equal(decoded.reshape(phone["shape"]), outputs[0][i, :frames].half().numpy())
    with pytest.raises(ValueError, match="too short"):
        worker._prepare_audio({"audio": [0.0] * 399})

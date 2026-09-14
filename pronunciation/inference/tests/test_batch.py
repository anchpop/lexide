from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor

from pronunciation.inference.infer import (
    decode_frames, plan_batches, read_audio, transcribe_audio_batch, transcribe_batch,
)


class Tokenizer:
    def convert_ids_to_tokens(self, index):
        return ["<pad>", "a", "b", "<unk>"][index]


def test_ctc_runs_stress_blanks_and_special_tokens():
    ids = np.array([1, 1, 0, 1, 3, 3, 2, 2])
    stress = np.log(np.array([[.1,.8,.1], [.9,.05,.05], [.3,.3,.4], [.1,.1,.8],
                              [.3,.3,.4], [.3,.3,.4], [.1,.8,.1], [.1,.8,.1]]))
    tokens = decode_frames(ids, stress, np.ones(8), Tokenizer(), 0)
    assert [(t['token'], t['stress'], t['frame']) for t in tokens] == [
        ('a', 0, 0), ('a', 2, 3), ('b', 1, 6),
    ]


def test_planner_budget_order_and_oversized_clip():
    lengths = [100, 20, 30, 10, 500, 20]
    batches = list(plan_batches(lengths, 3, 100))
    assert batches == [[3, 1, 5], [2], [0], [4]]
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert list(plan_batches([], 2, 100)) == []
    with pytest.raises(ValueError):
        list(plan_batches([10], 0))


class Model:
    blank_id = 0
    mel_sidechannel = True
    regularized_heads = False

    def __init__(self, norm="layer"):
        self.calls = []
        self.backbone = SimpleNamespace(
            config=SimpleNamespace(model_type="wav2vec2", feat_extract_norm=norm),
            _get_feat_extract_output_lengths=lambda lengths: lengths,
        )

    def __call__(self, values, attention_mask=None):
        self.calls.append((values.clone(), attention_mask))
        b, t = values.shape
        # Padding deliberately emits 'b', so failing to trim adds a token.
        ids = torch.where(values == 0, 2, 1)
        nonblank_logit = torch.where(values == 0, 1.0, values)
        phone_probs = torch.nn.functional.one_hot(ids, 4).float()
        log_probs = torch.nn.functional.logsigmoid(nonblank_logit)[..., None] + phone_probs.log()
        log_probs[..., self.blank_id] = torch.nn.functional.logsigmoid(-nonblank_logit)
        return {"log_probs": log_probs,
                "stress_logits": torch.zeros(b,t,3), "nonblank_logit": nonblank_logit}


def test_raw_waveform_mask_trimming_and_order(tmp_path):
    paths = []
    for i, length in enumerate([30, 10, 20]):
        p = tmp_path / f'{i}.wav'
        sf.write(p, np.full(length, .1 * (i + 1)), 16000, subtype='FLOAT')
        paths.append(p)
    model = Model()
    result = transcribe_batch(paths, model, SimpleNamespace(tokenizer=Tokenizer()),
                              torch.device('cpu'), batch_size=3)
    assert len(model.calls) == 1
    values, mask = model.calls[0]
    assert mask.sum(-1).tolist() == [10, 20, 30]
    assert values[0, :10].tolist() == pytest.approx([.2]*10)
    assert all([t['token'] for t in r] == ['a'] for r in result)
    assert [r[0]['nonblank_prob'] for r in result] == pytest.approx(
        torch.sigmoid(torch.tensor([.1,.2,.3])).tolist(), abs=1e-4)
    # Distinct lengths restored even when inference completes in sorted order.
    assert values.shape == (3, 30)


def test_group_norm_keeps_singleton_semantics():
    model = Model('group')
    transcribe_audio_batch([np.ones(10), np.ones(20)], model,
                           SimpleNamespace(tokenizer=Tokenizer()), torch.device('cpu'))
    assert [v.shape for v, _ in model.calls] == [(1, 10), (1, 20)]
    assert all(mask is None for _, mask in model.calls)


def test_normalization_happens_per_clip():
    model = Model()
    model.mel_sidechannel = False
    processor = Wav2Vec2FeatureExtractor(return_attention_mask=False)
    processor.tokenizer = Tokenizer()
    audios = [np.arange(10, dtype=np.float32), np.arange(20, dtype=np.float32)]
    transcribe_audio_batch(audios, model, processor, torch.device('cpu'))
    values, _ = model.calls[0]
    for i, audio in enumerate(audios):
        expected = processor(audio, sampling_rate=16000, return_tensors='pt').input_values[0]
        torch.testing.assert_close(values[i, :len(audio)], expected)


@pytest.mark.parametrize('sr,shape', [(8000,(100,)), (16000,(100,2)), (16000,(0,))])
def test_invalid_audio(tmp_path, sr, shape):
    path = tmp_path/'bad.wav'
    sf.write(path, np.zeros(shape), sr)
    with pytest.raises(ValueError, match='16kHz mono'):
        read_audio(path)


def test_real_encoder_padding_matches_singletons(monkeypatch):
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    from pronunciation.inference.infer import FactorizedCTCModel
    import src.factorized_ctc as module
    torch.manual_seed(7)
    torch.set_num_threads(2)
    config = Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, conv_dim=(8,8,8), conv_stride=(5,2,2),
        conv_kernel=(10,3,3), num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
        feat_extract_norm='layer', mask_time_prob=0, hidden_dropout=0,
        attention_dropout=0, feat_proj_dropout=0, final_dropout=0,
    )
    monkeypatch.setattr(module.AutoModel, 'from_pretrained', lambda _: Wav2Vec2Model(config))
    model = FactorizedCTCModel(model_name='tiny', vocab_size=4, language_head_specs={}, mel_sidechannel=False).eval()
    processor = Wav2Vec2FeatureExtractor(return_attention_mask=False)
    processor.tokenizer = Tokenizer()
    rng = np.random.default_rng(7)
    audios = [rng.normal(size=n).astype(np.float32) for n in (401, 997, 2103)]
    baseline = [transcribe_audio_batch([a], model, processor, torch.device('cpu'))[0] for a in audios]
    batched = transcribe_audio_batch(audios, model, processor, torch.device('cpu'))
    for a,b in zip(baseline, batched):
        assert [(t['token'],t['stress'],t['frame']) for t in a] == [
            (t['token'],t['stress'],t['frame']) for t in b]
        assert [t['nonblank_prob'] for t in a] == pytest.approx(
            [t['nonblank_prob'] for t in b], abs=1e-4)


def test_oom_splits_without_losing_or_reordering_clips(monkeypatch):
    import pronunciation.inference.infer as inference
    calls = []
    def run(audios, *args):
        calls.append(list(audios))
        if len(audios) > 2:
            raise torch.cuda.OutOfMemoryError('simulated')
        return audios
    monkeypatch.setattr(inference, '_transcribe_audio_batch', run)
    assert transcribe_audio_batch([0,1,2,3,4], None, None, None) == [0,1,2,3,4]
    assert calls == [[0,1,2,3,4], [0,1], [2,3,4], [2], [3,4]]


def test_singleton_oom_is_not_hidden(monkeypatch):
    import pronunciation.inference.infer as inference
    def fail(*args):
        raise torch.cuda.OutOfMemoryError('simulated')
    monkeypatch.setattr(inference, '_transcribe_audio_batch', fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        transcribe_audio_batch([0], None, None, None)

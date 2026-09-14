import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pronunciation.inference.infer import decide_frames, decode_frames, transcribe_audio_batch


class Tokenizer:
    tokens = ["<s>", "a", "<pad>", "b", "</s>", "c", "<unk>", "d"]

    def convert_ids_to_tokens(self, index):
        return self.tokens[index]


BLANK = 2


def factorized_outputs(probabilities):
    nonblank = torch.tensor(probabilities)
    phones = torch.tensor([0., .4, 0., .3, 0., .3, 0., 0.])
    log_probs = nonblank.log()[..., None] + phones.log()
    log_probs[..., BLANK] = (1 - nonblank).log()
    return log_probs, torch.logit(nonblank)


@pytest.mark.parametrize('shape', [(3,), (2, 3)])
@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_split_mass_threshold_shape_and_device(shape, device):
    probabilities = torch.tensor([.6, .4, .5]).expand(shape).tolist()
    log_probs, nonblank_logit = factorized_outputs(probabilities)
    log_probs, nonblank_logit = log_probs.to(device), nonblank_logit.to(device)
    original = log_probs.clone()
    # Even at p(nonblank)=.6 every individual phone loses to blank.
    assert log_probs.argmax(-1).eq(BLANK).all()
    ids = decide_frames(log_probs, nonblank_logit, Tokenizer(), BLANK)
    assert ids.tolist() == torch.tensor([1, BLANK, BLANK]).expand(shape).tolist()
    assert ids.shape == nonblank_logit.shape
    assert ids.device == log_probs.device
    assert ids.dtype == torch.long
    torch.testing.assert_close(log_probs, original)


def test_special_tokens_and_masked_phone_never_win():
    log_probs, nonblank_logit = factorized_outputs([.6])
    log_probs[..., [0, 4, 6]] = 0  # Legacy checkpoints may leave specials unmasked.
    log_probs[..., 1] = -torch.inf
    assert decide_frames(log_probs, nonblank_logit, Tokenizer(), BLANK).tolist() == [3]


def test_decisions_keep_ctc_runs_and_blank_separators():
    log_probs, nonblank_logit = factorized_outputs([.6, .6, .4, .6, .5, .6])
    ids = decide_frames(log_probs, nonblank_logit, Tokenizer(), BLANK)
    tokens = decode_frames(ids.numpy(), np.zeros((6, 3)),
                           torch.sigmoid(nonblank_logit).numpy(), Tokenizer(), BLANK)
    assert [(t['token'], t['frame']) for t in tokens] == [('a', 0), ('a', 3), ('a', 5)]


def test_batch_transcription_uses_factorized_decisions():
    log_probs, nonblank_logit = factorized_outputs([[.6, .6, .4, .6]])

    class Model:
        blank_id = BLANK
        mel_sidechannel = True
        backbone = SimpleNamespace(
            config=SimpleNamespace(model_type='wav2vec2', feat_extract_norm='layer'),
            _get_feat_extract_output_lengths=lambda lengths: lengths,
        )

        def __call__(self, values, attention_mask=None):
            return dict(log_probs=log_probs, nonblank_logit=nonblank_logit,
                        stress_logits=torch.zeros(1, 4, 3))

    result = transcribe_audio_batch([np.ones(4)], Model(),
                                    SimpleNamespace(tokenizer=Tokenizer()), torch.device('cpu'))
    assert [(t['token'], t['frame']) for t in result[0]] == [('a', 0), ('a', 3)]


def test_modal_reading_matches_shared_decisions_without_loading_modal():
    # Compile only the actual method: importing this module configures a Modal
    # app/image and requires Modal dependencies, neither needed for pure decoding.
    path = Path(__file__).resolve().parents[2] / 'espeak_audit' / 'modal_aligner.py'
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'VadCleanAligner')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_reading')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    aligner = SimpleNamespace(blank_id=BLANK, masked_slots=[BLANK, 7],
                              inv_vocab=dict(enumerate(Tokenizer.tokens)))
    log_probs, nonblank_logit = factorized_outputs([[.6, .6, .4, .6, .5, .6]])
    log_probs[..., [0, 4, 6]] = 0
    log_probs[..., 1] = -torch.inf
    original = log_probs.clone()
    assert namespace['_reading'](aligner, log_probs) == ['b', 'b', 'b']
    torch.testing.assert_close(log_probs, original)
    # Explicit checkpoint masking also excludes a slot even if its score is finite.
    log_probs[..., 7] = 0
    assert namespace['_reading'](aligner, log_probs) == ['b', 'b', 'b']

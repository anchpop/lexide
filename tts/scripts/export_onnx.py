"""Export one unpadded StyleVoca utterance at a time. Run with VOCA_LANGUAGE and PYTHONPATH=."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from voca.speak import load_model
from voca.text import LEXICAL, frontend_fingerprint, frontend_identity


class Stage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class Encode(Stage):
    def forward(self, tokens, lexical, speaker):
        m = self.model
        lengths = torch.ones_like(speaker) * tokens.shape[1]
        hidden, mask = m.encode(tokens, lexical, lengths, speaker)
        text = m.prosody_text(hidden, tokens, lexical, mask)
        mean, spread = m.delivery.prior(text, speaker, mask)
        return hidden, text, mean, spread.exp(), m.style[speaker]


class Predict(Stage):
    def forward(self, tokens, text, speaker, draw, speed, comma_pause, stop_pause):
        mask = torch.ones_like(tokens, dtype=torch.bool)
        words = self.model.words(tokens, mask)
        return (*self.model.predict_reading(text, speaker, draw, words, None, mask, tokens,
                                            speed, comma_pause, stop_pause), words.index)


class Melody(Stage):
    def forward(self, tokens, states, durations, owner, style, word_index, voiced_vowels, pitch_range):
        m = self.model
        mask = torch.ones_like(owner, dtype=torch.bool)
        prosody = m.reading_prosody(states, durations, owner, style, None, tokens, mask, voiced_vowels, pitch_range)
        final, rising = m.ending(prosody, word_index, owner)
        return prosody, final, rising


class Blend(Stage):
    def forward(self, prosody, durations, owner, final, typical, typical_durations):
        return self.model.blend_ending(prosody, durations, owner, final, typical, typical_durations)


class Render(Stage):
    """A real DFT filterbank and its overlap-add inverse for the short generator FFT."""
    def __init__(self, model):
        super().__init__(model)
        g = model.decoder.generator
        basis = torch.fft.rfft(torch.eye(g.fft, dtype=torch.float64)).T
        analysis = torch.cat((basis.real, basis.imag))
        scale = torch.full((len(basis), 1), 2 / g.fft)
        scale[[0, -1]] *= 0.5
        self.register_buffer('analysis', (analysis * g.window).float()[:, None])
        self.register_buffer('synthesis', (analysis * scale.repeat(2, 1) * g.window).float()[:, None])

    def forward(self, tokens, hidden, durations, owner, prosody, acoustic, pitch_shift, noise):
        m = self.model
        g = m.decoder.generator
        frame = torch.arange(owner.shape[1], device=owner.device)[None]
        text = m.decoder_text(hidden, tokens, durations, owner, frame)
        condition = m.prosody_condition(prosody)
        pitch = condition[..., 0] + pitch_shift
        condition = torch.cat((pitch[..., None], condition[..., 1:]), -1)
        features = m.decoder.features(text, condition, acoustic)
        source = g.m_source(m.frequency(pitch), condition[..., 1], features, noise)
        pad = g.fft // 2
        har = F.conv1d(F.pad(source[:, None], (pad, pad), mode='reflect'), self.analysis, stride=g.hop)
        mag, phase = g.coefficients(features, acoustic, har)
        spectrum = torch.cat((mag * phase.cos(), mag * phase.sin()), 1)
        wave = F.conv_transpose1d(spectrum, self.synthesis, stride=g.hop)
        envelope = F.conv_transpose1d(torch.ones_like(mag[:, :1]), g.window.square()[None, None], stride=g.hop)
        return (wave[:, 0, pad:-pad] / envelope[:, 0, pad:-pad])


def pack_weights(path, precision):
    """Compact weights on the wire, restored to float32 by constant folding at session load."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(path)
    unpack = []
    for initializer in list(model.graph.initializer):
        values = numpy_helper.to_array(initializer)
        if values.dtype != np.float32 or values.size < 512:
            continue
        name = initializer.name
        model.graph.initializer.remove(initializer)
        if precision == "fp16":
            model.graph.initializer.append(numpy_helper.from_array(values.astype(np.float16), name + ".half"))
            unpack.append(helper.make_node("Cast", [name + ".half"], [name], to=TensorProto.FLOAT))
        else:
            axis = int(np.argmax(values.shape)) if values.ndim == 3 and values.shape[0] == 2 else 0
            reduce = tuple(i for i in range(values.ndim) if i != axis)
            scale = np.maximum(np.max(np.abs(values), axis=reduce, keepdims=True) / 127, 1e-12)
            quantized = np.rint(values / scale).clip(-127, 127).astype(np.int8)
            data = [quantized, scale.ravel(), np.zeros(scale.size, dtype=np.int8)]
            inputs = [name + suffix for suffix in (".q", ".scale", ".zero")]
            model.graph.initializer.extend(numpy_helper.from_array(x, n) for x, n in zip(data, inputs))
            unpack.append(helper.make_node("DequantizeLinear", inputs, [name], axis=axis))
    for node in reversed(unpack):
        model.graph.node.insert(0, node)
    onnx.save(model, path)


@torch.no_grad()
def export(checkpoint, out, weights):
    m = load_model(checkpoint)
    for layer in m.modules():
        if isinstance(layer, nn.InstanceNorm1d):
            # ONNX InstanceNormalization requires explicit scale and offset.
            layer.weight = nn.Parameter(torch.ones(layer.num_features), requires_grad=False)
            layer.bias = nn.Parameter(torch.zeros(layer.num_features), requires_grad=False)
        if parametrize.is_parametrized(layer, "weight"):
            parametrize.remove_parametrizations(layer, "weight", leave_parametrized=True)
    out.mkdir(parents=True, exist_ok=True)
    c = m.config
    manifest = dict(sampleRate=c.sample_rate, hop=c.hop, speakers=c.speakers, maxFrames=c.max_frames,
                    maxTokens=c.max_tokens, lexical=list(LEXICAL), frontend=frontend_identity(),
                    fingerprint=frontend_fingerprint())
    (out / 'model.json').write_text(json.dumps(manifest))

    def stage(name, module, inputs, outputs, axis):
        module.eval()
        values = tuple(inputs.values())
        path = out / f'{name}.onnx'
        torch.onnx.export(module, values, path, dynamo=False, opset_version=18,
                          input_names=list(inputs), output_names=outputs, dynamic_axes=axis)
        precision = weights
        if weights == "render-fp16":
            precision = "fp16" if name == "render" else "fp32"
        if precision != "fp32":
            pack_weights(path, precision)
        return module(*values)

    tokens = torch.tensor([[1, 6, 7, 3, 8, 2]])
    speaker = torch.tensor([0])
    lexical = torch.zeros((*tokens.shape, len(LEXICAL)), dtype=torch.long)
    n, f = {1: 'tokens_n'}, {1: 'frames'}

    hidden, text, mean, _, acoustic = stage(
        'encode', Encode(m), dict(tokens=tokens, lexical=lexical, speaker=speaker),
        ['hidden', 'text', 'mean', 'scale', 'acoustic'], dict.fromkeys(['tokens', 'lexical', 'hidden', 'text'], n))
    style, states, durations, words = stage(
        'predict', Predict(m), dict(tokens=tokens, text=text, speaker=speaker, draw=mean,
                                    speed=torch.tensor(1.), comma_pause=torch.tensor(.15), stop_pause=torch.tensor(.3)),
        ['style', 'states', 'durations', 'word_index'], dict.fromkeys(['tokens', 'text', 'states', 'durations', 'word_index'], n))

    owner = torch.repeat_interleave(torch.arange(tokens.shape[1]), durations[0])[None]
    prosody, final, _ = stage(
        'melody', Melody(m), dict(tokens=tokens, states=states, durations=durations, owner=owner, style=style,
                                  word_index=words, voiced_vowels=torch.tensor(True), pitch_range=torch.tensor(1.1)),
        ['prosody', 'final', 'rising'], dict(tokens=n, states=n, durations=n, owner=f, word_index=n, prosody=f, final=f))
    stage('blend', Blend(m), dict(prosody=prosody, durations=durations, owner=owner, final=final,
                                  typical=prosody.clone(), typical_durations=durations.clone()),
          ['blended'], dict(prosody=f, durations=n, owner=f, final=f, typical={1: 'typical_frames'}, typical_durations=n, blended=f))
    noise = torch.randn(1, 1, owner.shape[1] * c.hop)
    stage('render', Render(m), dict(tokens=tokens, hidden=hidden, durations=durations, owner=owner, prosody=prosody,
                                    acoustic=acoustic, pitch_shift=torch.zeros(1, 1), noise=noise),
          ['wave'], dict(tokens=n, hidden=n, durations=n, owner=f, prosody=f, pitch_shift={1: 'pitch_offsets'},
                         noise={2: 'samples'}, wave={1: 'samples'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('output', type=Path)
    parser.add_argument('--weights', choices=['fp32', 'render-fp16', 'fp16', 'int8'], default='fp32')
    args = parser.parse_args()
    torch.set_num_threads(1)
    export(args.checkpoint, args.output, args.weights)
    if args.weights == 'int8':
        from scripts.bundle_voice import bundle
        print(bundle(args.output, args.output))

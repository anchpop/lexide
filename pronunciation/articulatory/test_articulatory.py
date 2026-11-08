"""CUDA numerical, closure, timing, data-integrity and inference-boundary tests."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from evaluate import edit_distance, summarize
from network import ControlNet, GestureLoss
from objective import AudioLoss, envelope, mel_bank
from prepare import AUDIO, MODEL, OUT
from maeda import NAMES, _native, _scaled_native, contours, preset
from synth import SR, Synth, effort, effort_weights, geometry, interpolate


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(11)

    def test_token_edits(self):
        self.assertEqual(edit_distance(['aɪ', 'tʃ'], ['aɪ', 'ʃ']), 1)
        self.assertEqual(edit_distance([], ['x']), 1)
        self.assertEqual(edit_distance(['a', 'b'], []), 2)
        result = summarize([dict(original_reading=['a', 'b'], fitted_reading=['a']),
                            dict(original_reading=['x'], fitted_reading=['x'])])
        self.assertEqual((result['edits'], result['reference_count']), (1, 3))
        self.assertAlmostEqual(result['per'], 1/3)

    def test_exact_100hz(self):
        ramp = torch.arange(101., device='cuda')[:, None]
        result = interpolate(ramp, 16159)
        self.assertEqual(float(result[16000, 0]), 100.)
        self.assertAlmostEqual(float(result[15920, 0]), 99.5)
        self.assertEqual(float(result[-1, 0]), 100.)

    def test_loss_and_envelope(self):
        target = torch.randn(SR, device='cuda')*.1
        prediction = (target*.8).requires_grad_()
        loss = AudioLoss(target, 'mel')(prediction)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        features = envelope(torch.rand(3, 513, device='cuda'), mel_bank(1024, 'cuda'))
        self.assertTrue(torch.isfinite(features).all())
        self.assertLess(float(features.mean(-1).abs().max()), 1e-6)

    def test_closure_and_cuda_gradients(self):
        synth = Synth(closures=True).cuda()
        art = torch.zeros(101, len(NAMES), device='cuda')
        art[:, 4] = -15.
        art[:, -1] = -15.
        areas, *_ = geometry(art, closures=True)
        self.assertAlmostEqual(float(areas.min()), .001)
        source = torch.tensor([-40., -40., 0., 0.], device='cuda').repeat(101, 1)
        f0 = torch.full((101,), 150., device='cuda')
        output = synth(art, source, f0, torch.zeros_like(f0), SR)
        self.assertLess(float(output.square().mean().sqrt()), 1e-12)
        art = (torch.randn(101, len(NAMES), device='cuda')*2).requires_grad_()
        anatomy = torch.tensor([.94, 1.06], device='cuda', requires_grad=True)
        output = synth(art, source*0, f0, torch.ones_like(f0), SR, anatomy)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(anatomy.grad).all())
        self.assertTrue((anatomy.grad.abs() > 0).all())
        self.assertTrue(torch.isfinite(art.grad).all())
        self.assertGreater(float(art.grad.abs().sum()), 0.)

    def test_maeda_geometry(self):
        for device in ('cpu', 'cuda'):
            raw = torch.zeros(2, 3, len(NAMES), device=device, dtype=torch.float64, requires_grad=True)
            scale = torch.tensor([[.9], [1.1]], device=device, dtype=raw.dtype, requires_grad=True)
            for closures in (False, True):
                areas, length, coupling, gap = geometry(raw, scale, closures)
                outer, inner = contours(raw, scale, closures)
                self.assertEqual(areas.shape, (2, 3, 40))
                self.assertEqual(outer.shape, (2, 3, 40, 2))
                self.assertTrue(torch.allclose(gap, (outer-inner).norm(dim=-1)))
                self.assertTrue(torch.allclose(outer[..., 0, :]+inner[..., 0, :], torch.zeros_like(outer[..., 0, :])))
                self.assertTrue((outer[..., -1, 0] > 0).all())
                self.assertTrue(torch.allclose(areas[1]/areas[0], torch.full_like(areas[0], (1.1/.9)**2)))
                self.assertTrue(torch.allclose(length[1]/length[0], torch.full_like(length[0], 1.1/.9)))
                loss = areas.sum()+length.sum()+coupling.sum()+gap.sum()+outer.sum()+inner.sum()
                gradients = torch.autograd.grad(loss, (raw, scale))
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue((gradients[0].abs().sum((0, 1)) > 0).all())
        # Double-precision finite differences of all controls AND anatomy scale.
        raw = preset('ah', dtype=torch.float64).requires_grad_()
        scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda r, s: geometry(r, s, True), (raw, scale)))
        # Saturated, collided shapes must remain finite, including backward at seals.
        raw = (torch.randn(256, len(NAMES))*20).requires_grad_()
        values = geometry(raw, closures=True)
        self.assertTrue(all(torch.isfinite(v).all() for v in values))
        sum(v.sum() for v in values).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreaterEqual(float(values[0].detach().min()), .001-1e-8)
        # Arc-length resampling retains native length and outlet, not index length.
        raw = preset('uw')
        _, _, lengths, areas = _native(raw)
        a, length, *_ = geometry(raw)
        self.assertAlmostEqual(float(length*100), float(lengths.sum()), places=5)
        self.assertAlmostEqual(float(a[-1]), float(areas[-1]), places=5)
        with self.assertRaises(ValueError):
            geometry(torch.zeros(9))

    def test_speaker_isolation(self):
        net = ControlNet(20, 3).cuda()
        # Nonzero randomized heads avoid a vacuous test of zero initialization.
        for head in (net.character, net.articulation, net.source):
            torch.nn.init.normal_(head.weight, std=.1)
        args = dict(phone=torch.arange(101, device='cuda') % 20,
                    relative=torch.linspace(0, 1, 101, device='cuda'),
                    language=torch.tensor(0, device='cuda'),
                    f0=torch.full((101,), 150., device='cuda'),
                    voiced=torch.ones(101, device='cuda'))
        a, s, scale = net(**args, speaker=torch.tensor(0, device='cuda'))
        b, t, other_scale = net(**args, speaker=torch.tensor(1, device='cuda'))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.allclose(s-t, (s-t)[0].expand_as(s), atol=1e-6))
        self.assertEqual(scale.shape, (2,))
        self.assertTrue(((scale >= .85) & (scale <= 1.15)).all())
        self.assertTrue(((other_scale >= .85) & (other_scale <= 1.15)).all())
        self.assertTrue((scale != other_scale).all())
        self.assertGreater(float((s-t).abs().sum().detach()), 0.)
        gradient, = torch.autograd.grad(a.sum(), net.speaker.weight, allow_unused=True)
        self.assertIsNone(gradient)

    def test_two_scale_anatomy(self):
        raw = preset('ah', dtype=torch.float64).requires_grad_()
        scale = torch.tensor([.94, 1.07], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda r, s: geometry(r, s, True), (raw, scale)))
        values = geometry(raw, scale)
        gradient, = torch.autograd.grad(sum(v.sum() for v in values), scale)
        self.assertTrue(torch.isfinite(gradient).all() and (gradient.abs() > 0).all())
        for s in (.85, 1., 1.15):
            for a, b in zip(geometry(raw, s), geometry(raw, [s, s])):
                self.assertTrue(torch.allclose(a, b, atol=1e-10))
            for a, b in zip(contours(raw, s), contours(raw, [s, s])):
                self.assertTrue(torch.allclose(a, b, atol=1e-10))
        o, i, lengths, native = _scaled_native(raw, scale)
        self.assertTrue(torch.allclose(((o+i)/2).diff(dim=-2).norm(dim=-1), lengths))
        areas, length, _, gap = values
        self.assertTrue(torch.allclose(length*100, lengths.sum()))
        self.assertTrue(torch.allclose(areas[-1], native[-1]))
        outer, inner = contours(raw, scale)
        self.assertTrue(torch.allclose(gap, (outer-inner).norm(dim=-1)))
        # Volume of integrated cells, except the deliberately exact outlet cell.
        edges = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        cutoff = lengths.sum()*39/40
        overlap = (torch.minimum(edges[1:], cutoff)-edges[:-1]).clamp_min(0)
        self.assertTrue(torch.allclose(areas[:-1].sum()*length*100/40, (overlap*native).sum()))

    def test_silence_effort(self):
        audio = torch.zeros(SR+37)
        audio[4000:12000] = .2
        weights = effort_weights(audio)
        self.assertEqual(len(weights), len(audio)//160+1)
        self.assertEqual(float(weights[5]), 20.)
        self.assertEqual(float(weights[50]), 1.)
        self.assertTrue(torch.equal(weights, effort_weights(audio*2)))
        art = torch.ones(len(weights), len(NAMES), requires_grad=True)
        self.assertGreater(float(effort(art, weights).detach()), float(effort(art).detach()))
        effort(art, weights).backward()
        self.assertTrue(torch.isfinite(art.grad).all())
        self.assertTrue((effort_weights(torch.zeros(SR)) == 20).all())

    def test_gesture_masks(self):
        vocabulary = ['<pad>', 'm', 'n', 'ŋ', 'b', 't', 'ɾ', 'k', 'ɡ', 'a', 'õ', 'aː̃', 'ɐ̃ʊ̃']
        loss = GestureLoss(vocabulary)
        self.assertEqual(loss.classes[0].sum(), 0)
        self.assertEqual(loss.classes[1].tolist(), [True, True, False, False, False])
        self.assertEqual(loss.classes[5].tolist(), [False, False, True, True, False])
        self.assertTrue(loss.classes[10:, 1].all())
        self.assertFalse(loss.classes[9].any())
        art = torch.zeros(len(vocabulary), len(NAMES), requires_grad=True)
        phones = torch.arange(len(vocabulary))
        masked = loss(art, 1., phones, torch.zeros(len(phones)))
        self.assertTrue((masked == 0).all())
        active = loss(art, 1., phones, torch.full((len(phones),), .5))
        self.assertGreater(float(active.sum().detach()), 0.)
        active.sum().backward()
        self.assertTrue(torch.isfinite(art.grad).all())
        self.assertGreater(float(art.grad.abs().sum()), 0.)

    def test_cache_integrity_and_split(self):
        destination = OUT / 'corpus'
        self.assertTrue((destination / 'clips.json').exists(), 'prepare corpus first')
        rows = json.loads((destination / 'clips.json').read_text())
        identity = hashlib.sha256((MODEL / 'model.onnx').read_bytes()).hexdigest()
        train = {r['sentence'] for r in rows if r['split'] == 'train'}
        heldout = {r['sentence'] for r in rows if r['split'] == 'heldout'}
        self.assertFalse(train & heldout)
        self.assertGreaterEqual(len(heldout), 2)
        self.assertEqual(len({r['file'] for r in rows}), len(rows))
        for row in rows:
            self.assertEqual(row['source'], 'tts')
            self.assertIn('Chirp3-HD', row['voice'])
            key_row = {k: v for k, v in row.items() if k not in ('cache_key', 'split')}
            key = hashlib.sha256(json.dumps([identity, key_row, 'praat-65-500-grid-v1'], sort_keys=True).encode()
                                 + (AUDIO / row['file']).read_bytes()).hexdigest()
            with np.load(destination / (Path(row['file']).stem+'.npz')) as data:
                self.assertEqual(key, row['cache_key'])
                self.assertEqual(key, str(data['cache_key']))
                frames = len(data['audio'])//160+1
                for name in ('f0', 'voiced', 'phone', 'relative'):
                    self.assertEqual(len(data[name]), frames)
                    self.assertTrue(np.isfinite(data[name]).all())
                self.assertTrue((np.diff(data['boundaries']) > 0).all())
                self.assertTrue(((data['relative'] >= 0) & (data['relative'] <= 1.00001)).all())


if __name__ == '__main__':
    unittest.main()

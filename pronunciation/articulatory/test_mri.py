"""MRI coordinate, curve metric, split and model invariance regressions."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import maeda
import mri_fit as fit
import usc


class Tests(unittest.TestCase):
    def test_pixels(self):
        pixels = np.array([[0, 0], [42, 42], [83, 83]])
        np.testing.assert_allclose(usc.to_pixels(usc.from_pixels(pixels)), pixels)
        np.testing.assert_allclose(usc.from_pixels(pixels)[0], [10.08, 10.08])

    def test_distance_and_gradient(self):
        a = fit.tensor([[[0, 0], [1, 0], [2, 0]]]).requires_grad_()
        b = fit.tensor([[[0, 1], [2, 1]]])
        self.assertAlmostEqual(float(fit.distance(a, b)[0].detach()), 1.)
        fit.distance(a, b).sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertLess(float(fit.distance(b, b)[0]), 1e-5)

    def test_crossing_port(self):
        a = fit.tensor([[[-1, 0], [1, 0]]])
        b = fit.tensor([[[0, -1], [0, 1]]])
        self.assertEqual(float(fit.minimum_gap(a, b)[0]), 0.)
        self.assertAlmostEqual(float(fit.minimum_gap(a, a + 2)[0]), 2.)

    def test_visible_surface(self):
        a = fit.tensor([[[0, 0], [1, 0], [2, 0], [3, 0]]])
        b = fit.tensor([[[.5, 1], [2.5, 1]]])
        sampled = fit.visible(a, b)
        np.testing.assert_allclose(sampled[0, [0, -1]].cpu(), [[.5, 0], [2.5, 0]])
        self.assertAlmostEqual(float(fit.distance(sampled, b)[0]), 1.)

    def test_no_anatomy_gauge_or_velum_geometry(self):
        g = fit.tensor([0., 0., 0., 0., 1.])
        self.assertAlmostEqual(float(fit.anatomy(g).prod()), 1., delta=1e-6)
        art = torch.zeros(2, 8, device='cuda'); art[1, 7] = 5.
        outer, inner = maeda.contours(art)
        torch.testing.assert_close(outer[0], outer[1]); torch.testing.assert_close(inner[0], inner[1])

    def test_textgrid_encodings(self):
        text = 'name = "phones"\nintervals [1]:\n xmin = 0\n xmax = 1\n text = "ŋ"\n'
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'grid'
            for encoding in ['utf-8', 'utf-16']:
                p.write_bytes(text.encode(encoding))
                self.assertEqual(usc.intervals(p), [(0., 1., 'ŋ')])

    def test_nasopharynx_excluded(self):
        curve=np.array([[0., 0.], [1., 1.], [2., 2.]])
        np.testing.assert_allclose(usc.below_height(curve, 1.5), [[0, 0], [1, 1], [1.5, 1.5]])

    def test_public_contours_same_physical_lip_extent(self):
        raw = torch.zeros(1, 7, device='cuda')
        global_ = fit.tensor([.1, .2, 1., -2., .3])
        native = fit.model(raw, global_)
        targets = {name: fit.uniform(native[side][:, a:b])
                   for name, (side, a, b) in fit.MODEL_REGIONS.items()}
        result = fit.public_regions(raw, global_, targets)
        art = torch.zeros(1, 8, device='cuda')
        lengths = maeda._scaled_native(art, fit.anatomy(global_))[2]
        position = float(lengths[0, :27].sum()/lengths.sum()*39)
        index = min(int(position), 38)
        public = fit.register(maeda.contours(art, fit.anatomy(global_)), global_)
        for name, side in [('upper_lip', 0), ('lower_lip', 1)]:
            expected = public[side][0, index] + (position-index)*(public[side][0, index+1]-public[side][0, index])
            torch.testing.assert_close(result[name][0, 0], expected)
            torch.testing.assert_close(result[name][0, -1], public[side][0, -1])
        self.assertTrue(all(v.shape == (1, 32, 2) and torch.isfinite(v).all() for v in result.values()))

    def test_lazy_audio_and_intervals(self):
        u = usc.load('sub013_2drt_01_vcv1_r2')
        self.assertNotIn('audio', u.__dict__)
        samples, rate = u.audio
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(samples.ndim, 2)
        self.assertGreater(rate, 0)
        self.assertTrue(np.isfinite(samples).all())
        self.assertGreater(float(np.abs(samples).max()), 0)
        stream = next(s for s in usc.probe(u.video) if s['codec_type'] == 'audio')
        self.assertEqual(rate, int(stream['sample_rate']))
        self.assertEqual(samples.shape[1], stream['channels'])
        self.assertLess(abs(len(samples)/rate - float(stream['duration'])), 1024/rate)
        self.assertGreater(len(samples)/rate, u.times[-1])
        self.assertIs(u.audio[0], samples)
        self.assertGreater(len(u.phone_intervals), 0)
        for a, b, phone in u.phone_intervals:
            self.assertLessEqual(a, b)
            self.assertTrue(np.all(u.phones[(u.times >= a) & (u.times < b)] == phone))

    def test_nonindexed_l2_alignment(self):
        u = usc.load('sub022_2drt_07_grandfather1_r1')
        self.assertTrue(np.any(u.phones == 'NG'))

    def test_actual_frame_numbers(self):
        u = usc.load('sub013_2drt_01_vcv1_r2')
        self.assertEqual(u.frames[0], 1)
        self.assertAlmostEqual(u.times[0], 1/u.fps)
        self.assertEqual(len(u.segments), len(u.frames))
        self.assertEqual(len(u.articulators[0]['pharyngeal_wall']), 15)
        np.testing.assert_allclose(u.articulators[0]['pharyngeal_wall'], u.segments[0][1][20:35])
        self.assertEqual(usc.curves(u, [0])['pharyngeal_wall'].shape, (1, 32, 2))
        self.assertTrue(np.any(u.phones != ''))


if __name__ == '__main__':
    unittest.main()

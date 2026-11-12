"""MRI coordinate, curve metric, split and model invariance regressions."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.nn import functional as F
import maeda
import mri_fit as fit
import mri_tract
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

    def test_registered_rigid_palate(self):
        raw = torch.zeros(3, 7, device='cuda')
        raw[:, 6] = fit.tensor([-1., 0., 1.])
        for ratio in (0., .3):
            global_ = fit.tensor([.1, .2, 1., -2., ratio])
            outer, inner = fit.model(raw, global_)
            torch.testing.assert_close(outer[:, 15:28], outer[1:2, 15:28].expand(3, -1, -1))
            self.assertGreater(float((inner[2, 0]-inner[0, 0]).norm()), .5)
            target = fit.uniform(outer[1:2, 15:28]).expand(3, -1, -1)
            visible = fit.visible(outer[:, 15:28], target)
            self.assertLess(float(fit.distance(visible, target).max()), 1e-5)

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

    def test_gridline_gaps(self):
        # A straight channel narrowing from 0.7 to 0.5 cm, tongue below the wall.
        x = torch.linspace(0, 10, 100)
        ceiling = torch.stack([x, torch.ones(100)], -1)
        floor = torch.stack([x, .3+.02*x], -1)
        def tissue(curve, far):
            return torch.cat([curve, torch.tensor([[10., far], [0., far]])])[None]
        origin, normal = mri_tract.gridlines(ceiling, [10])
        origin, normal = origin[1:-1], normal[1:-1]  # the end gridlines graze the tissue corners
        gap, mid = mri_tract.gaps(origin, normal, [tissue(ceiling, 5.)], tissue(floor, -5.))
        torch.testing.assert_close(gap[0], .7-.02*origin[:, 0], atol=1e-5, rtol=0)
        torch.testing.assert_close(mid[0, :, 1], 1-gap[0]/2, atol=1e-5, rtol=0)
        # Outlines that overlap are in contact.
        gap, _ = mri_tract.gaps(origin, normal, [tissue(ceiling, 5.)], tissue(floor+torch.tensor([0., .6]), -5.))
        self.assertEqual(float(gap[0, -1]), 0.)
        self.assertGreater(float(gap[0, 0]), 0.)
        # Maeda outlines: gaps along wall normals follow the model's own
        # paired-landmark gaps and find the same constriction.
        for name in ('iy', 'aa', 'uw'):
            raw = maeda.preset(name)[None]
            outer, inner = [c[0] for c in maeda.contours(raw)]
            away = 30*F.normalize(outer.mean(0)-inner.mean(0), dim=0)
            origin, normal = mri_tract.gridlines(torch.tensor(usc.resample(outer.numpy(), 100), dtype=torch.float32), [39], smooth=.3)
            gap, _ = mri_tract.gaps(origin, normal, [torch.cat([outer, (outer.mean(0)+away)[None]])[None]],
                                    torch.cat([inner, (inner.mean(0)-away)[None]])[None])
            expected = maeda.geometry(raw)[3][0]
            self.assertLess(float((gap[0]-expected).abs().median()), .15, name)
            self.assertLessEqual(abs(int(gap[0].argmin())-int(expected.argmin())), 3, name)

    def test_measured_tract(self):
        u = usc.load('sub013_2drt_01_vcv1_r2')
        measured = mri_tract.measure(u)
        self.assertEqual(measured['gap'].shape, (len(u.frames), sum(mri_tract.GRID)+2))
        self.assertTrue(np.isfinite(measured['gap']).all() and (measured['gap'] >= 0).all())
        # An adult vocal tract, and lips that both close and open in VCV speech.
        self.assertTrue(14 < measured['length'].sum(1).mean() < 21)
        self.assertLess(measured['gap'][:, -1].min(), .1)
        self.assertGreater(measured['gap'][:, -1].max(), 1.)


if __name__ == '__main__':
    unittest.main()

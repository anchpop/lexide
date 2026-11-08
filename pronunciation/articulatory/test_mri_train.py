"""Mapping, timing, isolation and direct-contour gradient tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import mri_train as mt
import mri_fit as fit


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_bilabial_lip_mask(self):
        gestures = mt.GestureLoss(['p', 'b', 'm', 'a'])
        phone = torch.tensor([0, 1, 2, 3])
        values = torch.tensor([9., 8., 7., 2.], requires_grad=True)
        errors = {k: values for k in (*mt.SURFACES, 'lip_gap')}
        weights = {k: 1. for k in errors}
        terms = mt.contour_terms(errors, phone, gestures, weights)
        self.assertEqual(float(terms['lip_gap'].detach()), 2.)
        self.assertEqual(float(terms['tongue'].detach()), 6.5)
        terms['lip_gap'].backward()
        torch.testing.assert_close(values.grad, torch.tensor([0., 0., 0., 1.]))
        # No relative-position input: edges and centers are equally excluded.
        all_bilabial = mt.contour_terms(errors, torch.zeros(4, dtype=torch.long), gestures, weights)
        self.assertEqual(float(all_bilabial['lip_gap'].detach()), 0.)

    def test_velar_dominance(self):
        gestures = mt.GestureLoss(['k'])
        art = torch.zeros(1, 8)
        phone = torch.zeros(1, dtype=torch.long)
        relative = torch.tensor([.5])
        for dorsal, front, expected in [(0.4, 1., 0.), (0.4, .2, np.log(2.5)**2),
                                         (1., 2., 1.)]:
            areas = torch.full((1, 40), 2.)
            areas[:, 17:29] = dorsal
            areas[:, 29:40] = front
            areas.requires_grad_()
            with patch('network.geometry', return_value=(areas, None, torch.zeros(1), None)):
                result = gestures(art, 1., phone, relative)
                self.assertAlmostEqual(float(result[4].detach()), expected, places=6)
                result.sum().backward()
                self.assertTrue(torch.isfinite(areas.grad).all())
                if dorsal > front:
                    self.assertGreater(float(areas.grad[:, 17:29].sum()), 0.)
                    self.assertLess(float(areas.grad[:, 29:40].sum()), 0.)
                self.assertEqual(float(gestures(art, 1., phone, torch.zeros(1)).sum().detach()), 0.)

    def test_vowel_pharynx_class_and_mask(self):
        phones = ['iː', 'ə', 'æ', 'aɪ', 'aʊ', 'ɛ', 'ã', 'ɑː', 'ɔɪ', 'ɒ', 'k', '<pad>']
        gestures = mt.GestureLoss(phones)
        torch.testing.assert_close(gestures.classes[:, -1],
            torch.tensor([True]*7 + [False]*5))
        count = len(phones)
        areas = torch.full((count, 40), 2.)
        areas[:, 5] = .25
        areas.requires_grad_()
        with patch('network.geometry', return_value=(areas, None, torch.zeros(count), None)):
            args = (torch.zeros(count, 8), 1., torch.arange(count))
            middle = torch.full((count,), .5)
            result = gestures(*args, middle)
            self.assertEqual(float(result[-1].detach()), .25)
            terms = mt.gesture_terms(gestures, *args, middle)
            self.assertEqual(set(terms), set(gestures.names))
            self.assertAlmostEqual(float(terms['vowel_pharynx'].detach()), .025)
            result[-1].backward()
            self.assertTrue(torch.isfinite(areas.grad).all())
            self.assertLess(float(areas.grad[:7, 5].sum()), 0.)
            self.assertEqual(float(areas.grad[7:].abs().sum()), 0.)
            self.assertEqual(float(gestures(*args, torch.zeros(count))[-1].detach()), 0.)

    def test_vowel_pharynx_section_bounds_and_floor(self):
        gestures = mt.GestureLoss(['ə'])
        for section in (3, 4, 10, 11):
            for area in (.25, .5, 1.):
                areas = torch.full((1, 40), 2.)
                areas[0, section] = area
                with patch('network.geometry', return_value=(areas, None, torch.zeros(1), None)):
                    value = gestures(torch.zeros(1, 8), 1., torch.tensor([0]), torch.tensor([.5]))[-1]
                expected = max(0., (.5-area)/.5)**2 if 4 <= section <= 10 else 0.
                self.assertEqual(float(value), expected)

    def test_cli_passes_output_directory(self):
        import mri_evaluate
        out = Path('/tmp/articulatory-output-test')
        for stage, module, name, arguments in [
                ('prepare', mt, 'prepare', (out,)),
                ('train', mt, 'run_training', (65., out)),
                ('evaluate', mri_evaluate, 'run', (out,))]:
            with patch.object(sys, 'argv', ['mri_train.py', stage, '--out', str(out)]), \
                    patch.object(module, name) as action:
                mt.main()
                action.assert_called_once_with(*arguments)
        self.assertNotEqual(mt.OUT, out)

    def test_mapping(self):
        self.assertEqual(mt.phone_label('AH0'), 'ə')
        self.assertEqual(mt.phone_label('AH1'), 'ʌ')
        self.assertEqual(mt.phone_label(' NG '), 'ŋ')
        self.assertEqual(mt.phone_label('sil'), '<pad>')
        self.assertEqual(mt.phone_label('AY2'), 'aɪ')
        self.assertIsNone(mt.phone_label('YUH1RUW0'))
        _, state = mt.checkpoint()
        self.assertTrue(set(mt.ARPABET.values()) <= set(state['vocabulary']))

    def test_grid_and_missing_frames(self):
        data = mt.timeline([(0,.02,'K'),(.02,.04,'AH0'),(.04,.06,'BAD')],
                           np.arange(6)*.01, ['<pad>', 'k','ə'])
        np.testing.assert_array_equal(data['phone'], [1,1,2,2,0,0])
        np.testing.assert_array_equal(data['valid'], [1,1,1,1,0,0])
        np.testing.assert_allclose(data['relative'], [0,.5,0,.5,0,.5])
        index, valid = mt.nearest(np.array([0,.01,.02,.03]), np.array([.01,.03]), .006)
        np.testing.assert_array_equal(valid,[0,1,0,1])
        np.testing.assert_array_equal(index[[1,3]],[0,1])

    def test_extension_preserves_tts(self):
        old, state = mt.checkpoint()
        new, extended = mt.checkpoint(extra=['sub009'])
        for key, value in old.state_dict().items():
            actual = new.state_dict()[key]
            if key == 'speaker.weight': actual = actual[:len(value)]
            torch.testing.assert_close(value,actual,rtol=0,atol=0)
        self.assertEqual(extended['speakers'][:-1],state['speakers'])
        torch.testing.assert_close(new.speaker.weight[-1],old.speaker.weight.mean(0))

    def test_fixed_anatomy_and_contour_gradients(self):
        net, state = mt.checkpoint(extra=['MRI'])
        data = dict(phone=np.array([state['vocabulary'].index('iː')]*60),
                    relative=np.linspace(0,1,60,dtype=np.float32),
                    f0=np.full(60,150,dtype=np.float32),voiced=np.ones(60,np.float32))
        art, source, _ = mt.inference(net,data,{'voice':'MRI'},state['speakers'])
        global_ = fit.tensor([.1,.2,0.,0.,.3])
        raw = torch.zeros(60,7,device='cuda')
        native = fit.model(raw, global_)
        targets = {name:fit.uniform(native[side][:,a:b]).detach()
                   for name,(side,a,b) in fit.MODEL_REGIONS.items()}
        terms = mt.geometry_errors(art, global_, targets)
        sum(v.mean() for v in terms.values()).backward()
        self.assertGreater(float(net.articulation.weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
        self.assertIsNone(net.speaker.weight.grad)
        self.assertIsNone(net.character.weight.grad)
        self.assertIsNone(global_.grad)
        before = {k:v.detach().clone() for k,v in terms.items()}
        with torch.no_grad(): net.character.weight.add_(10); net.speaker.weight.add_(10)
        changed, _, _ = mt.inference(net,data,{'voice':'MRI'},state['speakers'])
        for k,v in mt.geometry_errors(changed,global_,targets).items():
            torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_textgrid_tier_case_and_boundary(self):
        text = ('name = "Phones"\nintervals [1]:\n xmin = 0\n xmax = 1\n text = "K"\n'
                'item [2]:\n name = "Words"\nintervals [1]:\n xmin = 0\n xmax = 1\n text = "cat"\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'grid'
            path.write_text(text)
            self.assertEqual(mt.usc.intervals(path), [(0.,1.,'K')])

    def test_track_overrun_is_not_observed(self):
        u = mt.usc.load('sub023_2drt_01_vcv1_r2')
        self.assertEqual(u.discarded_track_frames, 1)
        self.assertEqual(u.frames[-1], 2618)
        self.assertEqual(len(u.frames), len(u.articulators))

    def test_duplicate_tracks_are_not_observations(self):
        u = mt.usc.load('sub070_2drt_14_picture3')
        self.assertNotIn(638,u.frames)
        self.assertGreaterEqual(u.discarded_track_frames,2)
        self.assertTrue((np.diff(u.frames)>0).all())

    def test_pitch_fallback_preserves_voicing_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'data').mkdir()
            rows=[]
            for i,hz in enumerate([100.,110.,400.]):
                rows.append(dict(stem=str(i),voice='speaker',f0_quantiles=[hz,hz,hz],voiced_fraction=.5))
                np.savez(out/'data'/f'{i}.npz',f0=np.full(2,hz,np.float32),
                         voiced=np.array([0.,1.],np.float32),phone=np.array([1,2]))
            mt.dump(out, 'data.json',rows)
            audit=mt.audit_pitch(out)
            self.assertEqual(len(audit['speaker']['replacements']),1)
            data=mt.load(rows[-1], out)
            np.testing.assert_array_equal(data['f0'],[110.,110.])
            np.testing.assert_array_equal(data['voiced'],[0.,1.])
            np.testing.assert_array_equal(data['phone'],[1,2])

    def test_speaker_split(self):
        rows = json.loads((mt.OUT/'data.json').read_text())
        train = {r['voice'] for r in rows if r['split']=='train'}
        heldout = {r['voice'] for r in rows if r['split']=='heldout'}
        self.assertFalse(train & heldout)
        self.assertEqual(heldout,set(mt.HOLDOUT))
        self.assertEqual(len(train),14)
        regs=json.loads((mt.OUT/'registration.json').read_text())
        self.assertEqual(regs['sub013']['demographics']['l1'],'English')
        self.assertNotEqual(regs['sub022']['demographics']['l1'],'English')


if __name__ == '__main__': unittest.main()

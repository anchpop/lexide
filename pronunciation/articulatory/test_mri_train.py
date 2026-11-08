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
        with tempfile.TemporaryDirectory() as directory, patch.object(mt,'OUT',Path(directory)):
            (mt.OUT/'data').mkdir()
            rows=[]
            for i,hz in enumerate([100.,110.,400.]):
                rows.append(dict(stem=str(i),voice='speaker',f0_quantiles=[hz,hz,hz],voiced_fraction=.5))
                np.savez(mt.OUT/'data'/f'{i}.npz',f0=np.full(2,hz,np.float32),
                         voiced=np.array([0.,1.],np.float32),phone=np.array([1,2]))
            mt.dump('data.json',rows)
            audit=mt.audit_pitch()
            self.assertEqual(len(audit['speaker']['replacements']),1)
            data=mt.load(rows[-1])
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

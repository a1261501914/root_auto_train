import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

from root_self_train.quality import DEFAULTS, inspect_prediction, assess, settings
from root_self_train.agent import validate_plan, missing_data_guidance


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV optional')
class QualityTests(unittest.TestCase):
    def root(self):
        p = np.full((64,64), .01, dtype=np.float32)
        p[10:54,30:34] = .99
        return p

    def test_stable_foreground_passes_screening_not_accuracy(self):
        p = self.root()
        row, maps = inspect_prediction(p,p,DEFAULTS)
        self.assertEqual(row['status'],'passed_screening')
        self.assertEqual(row['metrics']['flip_dice'],1)
        self.assertNotIn('accuracy',row['metrics'])
        self.assertFalse(maps['disagreement'].any())

    def test_empty_prediction_is_not_a_success(self):
        p = np.zeros((16,16),dtype=np.float32)
        row,_=inspect_prediction(p,p,DEFAULTS)
        self.assertEqual(row['status'],'insufficient_evidence')
        self.assertIsNone(row['metrics']['flip_dice'])

    def test_flip_loss_and_fragments_trigger_risk(self):
        p=self.root()
        row,_=inspect_prediction(p,np.zeros_like(p),DEFAULTS)
        self.assertIn('flip_disagreement',row['reasons'])
        self.assertIn('unstable_thin_regions',row['reasons'])
        p[:]=.01; p[::4,::4]=.99
        row,_=inspect_prediction(p,p,DEFAULTS)
        self.assertIn('small_fragments',row['reasons'])

    def test_background_cannot_hide_uncertainty(self):
        p=np.full((1000,1000),.01,dtype=np.float32);p[20:80,40:44]=.6
        row,_=inspect_prediction(p,p,DEFAULTS)
        self.assertEqual(row['metrics']['uncertain_candidate_fraction'],1)
        self.assertEqual(row['status'],'risk_detected')
        p[0,0]=np.nan
        with self.assertRaises(ValueError):inspect_prediction(p,p,DEFAULTS)

    def test_batch_reports_duplicates_and_corrupt_input_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'input'; source.mkdir()
            for folder in ('a','b'):
                (source/folder).mkdir()
                Image.fromarray(np.zeros((64,64,3),dtype=np.uint8)).save(source/folder/'same.png')
            (source/'bad.png').write_text('broken',encoding='utf-8')
            p=self.root()
            class Fake:
                def predict(self,image,flip=False):return p
            cfg=dict(unlabeled_dir=str(source),workdir=str(root/'output'),checkpoint='test.pt')
            with patch('root_self_train.dependencies.ensure_imports'),patch('builtins.print'):
                summary=assess(cfg,Fake())
            self.assertEqual(summary['total'],3)
            self.assertEqual(summary['counts']['error'],1)
            self.assertEqual(summary['status'],'insufficient_evidence')
            self.assertFalse(summary['complete'])
            self.assertEqual(len(list((root/'output/masks').glob('*.png'))),2)
            self.assertTrue((root/'output/quality_report.html').exists())
            rows=json.loads((root/'output/quality_samples.json').read_text(encoding='utf-8'))
            self.assertEqual(len({r['mask'] for r in rows if 'mask' in r}),2)
            with self.assertRaises(ValueError):assess(cfg,Fake())


class QualityRoutingTests(unittest.TestCase):
    def test_no_validation_manifest_required_and_training_options_rejected(self):
        plan=dict(action='assess',message='质检',options={'input_dir':'new_images'})
        validate_plan(plan,'检查 new_images',{})
        self.assertIsNone(missing_data_guidance('assess',{'val_manifest':'nonexistent'}))
        plan['options']['auto_train']=True
        with self.assertRaises(ValueError):validate_plan(plan,'检查 new_images',{})

    def test_invalid_thresholds(self):
        for options in ({'min_flip_dice':2},{'thin_radius':float('nan')},{'unknown':1},{'uncertainty_low':.8}):
            with self.assertRaises(ValueError):settings({'quality':options})

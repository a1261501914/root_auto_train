import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from root_self_train.core import counts, metrics, select
from root_self_train.data import validate
from root_self_train.__main__ import run, promoted

GATE = dict(root_confidence=.95, consistency=.95, pixel_confidence=.9,
            min_coverage=.5, min_root_pixels=1)


class Tests(unittest.TestCase):
    def test_ignore_and_metrics(self):
        result = metrics(counts([[1, 1, 0, 1]], [[1, 0, 1, 255]]))
        self.assertEqual(result['dice'], .5)
        self.assertAlmostEqual(result['iou'], 1/3)

    def test_gate(self):
        p = np.array([[.99, .6], [.01, .01]])
        mask, info = select(p, p, {**GATE, 'root_confidence': .7})
        self.assertTrue(info['accepted'])
        self.assertEqual(mask.tolist(), [[1,255],[0,0]])
        self.assertFalse(select(np.zeros((2,2)), np.zeros((2,2)), GATE)[1]['accepted'])
        self.assertFalse(select(p, 1-p, GATE)[1]['accepted'])
        with self.assertRaises(ValueError):
            select(p*np.nan, p, GATE)

    def test_promotion(self):
        self.assertFalse(promoted({'dice': .9}, {'dice': .9}, 0))
        self.assertFalse(promoted({'dice': float('nan')}, {'dice': .9}, 0))
        self.assertTrue(promoted({'dice': .91}, {'dice': .9}, .001))

    def test_full_orchestration_with_fake_backend(self):
        # Real image IO, gates, manifests, scoring and persistence; fake only the GPU backend.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask = np.array([[1,0],[0,0]], dtype=np.uint8)
            for i, name in enumerate(['train','val','unlabeled']):
                Image.fromarray(np.full((2,2,3), i*50, np.uint8)).save(root/f'{name}.png')
                Image.fromarray(mask).save(root/f'{name}_mask.png')
            for name in ('train', 'val'):
                (root/f'{name}.jsonl').write_text(json.dumps(dict(image=f'{name}.png',mask=f'{name}_mask.png')), encoding='utf-8')
            un = root/'un'
            un.mkdir()
            (root/'unlabeled.png').replace(un/'image.png')
            # 目录内故意放入训练图副本；显式未标注清单必须只读取指定图片。
            (un/'excluded.png').write_bytes((root/'train.png').read_bytes())
            (root/'unlabeled.jsonl').write_text(json.dumps(dict(image='un/image.png')), encoding='utf-8')
            (root/'model.yml').write_text('model: {}')
            (root/'teacher.pdparams').touch()
            cfg = dict(GATE, train_manifest=str(root/'train.jsonl'), val_manifest=str(root/'val.jsonl'),
                       unlabeled_dir=str(un), unlabeled_manifest=str(root/'unlabeled.jsonl'),
                       model_config=str(root/'model.yml'), checkpoint=str(root/'teacher.pdparams'),
                       workdir=str(root/'run'), device='cpu', rounds=2, iters=1, max_pseudo_ratio=1, min_delta=.001)
            class Fake:
                def __init__(self, config, checkpoint, device):
                    self.student = 'student' in str(checkpoint)
                def predict(self, image, flip=False):
                    p = np.array([[.99,.01],[.01,.01]])
                    if Path(image).name == 'val.png' and not self.student:
                        p[0,1] = .99
                    return p
            def fake_train(*args, **kwargs):
                command = args[0]
                job = json.loads(Path(command[-1]).read_text())
                checkpoint = Path(job['output'])/'iter_1'/'model.pdparams'
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
            with patch('root_self_train.backend.Predictor', Fake), patch('subprocess.run', fake_train):
                run(cfg)
            best = json.loads((root/'run'/'best.json').read_text())
            history = json.loads((root/'run'/'history.json').read_text())
            self.assertEqual(best['round'], 1)
            self.assertEqual(best['metrics']['dice'], 1)
            self.assertEqual([r['promoted'] for r in history], [True, False])
            with self.assertRaises(ValueError):
                validate([(root/'train.png',root/'train_mask.png')],
                         [(root/'train.png',root/'train_mask.png')], [])


if __name__ == '__main__':
    unittest.main()

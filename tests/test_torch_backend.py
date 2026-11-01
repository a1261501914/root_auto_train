import importlib.util
import io
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from root_self_train.backend import Predictor, training_sample
from root_self_train.convert_weights import ArrayReader


class InputTests(unittest.TestCase):
    def test_old_checkpoint_has_conversion_instruction(self):
        with self.assertRaisesRegex(ValueError, 'convert_weights'):
            Predictor('missing.yml', 'old.pdparams', 'cpu')

    def test_reader_rejects_arbitrary_classes(self):
        with self.assertRaises(ValueError):
            ArrayReader(io.BytesIO(pickle.dumps(Path('x')))).load()

    def test_mask_padding_is_ignored_and_flip_keeps_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root/'image.png')
            Image.fromarray(np.ones((8, 8), dtype=np.uint8)).save(root/'mask.png')
            _, mask = training_sample(root/'image.png', root/'mask.png', [dict(type='RandomPaddingCrop', crop_size=[16,16])], np.asarray)
            self.assertEqual(int((mask==1).sum()),64)
            self.assertEqual(int((mask==255).sum()),192)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch optional until inference')
class TorchTests(unittest.TestCase):
    def test_model_forward_backward_and_strict_reload(self):
        import torch
        from root_self_train.torch_model import SegFormerB2
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = SegFormerB2(embedding=16)
            result = model(torch.randn(1,3,32,40))
            self.assertEqual(tuple(result.shape),(1,2,32,40))
            mask = torch.zeros(1,32,40,dtype=torch.long)
            mask[:,:,0]=255
            loss=torch.nn.functional.cross_entropy(result,mask,ignore_index=255)
            loss.backward()
            self.assertTrue(torch.isfinite(model.linear_pred.weight.grad).all())
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.pt'
                torch.save(model.state_dict(),path)
                restored=SegFormerB2(embedding=16)
                restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
                bad=dict(model.state_dict()); bad.pop('linear_pred.weight')
                with self.assertRaises(RuntimeError):
                    restored.load_state_dict(bad,strict=True)
        finally:
            torch.set_num_threads(threads)

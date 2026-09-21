import unittest
import numpy as np
from root_self_train.tiling import slide_logits


class TilingTests(unittest.TestCase):
    def test_non_divisible_and_small_images(self):
        for h, w in [(13, 17), (2, 3), (8, 8)]:
            image = np.arange(3*h*w, dtype=np.float32).reshape(3,h,w)
            result = slide_logits(image, lambda tile: tile[:2], [8,8], [5,5])
            np.testing.assert_array_equal(result, image[:2])

    def test_gap_rejected(self):
        with self.assertRaises(ValueError):
            slide_logits(np.zeros((3,10,10)), lambda x: x[:2], [4,4], [5,5])

import unittest
import numpy as np
from kiu_drone_show.handwriting_preprocess import preprocess_handwriting

class TestHandwritingFailureCases(unittest.TestCase):
    def test_preprocess_empty_raises(self):
        img = np.ones((60, 60, 3), dtype=np.float32)  # pure white, no ink
        with self.assertRaises(ValueError):
            preprocess_handwriting(img, method="otsu", min_component=10, crop=True)

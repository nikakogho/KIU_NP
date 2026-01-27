import unittest
import numpy as np
from kiu_drone_show.handwriting_preprocess import preprocess_handwriting

class TestHandwritingPreprocess(unittest.TestCase):
    def test_preprocess_dark_ink_on_white_detects_stroke(self):
        img = np.ones((80, 120, 3), dtype=np.float32)
        img[30:35, 20:100, :] = 0.0  # black stroke
        mask, info = preprocess_handwriting(img, method="otsu", min_component=10, crop=False)
        assert mask.any()
        assert info.polarity == "dark_ink"

    def test_preprocess_light_ink_on_black_detects_stroke(self):
        img = np.zeros((80, 120, 3), dtype=np.float32)
        img[30:35, 20:100, :] = 1.0  # white stroke
        mask, info = preprocess_handwriting(img, method="otsu", min_component=10, crop=False)
        assert mask.any()
        assert info.polarity == "light_ink"
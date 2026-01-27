import unittest
import numpy as np

from kiu_drone_show.render_text_mask import render_text_mask


class TestRenderTextMask(unittest.TestCase):
    def test_basic_non_empty(self):
        M, info = render_text_mask("Happy New Year!", canvas_size=(200, 800), fontsize=80)
        self.assertEqual(M.shape, (200, 800))
        self.assertEqual(M.dtype, np.bool_)
        self.assertTrue(M.any())  # should contain some True pixels

    def test_deterministic(self):
        M1, _ = render_text_mask("Test", canvas_size=(200, 800), fontsize=80, threshold=0.2)
        M2, _ = render_text_mask("Test", canvas_size=(200, 800), fontsize=80, threshold=0.2)
        self.assertTrue(np.array_equal(M1, M2))

    def test_different_text_differs(self):
        M1, _ = render_text_mask("AAAA", canvas_size=(200, 800), fontsize=80)
        M2, _ = render_text_mask("BBBB", canvas_size=(200, 800), fontsize=80)
        
        # Not guaranteed to differ by huge amount, but should not be identical
        self.assertFalse(np.array_equal(M1, M2))

    def test_invalid_text_raises(self):
        with self.assertRaises(ValueError):
            render_text_mask("")
        with self.assertRaises(ValueError):
            render_text_mask("   ")

    def test_threshold_effect(self):
        # With very high threshold, fewer pixels should be True
        M_low, _ = render_text_mask("Hello", canvas_size=(200, 800), fontsize=80, threshold=0.1)
        M_high, _ = render_text_mask("Hello", canvas_size=(200, 800), fontsize=80, threshold=0.9)
        self.assertGreater(M_low.sum(), M_high.sum())


if __name__ == "__main__":
    unittest.main()

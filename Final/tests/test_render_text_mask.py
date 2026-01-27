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

    def test_long_text_not_cropped_when_autoscale(self):
        M, info = render_text_mask("Happy New Year!", canvas_size=(200, 800), fontsize=120, autoscale=True)

        ys, xs = np.nonzero(M)
        self.assertTrue(ys.size > 0)

        # Require some non-trivial margin; allow anti-aliasing but not edge-touching
        self.assertGreater(xs.min(), 1)
        self.assertLess(xs.max(), 800 - 2)
        self.assertGreater(ys.min(), 1)
        self.assertLess(ys.max(), 200 - 2)

    def test_autoscale_respects_safety_margin(self):
        margin = 6
        M, _ = render_text_mask(
            "Happy New Year!",
            canvas_size=(200, 800),
            fontsize=140,
            autoscale=True,
            safety_margin_px=margin,
        )
        ys, xs = np.nonzero(M)
        self.assertTrue(ys.size > 0)

        self.assertGreaterEqual(xs.min(), margin)
        self.assertLessEqual(xs.max(), 800 - 1 - margin)
        self.assertGreaterEqual(ys.min(), margin)
        self.assertLessEqual(ys.max(), 200 - 1 - margin)



if __name__ == "__main__":
    unittest.main()

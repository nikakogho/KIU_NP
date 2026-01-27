import unittest
import numpy as np

from kiu_drone_show.anchoring import (
    compute_boundary, compute_fill, compute_skeleton,
    allocate_bkf_raw, anchors_from_mask, BKFPolicy
)


class TestAnchoring(unittest.TestCase):
    def test_boundary_subset(self):
        M = np.zeros((10, 12), dtype=bool)
        M[2:8, 3:10] = True
        B = compute_boundary(M)
        self.assertTrue(B.any())
        self.assertTrue(np.all(B <= M))
        self.assertFalse(B[4, 6])  # interior shouldn't be boundary

    def test_fill_excludes_boundary(self):
        M = np.zeros((10, 12), dtype=bool)
        M[2:8, 3:10] = True
        B = compute_boundary(M)
        F = compute_fill(M, boundary=B)
        self.assertTrue(np.all(F <= M))
        self.assertFalse(np.any(F & B))
        self.assertTrue(F[4, 6])

    def test_skeleton_subset_and_smaller(self):
        M = np.zeros((25, 25), dtype=bool)
        M[5:20, 8:17] = True
        K = compute_skeleton(M, max_iters=2000)
        self.assertTrue(np.all(K <= M))
        self.assertLess(K.sum(), M.sum())

    def test_allocate_density_regimes(self):
        policy = BKFPolicy()
        # Very sparse
        c = allocate_bkf_raw(N=10, P=1000, L=1000, policy=policy)
        self.assertEqual(c.regime, "sparse")
        self.assertEqual(c.nK, 10)

        # Medium
        c = allocate_bkf_raw(N=200, P=200, L=200, policy=policy)
        self.assertEqual(c.regime, "medium")
        self.assertEqual(c.nB + c.nK + c.nF, 200)

        # Dense
        c = allocate_bkf_raw(N=500, P=50, L=50, policy=policy)
        self.assertEqual(c.regime, "dense")
        self.assertEqual(c.nB + c.nK + c.nF, 500)

    def test_anchors_from_mask_count_and_on_foreground(self):
        rng = np.random.default_rng(123)
        M = np.zeros((60, 200), dtype=bool)
        # crude "HI"
        M[10:50, 20:30] = True
        M[10:50, 50:60] = True
        M[28:32, 20:60] = True

        out = anchors_from_mask(M, N=120, rng=rng, max_candidates=10000)
        anchors = out["anchors"]
        self.assertEqual(anchors.shape, (120, 2))
        yy, xx = anchors[:, 0], anchors[:, 1]
        self.assertTrue(np.all(M[yy, xx]))

    def test_thin_stroke_no_fill_still_works(self):
        rng = np.random.default_rng(0)
        M = np.zeros((50, 200), dtype=bool)
        # 1-pixel-thick line (fill is empty; boundary ~ mask)
        M[25, 20:180] = True

        # Choose N smaller than available pixels so sampling is possible
        out = anchors_from_mask(M, N=80, rng=rng, max_candidates=10000)
        anchors = out["anchors"]
        self.assertEqual(anchors.shape, (80, 2))
        yy, xx = anchors[:, 0], anchors[:, 1]
        self.assertTrue(np.all(M[yy, xx]))


if __name__ == "__main__":
    unittest.main()

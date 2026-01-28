import unittest
import numpy as np

from kiu_drone_show.pipelines import run_problem3_tracking_synthetic_frames
from kiu_drone_show.video_tracking import TrackConfig


def pairwise_dists(P: np.ndarray) -> np.ndarray:
    # P: (N,3)
    diff = P[:, None, :] - P[None, :, :]
    D2 = np.einsum("ijk,ijk->ij", diff, diff)
    return np.sqrt(D2)


class TestProblem3Pipeline(unittest.TestCase):
    def test_problem3_targets_preserve_shape_and_start_matches_base(self):
        # Base formation: small grid around (50,50,0)
        xs = np.linspace(45, 55, 5)
        ys = np.linspace(45, 55, 5)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        base = np.stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)], axis=1)  # (25,3)
        N = base.shape[0]

        # Synthetic frames: moving bright square on black bg
        K, H, W = 12, 60, 80
        frames = np.zeros((K, H, W, 3), dtype=np.float32)
        for k in range(K):
            y = 10 + k
            x = 10 + 2 * k
            frames[k, y:y+6, x:x+6, :] = 1.0

        out = run_problem3_tracking_synthetic_frames(
            frames=frames,
            base_formation_xyz=base,
            frame_dt=0.10,
            dt=0.05,  # coarse sim ok for test
            record_every=1,
            track_cfg=TrackConfig(bg_frames=1, diff_thresh=0.05, min_area=10, open_iter=0, close_iter=0),
            mapping_margin=10.0,  # avoid clipping
            invert_y=True,
            z_mode="zero",
        )

        targets_time = out["targets_time"]  # (K,N,3)
        self.assertEqual(targets_time.shape, (K, N, 3))

        # At t=0, after shifting centroids, targets should equal the base formation (translation-only)
        self.assertTrue(np.allclose(targets_time[0], base, atol=1e-6))

        # Pairwise distances between targets must be constant over time (rigid translation)
        D0 = pairwise_dists(targets_time[0])
        Dm = pairwise_dists(targets_time[K // 2])
        Dlast = pairwise_dists(targets_time[-1])
        self.assertTrue(np.allclose(D0, Dm, atol=1e-6))
        self.assertTrue(np.allclose(D0, Dlast, atol=1e-6))

    def test_problem3_simulation_shapes(self):
        # Very small N to keep test cheap
        base = np.array([[48, 50, 0], [52, 50, 0], [50, 52, 0], [50, 48, 0]], dtype=float)

        K, H, W = 8, 50, 70
        frames = np.zeros((K, H, W, 3), dtype=np.float32)
        for k in range(K):
            frames[k, 20:26, (10 + k):(16 + k), :] = 1.0

        out = run_problem3_tracking_synthetic_frames(
            frames=frames,
            base_formation_xyz=base,
            frame_dt=0.10,
            dt=0.02,
            record_every=2,
            track_cfg=TrackConfig(bg_frames=1, diff_thresh=0.05, min_area=10, open_iter=0, close_iter=0),
            mapping_margin=10.0,
            invert_y=True,
            z_mode="zero",
        )

        X = out["X"]
        V = out["V"]
        self.assertEqual(X.ndim, 3)
        self.assertEqual(V.ndim, 3)
        self.assertEqual(X.shape[1:], (base.shape[0], 3))
        self.assertEqual(V.shape[1:], (base.shape[0], 3))
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue(np.isfinite(V).all())

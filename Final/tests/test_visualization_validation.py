import unittest
import numpy as np

import matplotlib
matplotlib.use("Agg")

from kiu_drone_show.visualization import animate_swarm_3d, AnimationConfig


class TestVisualizationValidation(unittest.TestCase):
    def test_rejects_bad_X_shape(self):
        times = np.linspace(0, 1, 5)
        X_bad = np.zeros((5, 10), dtype=float)  # missing last dim
        with self.assertRaises(ValueError):
            animate_swarm_3d(times, X_bad, out_path=None, cfg=AnimationConfig())

    def test_rejects_time_length_mismatch(self):
        times = np.linspace(0, 1, 4)
        X = np.zeros((5, 10, 3), dtype=float)
        with self.assertRaises(ValueError):
            animate_swarm_3d(times, X, out_path=None, cfg=AnimationConfig())

    def test_rejects_bad_every(self):
        times = np.linspace(0, 1, 5)
        X = np.zeros((5, 10, 3), dtype=float)
        cfg = AnimationConfig(every=0)
        with self.assertRaises(ValueError):
            animate_swarm_3d(times, X, out_path=None, cfg=cfg)

    def test_rejects_bad_targets_shape(self):
        times = np.linspace(0, 1, 5)
        X = np.zeros((5, 10, 3), dtype=float)
        targets_bad = np.zeros((10, 2), dtype=float)
        with self.assertRaises(ValueError):
            animate_swarm_3d(times, X, targets=targets_bad, out_path=None, cfg=AnimationConfig())


if __name__ == "__main__":
    unittest.main()

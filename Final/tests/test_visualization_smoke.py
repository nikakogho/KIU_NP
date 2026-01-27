import unittest
import numpy as np
import tempfile
import os

import matplotlib
matplotlib.use("Agg")

from kiu_drone_show.visualization import animate_swarm_3d, AnimationConfig


class TestVisualizationSmoke(unittest.TestCase):
    def test_animation_constructs_no_targets(self):
        K, N = 3, 12
        times = np.linspace(0, 0.2, K)
        X = np.zeros((K, N, 3), dtype=float)
        X[:, :, 2] = np.linspace(0, 30, K)[:, None]  # world Z
        X[:, :, 1] = np.linspace(0, 10, K)[:, None]  # world Y

        cfg = AnimationConfig(fps=10, every=1, trail=0, show_targets=False)

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "smoke.gif")
            anim = animate_swarm_3d(times, X, targets=None, out_path=out, cfg=cfg)
            self.assertIsNotNone(anim)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)

    def test_animation_constructs_with_targets_and_trail(self):
        K, N = 3, 10
        times = np.linspace(0, 0.2, K)
        X = np.random.default_rng(0).uniform(0, 100, size=(K, N, 3))
        targets = np.random.default_rng(1).uniform(0, 100, size=(N, 3))

        cfg = AnimationConfig(fps=10, every=1, trail=2, show_targets=True)

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "smoke2.gif")
            anim = animate_swarm_3d(times, X, targets=targets, out_path=out, cfg=cfg)
            self.assertIsNotNone(anim)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == "__main__":
    unittest.main()

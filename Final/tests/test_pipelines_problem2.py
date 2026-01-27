import unittest
import numpy as np

from kiu_drone_show.dynamics import DynamicsParams
from kiu_drone_show.pipelines import run_problem2_static_text


class TestPipelinesProblem2(unittest.TestCase):
    def test_problem2_runs_and_shapes_ok(self):
        params = DynamicsParams(m=1.0, kp=2.0, kd=1.0, krep=0.0, rsafe=1.0, vmax=5.0)
        out = run_problem2_static_text(
            text="Happy New Year!",
            N=40,
            init="cube",
            seed=0,
            dt=0.05,
            T_total=1.0,
            record_every=1,
            params=params,
            stop_early=False,
        )
        times = out["times"]
        X = out["X"]
        V = out["V"]

        self.assertEqual(X.ndim, 3)
        self.assertEqual(V.ndim, 3)
        self.assertEqual(X.shape[1:], (40, 3))
        self.assertEqual(V.shape[1:], (40, 3))
        self.assertEqual(times.shape[0], X.shape[0])

        # speed cap respected at final frame (semi-implicit Euler enforces it every step)
        speeds = np.linalg.norm(V[-1], axis=1)
        self.assertTrue(np.all(speeds <= params.vmax + 1e-9))

    def test_problem2_rms_decreases_somewhat(self):
        params = DynamicsParams(m=1.0, kp=2.5, kd=1.2, krep=0.0, rsafe=1.0, vmax=8.0)
        out = run_problem2_static_text(
            text="Happy New Year!",
            N=30,
            init="cube",
            seed=1,
            dt=0.05,
            T_total=2.0,
            record_every=1,
            params=params,
            stop_early=False,
        )
        X = out["X"]
        targets = out["targets_assigned"]

        d0 = np.sqrt(np.mean(np.sum((X[0] - targets) ** 2, axis=1)))
        d1 = np.sqrt(np.mean(np.sum((X[-1] - targets) ** 2, axis=1)))
        self.assertLess(d1, d0)


if __name__ == "__main__":
    unittest.main()

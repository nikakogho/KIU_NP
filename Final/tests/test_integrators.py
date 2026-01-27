import unittest
import numpy as np

from kiu_drone_show.dynamics import DynamicsParams
from kiu_drone_show.integrators import step_semi_implicit_euler_method1, rollout_method1


class TestIntegrators(unittest.TestCase):
    def test_speed_is_capped(self):
        params = DynamicsParams(m=1.0, kp=1000.0, kd=0.0, krep=0.0, rsafe=1.0, vmax=1.5)
        x = np.array([[0.0, 0.0, 0.0]], dtype=float)
        v = np.array([[0.0, 0.0, 0.0]], dtype=float)
        T = np.array([[100.0, 0.0, 0.0]], dtype=float)

        res = step_semi_implicit_euler_method1(x, v, T, dt=0.1, params=params)
        speed = np.linalg.norm(res.v_next[0])
        self.assertLessEqual(speed, params.vmax + 1e-9)

    def test_single_drone_moves_toward_target(self):
        params = DynamicsParams(m=1.0, kp=2.0, kd=0.5, krep=0.0, rsafe=1.0, vmax=10.0)
        x0 = np.array([[0.0, 0.0, 0.0]], dtype=float)
        v0 = np.array([[0.0, 0.0, 0.0]], dtype=float)
        target = np.array([[5.0, 0.0, 0.0]], dtype=float)

        def targets_fn(t: float) -> np.ndarray:
            return target

        times, X, V = rollout_method1(x0, v0, targets_fn, T_total=2.0, dt=0.02, params=params, record_every=10)

        # distance should generally decrease (allow minor numerical wiggles)
        d0 = np.linalg.norm(X[0, 0] - target[0])
        d_end = np.linalg.norm(X[-1, 0] - target[0])
        self.assertLess(d_end, d0)

    def test_rollout_shapes(self):
        params = DynamicsParams(krep=0.0, vmax=5.0)
        x0 = np.zeros((3, 3), dtype=float)
        v0 = np.zeros((3, 3), dtype=float)

        def targets_fn(t: float) -> np.ndarray:
            return np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]], dtype=float)

        times, X, V = rollout_method1(x0, v0, targets_fn, T_total=1.0, dt=0.1, params=params, record_every=1)
        self.assertEqual(X.ndim, 3)
        self.assertEqual(V.ndim, 3)
        self.assertEqual(X.shape[1:], (3, 3))
        self.assertEqual(V.shape[1:], (3, 3))
        self.assertEqual(times.shape[0], X.shape[0])
        self.assertEqual(times.shape[0], V.shape[0])


if __name__ == "__main__":
    unittest.main()

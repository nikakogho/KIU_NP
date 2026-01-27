import unittest
import numpy as np

from kiu_drone_show.dynamics import (
    DynamicsParams, saturate_vectors, repulsion_sum, acceleration_method1
)


class TestDynamics(unittest.TestCase):
    def test_saturate_vectors(self):
        V = np.array([[3.0, 4.0, 0.0],   # norm=5
                      [0.0, 0.0, 0.0]], dtype=float)
        out = saturate_vectors(V, vmax=4.0)
        n0 = np.linalg.norm(out[0])
        n1 = np.linalg.norm(out[1])
        self.assertLessEqual(n0, 4.0 + 1e-9)
        self.assertAlmostEqual(n1, 0.0, places=9)

    def test_repulsion_two_drones_equal_opposite(self):
        params = DynamicsParams(krep=2.0, rsafe=10.0)
        x = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]], dtype=float)
        F = repulsion_sum(x, params)

        # Components orthogonal to x should be ~0
        self.assertAlmostEqual(F[0, 1], 0.0, places=9)
        self.assertAlmostEqual(F[0, 2], 0.0, places=9)
        self.assertAlmostEqual(F[1, 1], 0.0, places=9)
        self.assertAlmostEqual(F[1, 2], 0.0, places=9)

        # Equal and opposite along x
        self.assertAlmostEqual(F[0, 0], -F[1, 0], places=9)

        # Direction check: drone0 at x=0 is pushed left (negative),
        # drone1 at x=1 is pushed right (positive)
        self.assertLess(F[0, 0], 0.0)
        self.assertGreater(F[1, 0], 0.0)


    def test_repulsion_zero_outside_rsafe(self):
        params = DynamicsParams(krep=2.0, rsafe=0.5)
        x = np.array([[0.0, 0.0, 0.0],
                      [10.0, 0.0, 0.0]], dtype=float)
        F = repulsion_sum(x, params)
        self.assertTrue(np.allclose(F, 0.0))

    def test_acceleration_toward_target(self):
        params = DynamicsParams(m=1.0, kp=2.0, kd=0.0, krep=0.0, rsafe=1.0)
        x = np.array([[0.0, 0.0, 0.0]], dtype=float)
        v = np.array([[0.0, 0.0, 0.0]], dtype=float)
        T = np.array([[1.0, 0.0, 0.0]], dtype=float)
        a = acceleration_method1(x, v, T, params)
        self.assertGreater(a[0, 0], 0.0)
        self.assertAlmostEqual(a[0, 1], 0.0, places=9)
        self.assertAlmostEqual(a[0, 2], 0.0, places=9)

    def test_damping_only(self):
        params = DynamicsParams(m=2.0, kp=0.0, kd=4.0, krep=0.0, rsafe=1.0)
        x = np.array([[0.0, 0.0, 0.0]], dtype=float)
        v = np.array([[1.0, -2.0, 3.0]], dtype=float)
        T = np.array([[0.0, 0.0, 0.0]], dtype=float)
        a = acceleration_method1(x, v, T, params)
        # a = (-kd*v)/m = -4/2 * v = -2v
        self.assertTrue(np.allclose(a, -2.0 * v))


if __name__ == "__main__":
    unittest.main()

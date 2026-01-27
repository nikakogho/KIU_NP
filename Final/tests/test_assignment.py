import unittest
import numpy as np

from kiu_drone_show.assignment import hungarian_assignment, apply_assignment


class TestAssignment(unittest.TestCase):
    def test_identity_assignment(self):
        D = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]], dtype=float)
        T = D.copy()
        res = hungarian_assignment(D, T, cost="sqeuclidean")
        self.assertTrue(np.array_equal(res.perm, np.array([0, 1, 2])))
        self.assertAlmostEqual(res.total_cost, 0.0, places=9)

    def test_known_optimal(self):
        # Two drones swapped targets
        D = np.array([[0, 0, 0],
                      [10, 0, 0]], dtype=float)
        T = np.array([[10, 0, 0],
                      [0, 0, 0]], dtype=float)
        res = hungarian_assignment(D, T, cost="euclidean")
        
        # optimal is swap: drone0->target1, drone1->target0
        self.assertTrue(np.array_equal(res.perm, np.array([1, 0])))
        self.assertAlmostEqual(res.total_cost, 0.0, places=9)

    def test_apply_assignment(self):
        T = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [2, 0, 0]], dtype=float)
        perm = np.array([2, 0, 1])
        out = apply_assignment(T, perm)
        self.assertTrue(np.array_equal(out, np.array([[2, 0, 0],
                                                      [0, 0, 0],
                                                      [1, 0, 0]], dtype=float)))

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            hungarian_assignment(np.zeros((3, 2)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            hungarian_assignment(np.zeros((3, 3)), np.zeros((4, 3)))

    def test_max_cost_guard(self):
        D = np.array([[0, 0, 0],
                      [100, 0, 0]], dtype=float)
        T = np.array([[0, 0, 0],
                      [0, 0, 0]], dtype=float)
        with self.assertRaises(ValueError):
            hungarian_assignment(D, T, cost="euclidean", max_cost=50.0)


if __name__ == "__main__":
    unittest.main()

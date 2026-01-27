import unittest
import numpy as np

from kiu_drone_show.initial_conditions import init_line, init_square, init_cube


class TestInitialConditions(unittest.TestCase):
    def test_init_line_shapes_and_bounds(self):
        res = init_line(10)
        self.assertEqual(res.x0.shape, (10, 3))
        self.assertEqual(res.v0.shape, (10, 3))
        self.assertTrue((res.x0 >= 0).all() and (res.x0 <= 100).all())

    def test_init_line_collinear(self):
        # points should lie on the line between start and end
        start = (10.0, 20.0, 30.0)
        end = (90.0, 80.0, 70.0)
        res = init_line(20, start_xyz=start, end_xyz=end)
        x = res.x0
        s = np.array(start)
        e = np.array(end)
        d = e - s

        # For each point, (x - s) should be parallel to d, so cross product ~ 0
        # In 3D: ||(x-s) x d|| should be ~0
        cross = np.cross(x - s[None, :], d[None, :])
        norms = np.linalg.norm(cross, axis=1)
        self.assertTrue(np.all(norms < 1e-6))

    def test_init_square_random_z_constant(self):
        rng = np.random.default_rng(123)
        res = init_square(50, center_xy=(50, 50), side=20, z=42.0, mode="random", rng=rng)
        self.assertEqual(res.x0.shape, (50, 3))
        self.assertTrue(np.allclose(res.x0[:, 2], 42.0))
        self.assertTrue((res.x0 >= 0).all() and (res.x0 <= 100).all())

    def test_init_square_grid_deterministic(self):
        res1 = init_square(30, side=40, z=10.0, mode="grid")
        res2 = init_square(30, side=40, z=10.0, mode="grid")
        self.assertTrue(np.array_equal(res1.x0, res2.x0))

    def test_init_cube_random_reproducible(self):
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        res1 = init_cube(100, side=30, rng=rng1)
        res2 = init_cube(100, side=30, rng=rng2)
        self.assertTrue(np.allclose(res1.x0, res2.x0))

    def test_init_cube_bounds(self):
        rng = np.random.default_rng(0)
        res = init_cube(1000, center_xyz=(50, 50, 50), side=90, rng=rng)
        self.assertTrue((res.x0 >= 0).all() and (res.x0 <= 100).all())

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            init_line(0)
        with self.assertRaises(ValueError):
            init_square(10, side=-1)
        with self.assertRaises(ValueError):
            init_cube(10, side=0)


if __name__ == "__main__":
    unittest.main()

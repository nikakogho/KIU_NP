import unittest
import numpy as np

from kiu_drone_show.world_mapping import pixels_to_world


class TestWorldMapping(unittest.TestCase):
    def test_bounds(self):
        # rectangle in pixel space
        pts = np.array([
            [10, 20],
            [10, 120],
            [60, 20],
            [60, 120],
        ], dtype=float)

        world, info = pixels_to_world(pts, world_min=0, world_max=100, margin=5, z_plane=0, invert_y=True)
        self.assertEqual(world.shape, (4, 3))
        self.assertTrue(np.all(world >= 0.0))
        self.assertTrue(np.all(world <= 100.0))
        self.assertTrue(np.allclose(world[:, 2], 0.0))

    def test_aspect_ratio_preserved(self):
        # non-square bbox: width=200, height=50
        pts = np.array([
            [0, 0],
            [0, 200],
            [50, 0],
            [50, 200],
        ], dtype=float)

        world, info = pixels_to_world(pts, world_min=0, world_max=100, margin=5, invert_y=True)
        # world bbox
        w = world[:, 0].max() - world[:, 0].min()
        h = world[:, 1].max() - world[:, 1].min()
        # ratio should match original (within tolerance)
        self.assertAlmostEqual(w / h, 200 / 50, places=6)

    def test_centering(self):
        # symmetric bbox should center near 50 in both axes
        pts = np.array([
            [10, 10],
            [10, 30],
            [30, 10],
            [30, 30],
        ], dtype=float)

        world, info = pixels_to_world(pts, world_min=0, world_max=100, margin=5, invert_y=True)
        cx = (world[:, 0].min() + world[:, 0].max()) / 2.0
        cy = (world[:, 1].min() + world[:, 1].max()) / 2.0
        self.assertAlmostEqual(cx, 50.0, places=6)
        self.assertAlmostEqual(cy, 50.0, places=6)

    def test_invert_y_effect(self):
        # two points with different y should flip order when invert_y changes
        pts = np.array([
            [0, 0],   # top
            [10, 0],  # lower in image
        ], dtype=float)

        w1, _ = pixels_to_world(pts, invert_y=True)
        w2, _ = pixels_to_world(pts, invert_y=False)

        # With invert_y=True, the lower image y should map to smaller world Y (because y-down -> y-up)
        # Specifically: y=10 (lower) becomes smaller Y than y=0 (upper) after inversion.
        self.assertLess(w1[1, 1], w1[0, 1])

        # Without invert, y=10 should map to larger world Y than y=0
        self.assertGreater(w2[1, 1], w2[0, 1])


if __name__ == "__main__":
    unittest.main()

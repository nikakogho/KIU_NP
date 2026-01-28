import unittest
import numpy as np

from kiu_drone_show.world_mapping import centroids_pixels_to_world


class TestCentroidWorldMapping(unittest.TestCase):
    def test_endpoints_map_to_margins(self):
        H, W = 60, 80
        centroids = np.array([
            [0.0, 0.0],          # top-left
            [H-1.0, W-1.0],      # bottom-right
        ])

        world, info = centroids_pixels_to_world(centroids, frame_H=H, frame_W=W, margin=5.0, invert_y=True)
        # x: 0 -> 5, W-1 -> 95
        self.assertAlmostEqual(world[0, 0], 5.0, places=6)
        self.assertAlmostEqual(world[1, 0], 95.0, places=6)

        # y inverted: y=0 (top) -> 95, y=H-1 (bottom) -> 5
        self.assertAlmostEqual(world[0, 1], 95.0, places=6)
        self.assertAlmostEqual(world[1, 1], 5.0, places=6)

        # z plane
        self.assertTrue(np.allclose(world[:, 2], 0.0))

    def test_monotonic_mapping_no_invert(self):
        H, W = 10, 10
        centroids = np.array([[0.0, 0.0], [9.0, 9.0]])
        world, _ = centroids_pixels_to_world(centroids, frame_H=H, frame_W=W, margin=0.0, invert_y=False)
        self.assertTrue(world[1, 0] > world[0, 0])  # x increases
        self.assertTrue(world[1, 1] > world[0, 1])  # y increases

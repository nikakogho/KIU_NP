import unittest
import numpy as np

from kiu_drone_show.shape_preservation import compute_rigid_offsets, rigid_targets_from_centroids


class TestShapePreservation(unittest.TestCase):
    def test_offsets_centroid_is_zero(self):
        P = np.array([[10, 10, 0], [20, 10, 0], [10, 20, 0], [20, 20, 0]], dtype=float)
        R, info = compute_rigid_offsets(P, z_mode="keep")
        self.assertTrue(np.allclose(R.mean(axis=0), 0.0))
        self.assertTrue(np.allclose(info.centroid, P.mean(axis=0)))

    def test_rigid_targets_translation(self):
        # base formation
        P = np.array([[10, 10, 0], [20, 10, 0], [10, 20, 0], [20, 20, 0]], dtype=float)
        R, info = compute_rigid_offsets(P, z_mode="zero")  # pure XY

        # centroid path (two frames)
        c0 = np.array([50.0, 50.0, 0.0])
        c1 = np.array([60.0, 40.0, 0.0])
        C = np.stack([c0, c1], axis=0)  # (2,3)

        T = rigid_targets_from_centroids(C, R, clip=False)  # (2,N,3)

        # Difference between frames should equal centroid delta for every drone
        delta = c1 - c0
        self.assertTrue(np.allclose(T[1] - T[0], delta[None, :]))

    def test_z_mode_zero_forces_planar(self):
        P = np.array([[10, 10, 5], [20, 10, 6], [10, 20, 7]], dtype=float)
        R, _ = compute_rigid_offsets(P, z_mode="zero")
        self.assertTrue(np.allclose(R[:, 2], 0.0))

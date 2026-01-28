import unittest
import numpy as np

from kiu_drone_show.video_tracking import track_centroids_bgdiff, TrackConfig

class TestVideoTracking(unittest.TestCase):
    def test_track_centroids_bgdiff_moving_square(self):
        K, H, W = 12, 60, 80
        frames = np.zeros((K, H, W, 3), dtype=np.float32)

        for k in range(K):
            y = 10 + k
            x = 20 + 2 * k
            frames[k, y:y+6, x:x+6, :] = 1.0  # bright square

        cfg = TrackConfig(bg_frames=1, diff_thresh=0.05, min_area=10, open_iter=0, close_iter=0)
        cyx = track_centroids_bgdiff(frames, cfg)

        self.assertEqual(cyx.shape, (K, 2))
        self.assertTrue(np.all(np.diff(cyx[:, 0]) > 0))  # y increasing
        self.assertTrue(np.all(np.diff(cyx[:, 1]) > 0))  # x increasing

    def test_track_centroids_raises_on_empty(self):
        K, H, W = 5, 40, 40
        frames = np.zeros((K, H, W, 3), dtype=np.float32)  # no moving object

        cfg = TrackConfig(bg_frames=1, diff_thresh=0.05, min_area=10)
        with self.assertRaises(ValueError):
            track_centroids_bgdiff(frames, cfg)

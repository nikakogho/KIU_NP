import unittest
import numpy as np
import tempfile
import os

from kiu_drone_show.visualization import _map_points_for_plot, load_run_npz


class TestVisualizationUtils(unittest.TestCase):
    def test_map_points_for_plot_unity_mapping(self):
        # world: (X, Y, Z) -> plot: (X, Z, Y)
        P = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ], dtype=float)

        x, y, z = _map_points_for_plot(P)
        self.assertTrue(np.allclose(x, [1.0, 10.0]))
        self.assertTrue(np.allclose(y, [3.0, 30.0]))
        self.assertTrue(np.allclose(z, [2.0, 20.0]))

    def test_load_run_npz_decodes_metrics_object(self):
        # Create a tiny .npz like the pipeline saves: metrics as object array containing 1 dict
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.npz")
            times = np.array([0.0, 1.0], dtype=float)
            X = np.zeros((2, 1, 3), dtype=float)
            metrics = np.array([{"a": 1, "b": 2.5}], dtype=object)

            np.savez_compressed(path, times=times, X=X, metrics=metrics)

            loaded = load_run_npz(path)
            self.assertIn("metrics", loaded)
            self.assertIsInstance(loaded["metrics"], dict)
            self.assertEqual(loaded["metrics"]["a"], 1)
            self.assertAlmostEqual(loaded["metrics"]["b"], 2.5)


if __name__ == "__main__":
    unittest.main()

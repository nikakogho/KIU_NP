import unittest
import numpy as np

from kiu_drone_show.pipelines import run_problem1_handwriting
from kiu_drone_show.io_utils import save_run_npz
from kiu_drone_show.visualization import load_run_npz

class TestProblem1SaveLoad(unittest.TestCase):
    def test_save_and_load_npz_roundtrip(self):
        img = np.ones((80, 160, 3), dtype=np.float32)
        img[35:40, 20:140, :] = 0.0

        run = run_problem1_handwriting(img=img, N=60, T_total=0.2, dt=0.02, record_every=1)

        path = "tests/_tmp_problem1_run.npz"
        save_run_npz(path, run)

        loaded = load_run_npz(path)
        self.assertIn("times", loaded)
        self.assertIn("X", loaded)
        self.assertIn("metrics", loaded)

        self.assertEqual(loaded["X"].shape[1:], (60, 3))
        self.assertIsInstance(loaded["metrics"], dict)

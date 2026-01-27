import unittest
import numpy as np
from kiu_drone_show.pipelines import run_problem1_handwriting

class TestProblem1Pipeline(unittest.TestCase):
    def test_problem1_pipeline_smoke_synthetic(self):
        # synthetic "handwriting": black line on white
        img = np.ones((80, 160, 3), dtype=np.float32)
        img[35:40, 20:140, :] = 0.0

        out = run_problem1_handwriting(img=img, N=80, T_total=0.2, dt=0.02, record_every=1)
        X = out["X"]
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[1:], (80, 3))
        # targets must be in world bounds and z=0
        T = out["targets_assigned"]
        self.assertTrue(np.all(T >= 0.0) and np.all(T <= 100.0))
        self.assertTrue(np.allclose(T[:,2], 0.0))

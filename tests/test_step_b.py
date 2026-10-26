"""Tests for Step B: baseline evaluation helpers (Acc, farneback_midpoint)."""
import os
import sys
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import eval_baselines as eb  # noqa: E402


class TestStepB(unittest.TestCase):
    def test_acc_summary(self):
        acc = eb.Acc()
        # Add synthetic stats for linear on gap 30
        acc.add(
            ("linear", 30),
            n=100,
            ae0=100.0,
            ae1=200.0,
            ae2=300.0,
            ae3=400.0,
            se1=500.0,
            bias1=-50.0,
            n_strong=10,
            ae_strong=50.0,
            n_weak=20,
            ae_weak=60.0,
            n_ambient=70,
            ae_ambient=90.0,
            hit241=15,
            miss241=5,
            fa241=5,
            hit221=8,
            miss221=2,
            fa221=2,
        )
        res = acc.summary()
        self.assertIn(("linear", 30), res)
        r = res[("linear", 30)]
        self.assertAlmostEqual(r["mae_b9"], 1.0)
        self.assertAlmostEqual(r["mae_b13"], 2.0)
        self.assertAlmostEqual(r["mae_b14"], 3.0)
        self.assertAlmostEqual(r["mae_b15"], 4.0)
        self.assertAlmostEqual(r["mae_mean4"], 2.5)
        self.assertAlmostEqual(r["rmse_b13"], (500.0 / 100.0) ** 0.5)
        self.assertAlmostEqual(r["bias_b13"], -0.5)
        self.assertAlmostEqual(r["mae_b13_strong"], 5.0)
        self.assertAlmostEqual(r["mae_b13_weak"], 3.0)
        # CSI241 = hit / (hit + miss + fa) = 15 / (15 + 5 + 5) = 15 / 25 = 0.6
        self.assertAlmostEqual(r["CSI241"], 0.6)
        # CSI221 = 8 / (8 + 2 + 2) = 8 / 12 = 0.6666667
        self.assertAlmostEqual(r["CSI221"], 8.0 / 12.0)

    def test_farneback_midpoint(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("cv2 not available")
        # Identical constant frames -> midpoint should be identical
        a = np.full((4, 64, 64), 250.0, dtype=np.float32)
        b = np.full((4, 64, 64), 250.0, dtype=np.float32)
        mid = eb.farneback_midpoint(a, b)
        self.assertEqual(mid.shape, (4, 64, 64))
        self.assertTrue(np.allclose(mid, 250.0, atol=1e-3))


if __name__ == "__main__":
    unittest.main()

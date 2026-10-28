import unittest
import numpy as np
from src.eval import utility as U


class TestUtility(unittest.TestCase):
    def test_coldcore(self):
        a = np.full((10, 10), 250.0); a[0, :5] = [200, 201, 202, 203, 204]
        self.assertEqual(U.coldcore(a), 204.0)

    def test_first_cross_and_delay(self):
        t = [0, 30, 60, 90]
        self.assertEqual(U.first_cross(t, [250, 245, 220, 210], 221.0), 60.0)
        self.assertIsNone(U.first_cross(t, [250, 245, 240, 230], 221.0))
        s = U.delay_summary([0.0, 10.0, None, -10.0])
        self.assertEqual(s["detected"], 3)
        self.assertAlmostEqual(s["detection_rate"], 0.75)
        self.assertAlmostEqual(s["early_gt5min"], 1 / 3)

    def test_rapid(self):
        tr = np.array([-6.0, -1.0, -5.0, -7.0])
        pr = np.array([-5.5, -4.5, -1.0, -8.0])
        r = U.rapid_scores(pr, tr, 4.0)
        self.assertEqual(r["n_true"], 3)
        self.assertEqual(r["hits"], 2)
        self.assertAlmostEqual(r["POD"], 2 / 3)
        self.assertAlmostEqual(r["FAR"], 1 / 3)


if __name__ == "__main__":
    unittest.main()

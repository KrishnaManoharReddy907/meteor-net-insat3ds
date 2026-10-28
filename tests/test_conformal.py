import unittest
import numpy as np
from src.eval import conformal as C


class TestConformal(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_quantile_finite_sample(self):
        s = np.arange(1, 10)            # n=9, alpha=0.1 -> k=ceil(10*0.9)=9 -> 9
        self.assertEqual(C.conformal_quantile(s, 0.1), 9.0)
        self.assertTrue(np.isinf(C.conformal_quantile(s, 0.05)))    # k=10 > n

    def test_global_coverage_after_calibration(self):
        n = 40000
        b_true = self.rng.uniform(0.5, 3, n)
        e = self.rng.laplace(0, b_true)
        bhat = 0.5 * b_true                                         # badly miscalibrated scale
        raw_cov = C.coverage(np.abs(e), bhat, np.log(1 / 0.1))
        self.assertLess(raw_cov, 0.8)
        h = n // 2
        q = C.conformal_quantile(np.abs(e[:h]) / bhat[:h], 0.1)
        cov = C.coverage(np.abs(e[h:]), bhat[h:], q)
        self.assertAlmostEqual(cov, 0.9, delta=0.012)

    def test_mondrian_fixes_group_coverage(self):
        n = 60000
        grp = self.rng.integers(0, 2, n)
        b_true = np.where(grp == 0, 1.0, 4.0)
        bhat = np.ones(n)                                           # same predicted scale, groups differ
        e = self.rng.laplace(0, b_true)
        keys = np.array([f"30|{C.CLASSES[g]}" for g in grp])
        h = n // 2
        cal = C.fit_mondrian(np.abs(e[:h]) / bhat[:h], keys[:h], 0.1, min_n=500)
        for g in (0, 1):
            m = keys[h:] == f"30|{C.CLASSES[g]}"
            q = C.q_for(cal, keys[h:][m])
            self.assertAlmostEqual(C.coverage(np.abs(e[h:][m]), bhat[h:][m], q), 0.9, delta=0.015)
        qg = np.full(n - h, cal["global"]["q"])
        m0 = keys[h:] == "30|strong"
        self.assertLess(C.coverage(np.abs(e[h:][~m0]), bhat[h:][~m0], qg[~m0]), 0.85)   # global q under-covers group 1

    def test_small_group_falls_back(self):
        sc = np.concatenate([np.ones(1000), np.ones(10) * 5])
        keys = np.array(["a"] * 1000 + ["b"] * 10)
        cal = C.fit_mondrian(sc, keys, 0.1, min_n=500)
        self.assertTrue(cal["groups"]["b"]["fallback"])
        self.assertFalse(cal["groups"]["a"]["fallback"])

    def test_classes_spearman_deciles(self):
        t = np.array([200.0, 230.0, 260.0, 220.9, 241.0])
        self.assertEqual(C.intensity_class(t).tolist(), [0, 1, 2, 0, 2])
        x = self.rng.normal(size=2000)
        self.assertGreater(C.spearman(x, x + 0.1 * self.rng.normal(size=2000)), 0.95)
        err = np.abs(x)
        d = C.decile_table(np.abs(x), err)
        self.assertEqual(len(d), 10)
        self.assertEqual(d, sorted(d))


if __name__ == "__main__":
    unittest.main()

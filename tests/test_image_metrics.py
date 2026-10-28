import unittest
import numpy as np
from src.eval import image_metrics as M


class TestImageMetrics(unittest.TestCase):
    def test_identity_and_noise(self):
        rng = np.random.default_rng(0)
        a = rng.normal(250, 15, (128, 128))
        self.assertAlmostEqual(M.ssim(a, a), 1.0, places=6)
        self.assertTrue(np.isinf(M.psnr(a, a)))
        b = a + rng.normal(0, 2, a.shape)
        self.assertLess(M.ssim(a, b), 1.0)
        self.assertAlmostEqual(M.psnr(a, b), 10 * np.log10(100.0 ** 2 / np.mean((a - b) ** 2)), places=6)

    def test_matches_skimage(self):
        try:
            from skimage.metrics import structural_similarity as sk
        except Exception:
            self.skipTest("scikit-image not installed")
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(1)
        a = gaussian_filter(rng.normal(size=(128, 128)), 4) * 60 + 250
        b = a + gaussian_filter(rng.normal(size=(128, 128)), 1) * 5
        ref = sk(a, b, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11,
                 data_range=100.0)
        self.assertAlmostEqual(M.ssim(a, b), ref, delta=2e-3)


if __name__ == "__main__":
    unittest.main()

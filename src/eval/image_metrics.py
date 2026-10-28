"""src/eval/image_metrics.py -- numpy PSNR / SSIM (Gaussian-window, Wang et al. 2004) for brightness-temperature maps.

Data range is fixed (default 100 K) so scores are comparable across methods and events.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

DATA_RANGE_K = 100.0


def psnr(pred: np.ndarray, tgt: np.ndarray, data_range: float = DATA_RANGE_K) -> float:
    mse = float(np.mean((pred.astype(np.float64) - tgt.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(pred: np.ndarray, tgt: np.ndarray, data_range: float = DATA_RANGE_K, sigma: float = 1.5, border: int = 5) -> float:
    """Mean SSIM of two 2-D arrays; `border` pixels are excluded to avoid padding artefacts."""
    x, y = pred.astype(np.float64), tgt.astype(np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    f = lambda a: gaussian_filter(a, sigma, truncate=3.5, mode="reflect")
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s[border:-border, border:-border].mean())

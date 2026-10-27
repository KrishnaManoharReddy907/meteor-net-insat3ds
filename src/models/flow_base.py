"""src/models/flow_base.py -- classical optical-flow midpoint (reference baseline / optional hybrid base)."""
import numpy as np

FILL_K = 250.0


def farneback_midpoint(a: np.ndarray, b: np.ndarray, ref: int = 1) -> np.ndarray:
    """a, b: [C,H,W] Kelvin at t and t+G. Returns midpoint estimate [C,H,W] (flow from channel `ref`)."""
    import cv2
    a = np.nan_to_num(a, nan=FILL_K).astype(np.float32)
    b = np.nan_to_num(b, nan=FILL_K).astype(np.float32)
    lo, hi = min(a[ref].min(), b[ref].min()), max(a[ref].max(), b[ref].max())
    sc = 255.0 / max(hi - lo, 1e-3)
    ua = ((a[ref] - lo) * sc).astype(np.uint8)
    ub = ((b[ref] - lo) * sc).astype(np.uint8)
    f = cv2.calcOpticalFlowFarneback(ua, ub, None, 0.5, 3, 15, 3, 5, 1.2, 0)   # ua(x) ~ ub(x + f)
    h, w = ua.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = np.empty_like(a)
    for c in range(a.shape[0]):
        wa = cv2.remap(a[c], gx - 0.5 * f[..., 0], gy - 0.5 * f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(b[c], gx + 0.5 * f[..., 0], gy + 0.5 * f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[c] = 0.5 * (wa + wb)
    return out


def farneback_batch(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """xa, xb: [B,C,H,W] -> [B,C,H,W]."""
    return np.stack([farneback_midpoint(xa[i], xb[i]) for i in range(len(xa))])

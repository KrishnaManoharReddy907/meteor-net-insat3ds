"""src/eval/utility.py -- numpy helpers for the utility evaluation (cold-core detection timing, cooling rates)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def coldcore(b13_patch: np.ndarray, k: int = 5) -> float:
    """k-th coldest pixel of a patch (robust 'cold-core temperature'; k=5 matches the >=5 pixel convective rule)."""
    v = np.asarray(b13_patch, dtype=np.float64).reshape(-1)
    return float(np.partition(v, k - 1)[k - 1])


def first_cross(times: Sequence[float], c: Sequence[float], thr: float) -> Optional[float]:
    idx = np.flatnonzero(np.asarray(c) < thr)
    return float(np.asarray(times)[idx[0]]) if len(idx) else None


def delay_summary(delays: List[Optional[float]]) -> Dict[str, float]:
    """delays: minutes (detect - truth) per onset event, None = never detected inside the window."""
    n = len(delays)
    got = np.array([d for d in delays if d is not None], dtype=np.float64)
    out = {"n_events": n, "detected": int(len(got)), "detection_rate": len(got) / n if n else float("nan")}
    if len(got):
        out.update({"mean_delay_min": float(got.mean()), "median_delay_min": float(np.median(got)),
                    "within_10min": float(np.mean(got <= 10.0)), "early_gt5min": float(np.mean(got < -5.0))})
    return out


def rapid_scores(pred_rate: np.ndarray, true_rate: np.ndarray, k: float) -> Dict[str, float]:
    """Rapid cooling = rate <= -k (K per interval). Returns POD/FAR/CSI and counts."""
    ev, pr = true_rate <= -k, pred_rate <= -k
    h, m, f = int((ev & pr).sum()), int((ev & ~pr).sum()), int((~ev & pr).sum())
    return {"n_true": int(ev.sum()), "hits": h, "POD": h / (h + m) if h + m else float("nan"),
            "FAR": f / (h + f) if h + f else float("nan"), "CSI": h / (h + m + f) if h + m + f else float("nan")}

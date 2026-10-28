"""src/eval/conformal.py -- split conformal prediction helpers (numpy only, unit-tested).

Nonconformity score s = |y - yhat| / scale (scale = the head's predicted Laplace scale).
Interval at level 1-alpha:  yhat +/- q * scale,  q = ceil((n+1)(1-alpha))-th smallest calibration score.
Mondrian (group-conditional) version: one q per group, e.g. (gap, predicted-intensity class).
Guarantee: marginal (per group) coverage >= 1-alpha if calibration and test samples are exchangeable.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

CLASSES = ("strong", "weak", "ambient")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])


def intensity_class(t13_k: np.ndarray) -> np.ndarray:
    """0 strong (<221 K), 1 weak (221-241 K), 2 ambient (>=241 K)."""
    t = np.asarray(t13_k)
    return np.where(t < 221.0, 0, np.where(t < 241.0, 1, 2)).astype(np.int64)


def group_keys(gap_min: np.ndarray, cls: np.ndarray) -> np.ndarray:
    return np.array([f"{int(g)}|{CLASSES[int(c)]}" for g, c in zip(gap_min, cls)])


def fit_mondrian(scores: np.ndarray, groups: np.ndarray, alpha: float, min_n: int = 500) -> Dict:
    """Returns {'global': {q, n}, 'groups': {key: {q, n, fallback}}}. Groups with < min_n scores use the global q."""
    scores = np.asarray(scores)
    out = {"alpha": alpha, "global": {"q": conformal_quantile(scores, alpha), "n": int(len(scores))}, "groups": {}}
    for key in np.unique(groups):
        s = scores[groups == key]
        if len(s) >= min_n:
            out["groups"][str(key)] = {"q": conformal_quantile(s, alpha), "n": int(len(s)), "fallback": False}
        else:
            out["groups"][str(key)] = {"q": out["global"]["q"], "n": int(len(s)), "fallback": True}
    return out


def q_for(cal: Dict, groups: np.ndarray) -> np.ndarray:
    gq = {k: v["q"] for k, v in cal["groups"].items()}
    return np.array([gq.get(str(k), cal["global"]["q"]) for k in groups], dtype=np.float64)


def coverage(abs_err: np.ndarray, scale: np.ndarray, q) -> float:
    if len(abs_err) == 0:
        return float("nan")
    return float(np.mean(abs_err <= np.asarray(q) * scale))


def rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def decile_table(scale: np.ndarray, abs_err: np.ndarray, nbins: int = 10):
    """Mean |error| per bin of predicted scale (low -> high). A useful uncertainty rises monotonically."""
    order = np.argsort(scale)
    return [float(abs_err[idx].mean()) for idx in np.array_split(order, nbins) if len(idx)]

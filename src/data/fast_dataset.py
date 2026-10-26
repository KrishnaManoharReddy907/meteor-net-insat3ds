"""
src/data/fast_dataset.py
Fast, cache-backed patch store for METEOR-Net with MIXED temporal gaps.

A training sample for gap G minutes (G in {10, 20, 30}; nominal cadence 5 min) is:
    inputs : frames at  t-2G, t-G, t, t+G          (4 frames)
    target : frame  at  t+G/2                       (midpoint)
G = 30 reproduces the original [-60,-30,0,+30] -> +15 task (INSAT-3DS cadence).
G = 20, 10 give additional training diversity and let the SAME weights be applied
recursively (e.g. 20 -> 10 -> 5 validation cascade; 30 -> 15 -> 7.5 at INSAT inference).

Validity is STRICT: a patch is usable only if every pixel of all 4 channels is valid in all
5 frames, and all 5 frame times match the nominal offsets within `time_tol_s`.

numpy-only core (FastSequenceStore); torch is imported lazily in `to_device`.
No DataLoader / worker processes are needed: one batch is ~20 MB of uint16 slicing.
"""

import json
import os
import queue
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src.data.fast_cache import (
    SCALE_K,
    OFFSET_K,
    SENTINEL,
    kelvin_to_raw_threshold,
    load_event_cache,
    parse_iso,
)

T241_RAW = kelvin_to_raw_threshold(241.0)
T221_RAW = kelvin_to_raw_threshold(221.0)


def _patch_grid(h: int, w: int, ps: int, stride: int) -> Tuple[List[int], List[int]]:
    return list(range(0, h - ps + 1, stride)), list(range(0, w - ps + 1, stride))


class FastSequenceStore:
    """
    Indexes every valid (event, anchor time, gap, patch) sample and slices patches from cache.

    Item arrays (aligned, length N):
        it_ev, it_t, it_s (gap in steps), it_p (patch id), it_conv, it_gap_min, it_tmin_k
    """

    def __init__(
        self,
        cache_dir: str,
        event_ids: Sequence[str],
        gaps_min: Sequence[int] = (10, 20, 30),
        patch_size: int = 128,
        stride: int = 64,
        nominal_step_min: float = 5.0,
        time_tol_s: float = 90.0,
        in_ram: bool = False,
        b13_index: int = 1,
        use_patch_stats_cache: bool = True,
    ):
        if len(event_ids) == 0:
            raise ValueError("event_ids is empty.")
        self.cache_dir = cache_dir
        self.event_ids = list(event_ids)
        self.gaps_min = [int(g) for g in gaps_min]
        self.ps = int(patch_size)
        self.stride = int(stride)
        self.step_min = float(nominal_step_min)
        self.time_tol_s = float(time_tol_s)
        self.b13 = int(b13_index)
        self.gap_steps: Dict[int, int] = {}
        for g in self.gaps_min:
            s = g / self.step_min
            if abs(s - round(s)) > 1e-9 or int(round(s)) % 2 != 0 or int(round(s)) < 2:
                raise ValueError(
                    f"Gap {g} min is invalid: gap/step must be an EVEN integer >= 2 so the "
                    f"midpoint lies on the {self.step_min:g}-min grid (valid: 10, 20, 30, 40...)."
                )
            self.gap_steps[g] = int(round(s))

        self.arrays: List[np.ndarray] = []
        self.metas: List[Dict[str, Any]] = []
        self.times_s: List[np.ndarray] = []
        self.grids: List[Tuple[List[int], List[int]]] = []
        self.channel_order: Optional[Tuple[int, ...]] = None
        for eid in self.event_ids:
            arr, meta = load_event_cache(cache_dir, eid, in_ram=in_ram)
            ch = tuple(meta["channel_order"])
            if self.channel_order is None:
                self.channel_order = ch
            elif ch != self.channel_order:
                raise ValueError(f"Channel order mismatch in event {eid}: {ch} vs {self.channel_order}")
            self.arrays.append(arr)
            self.metas.append(meta)
            ts = [parse_iso(t) for t in meta["timestamps"]]
            t0 = ts[0]
            self.times_s.append(np.array([(t - t0).total_seconds() for t in ts], dtype=np.float64))
            self.grids.append(_patch_grid(arr.shape[2], arr.shape[3], self.ps, self.stride))
        self.n_ch = len(self.channel_order)

        parts = []
        for ev_i in range(len(self.event_ids)):
            parts.append(self._index_event(ev_i, use_patch_stats_cache))

        def cat(key, dtype):
            return np.concatenate([p[key] for p in parts]).astype(dtype) if parts else np.zeros(0, dtype)

        self.it_ev = cat("ev", np.int16)
        self.it_t = cat("t", np.int32)
        self.it_s = cat("s", np.int16)
        self.it_p = cat("p", np.int32)
        self.it_conv = cat("conv", bool)
        self.it_tmin_k = cat("tmin", np.float32)
        self.it_gap_min = np.rint(self.it_s.astype(np.float64) * self.step_min).astype(np.int16)

        self._pools: Dict[Tuple[int, bool], np.ndarray] = {}
        for g in self.gaps_min:
            for c in (False, True):
                self._pools[(g, c)] = np.nonzero((self.it_gap_min == g) & (self.it_conv == c))[0]

    # ------------------------------------------------------------------ indexing
    def _patch_stats(self, ev_i: int, use_cache: bool):
        """Per (timestamp, patch): fully-valid flag, min raw B13, #px<241K, #px<221K."""
        arr, meta = self.arrays[ev_i], self.metas[ev_i]
        rows, cols = self.grids[ev_i]
        P = len(rows) * len(cols)
        T = arr.shape[0]
        cache_file = os.path.join(
            self.cache_dir, f"{self.event_ids[ev_i]}.pstats_p{self.ps}_s{self.stride}_b{self.b13}.npz"
        )
        if use_cache and os.path.exists(cache_file):
            z = np.load(cache_file, allow_pickle=False)
            if str(z["array_sha256"]) == meta["array_sha256"] and z["pv"].shape == (T, P):
                return z["pv"], z["bmin"], z["n241"], z["n221"]
        pv = np.zeros((T, P), dtype=bool)
        bmin = np.full((T, P), 65535, dtype=np.int32)
        n241 = np.zeros((T, P), dtype=np.int32)
        n221 = np.zeros((T, P), dtype=np.int32)
        ps = self.ps
        for t in range(T):
            frame = np.asarray(arr[t])
            valid_all = (frame != SENTINEL).all(axis=0)
            b13 = frame[self.b13].astype(np.int32)
            p = 0
            for r in rows:
                for c in cols:
                    if valid_all[r:r + ps, c:c + ps].all():
                        blk = b13[r:r + ps, c:c + ps]
                        pv[t, p] = True
                        bmin[t, p] = blk.min()
                        n241[t, p] = int((blk < T241_RAW).sum())
                        n221[t, p] = int((blk < T221_RAW).sum())
                    p += 1
        if use_cache:
            try:
                np.savez(cache_file, pv=pv, bmin=bmin, n241=n241, n221=n221,
                         array_sha256=np.array(meta["array_sha256"]))
            except OSError:
                pass  # read-only cache dir: just recompute next time
        return pv, bmin, n241, n221

    def _index_event(self, ev_i: int, use_cache: bool) -> Dict[str, np.ndarray]:
        pv, bmin, n241, n221 = self._patch_stats(ev_i, use_cache)
        tsec = self.times_s[ev_i]
        T = len(tsec)
        out = {k: [] for k in ("ev", "t", "s", "p", "conv", "tmin")}
        for g in self.gaps_min:
            s = self.gap_steps[g]
            h = s // 2
            anchors = np.arange(2 * s, T - s)
            if len(anchors) == 0:
                continue
            ok = np.ones(len(anchors), dtype=bool)
            for k in (-2 * s, -s, s, h):
                delta = tsec[anchors + k] - tsec[anchors]
                ok &= np.abs(delta - k * self.step_min * 60.0) <= self.time_tol_s
            valid = (pv[anchors - 2 * s] & pv[anchors - s] & pv[anchors]
                     & pv[anchors + s] & pv[anchors + h])
            valid &= ok[:, None]
            ti, pi = np.nonzero(valid)
            tgt = anchors[ti] + h
            tmin_raw = bmin[tgt, pi]
            frac241 = n241[tgt, pi] / float(self.ps * self.ps)
            conv = (tmin_raw < T241_RAW) & ((frac241 >= 0.02) | (n221[tgt, pi] >= 5))
            out["ev"].append(np.full(len(ti), ev_i))
            out["t"].append(anchors[ti])
            out["s"].append(np.full(len(ti), s))
            out["p"].append(pi)
            out["conv"].append(conv)
            out["tmin"].append(tmin_raw.astype(np.float32) * np.float32(SCALE_K) + np.float32(OFFSET_K))
        return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}

    # ------------------------------------------------------------------ queries
    def __len__(self) -> int:
        return int(self.it_ev.shape[0])

    def describe(self) -> Dict[str, Any]:
        rows = []
        for ev_i, eid in enumerate(self.event_ids):
            for g in self.gaps_min:
                m = (self.it_ev == ev_i) & (self.it_gap_min == g)
                rows.append({
                    "event_id": eid, "gap_min": g, "samples": int(m.sum()),
                    "convective": int((m & self.it_conv).sum()),
                })
        return {
            "n_events": len(self.event_ids),
            "n_samples": len(self),
            "per_event_gap": rows,
            "pool_sizes": {f"gap{g}_{'conv' if c else 'amb'}": int(len(v)) for (g, c), v in self._pools.items()},
        }

    def select(self, gap_min: Optional[int] = None, event_ids: Optional[Sequence[str]] = None,
               convective: Optional[bool] = None) -> np.ndarray:
        m = np.ones(len(self), dtype=bool)
        if gap_min is not None:
            m &= self.it_gap_min == int(gap_min)
        if event_ids is not None:
            idx = [self.event_ids.index(e) for e in event_ids]
            m &= np.isin(self.it_ev, idx)
        if convective is not None:
            m &= self.it_conv == bool(convective)
        return np.nonzero(m)[0]

    def sample_train_ids(self, rng: np.random.Generator, batch_size: int, convective_frac: float = 0.5,
                         gap_weights: Optional[Dict[int, float]] = None) -> np.ndarray:
        """Random batch with a target convective fraction and gap mixture (with replacement)."""
        if gap_weights is None:
            # Default: gap 30 (the INSAT-relevant task) gets 50% of samples, the rest share 50%.
            others = [g for g in self.gaps_min if g != 30]
            if 30 in self.gaps_min and others:
                gap_weights = {30: 0.5, **{g: 0.5 / len(others) for g in others}}
            else:
                gap_weights = {g: 1.0 for g in self.gaps_min}
        gaps = [g for g in self.gaps_min if gap_weights.get(g, 0.0) > 0]
        w = np.array([gap_weights[g] for g in gaps], dtype=np.float64)
        w /= w.sum()
        n_conv = int(round(convective_frac * batch_size))
        want_conv = np.zeros(batch_size, dtype=bool)
        want_conv[:n_conv] = True
        rng.shuffle(want_conv)
        chosen_gap = rng.choice(gaps, size=batch_size, p=w)
        ids = np.empty(batch_size, dtype=np.int64)
        for b in range(batch_size):
            g, c = int(chosen_gap[b]), bool(want_conv[b])
            pool = self._pools[(g, c)]
            if len(pool) == 0:
                pool = self._pools[(g, not c)]
            if len(pool) == 0:
                alt = [self._pools[(gg, cc)] for gg in gaps for cc in (False, True) if len(self._pools[(gg, cc)])]
                if not alt:
                    raise RuntimeError("No valid samples available.")
                pool = alt[int(rng.integers(len(alt)))]
            ids[b] = pool[int(rng.integers(len(pool)))]
        return ids

    # ------------------------------------------------------------------ slicing
    def gather(self, item_ids: Sequence[int]) -> Dict[str, np.ndarray]:
        ids = np.asarray(item_ids, dtype=np.int64)
        B, ps, C = len(ids), self.ps, self.n_ch
        raw_in = np.empty((B, 4, C, ps, ps), dtype=np.uint16)
        raw_tgt = np.empty((B, C, ps, ps), dtype=np.uint16)
        rr = np.empty(B, dtype=np.int32)
        cc = np.empty(B, dtype=np.int32)
        for b, i in enumerate(ids):
            ev, t, s, p = int(self.it_ev[i]), int(self.it_t[i]), int(self.it_s[i]), int(self.it_p[i])
            rows, cols = self.grids[ev]
            r, c = rows[p // len(cols)], cols[p % len(cols)]
            arr = self.arrays[ev]
            for k, off in enumerate((-2 * s, -s, 0, s)):
                raw_in[b, k] = arr[t + off, :, r:r + ps, c:c + ps]
            raw_tgt[b] = arr[t + s // 2, :, r:r + ps, c:c + ps]
            rr[b], cc[b] = r, c
        return {
            "raw_in": raw_in, "raw_tgt": raw_tgt, "item_id": ids,
            "event_idx": self.it_ev[ids].astype(np.int32), "t_index": self.it_t[ids],
            "gap_min": self.it_gap_min[ids].astype(np.int32), "row": rr, "col": cc,
            "is_convective": self.it_conv[ids], "b13_target_min_k": self.it_tmin_k[ids],
        }

    def iter_eval_batches(self, item_ids: Sequence[int], batch_size: int) -> Iterator[Dict[str, np.ndarray]]:
        ids = np.asarray(item_ids)
        for i in range(0, len(ids), batch_size):
            yield self.gather(ids[i:i + batch_size])


# ---------------------------------------------------------------------- torch glue
def to_device(batch: Dict[str, np.ndarray], device, standardizer=None) -> Dict[str, Any]:
    """
    Moves a numpy batch to `device` and decodes uint16 -> Kelvin there.
    Returns input_kelvin [B,4,C,H,W], input_mask [B,4,H,W], target_kelvin [B,C,H,W],
    target_mask [B,H,W], combined_mask [B,H,W] and (if standardizer) input_norm.
    """
    import torch

    def dec(a: np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).to(device, non_blocking=True)
        x = t.to(torch.int32) & 0xFFFF
        valid = x != SENTINEL
        k = x.float() * SCALE_K + OFFSET_K
        k = torch.where(valid, k, torch.full_like(k, float("nan")))
        return k, valid

    in_k, in_v = dec(batch["raw_in"])
    tg_k, tg_v = dec(batch["raw_tgt"])
    in_mask = in_v.all(dim=2)
    tg_mask = tg_v.all(dim=1)
    out = {
        "input_kelvin": in_k, "input_mask": in_mask,
        "target_kelvin": tg_k, "target_mask": tg_mask,
        "combined_mask": in_mask.all(dim=1) & tg_mask,
        "gap_min": torch.from_numpy(batch["gap_min"]).to(device),
        "is_convective": torch.from_numpy(batch["is_convective"]).to(device),
    }
    if standardizer is not None:
        out["input_norm"] = standardizer.normalize_input(in_k, in_mask)
    return out


class BatchPrefetcher:
    """Background thread that builds numpy batches ahead of the training loop."""

    _END = object()

    def __init__(self, make_batch, num_batches: int, depth: int = 4):
        self._make, self._n = make_batch, int(num_batches)
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        self._err: Optional[BaseException] = None
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        try:
            for i in range(self._n):
                self._q.put(self._make(i))
        except BaseException as e:  # propagate to consumer
            self._err = e
        finally:
            self._q.put(self._END)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is self._END:
                if self._err is not None:
                    raise self._err
                return
            yield item

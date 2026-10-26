"""
src/data/fast_cache.py
Compact on-disk cache for cropped GOES (or INSAT) brightness-temperature frames.

Why this exists
---------------
The legacy GOESSequencePatchDataset re-reads ~20 NetCDF files (5 timestamps x 4 bands)
every time its tiny LRU cache misses. With a shuffled sampler that is almost every
sample, so training became I/O bound (~2 s per microbatch vs ~35 ms of GPU compute).

This module stores each event ONCE as a single uint16 array [T, C, H, W] already cropped
to the study region, plus a JSON sidecar. Patches are then sliced straight from memory.

Encoding
--------
    raw = round((K - 100.0) / 0.01)      K = raw * 0.01 + 100.0
    raw == 65535  -> invalid pixel (NaN / fill / DQF != 0 in ANY channel)
Resolution 0.01 K (max round-off 0.005 K), range 100.00 K .. 755.34 K.
A timestamp is invalid at a pixel if ANY channel is invalid there (same semantics as the
legacy combined mask), so validity can be recovered from the array alone.

The module is numpy-only (no torch, no h5py) so it can be imported and tested anywhere.
"""

import datetime
import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

CACHE_FORMAT_VERSION = 1
SCALE_K = 0.01
OFFSET_K = 100.0
SENTINEL = 65535
MAX_ENCODABLE = 65534


def kelvin_to_raw_threshold(kelvin: float) -> int:
    """Raw uint16 code equal to `kelvin`. Use `raw < code` for 'strictly colder than'."""
    return int(round((float(kelvin) - OFFSET_K) / SCALE_K))


def encode_kelvin(k: np.ndarray) -> np.ndarray:
    """float Kelvin (NaN/inf = invalid) -> uint16 with SENTINEL for invalid."""
    k = np.asarray(k, dtype=np.float32)
    out = np.full(k.shape, SENTINEL, dtype=np.uint16)
    finite = np.isfinite(k)
    q = np.rint((k[finite].astype(np.float64) - OFFSET_K) / SCALE_K)
    q = np.clip(q, 0, MAX_ENCODABLE)
    out[finite] = q.astype(np.uint16)
    return out


def decode_kelvin(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """uint16 -> (float32 Kelvin with NaN at invalid, bool valid mask)."""
    raw = np.asarray(raw)
    valid = raw != SENTINEL
    k = raw.astype(np.float32) * np.float32(SCALE_K) + np.float32(OFFSET_K)
    k[~valid] = np.nan
    return k, valid


def array_path(cache_dir: str, event_id: str) -> str:
    return os.path.join(cache_dir, f"{event_id}.u16.npy")


def meta_path(cache_dir: str, event_id: str) -> str:
    return os.path.join(cache_dir, f"{event_id}.meta.json")


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iso(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def write_event_cache(
    cache_dir: str,
    event_id: str,
    timestamps: Sequence[datetime.datetime],
    channel_order: Sequence[int],
    crop_info: Dict[str, Any],
    frame_fn: Callable[[int], np.ndarray],
    frame_shape_chw: Tuple[int, int, int],
    already_encoded: bool = False,
    overwrite: bool = False,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Writes <event_id>.u16.npy and <event_id>.meta.json atomically.

    frame_fn(i) must return the i-th timestamp as [C, H, W]: float32 Kelvin (NaN = invalid)
    unless already_encoded=True, in which case it must be uint16.
    Timestamps must be strictly increasing.
    """
    os.makedirs(cache_dir, exist_ok=True)
    final_arr = array_path(cache_dir, event_id)
    final_meta = meta_path(cache_dir, event_id)
    if (os.path.exists(final_arr) or os.path.exists(final_meta)) and not overwrite:
        raise FileExistsError(f"Cache for '{event_id}' already exists in {cache_dir} (use overwrite=True).")

    ts = list(timestamps)
    if any(ts[i + 1] <= ts[i] for i in range(len(ts) - 1)):
        raise ValueError("Timestamps must be strictly increasing.")
    T = len(ts)
    C, H, W = (int(v) for v in frame_shape_chw)
    if C != len(channel_order):
        raise ValueError(f"frame_shape C={C} does not match channel_order length {len(channel_order)}")

    tmp_arr = final_arr + ".partial"
    mm = np.lib.format.open_memmap(tmp_arr, mode="w+", dtype=np.uint16, shape=(T, C, H, W))
    try:
        for i in range(T):
            fr = np.asarray(frame_fn(i))
            if fr.shape != (C, H, W):
                raise ValueError(f"Frame {i} shape {fr.shape} != expected {(C, H, W)}")
            mm[i] = fr.astype(np.uint16, copy=False) if already_encoded else encode_kelvin(fr)
        mm.flush()
    finally:
        del mm
    os.replace(tmp_arr, final_arr)

    meta = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "event_id": event_id,
        "dtype": "uint16",
        "shape": [T, C, H, W],
        "channel_order": [int(c) for c in channel_order],
        "scale_k": SCALE_K,
        "offset_k": OFFSET_K,
        "sentinel": SENTINEL,
        "timestamps": [_iso(t) for t in ts],
        "crop_info": crop_info,
        "array_sha256": file_sha256(final_arr),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra_meta:
        meta["extra"] = extra_meta
    tmp_meta = final_meta + ".partial"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta, final_meta)
    return meta


def load_event_cache(
    cache_dir: str, event_id: str, in_ram: bool = False, verify_hash: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Returns (array [T,C,H,W] uint16, meta). Memory-mapped unless in_ram=True."""
    mp, ap = meta_path(cache_dir, event_id), array_path(cache_dir, event_id)
    if not (os.path.exists(mp) and os.path.exists(ap)):
        raise FileNotFoundError(f"Cache files for '{event_id}' not found in {cache_dir}")
    with open(mp, "r") as f:
        meta = json.load(f)
    if meta.get("cache_format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported cache format version: {meta.get('cache_format_version')}")
    if (meta["scale_k"], meta["offset_k"], meta["sentinel"]) != (SCALE_K, OFFSET_K, SENTINEL):
        raise ValueError("Cache encoding constants do not match this code version.")
    if verify_hash and file_sha256(ap) != meta["array_sha256"]:
        raise ValueError(f"SHA-256 mismatch for {ap}: cache is corrupt or was modified.")
    arr = np.load(ap, mmap_mode=None if in_ram else "r")
    if list(arr.shape) != list(meta["shape"]) or arr.dtype != np.uint16:
        raise ValueError(f"Array {arr.shape}/{arr.dtype} does not match meta {meta['shape']}/uint16")
    return arr, meta


def parse_iso(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

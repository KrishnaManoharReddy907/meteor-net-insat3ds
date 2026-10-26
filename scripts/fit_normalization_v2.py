"""
scripts/fit_normalization_v2.py
Fit per-channel mean/std (Kelvin) on TRAINING events of the v2 cache only, and write a stats
file in the exact schema of outputs/normalization/frozen_global_train_stats.json (loadable by
ChannelStandardizer.from_stats_file). numpy-only.

Safety
  * only events whose split == 'train' in the events JSON are used (val/cal/test never touched)
  * refuses to run on a 'test' cache directory
  * never overwrites an existing file unless --overwrite, and never the v1 frozen file
  * statistics use pixels that are valid in ALL channels (same strict validity as training)

Example
  python scripts/fit_normalization_v2.py --cache_dir /content/cache/dev \
      --events_json configs/events_v2.json --out outputs/normalization/v2_train_stats.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.fast_cache import OFFSET_K, SCALE_K, SENTINEL, load_event_cache  # noqa: E402

PROTECTED_NAMES = {"frozen_global_train_stats.json"}
SHIFT_K = 250.0     # shift before squaring: avoids catastrophic cancellation in float64 sums


def fit_stats(cache_dir: str, event_ids: Sequence[str], frame_stride: int = 2) -> Dict[str, Any]:
    n = None
    s1 = s2 = None
    chan_order = None
    hashes = {}
    for eid in event_ids:
        arr, meta = load_event_cache(cache_dir, eid)
        if chan_order is None:
            chan_order = list(meta["channel_order"])
            C = len(chan_order)
            n, s1, s2 = 0, np.zeros(C), np.zeros(C)
        elif list(meta["channel_order"]) != chan_order:
            raise ValueError(f"channel order mismatch in {eid}")
        hashes[eid] = meta["array_sha256"]
        for t in range(0, arr.shape[0], frame_stride):
            fr = np.asarray(arr[t])                       # [C,H,W] uint16
            valid = (fr != SENTINEL).all(axis=0)
            if not valid.any():
                continue
            k = fr[:, valid].astype(np.float64) * SCALE_K + OFFSET_K - SHIFT_K   # [C, Nvalid]
            n += k.shape[1]
            s1 += k.sum(axis=1)
            s2 += (k * k).sum(axis=1)
    if not n:
        raise RuntimeError("no valid pixels found")
    mean_shift = s1 / n
    var = s2 / n - mean_shift ** 2
    return {"channel_order": chan_order, "means": (mean_shift + SHIFT_K).tolist(),
            "stds": np.sqrt(np.maximum(var, 0.0)).tolist(), "n_pixels": int(n), "hashes": hashes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--out", default="outputs/normalization/v2_train_stats.json")
    ap.add_argument("--frame_stride", type=int, default=2, help="use every Nth timestamp (speed)")
    ap.add_argument("--compare_with", default="outputs/normalization/frozen_global_train_stats.json")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.basename(args.out) in PROTECTED_NAMES:
        sys.exit(f"REFUSED: {args.out} is a protected v1 file.")
    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f"REFUSED: {args.out} exists (use --overwrite).")
    idx_path = os.path.join(args.cache_dir, "cache_index.json")
    if os.path.exists(idx_path) and json.load(open(idx_path)).get("cache_role") == "test":
        sys.exit("REFUSED: cache_dir is a TEST cache.")

    with open(args.events_json) as f:
        raw = json.load(f)
    events = raw["events"] if isinstance(raw, dict) else raw
    train_ids: List[str] = [e["event_id"] for e in events if e["split"] == "train"]
    if not train_ids:
        sys.exit("No events with split == 'train' in the events JSON.")
    have = [e for e in train_ids if os.path.exists(os.path.join(args.cache_dir, f"{e}.meta.json"))]
    missing = sorted(set(train_ids) - set(have))
    if missing:
        sys.exit(f"Train events missing from cache: {missing}. Build them first.")
    res = fit_stats(args.cache_dir, have, args.frame_stride)

    payload = {
        "num_channels": len(res["channel_order"]),
        "channel_order": res["channel_order"],
        "means": res["means"], "stds": res["stds"], "eps": 1e-6,
        "provenance": {
            "description": f"Fitted on v2 TRAIN events only ({len(have)} events), strict all-channel validity",
            "training_events": have, "cache_array_sha256": res["hashes"],
            "n_pixels": res["n_pixels"], "frame_stride": args.frame_stride,
            "created_at": datetime.datetime.now().isoformat(),
            "channel_order": res["channel_order"],
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out} from {len(have)} train events, {res['n_pixels']:,} pixel samples")
    print("band   mean_K    std_K")
    for b, m, s in zip(res["channel_order"], res["means"], res["stds"]):
        print(f"{b:>4} {m:9.3f} {s:8.3f}")
    if os.path.exists(args.compare_with):
        old = json.load(open(args.compare_with))
        print("\nshift vs v1 frozen stats (mean, std):")
        for b, m, s, m0, s0 in zip(res["channel_order"], res["means"], res["stds"], old["means"], old["stds"]):
            print(f"{b:>4} {m - m0:+8.2f} K {s - s0:+8.2f} K")


if __name__ == "__main__":
    main()

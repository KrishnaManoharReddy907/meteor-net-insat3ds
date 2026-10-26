"""
scripts/build_cache_s3.py
Download GOES-19 CONUS scans for the events in an events JSON, crop them, write one compact
uint16 cache per event (same format as build_fast_cache.py), and delete the raw NetCDFs.
Runs fine on a FREE Colab CPU runtime (no GPU units burnt).

Cache role guard
  A cache directory is either 'dev' (train/val/cal events) or 'test'. Mixing is refused, so the
  held-out test arrays can live in a physically separate folder that training never mounts.

Examples
  # preview only (lists S3, counts scans, downloads nothing)
  python scripts/build_cache_s3.py --events_json configs/events_v2.json --cache_dir /content/cache/dev --dry_run

  # build everything except test, sync each finished event to Drive
  python scripts/build_cache_s3.py --events_json configs/events_v2.json --cache_dir /content/cache/dev \
      --splits train,val,cal --sync_dir /content/drive/MyDrive/meteor_net_cache/goes19_v2_dev

  # split the work across teammates
  python scripts/build_cache_s3.py ... --only_events event_2025-05-18,event_2025-06-03

  # held-out test events (separate folder!)
  python scripts/build_cache_s3.py --events_json configs/events_v2.json --cache_dir /content/cache/test --splits test
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.fast_cache import array_path, meta_path, write_event_cache  # noqa: E402
from src.data.goes_s3 import download_file, group_scans, list_keys, list_window_keys  # noqa: E402

TIME_FMT = "%Y-%m-%d %H:%M:%S"
MIN_SCANS_DEFAULT = 66   # of 72 nominal; a few gaps are tolerated (affected windows are dropped)


# ----------------------------------------------------------------------------- events / roles
def load_events(path: str, splits: Optional[Sequence[str]] = None,
                only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        raw = json.load(f)
    events = raw["events"] if isinstance(raw, dict) else raw
    out = []
    for e in events:
        if splits and e["split"] not in splits:
            continue
        if only and e["event_id"] not in only:
            continue
        out.append({**e,
                    "start_dt": datetime.datetime.strptime(e["start"], TIME_FMT),
                    "end_dt": datetime.datetime.strptime(e["end"], TIME_FMT)})
    return out


def role_of_split(split: str) -> str:
    return "test" if split == "test" else "dev"


def index_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "cache_index.json")


def read_index(cache_dir: str) -> Dict[str, Any]:
    p = index_path(cache_dir)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {"events": {}}


def write_index(cache_dir: str, index: Dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    tmp = index_path(cache_dir) + ".partial"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, index_path(cache_dir))


def enforce_cache_role(cache_dir: str, events: Sequence[Dict[str, Any]]) -> str:
    """Refuse to mix dev and test events in one cache folder. Returns the role."""
    roles = {role_of_split(e["split"]) for e in events}
    if len(roles) > 1:
        raise SystemExit("REFUSED: request mixes test and non-test events. Build them in separate cache dirs.")
    want = roles.pop() if roles else "dev"
    idx = read_index(cache_dir)
    have = idx.get("cache_role")
    if have is None:
        existing = {role_of_split(v.get("split", "train")) for v in idx.get("events", {}).values()}
        have = existing.pop() if len(existing) == 1 else None
    if have is not None and have != want:
        raise SystemExit(f"REFUSED: cache dir '{cache_dir}' has role '{have}', but this request is '{want}'.")
    idx["cache_role"] = want
    write_index(cache_dir, idx)
    return want


# ----------------------------------------------------------------------------- build one event
def build_event(
    spec: Dict[str, Any],
    cache_dir: str,
    work_dir: str,
    channels: Sequence[int],
    crop_info: Dict[str, Any],
    list_fn: Callable[[str], List[str]],
    download_fn: Callable[[str, str], int],
    read_fn: Callable[[Dict[int, str]], np.ndarray],
    workers: int = 12,
    keep_raw: bool = False,
    overwrite: bool = False,
    min_scans: int = MIN_SCANS_DEFAULT,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    eid = spec["event_id"]
    if os.path.exists(array_path(cache_dir, eid)) and not overwrite:
        return {"event_id": eid, "status": "exists"}

    keys = list_window_keys(spec["start_dt"], spec["end_dt"], list_fn)
    scans = group_scans(keys, channels, spec["start_dt"], spec["end_dt"])
    modes = sorted({m for s in scans for m in s["modes"]})
    steps = np.diff([s["datetime"].timestamp() for s in scans]) if len(scans) > 1 else np.array([])
    irregular = int((np.abs(steps - 300.0) > 90).sum()) if len(steps) else 0
    info = {"event_id": eid, "n_scans": len(scans), "scan_modes": modes, "irregular_steps": irregular}
    if len(scans) < min_scans:
        log(f"[skip] {eid}: only {len(scans)} complete scans (< {min_scans})")
        return {**info, "status": "insufficient"}
    if dry_run:
        return {**info, "status": "dry_run_ok"}

    ev_dir = os.path.join(work_dir, eid)
    tasks = [(s["channels"][c], os.path.join(ev_dir, os.path.basename(s["channels"][c])))
             for s in scans for c in channels]
    sizes: Dict[str, int] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_fn, k, d): (k, d) for k, d in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            k, d = futs[fu]
            try:
                sizes[k] = fu.result()
            except Exception as e:  # noqa: BLE001
                failures.append((k, str(e)[:120]))
            if i % 50 == 0:
                log(f"    {eid}: downloaded {i}/{len(tasks)} files")
    if failures:
        log(f"[fail] {eid}: {len(failures)} downloads failed (first: {failures[0]}); re-run to resume")
        return {**info, "status": "download_failed", "n_failures": len(failures)}

    def frame_fn(i: int) -> np.ndarray:
        paths = {c: os.path.join(ev_dir, os.path.basename(scans[i]["channels"][c])) for c in channels}
        return read_fn(paths)

    H, W = (int(v) for v in crop_info["crop_shape"])
    meta = write_event_cache(
        cache_dir, eid, [s["datetime"] for s in scans], channels, crop_info, frame_fn,
        (len(channels), H, W), overwrite=overwrite,
        extra_meta={"split": spec["split"], "irregular_steps": irregular, "scan_modes": modes,
                    "stratum": spec.get("stratum"), "source": "noaa-goes19 S3"},
    )
    src_path = os.path.join(cache_dir, f"{eid}.sources.json")
    with open(src_path, "w") as f:
        json.dump({"event_id": eid, "files": [{"key": k, "bytes": sizes[k]} for k, _ in tasks]}, f, indent=1)
    if not keep_raw:
        shutil.rmtree(ev_dir, ignore_errors=True)
    idx = read_index(cache_dir)
    idx.setdefault("events", {})[eid] = {
        "split": spec["split"], "n_timestamps": len(scans), "shape": meta["shape"],
        "array_sha256": meta["array_sha256"], "start": meta["timestamps"][0], "end": meta["timestamps"][-1]}
    write_index(cache_dir, idx)
    log(f"[done] {eid}: {len(scans)} scans, irregular={irregular}, modes={modes}, sha {meta['array_sha256'][:12]}")
    return {**info, "status": "built"}


# ----------------------------------------------------------------------------- sync / reindex
def sync_event(cache_dir: str, sync_dir: str, event_id: str) -> None:
    os.makedirs(sync_dir, exist_ok=True)
    for src in (array_path(cache_dir, event_id), meta_path(cache_dir, event_id),
                os.path.join(cache_dir, f"{event_id}.sources.json")):
        if not os.path.exists(src):
            continue
        dst = os.path.join(sync_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        if os.path.getsize(dst) != os.path.getsize(src):
            raise IOError(f"sync size mismatch for {dst}")
    shutil.copy2(index_path(cache_dir), os.path.join(sync_dir, "cache_index.json"))


def reindex(cache_dir: str, events_json: Optional[str] = None) -> Dict[str, Any]:
    """Rebuild cache_index.json from *.meta.json files (use after merging teammates' folders)."""
    splits = {}
    if events_json:
        splits = {e["event_id"]: e["split"] for e in load_events(events_json)}
    idx = read_index(cache_dir)
    idx["events"] = {}
    for mp in sorted(glob.glob(os.path.join(cache_dir, "*.meta.json"))):
        with open(mp, "r") as f:
            m = json.load(f)
        eid = m["event_id"]
        idx["events"][eid] = {
            "split": splits.get(eid, (m.get("extra") or {}).get("split", "unknown")),
            "n_timestamps": m["shape"][0], "shape": m["shape"], "array_sha256": m["array_sha256"],
            "start": m["timestamps"][0], "end": m["timestamps"][-1]}
    write_index(cache_dir, idx)
    return idx


# ----------------------------------------------------------------------------- real reader
def make_real_read_fn(cropper, channels: Sequence[int]) -> Callable[[Dict[int, str]], np.ndarray]:
    from src.data.goes_reader import read_multichannel_timestamp

    def read_fn(paths: Dict[int, str]) -> np.ndarray:
        res = read_multichannel_timestamp({int(b): p for b, p in paths.items()}, channel_order=tuple(channels))
        data = cropper.crop_array(res["data_kelvin"]).astype(np.float32, copy=True)
        valid = cropper.crop_array(res["valid_mask"])
        data[:, ~valid] = np.nan     # pixel invalid if ANY channel invalid (same as v1 cache)
        return data
    return read_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--splits", default="train,val,cal", help="comma list; use 'test' alone for the test cache")
    ap.add_argument("--only_events", default="")
    ap.add_argument("--work_dir", default="/content/raw_goes19")
    ap.add_argument("--sync_dir", default="")
    ap.add_argument("--channels", default="9,13,14,15")
    ap.add_argument("--lat_min", type=float, default=25.0)
    ap.add_argument("--lat_max", type=float, default=35.0)
    ap.add_argument("--lon_min", type=float, default=-95.0)
    ap.add_argument("--lon_max", type=float, default=-80.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min_scans", type=int, default=MIN_SCANS_DEFAULT)
    ap.add_argument("--keep_raw", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--reindex", action="store_true", help="only rebuild cache_index.json and exit")
    args = ap.parse_args()

    if args.reindex:
        idx = reindex(args.cache_dir, args.events_json)
        print(f"Reindexed {len(idx['events'])} events in {args.cache_dir}")
        return

    splits = [s for s in args.splits.split(",") if s]
    only = [s for s in args.only_events.split(",") if s]
    events = load_events(args.events_json, splits, only)
    if not events:
        sys.exit("No events selected.")
    enforce_cache_role(args.cache_dir, events)

    from src.data.spatial_crop import GeographicCropper
    channels = tuple(int(c) for c in args.channels.split(","))
    cropper = GeographicCropper(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    crop_info = cropper.get_crop_info()
    print(f"Crop {crop_info['crop_shape']} px; {len(events)} events; role-checked cache: {args.cache_dir}")
    read_fn = make_real_read_fn(cropper, channels)

    results = []
    for spec in events:
        res = build_event(spec, args.cache_dir, args.work_dir, channels, crop_info, list_keys,
                          download_file, read_fn, workers=args.workers, keep_raw=args.keep_raw,
                          overwrite=args.overwrite, min_scans=args.min_scans, dry_run=args.dry_run)
        results.append(res)
        print(f"  {spec['event_id']:<18} {spec['split']:<5} -> {res['status']} "
              f"(scans={res.get('n_scans', '-')}, irregular={res.get('irregular_steps', '-')}, "
              f"modes={res.get('scan_modes', '-')})")
        if res["status"] == "built" and args.sync_dir:
            sync_event(args.cache_dir, args.sync_dir, spec["event_id"])
    bad = [r for r in results if r["status"] in ("insufficient", "download_failed")]
    print(f"\nSummary: {len(results)} events, {len(bad)} need attention: {[r['event_id'] for r in bad]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

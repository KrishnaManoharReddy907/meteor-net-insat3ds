"""
scripts/build_fast_cache.py
One-time conversion: GOES NetCDF files -> one compact uint16 array per event.

Examples
--------
# Rebuild the 4 existing events from the locked v1 manifest (test event is SKIPPED by default):
python scripts/build_fast_cache.py --data_dir data --cache_dir cache/goes19_v1 \
    --from_manifest outputs/dataset_splits/splits_manifest.json

# New events from a JSON spec (list of {"event_id","start","end","split"} with UTC times):
python scripts/build_fast_cache.py --data_dir data_new --cache_dir cache/goes19_v2 \
    --events_json configs/events_v2.json --lat_min 28 --lat_max 40 --lon_min -100 --lon_max -85

Safety: never modifies existing caches unless --overwrite; never touches raw data, the manifest,
or normalization stats. The held-out test event is only cached with --include_test.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.fast_cache import write_event_cache, array_path  # noqa: E402


def _parse_manifest_time(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s.replace(" UTC", "").strip(), "%Y-%m-%d %H:%M:%S")


def specs_from_manifest(path: str, include_test: bool):
    with open(path, "r") as f:
        m = json.load(f)
    spans = {}
    for split, windows in m["splits"].items():
        if split == "test" and not include_test:
            continue
        for w in windows:
            gid = w["group_id"]
            times = [_parse_manifest_time(x["datetime"]) for x in w["source_files"]["input"] + w["source_files"]["target"]]
            lo, hi = min(times), max(times)
            if gid in spans:
                lo, hi = min(lo, spans[gid]["start"]), max(hi, spans[gid]["end"])
            spans[gid] = {"event_id": gid, "split": split, "start": lo, "end": hi}
    return list(spans.values())


def specs_from_json(path: str):
    with open(path, "r") as f:
        raw = json.load(f)
    out = []
    for e in raw:
        out.append({
            "event_id": e["event_id"], "split": e.get("split", "unspecified"),
            "start": datetime.datetime.strptime(e["start"], "%Y-%m-%d %H:%M:%S"),
            "end": datetime.datetime.strptime(e["end"], "%Y-%m-%d %H:%M:%S"),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from_manifest")
    src.add_argument("--events_json")
    ap.add_argument("--include_test", action="store_true", help="Also cache events in the manifest 'test' split.")
    ap.add_argument("--channels", default="9,13,14,15")
    ap.add_argument("--lat_min", type=float, default=25.0)
    ap.add_argument("--lat_max", type=float, default=35.0)
    ap.add_argument("--lon_min", type=float, default=-95.0)
    ap.add_argument("--lon_max", type=float, default=-80.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Heavy imports only when actually building (keeps --help fast, h5py required here only).
    from src.data.goes_reader import build_timestamp_catalog, read_multichannel_timestamp
    from src.data.spatial_crop import GeographicCropper

    channels = tuple(int(c) for c in args.channels.split(","))
    specs = specs_from_manifest(args.from_manifest, args.include_test) if args.from_manifest \
        else specs_from_json(args.events_json)
    if not specs:
        sys.exit("No events to build.")

    print(f"Scanning {args.data_dir} ...")
    catalog, audit = build_timestamp_catalog(args.data_dir, channels=channels, on_conflict="exclude", return_audit=True)
    print(f"  synchronized timestamps: {len(catalog)} | exact dups: {audit['exact_duplicate_slots']} "
          f"| conflicts: {audit['conflicting_slots']}")

    cropper = GeographicCropper(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    crop = cropper.get_crop_info()
    H, W = crop["crop_shape"]
    print(f"Crop {crop['lat_bounds']} N, {crop['lon_bounds']} E -> {H}x{W} px "
          f"(~{H * W * len(channels) * 2 / 1e6:.1f} MB per timestamp)")

    os.makedirs(args.cache_dir, exist_ok=True)
    index_path = os.path.join(args.cache_dir, "cache_index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {"events": {}}

    for spec in specs:
        eid = spec["event_id"]
        if os.path.exists(array_path(args.cache_dir, eid)) and not args.overwrite:
            print(f"[skip] {eid}: cache exists (use --overwrite)")
            continue
        entries = [c for c in catalog if spec["start"] <= c["datetime"] <= spec["end"]]
        if len(entries) < 8:
            print(f"[WARN] {eid}: only {len(entries)} timestamps in window; skipping.")
            continue
        dts = [e["datetime"] for e in entries]
        deltas = np.diff([d.timestamp() for d in dts])
        off_grid = int((np.abs(deltas - 300.0) > 90).sum())
        print(f"[build] {eid} ({spec['split']}): {len(entries)} timestamps, "
              f"{off_grid} irregular steps (these windows are excluded automatically)")

        def frame_fn(i, entries=entries):
            res = read_multichannel_timestamp(entries[i]["channels"], channel_order=channels)
            data = cropper.crop_array(res["data_kelvin"]).astype(np.float32, copy=True)
            valid = cropper.crop_array(res["valid_mask"])
            data[:, ~valid] = np.nan  # a pixel is invalid if ANY channel is invalid
            return data

        meta = write_event_cache(
            args.cache_dir, eid, dts, channels, crop, frame_fn, (len(channels), H, W),
            overwrite=args.overwrite,
            extra_meta={"split": spec["split"], "irregular_steps": off_grid, "source_data_dir": os.path.abspath(args.data_dir)},
        )
        index["events"][eid] = {"split": spec["split"], "n_timestamps": len(dts), "shape": meta["shape"],
                                "array_sha256": meta["array_sha256"], "start": meta["timestamps"][0],
                                "end": meta["timestamps"][-1]}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"        done -> shape {meta['shape']}  sha256 {meta['array_sha256'][:12]}...")
    print("Finished. Index:", index_path)


if __name__ == "__main__":
    main()

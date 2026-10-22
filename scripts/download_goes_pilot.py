import os
import sys
import time
import hashlib
import json
import urllib.request
import h5py
import numpy as np
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.goes_reader import (
    build_timestamp_catalog,
    read_multichannel_timestamp,
    read_goes_single_file,
    compute_file_sha256
)
from src.data.sequence_generator import TemporalSequenceGenerator

BASE_URL = "https://noaa-goes19.s3.amazonaws.com/"

PILOT_KEYS = [
    # Input 1: 21:01 UTC (t-60)
    "ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C09_G19_s20261402101179_e20261402103558_c20261402104044.nc",
    "ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C13_G19_s20261402101179_e20261402103563_c20261402104047.nc",
    "ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C14_G19_s20261402101179_e20261402103552_c20261402104053.nc",
    "ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C15_G19_s20261402101179_e20261402103558_c20261402104036.nc",
    
    # Input 2: 21:31 UTC (t-30)
    "ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C09_G19_s20261402131179_e20261402133558_c20261402134034.nc",
    "ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C13_G19_s20261402131179_e20261402133563_c20261402134050.nc",
    "ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C14_G19_s20261402131179_e20261402133552_c20261402134026.nc",
    "ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C15_G19_s20261402131179_e20261402133558_c20261402134038.nc",

    # Input 3: 22:01 UTC (t, Anchor)
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C09_G19_s20261402201179_e20261402203557_c20261402204034.nc",
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C13_G19_s20261402201179_e20261402203564_c20261402204027.nc",
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C14_G19_s20261402201179_e20261402203552_c20261402204008.nc",
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C15_G19_s20261402201179_e20261402203558_c20261402204046.nc",

    # Target: 22:16 UTC (t+15)
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C09_G19_s20261402216179_e20261402218558_c20261402219049.nc",
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C13_G19_s20261402216179_e20261402218563_c20261402219046.nc",
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C14_G19_s20261402216179_e20261402218552_c20261402219038.nc",
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C15_G19_s20261402216179_e20261402218557_c20261402219031.nc",

    # Input 4: 22:31 UTC (t+30)
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C09_G19_s20261402231179_e20261402233558_c20261402234024.nc",
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C13_G19_s20261402231179_e20261402233563_c20261402234042.nc",
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C14_G19_s20261402231179_e20261402233552_c20261402234028.nc",
    "ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C15_G19_s20261402231179_e20261402233557_c20261402234048.nc",
]

def verify_netcdf_readable(filepath: str) -> bool:
    try:
        with h5py.File(filepath, "r") as h5:
            if "CMI" not in h5:
                return False
            _ = h5["CMI"].shape
        return True
    except Exception:
        return False

def download_file_atomic(url: str, dest_path: str, chunk_size: int = 1024 * 256) -> dict:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Check if already exists and valid
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 0 and verify_netcdf_readable(dest_path):
            hash_val = compute_file_sha256(dest_path)
            return {
                "status": "reused",
                "path": dest_path,
                "size_bytes": size,
                "sha256": hash_val,
                "duration_sec": 0.0
            }
        else:
            print(f"Warning: Corrupt or zero-size file found at {dest_path}, re-downloading...")
            os.remove(dest_path)

    part_path = dest_path + ".part"
    if os.path.exists(part_path):
        os.remove(part_path)

    start_t = time.time()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "METEOR-Net-Pilot/1.0 (NOAA Public Open Data)"}
    )
    
    with urllib.request.urlopen(req, timeout=30) as response:
        total_len = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(part_path, "wb") as f_out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)

    if total_len > 0 and downloaded != total_len:
        if os.path.exists(part_path):
            os.remove(part_path)
        raise IOError(f"Incomplete download for {url}: expected {total_len} bytes, got {downloaded}")

    # Atomic rename
    os.replace(part_path, dest_path)
    dur = time.time() - start_t
    size = os.path.getsize(dest_path)

    # Verify readable
    if not verify_netcdf_readable(dest_path):
        os.remove(dest_path)
        raise IOError(f"Downloaded file {dest_path} failed NetCDF/HDF5 readability check.")

    hash_val = compute_file_sha256(dest_path)
    return {
        "status": "downloaded",
        "path": dest_path,
        "size_bytes": size,
        "sha256": hash_val,
        "duration_sec": dur
    }

def run_pilot():
    print("=" * 80)
    print("PHASE 5D: GOES-19 20-FILE PILOT ACQUISITION AND VALIDATION")
    print("=" * 80)
    
    dest_base = os.path.join(PROJECT_ROOT, "data", "goes19", "2026", "140")
    print(f"Destination directory: {dest_base}")
    print(f"Total files in pilot inventory: {len(PILOT_KEYS)}")
    
    download_records = []
    total_bytes = 0
    start_total_t = time.time()

    for idx, key in enumerate(PILOT_KEYS, 1):
        rel_path = "/".join(key.split("/")[3:]) # e.g. 21/OR_...nc
        local_path = os.path.join(dest_base, rel_path.replace("/", os.sep))
        url = BASE_URL + key
        
        print(f"[{idx:02d}/20] Ingesting: {key.split('/')[-1]}...")
        rec = download_file_atomic(url, local_path)
        rec["object_key"] = key
        rec["url"] = url
        download_records.append(rec)
        total_bytes += rec["size_bytes"]
        print(f"       -> Status: {rec['status']}, Size: {rec['size_bytes'] / (1024*1024):.2f} MB, Time: {rec['duration_sec']:.2f}s")

    elapsed_total = time.time() - start_total_t
    print("\n" + "-" * 80)
    print(f"Download complete: 20 files, {total_bytes / (1024*1024):.2f} MB in {elapsed_total:.2f} seconds.")
    print("-" * 80)

    # Step 2: Validate catalog and timestamp parsing
    print("\n[Step 2] Auditing timestamp catalog and duplicate detection...")
    catalog, audit_summary = build_timestamp_catalog(
        data_dir=dest_base,
        channels=(9, 13, 14, 15),
        on_conflict="raise",
        return_audit=True
    )
    print(f"Catalog scan result: {len(catalog)} complete timestamps discovered.")
    for entry in catalog:
        dt_str = entry["datetime"].strftime("%Y-%m-%d %H:%M:%S UTC")
        chans = list(entry["channels"].keys())
        print(f"  - Timestamp: {dt_str} | Channels: {chans}")

    if len(catalog) != 5:
        raise ValueError(f"Expected exactly 5 synchronized timestamps, found {len(catalog)}!")

    # Step 3: Calibrate temperatures, check quality flags, and compute distributions
    print("\n[Step 3] Calibrating physical temperatures and analyzing convective distributions...")
    temp_analysis = {}
    for entry in catalog:
        dt = entry["datetime"]
        dt_key = dt.strftime("%Y-%m-%d_%H%M%S")
        mc_data = read_multichannel_timestamp(entry["channels"], channel_order=(9, 13, 14, 15))
        
        data_k = mc_data["data_kelvin"] # (4, H, W)
        mask = mc_data["valid_mask"]     # (H, W)
        
        # Band 13 is index 1
        b13 = data_k[1]
        b13_valid = b13[mask]
        
        min_k = float(np.nanmin(b13_valid))
        max_k = float(np.nanmax(b13_valid))
        mean_k = float(np.nanmean(b13_valid))
        p01_k = float(np.percentile(b13_valid, 1))
        p05_k = float(np.percentile(b13_valid, 5))
        p50_k = float(np.percentile(b13_valid, 50))
        
        frac_below_241 = float(np.mean(b13_valid < 241.0))
        frac_below_221 = float(np.mean(b13_valid < 221.0))
        count_below_221 = int(np.sum(b13_valid < 221.0))
        
        temp_analysis[dt_key] = {
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "valid_pixel_count": int(np.sum(mask)),
            "valid_pixel_fraction": float(np.mean(mask)),
            "b13_min_k": min_k,
            "b13_max_k": max_k,
            "b13_mean_k": mean_k,
            "b13_p01_k": p01_k,
            "b13_p05_k": p05_k,
            "b13_p50_k": p50_k,
            "frac_below_241k": frac_below_241,
            "frac_below_221k": frac_below_221,
            "count_below_221k": count_below_221
        }
        print(f"  [{dt_key}] Band 13 Kelvin: min={min_k:.2f}K, mean={mean_k:.2f}K, max={max_k:.2f}K | <241K: {frac_below_241*100:.2f}%, <221K: {count_below_221} px ({frac_below_221*100:.3f}%)")

    # Step 4: Construct pilot temporal sequence
    print("\n[Step 4] Constructing pilot sequence with TemporalSequenceGenerator...")
    seq_gen = TemporalSequenceGenerator(
        catalog=catalog,
        input_offsets_min=(-60, -30, 0, 30),
        target_offsets_min=(15,),
        channel_order=(9, 13, 14, 15)
    )
    seq_windows = seq_gen.valid_windows
    print(f"Discovered {len(seq_windows)} valid sequence windows.")
    
    if len(seq_windows) != 1:
        raise ValueError(f"Expected exactly 1 sequence window, found {len(seq_windows)}!")

    win = seq_windows[0]
    print(f"Sequence window details:")
    print(f"  Anchor (t):      {win['anchor_datetime']}")
    print(f"  Input frames:    {[f['datetime'].strftime('%H:%M:%S UTC') for f in win['input_frames']]}")
    print(f"  Target frames:   {[f['datetime'].strftime('%H:%M:%S UTC') for f in win['target_frames']]}")

    # Load window tensors and verify shapes and validity masks
    print("\n  Loading window tensors (full cropped CONUS)...")
    win_tensors = seq_gen.load_window_tensors(0)
    in_tensors = win_tensors["input_tensors"]       # (4, 4, H, W)
    tgt_tensors = win_tensors["target_tensors"]     # (1, 4, H, W)
    comb_mask = win_tensors["combined_valid_mask"]  # (H, W)
    
    print(f"  Input tensors shape:   {in_tensors.shape} (dtype={in_tensors.dtype})")
    print(f"  Target tensors shape:  {tgt_tensors.shape} (dtype={tgt_tensors.dtype})")
    print(f"  Combined valid mask:   {comb_mask.shape} (valid px: {np.sum(comb_mask)} / {comb_mask.size}, {np.mean(comb_mask)*100:.2f}%)")
    
    # Extract patches and assess convective patches
    patches = seq_gen.extract_sequence_patches(0, patch_size=128, stride=64, require_full_valid=True)
    print(f"  Extracted {len(patches)} fully-valid 128x128 patches across sequence.")
    
    convective_patches = []
    for idx, p in enumerate(patches):
        # Band 13 target patch is p['target_patches'][0, 1]
        tgt_b13 = p["target_patches"][0, 1]
        p_min = float(np.min(tgt_b13))
        p_frac241 = float(np.mean(tgt_b13 < 241.0))
        p_count221 = int(np.sum(tgt_b13 < 221.0))
        if p_min < 241.0 and (p_frac241 >= 0.02 or p_count221 >= 5):
            convective_patches.append({
                "patch_index": idx,
                "row_slice": p["row_slice"],
                "col_slice": p["col_slice"],
                "b13_min_k": p_min,
                "frac_below_241k": p_frac241,
                "count_below_221k": p_count221
            })
    print(f"  Convective patches meeting Phase 5C criteria: {len(convective_patches)} / {len(patches)}")
    if convective_patches:
        top_c = min(convective_patches, key=lambda x: x["b13_min_k"])
        print(f"  Most intense convective patch: index {top_c['patch_index']} with min Band 13 T = {top_c['b13_min_k']:.2f} K, {top_c['count_below_221k']} px < 221K")

    # Step 5: Save manifest report
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "pilot_acquisition")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "pilot_manifest.json")
    
    manifest_data = {
        "event_id": "Event 2 (Day 140 / May 20, 2026)",
        "download_summary": {
            "total_files": len(download_records),
            "total_bytes": total_bytes,
            "total_megabytes": round(total_bytes / (1024 * 1024), 2),
            "elapsed_seconds": round(elapsed_total, 2),
            "reused_count": sum(1 for r in download_records if r["status"] == "reused"),
            "downloaded_count": sum(1 for r in download_records if r["status"] == "downloaded"),
            "base_url": BASE_URL,
            "destination_dir": dest_base
        },
        "file_inventory": download_records,
        "timestamp_catalog": [
            {
                "datetime": e["datetime"].isoformat(),
                "channels": {str(c): os.path.basename(p) for c, p in e["channels"].items()}
            } for e in catalog
        ],
        "temperature_distributions": temp_analysis,
        "sequence_window": {
            "anchor_datetime": win["anchor_datetime"].isoformat(),
            "input_datetimes": [f["datetime"].isoformat() for f in win["input_frames"]],
            "target_datetimes": [f["datetime"].isoformat() for f in win["target_frames"]],
            "input_tensors_shape": list(in_tensors.shape),
            "target_tensors_shape": list(tgt_tensors.shape),
            "combined_valid_fraction": float(np.mean(comb_mask)),
            "total_128x128_patches": len(patches),
            "convective_128x128_patches": len(convective_patches),
            "top_convective_patch": top_c if convective_patches else None
        }
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\n[Saved] Pilot manifest saved to {manifest_path}")
    print("=" * 80)
    print("PHASE 5D PILOT VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_pilot()


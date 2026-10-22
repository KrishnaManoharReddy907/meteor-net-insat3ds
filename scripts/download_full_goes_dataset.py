import os
import sys
import time
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.goes_reader import (
    build_timestamp_catalog,
    read_multichannel_timestamp,
    compute_file_sha256,
)
from src.data.sequence_generator import TemporalSequenceGenerator

BASE_URL = "https://noaa-goes19.s3.amazonaws.com/"

EVENTS_CONFIG = [
    {
        "event_id": "event_2026_01_01",
        "description": "Event 1: Winter Pacific Trough / AR (January 1, 2026, 02:00-08:00 UTC)",
        "year": 2026,
        "day_hours": [(1, [2, 3, 4, 5, 6, 7])],
        "split_target": "train"
    },
    {
        "event_id": "event_2026_05_20",
        "description": "Event 2: Spring Severe Convective Outbreak (May 20, 2026, 18:00-24:00 UTC)",
        "year": 2026,
        "day_hours": [(140, [18, 19, 20, 21, 22, 23])],
        "split_target": "train"
    },
    {
        "event_id": "event_2026_07_19",
        "description": "Event 3: North American Monsoon Diurnal Clusters (July 19 20:00 - July 20 02:00 UTC)",
        "year": 2026,
        "day_hours": [(200, [20, 21, 22, 23]), (201, [0, 1])],
        "split_target": "val"
    },
    {
        "event_id": "event_2026_09_07",
        "description": "Event 4: Gulf Coastal / Maritime Convective Bands (September 7, 2026, 16:00-22:00 UTC)",
        "year": 2026,
        "day_hours": [(250, [16, 17, 18, 19, 20, 21])],
        "split_target": "test"
    }
]

TARGET_CHANNELS = {"09", "13", "14", "15"}

def verify_netcdf_readable(filepath: str) -> bool:
    try:
        with h5py.File(filepath, "r") as h5:
            if "CMI" not in h5:
                return False
            _ = h5["CMI"].shape
        return True
    except Exception:
        return False

def enumerate_event_keys(ev_cfg: dict) -> list:
    keys = []
    year = ev_cfg["year"]
    for day, hours in ev_cfg["day_hours"]:
        for h in hours:
            prefix = f"ABI-L2-CMIPC/{year}/{day:03d}/{h:02d}/"
            url = f"{BASE_URL}?prefix={prefix}"
            req = urllib.request.Request(url, headers={"User-Agent": "FullAcquisition/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                tree = ET.fromstring(resp.read())
                ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
                for item in tree.findall("s3:Contents", ns):
                    key = item.find("s3:Key", ns).text
                    for c in TARGET_CHANNELS:
                        if f"-M6C{c}_G19_" in key:
                            keys.append(key)
    return keys

def process_file_task(key: str, dest_base: str) -> dict:
    # Key format: ABI-L2-CMIPC/<YYYY>/<JJJ>/<HH>/<filename>.nc
    parts = key.split("/")
    year = parts[1]
    julian_day = parts[2]
    hour = parts[3]
    filename = parts[4]
    
    local_dir = os.path.join(dest_base, year, julian_day, hour)
    local_path = os.path.join(local_dir, filename)
    url = BASE_URL + key
    
    # 1. Check if already exists and readable
    if os.path.exists(local_path):
        size = os.path.getsize(local_path)
        if size > 0 and verify_netcdf_readable(local_path):
            sha = compute_file_sha256(local_path)
            return {
                "status": "reused",
                "object_key": key,
                "local_path": local_path,
                "size_bytes": size,
                "sha256": sha,
                "duration_sec": 0.0
            }
        else:
            try:
                os.remove(local_path)
            except Exception:
                pass

    # 2. Download to .part
    os.makedirs(local_dir, exist_ok=True)
    part_path = local_path + ".part"
    if os.path.exists(part_path):
        try:
            os.remove(part_path)
        except Exception:
            pass

    start_t = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "FullAcquisition/1.0"})
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                total_len = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with open(part_path, "wb") as f_out:
                    while True:
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        downloaded += len(chunk)
            
            if total_len > 0 and downloaded != total_len:
                raise IOError(f"Byte count mismatch: expected {total_len}, got {downloaded}")
            
            # Atomic rename
            os.replace(part_path, local_path)
            dur = time.time() - start_t
            
            # Readability check
            if not verify_netcdf_readable(local_path):
                raise IOError("NetCDF readability check failed after download.")
                
            sha = compute_file_sha256(local_path)
            size = os.path.getsize(local_path)
            return {
                "status": "downloaded",
                "object_key": key,
                "local_path": local_path,
                "size_bytes": size,
                "sha256": sha,
                "duration_sec": dur
            }
        except Exception as err:
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except Exception:
                    pass
            if attempt == max_retries - 1:
                return {
                    "status": "failed",
                    "object_key": key,
                    "local_path": local_path,
                    "error": str(err),
                    "duration_sec": time.time() - start_t
                }
            time.sleep(1.0)

def main():
    print("=" * 80)
    print("PHASE 6A: FULL FOUR-EVENT GOES-19 ACQUISITION AND DATASET PREPARATION")
    print("=" * 80)
    
    dest_base = os.path.join(PROJECT_ROOT, "data", "goes19")
    print(f"Base data directory: {dest_base}")
    
    # 1. Enumerate all keys
    all_tasks = []
    print("\n[Step 1] Enumerating remote object keys across all 4 events...")
    for ev in EVENTS_CONFIG:
        t0 = time.time()
        keys = enumerate_event_keys(ev)
        print(f"  {ev['description']}: {len(keys)} keys discovered ({time.time()-t0:.2f}s)")
        for k in keys:
            all_tasks.append((k, ev["event_id"]))
            
    total_files = len(all_tasks)
    print(f"\nTotal remote files to ingest / verify: {total_files}")
    if total_files != 1152:
        raise ValueError(f"Expected exactly 1152 files across 4 events, got {total_files}!")

    # 2. Concurrent ingestion with thread pool
    print("\n[Step 2] Executing atomic ingestion and verification with 12 parallel workers...")
    start_all = time.time()
    results = []
    reused_count = 0
    downloaded_count = 0
    failed_count = 0
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_task = {
            executor.submit(process_file_task, key, dest_base): (key, ev_id)
            for key, ev_id in all_tasks
        }
        
        completed = 0
        last_log = time.time()
        for fut in as_completed(future_to_task):
            key, ev_id = future_to_task[fut]
            rec = fut.result()
            rec["event_id"] = ev_id
            results.append(rec)
            completed += 1
            
            if rec["status"] == "reused":
                reused_count += 1
                total_bytes += rec.get("size_bytes", 0)
            elif rec["status"] == "downloaded":
                downloaded_count += 1
                total_bytes += rec.get("size_bytes", 0)
            else:
                failed_count += 1
                print(f"  [ERROR] Failed to ingest {key}: {rec.get('error')}")

            # Print progress every 5 seconds or on milestone
            if time.time() - last_log > 5.0 or completed == total_files:
                elapsed = time.time() - start_all
                rate_mb = (total_bytes / (1024 * 1024)) / max(1.0, elapsed)
                pct = (completed / total_files) * 100.0
                print(f"  Progress: {completed}/{total_files} ({pct:.1f}%) | "
                      f"Reused: {reused_count}, Downloaded: {downloaded_count}, Failed: {failed_count} | "
                      f"Elapsed: {elapsed:.1f}s")
                last_log = time.time()

    elapsed_total = time.time() - start_all
    print("\n" + "-" * 80)
    print(f"Ingestion completed in {elapsed_total:.2f} seconds.")
    print(f"Total: {total_files} | Reused: {reused_count} | Downloaded: {downloaded_count} | Failed: {failed_count}")
    print(f"Total Data Volume: {total_bytes / (1024*1024):.2f} MB ({total_bytes / (1024*1024*1024):.3f} GB)")
    print("-" * 80)

    if failed_count > 0:
        raise RuntimeError(f"Acquisition aborted with {failed_count} failed downloads!")

    # 3. Save ingestion manifest
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "full_acquisition")
    os.makedirs(out_dir, exist_ok=True)
    inv_path = os.path.join(out_dir, "acquisition_inventory.json")
    with open(inv_path, "w") as f:
        json.dump({
            "total_files": total_files,
            "reused_count": reused_count,
            "downloaded_count": downloaded_count,
            "failed_count": failed_count,
            "total_bytes": total_bytes,
            "elapsed_seconds": round(elapsed_total, 2),
            "files": results
        }, f, indent=2)
    print(f"\n[Saved] Acquisition inventory saved to {inv_path}")

if __name__ == "__main__":
    main()

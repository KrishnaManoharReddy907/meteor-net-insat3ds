"""
scripts/archive_dataset.py
Reproducible Dataset Archiving Utility for Google Drive Transfer.

Packages the 4-event GOES-19 dataset (1,152 NetCDF files) into a compressed
tar.gz archive and produces a cryptographic SHA-256 verification manifest.
"""

import os
import sys
import tarfile
import hashlib
import json
import time
import argparse
from typing import Dict, Any, List


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash in 4 MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def create_dataset_archive(
    data_dir: str = "data",
    archive_out: str = "data_goes19_4events.tar.gz",
    manifest_out: str = "outputs/full_acquisition/data_archive_manifest.json"
) -> Dict[str, Any]:
    print("=" * 80)
    print("METEOR-NET: DATASET ARCHIVE GENERATOR FOR GOOGLE DRIVE TRANSFER")
    print("=" * 80)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Source data directory not found: {data_dir}")

    # Discover all files under data_dir
    files_to_pack: List[str] = []
    for root, _, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith(".nc"):
                files_to_pack.append(os.path.join(root, f))

    files_to_pack.sort()
    print(f"Discovered {len(files_to_pack)} NetCDF files under '{data_dir}'.")
    assert len(files_to_pack) == 1152, f"Expected 1,152 files, found {len(files_to_pack)}"

    total_uncompressed_bytes = sum(os.path.getsize(f) for f in files_to_pack)
    print(f"Total uncompressed size: {total_uncompressed_bytes / (1024**3):.3f} GB ({total_uncompressed_bytes:,} bytes)")

    print(f"\nCreating compressed archive: '{archive_out}'...")
    t0 = time.perf_counter()
    with tarfile.open(archive_out, "w:gz") as tar:
        for idx, fpath in enumerate(files_to_pack):
            rel_name = os.path.relpath(fpath, ".")
            tar.add(fpath, arcname=rel_name.replace("\\", "/"))
            if (idx + 1) % 200 == 0 or (idx + 1) == len(files_to_pack):
                print(f"  Archived {idx + 1:04d}/{len(files_to_pack)} files...")

    t_compress = time.perf_counter() - t0
    archive_size_bytes = os.path.getsize(archive_out)
    print(f"Compression completed in {t_compress:.1f} s.")
    print(f"Compressed archive size: {archive_size_bytes / (1024**3):.3f} GB ({archive_size_bytes:,} bytes)")

    print("\nComputing archive SHA-256 hash (this guarantees tamper-free Colab extraction)...")
    archive_sha256 = compute_sha256(archive_out)
    print(f"  SHA-256: {archive_sha256}")

    # Load verified per-file hashes from acquisition inventory
    inventory_path = "outputs/full_acquisition/acquisition_inventory.json"
    files_detail = {}
    if os.path.exists(inventory_path):
        with open(inventory_path, "r") as f:
            inv = json.load(f)
        for item in inv.get("files", []):
            rel_p = os.path.relpath(item["local_path"], ".").replace("\\", "/")
            files_detail[rel_p] = {
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"]
            }

    manifest_data = {
        "manifest_version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive_file": os.path.basename(archive_out),
        "archive_size_bytes": archive_size_bytes,
        "archive_size_gb": round(archive_size_bytes / (1024**3), 3),
        "archive_sha256": archive_sha256,
        "uncompressed_bytes": total_uncompressed_bytes,
        "uncompressed_gb": round(total_uncompressed_bytes / (1024**3), 3),
        "total_files": len(files_to_pack),
        "events": {
            "event_2026-01-01": 288,
            "event_2026-05-20": 288,
            "event_2026-07-19": 288,
            "event_2026-09-07": 288
        },
        "files": files_detail
    }

    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Archive manifest written to: '{manifest_out}'")
    print("=" * 80)
    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="METEOR-Net Dataset Archiver")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--archive_out", type=str, default="data_goes19_4events.tar.gz")
    parser.add_argument("--manifest_out", type=str, default="outputs/full_acquisition/data_archive_manifest.json")
    args = parser.parse_args()

    create_dataset_archive(
        data_dir=args.data_dir,
        archive_out=args.archive_out,
        manifest_out=args.manifest_out
    )

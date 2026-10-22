"""
scripts/package_source_bundle.py
Reproducible Source Bundle Packaging Utility for Google Colab Migration.

Packages the verified METEOR-Net codebase, documentation, tests, and metadata
manifests into an offline zip bundle ('meteor_net_source_bundle.zip') for transfer
to Google Drive. Excludes raw NetCDF datasets, large model checkpoints (*.pt),
caches, and temporary files.
"""

import os
import sys
import zipfile
import hashlib
import time
from typing import List, Tuple

EXCLUDE_ROOT_DIRS = {
    ".git",
    "data",
    "notebooks",
    "docs",
    "reports",
    ".pytest_cache",
    ".system_generated",
    ".tempmediaStorage",
    "scratch",
    "colab_t4_smoke_test",
    "colab_t4_phase8"
}

EXCLUDE_EXTENSIONS = (
    ".pyc",
    ".pt",
    ".tar.gz",
    ".gz",
    ".zip",
    ".log",
    ".tmp"
)

# Explicitly allowed non-source metadata artifacts under outputs/
ALLOWED_OUTPUT_FILES = {
    "outputs/dataset_splits/splits_manifest.json",
    "outputs/normalization/frozen_global_train_stats.json",
    "outputs/full_acquisition/acquisition_inventory.json",
    "outputs/full_acquisition/data_archive_manifest.json",
    "outputs/phase7b/patch_classification_audit.json",
    "outputs/phase7b/baseline_metrics.json",
    "outputs/phase7b/training_history.json"
}


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def collect_files(repo_root: str = ".") -> List[str]:
    included_files = []
    for root, dirs, files in os.walk(repo_root):
        rel_root = os.path.relpath(root, repo_root).replace("\\", "/")
        # Prune excluded directories
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d not in EXCLUDE_ROOT_DIRS]
        else:
            dirs[:] = [d for d in dirs if d != "__pycache__"]

        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), repo_root).replace("\\", "/")
            ext = os.path.splitext(f)[1].lower()

            if f.endswith(EXCLUDE_EXTENSIONS) or ext in EXCLUDE_EXTENSIONS:
                continue

            # Special filtering for outputs/
            if rel_path.startswith("outputs/"):
                if rel_path not in ALLOWED_OUTPUT_FILES:
                    continue

            included_files.append(rel_path)

    included_files.sort()
    return included_files


def create_source_bundle(
    output_zip: str = "meteor_net_source_bundle.zip",
    repo_root: str = "."
) -> Tuple[int, int, str]:
    print("=" * 80)
    print("METEOR-NET: REPRODUCIBLE SOURCE BUNDLE PACKAGER")
    print("=" * 80)

    files = collect_files(repo_root)
    print(f"Collected {len(files)} files for source bundle.")

    # Sanity checks on critical files
    required_files = [
        "src/models/meteor_net.py",
        "src/models/losses.py",
        "src/models/normalization.py",
        "src/data/goes_dataset.py",
        "src/data/goes_reader.py",
        "scripts/train_meteor_net.py",
        "scripts/run_gpu_smoke_test.py",
        "requirements.txt",
        "outputs/dataset_splits/splits_manifest.json",
        "outputs/normalization/frozen_global_train_stats.json",
        "outputs/full_acquisition/data_archive_manifest.json"
    ]
    for rf in required_files:
        assert rf in files, f"CRITICAL: Required file {rf} is missing from bundle!"

    # Ensure no prohibited items
    for f in files:
        assert not f.startswith("data/"), f"Prohibited raw data file included: {f}"
        assert not f.endswith(".pt"), f"Prohibited model checkpoint included: {f}"

    total_uncompressed = 0
    t0 = time.perf_counter()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files:
            abs_path = os.path.normpath(os.path.join(repo_root, f))
            total_uncompressed += os.path.getsize(abs_path)
            # Normalize archive entry name with forward slashes
            zf.write(abs_path, arcname=f)

    duration = time.perf_counter() - t0
    bundle_size = os.path.getsize(output_zip)
    bundle_sha256 = compute_file_sha256(output_zip)

    print(f"Packaging complete in {duration:.2f} s.")
    print(f"Files archived:     {len(files)}")
    print(f"Uncompressed size:  {total_uncompressed / (1024*1024):.2f} MB ({total_uncompressed:,} bytes)")
    print(f"Compressed size:    {bundle_size / (1024*1024):.2f} MB ({bundle_size:,} bytes)")
    print(f"Bundle SHA-256:     {bundle_sha256}")
    print("=" * 80)
    return len(files), bundle_size, bundle_sha256


if __name__ == "__main__":
    create_source_bundle()

"""
scripts/audit_goes_sample_readiness.py
Comprehensive empirical audit of the GOES-19 sample for Phase 3 readiness assessment.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import glob
import datetime
import numpy as np
from src.data.goes_reader import (
    build_timestamp_catalog,
    read_multichannel_timestamp,
    read_goes_single_file,
    GOESProjection
)
from src.data.spatial_crop import GeographicCropper, extract_patches
from src.data.sequence_generator import TemporalSequenceGenerator

def run_audit():
    print("=" * 80)
    print("PHASE 3: GOES-19 SAMPLE AND SEQUENCE READINESS AUDIT")
    print("=" * 80)

    # 1. Inspect raw files inventory
    base_dir = os.path.join("data", "goes19", "2026", "001")
    pattern = os.path.join(base_dir, "**", "*.nc")
    all_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Total NetCDF files found: {len(all_files)}")

    channels_found = {9: 0, 13: 0, 14: 0, 15: 0}
    hours_found = {}
    for f in all_files:
        bn = os.path.basename(f)
        for ch in [9, 13, 14, 15]:
            if f"M6C{ch:02d}" in bn:
                channels_found[ch] += 1
        hour = os.path.basename(os.path.dirname(f))
        hours_found[hour] = hours_found.get(hour, 0) + 1

    print(f"Channels distribution: {channels_found}")
    print(f"Hours distribution: {hours_found}")

    # Build catalog
    catalog = build_timestamp_catalog(base_dir, channels=(9, 13, 14, 15))
    print(f"Distinct synchronized timestamps in catalog: {len(catalog)}")
    if catalog:
        print(f"  First scan: {catalog[0]['datetime'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Last scan:  {catalog[-1]['datetime'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        span_min = (catalog[-1]['datetime'] - catalog[0]['datetime']).total_seconds() / 60.0
        print(f"  Total elapsed time: {span_min:.1f} minutes")

    # Time cadence check
    deltas = []
    for i in range(len(catalog) - 1):
        dt = (catalog[i+1]['datetime'] - catalog[i]['datetime']).total_seconds() / 60.0
        deltas.append(dt)
    print(f"Inter-scan interval deltas (min): mean={np.mean(deltas):.2f}, min={np.min(deltas):.2f}, max={np.max(deltas):.2f}, all_5min={all(abs(d-5.0)<0.05 for d in deltas)}")

    # 2. Inspect Sequence Windows
    cropper = GeographicCropper(lat_min=25.0, lat_max=35.0, lon_min=-95.0, lon_max=-80.0)
    seq_gen = TemporalSequenceGenerator(
        catalog=catalog,
        input_offsets_min=(-60, -30, 0, 30),
        target_offsets_min=(15,),
        cropper=cropper
    )
    print(f"\nDiscovered valid sequence windows: {len(seq_gen.valid_windows)}")
    for idx, win in enumerate(seq_gen.valid_windows):
        in_times = [f['datetime'].strftime('%H:%M') for f in win['input_frames']]
        tgt_time = win['target_frames'][0]['datetime'].strftime('%H:%M')
        print(f"  Window {idx:02d}: inputs=[{', '.join(in_times)}] -> target={tgt_time} (Anchor {win['anchor_datetime'].strftime('%H:%M')})")

    # 3. Patch Extraction Audit across ALL windows
    print("\n" + "=" * 80)
    print("PATCH AUDIT ON SOUTHEAST US / GULF CROPPED REGION (453 x 740)")
    print("=" * 80)

    for stride in [64, 128]:
        all_patches_stride = []
        for w_idx in range(len(seq_gen.valid_windows)):
            patches = seq_gen.extract_sequence_patches(w_idx, patch_size=128, stride=stride, require_full_valid=True)
            all_patches_stride.extend(patches)
        
        # Calculate statistics
        t13_mins = [p['target_min_temp_k'] for p in all_patches_stride]
        t13_means = [p['target_mean_temp_k'] for p in all_patches_stride]
        conv_ratios = [p['convective_ratio_241'] for p in all_patches_stride]
        deep_counts = [p['deep_convective_count_221'] for p in all_patches_stride]
        is_conv_list = [p['is_convective'] for p in all_patches_stride]

        print(f"\nPatch extraction with stride={stride} (patch_size=128):")
        print(f"  Total accepted patches across all 6 windows: {len(all_patches_stride)}")
        print(f"  Patches per window: {len(all_patches_stride) // len(seq_gen.valid_windows)}")
        print(f"  Target Band 13 Min Temp: min={np.min(t13_mins):.2f} K, median={np.median(t13_mins):.2f} K, max={np.max(t13_mins):.2f} K")
        print(f"  Target Band 13 Mean Temp: min={np.min(t13_means):.2f} K, median={np.median(t13_means):.2f} K, max={np.max(t13_means):.2f} K")
        print(f"  Fraction of patches with ANY pixel < 273 K: {np.mean([t < 273.0 for t in t13_mins]):.4f}")
        print(f"  Fraction of patches with ANY pixel < 241 K: {np.mean([t < 241.0 for t in t13_mins]):.4f}")
        print(f"  Fraction of patches with ANY pixel < 221 K: {np.mean([t < 221.0 for t in t13_mins]):.4f}")
        print(f"  Patches marked 'is_convective' (ratio_241 >= 0.02 or deep >= 5): {sum(is_conv_list)} / {len(is_conv_list)}")

    # 4. Check FULL CONUS Grid Temperature Distribution for Band 13
    print("\n" + "=" * 80)
    print("FULL CONUS DISK CONVECTIVE AUDIT (ALL 1500 x 2500 PIXELS)")
    print("=" * 80)
    # Check target of Window 00: 05:16:19 UTC
    target_dt = seq_gen.valid_windows[0]['target_frames'][0]['datetime']
    target_channels = seq_gen.dt_map[target_dt]['channels']
    ch13_info = read_goes_single_file(target_channels[13])
    t13_full = ch13_info['data_kelvin']
    mask_full = ch13_info['valid_mask']
    
    valid_t13 = t13_full[mask_full]
    total_valid = len(valid_t13)
    c_273 = np.count_nonzero(valid_t13 < 273.0)
    c_241 = np.count_nonzero(valid_t13 < 241.0)
    c_221 = np.count_nonzero(valid_t13 < 221.0)
    
    print(f"CONUS Scan at {target_dt.strftime('%H:%M:%S UTC')}:")
    print(f"  Total Valid Earth Pixels: {total_valid:,} ({total_valid / t13_full.size * 100:.2f}%)")
    print(f"  Band 13 Stats: min={np.min(valid_t13):.2f} K, p5={np.percentile(valid_t13, 5):.2f} K, median={np.median(valid_t13):.2f} K, p95={np.percentile(valid_t13, 95):.2f} K, max={np.max(valid_t13):.2f} K")
    print(f"  Cold Cloud Pixels (T < 273 K): {c_273:,} ({c_273 / total_valid * 100:.2f}%)")
    print(f"  Convective Pixels (T < 241 K):  {c_241:,} ({c_241 / total_valid * 100:.2f}%)")
    print(f"  Deep Convective Pixels (T < 221 K): {c_221:,} ({c_221 / total_valid * 100:.2f}%)")

    # Find where the deep convective pixels are located
    deep_rows, deep_cols = np.where((t13_full < 221.0) & mask_full)
    proj = GOESProjection()
    lats, lons = proj.rowcol_to_latlon(deep_rows, deep_cols)
    print(f"  Deep Convective Core Location (T < 221 K):")
    print(f"    Latitude range:  [{np.min(lats):.2f} N, {np.max(lats):.2f} N] (mean={np.mean(lats):.2f} N)")
    print(f"    Longitude range: [{np.min(lons):.2f} E, {np.max(lons):.2f} E] (mean={np.mean(lons):.2f} E)")
    print(f"    Row range:       [{np.min(deep_rows)}, {np.max(deep_rows)}]")
    print(f"    Col range:       [{np.min(deep_cols)}, {np.max(deep_cols)}]")

if __name__ == '__main__':
    run_audit()

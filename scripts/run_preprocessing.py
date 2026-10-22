"""
scripts/run_preprocessing.py
CLI script to run and validate the GOES-19 multichannel preprocessing prototype:
- Reads and validates 4 channels (9, 13, 14, 15)
- Generates scientific visualizations under outputs/preprocessing/
- Executes geographic cropping (Gulf / Southeast US)
- Extracts and filters 128x128 patches
- Constructs temporal sequence windows (4 inputs -> 1 target midpoint)
- Saves a prototype processed sample in outputs/processed_sample/
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.goes_reader import (
    build_timestamp_catalog,
    read_goes_single_file,
    read_multichannel_timestamp,
    GOESProjection,
)
from src.data.spatial_crop import (
    GeographicCropper,
    extract_patches,
)
from src.data.sequence_generator import (
    TemporalSequenceGenerator,
)


def plot_four_channel_full_disk(multichannel_dict: dict, output_path: str):
    """
    Plots the 4 calibrated channels side-by-side with scientific colormaps and metadata.
    """
    data = multichannel_dict["data_kelvin"]  # (4, 1500, 2500)
    channels = multichannel_dict["channels"]  # [9, 13, 14, 15]
    dt = multichannel_dict["start_datetime"]
    
    channel_titles = {
        9: "Ch 09 - Mid-Level Water Vapor (6.93 um)\n[Atmospheric Moisture & Pre-convective Dynamics]",
        13: "Ch 13 - Clean IR Longwave Window (10.33 um)\n[Cloud-Top Temperature & Core Convection]",
        14: "Ch 14 - Traditional IR Window (11.19 um)\n[Longwave Window & Cloud Phase Tracking]",
        15: "Ch 15 - Dirty IR Split Window (12.27 um)\n[Low-level Moisture & Attenuation Split-Window]",
    }
    
    # Colormaps: 'magma' or 'turbo_r' so cold clouds stand out
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=150)
    axes = axes.flatten()
    
    for idx, c in enumerate(channels):
        ax = axes[idx]
        ch_img = data[idx]
        
        # Color scale tuned to meteorological temperature limits
        vmin = 200.0 if c in [13, 14, 15] else 200.0
        vmax = 305.0 if c in [13, 14, 15] else 270.0
        cmap = "turbo_r" if c in [13, 14, 15] else "Blues_r"
        
        im = ax.imshow(ch_img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(channel_titles.get(c, f"Channel {c}"), fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("CONUS Column Index (Fixed Grid)", fontsize=9)
        ax.set_ylabel("CONUS Row Index (Fixed Grid)", fontsize=9)
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("Brightness Temperature (K)", fontsize=9)
        
        valid_count = int(np.count_nonzero(~np.isnan(ch_img)))
        fill_count = ch_img.size - valid_count
        t_min = np.nanmin(ch_img)
        t_max = np.nanmax(ch_img)
        t_mean = np.nanmean(ch_img)
        
        stats_str = f"Min: {t_min:.1f} K | Max: {t_max:.1f} K | Mean: {t_mean:.1f} K\nValid: {valid_count:,} ({valid_count/ch_img.size*100:.1f}%) | Masked: {fill_count:,}"
        ax.text(0.02, 0.03, stats_str, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

    fig.suptitle(f"GOES-19 ABI CONUS Calibrated 4-Channel Inspection\nScan Acquisition: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [Visualization Saved]: {output_path}")


def plot_geographic_crop_comparison(full_dict: dict, crop_data: np.ndarray, cropper: GeographicCropper, output_path: str):
    """
    Visualizes the cropped region (Gulf / Southeast US) alongside CONUS reference.
    """
    dt = full_dict["start_datetime"]
    ch13_full = full_dict["data_kelvin"][1]   # Ch 13
    ch13_crop = crop_data[1]                  # Ch 13 crop
    ch09_crop = crop_data[0]                  # Ch 09 crop
    ch15_crop = crop_data[3]                  # Ch 15 crop
    
    fig = plt.figure(figsize=(18, 10), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.2])
    
    # 1. Full CONUS with Bounding Box
    ax0 = fig.add_subplot(gs[:, 0])
    im0 = ax0.imshow(ch13_full, cmap="turbo_r", vmin=205, vmax=300)
    rect = plt.Rectangle((cropper.c_min, cropper.r_min),
                         cropper.c_max - cropper.c_min,
                         cropper.r_max - cropper.r_min,
                         linewidth=2.5, edgecolor="lime", facecolor="none", linestyle="--")
    ax0.add_patch(rect)
    ax0.set_title("Full CONUS Grid (Channel 13, 10.33 um)\n[Green Dashed Box: Target Gulf/SE-US Subgrid]", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Column Index")
    ax0.set_ylabel("Row Index")
    fig.colorbar(im0, ax=ax0, fraction=0.035, pad=0.03, label="Brightness Temp (K)")
    
    # 2. Cropped Subgrid Ch 13
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(ch13_crop, cmap="turbo_r", vmin=205, vmax=300)
    ax1.set_title(f"Cropped Subgrid: Ch 13 Clean IR (10.33 um) | Shape: {ch13_crop.shape}\n[Lat: {cropper.lat_min}°N - {cropper.lat_max}°N, Lon: {cropper.lon_min}°E - {cropper.lon_max}°E]",
                  fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.03, label="Kelvin")
    
    # 3. Cropped Subgrid Ch 09 vs Split Window Ch 15
    ax2 = fig.add_subplot(gs[1, 1])
    split_diff = ch13_crop - ch15_crop  # Brightness Temperature Difference (BTD)
    im2 = ax2.imshow(split_diff, cmap="RdBu_r", vmin=-2.0, vmax=6.0)
    ax2.set_title("Split-Window BTD: (Ch 13 - Ch 15) [Kelvin]\n[Low-Level Water Vapor & Cloud Mask Discrimination]",
                  fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.03, label="BTD (K)")
    
    fig.suptitle(f"Geographic Bounding Box Crop Validation (Gulf / Southeast US)\nTimestamp: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [Visualization Saved]: {output_path}")


def plot_patch_gallery(patches: list, output_path: str):
    """
    Plots a gallery of sample 128x128 patches demonstrating convective and ambient structures.
    """
    if not patches:
        print("  [Warning]: No patches provided to plot.")
        return

    # Select up to 4 representative patches: convective ones first, then ambient
    conv_patches = [p for p in patches if p["is_convective"]]
    ambient_patches = [p for p in patches if not p["is_convective"]]
    
    selected = (conv_patches[:3] + ambient_patches[:1]) if conv_patches else patches[:4]
    if len(selected) < 4:
        selected = patches[:4]

    fig, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 4.5), dpi=150)
    if len(selected) == 1:
        axes = [axes]
        
    for idx, p in enumerate(selected):
        ax = axes[idx]
        ch13_patch = p["patch_data"][1]  # Ch 13
        tag = "CONVECTIVE" if p["is_convective"] else "AMBIENT/CLEAR"
        
        im = ax.imshow(ch13_patch, cmap="turbo_r", vmin=205, vmax=300)
        ax.set_title(f"Patch {idx+1}: {tag}\nRows {p['row_slice']} | Cols {p['col_slice']}\nMin T: {p['min_temp_k']:.1f} K | Conv: {p['convective_ratio_241']*100:.1f}%",
                     fontsize=9, fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Kelvin")

    fig.suptitle("Representative 128x128 Cropped Patches (Channel 13 Brightness Temperature)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [Visualization Saved]: {output_path}")


def plot_temporal_sequence_evolution(window_data: dict, patch_size: int, output_path: str):
    """
    Plots the 5-frame temporal sequence progression for a representative convective patch:
    Inputs: [t-60m, t-30m, t, t+30m] and Target: [t+15m]
    """
    in_tensors = window_data["input_tensors"]    # (4, 4, H, W)
    tgt_tensors = window_data["target_tensors"]  # (1, 4, H, W)
    in_dts = window_data["input_timestamps"]
    tgt_dts = window_data["target_timestamps"]
    
    # Pick central 128x128 patch
    H, W = in_tensors.shape[2], in_tensors.shape[3]
    r0 = (H - patch_size) // 2
    c0 = (W - patch_size) // 2
    
    # Extract Channel 13 (index 1) for all 5 frames
    frames = [
        ("Input 1 (t - 60m)", in_tensors[0, 1, r0:r0+patch_size, c0:c0+patch_size], in_dts[0]),
        ("Input 2 (t - 30m)", in_tensors[1, 1, r0:r0+patch_size, c0:c0+patch_size], in_dts[1]),
        ("Input 3 (Anchor t)", in_tensors[2, 1, r0:r0+patch_size, c0:c0+patch_size], in_dts[2]),
        ("TARGET (Midpoint t + 15m)", tgt_tensors[0, 1, r0:r0+patch_size, c0:c0+patch_size], tgt_dts[0]),
        ("Input 4 (Future t + 30m)", in_tensors[3, 1, r0:r0+patch_size, c0:c0+patch_size], in_dts[3]),
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=150)
    
    for idx, (role, img, dt) in enumerate(frames):
        ax = axes[idx]
        is_tgt = "TARGET" in role
        
        cmap = "turbo_r"
        im = ax.imshow(img, cmap=cmap, vmin=205, vmax=295)
        
        border_color = "red" if is_tgt else "blue"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3 if is_tgt else 1.5)
            
        ax.set_title(f"{role}\n{dt.strftime('%H:%M:%S UTC')}",
                     fontsize=10, fontweight="bold", color="crimson" if is_tgt else "navy")
        ax.axis("off")
        
        t_min = np.nanmin(img)
        t_mean = np.nanmean(img)
        ax.text(0.05, 0.05, f"Min: {t_min:.1f} K\nMean: {t_mean:.1f} K", transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    fig.suptitle("METEOR-Net Sequence-Flow Formulation: 4 Inputs + 1 Ground-Truth Midpoint Target (128x128 Patch)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [Visualization Saved]: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GOES-19 multichannel preprocessing prototype.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root raw NetCDF directory")
    parser.add_argument("--output-dir", type=str, default="outputs/preprocessing", help="Visualizations output dir")
    parser.add_argument("--processed-dir", type=str, default="outputs/processed_sample", help="Processed tensor sample output dir")
    parser.add_argument("--lat-min", type=float, default=25.0)
    parser.add_argument("--lat-max", type=float, default=35.0)
    parser.add_argument("--lon-min", type=float, default=-95.0)
    parser.add_argument("--lon-max", type=float, default=-80.0)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 2B: GOES-19 PREPROCESSING PROTOTYPE PIPELINE")
    print("=" * 80)

    # 1. Catalog scan
    print("\n[Step 1] Building synchronized 4-channel catalog...")
    catalog = build_timestamp_catalog(args.data_dir, channels=(9, 13, 14, 15))
    print(f"  Discovered {len(catalog)} fully synchronized observation timestamps across 4 channels.")
    if not catalog:
        print("  Error: No synchronized timestamps found. Exiting.")
        return

    # 2. Read representative timestamp (Scan 00)
    print("\n[Step 2] Reading and validating representative timestamp (Scan 00)...")
    sample_ts = catalog[0]
    multi_data = read_multichannel_timestamp(sample_ts["channels"], channel_order=(9, 13, 14, 15))
    print(f"  Timestamp:     {multi_data['start_datetime']} UTC")
    print(f"  Stacked Shape: {multi_data['data_kelvin'].shape} (Channels, Height, Width)")
    print(f"  Data Type:     {multi_data['data_kelvin'].dtype}")
    print(f"  Valid Pixels:  {np.count_nonzero(multi_data['valid_mask']):,} / {multi_data['valid_mask'].size:,} ({np.mean(multi_data['valid_mask'])*100:.2f}%)")

    # 3. Visualizations of 4-channel full CONUS
    print("\n[Step 3] Generating full CONUS 4-channel visualization...")
    os.makedirs(args.output_dir, exist_ok=True)
    full_vis_path = os.path.join(args.output_dir, "01_full_conus_4channels.png")
    plot_four_channel_full_disk(multi_data, full_vis_path)

    # 4. Geographic crop
    print("\n[Step 4] Executing geographic crop (Gulf / Southeast US)...")
    cropper = GeographicCropper(
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        projection=multi_data["projection"]
    )
    crop_info = cropper.get_crop_info()
    print(f"  Target Latitudes:  [{cropper.lat_min}°N, {cropper.lat_max}°N]")
    print(f"  Target Longitudes: [{cropper.lon_min}°E, {cropper.lon_max}°E]")
    print(f"  Pixel Row Bounds:  {crop_info['row_bounds']} (Height: {crop_info['crop_shape'][0]})")
    print(f"  Pixel Col Bounds:  {crop_info['col_bounds']} (Width:  {crop_info['crop_shape'][1]})")

    cropped_data = cropper.crop_array(multi_data["data_kelvin"])
    cropped_mask = cropper.crop_array(multi_data["valid_mask"])
    print(f"  Cropped Array Shape: {cropped_data.shape}")
    print(f"  Crop Valid Ratio:    {np.mean(cropped_mask)*100:.2f}% (Fill/Space Pixels: {np.count_nonzero(~cropped_mask)})")

    crop_vis_path = os.path.join(args.output_dir, "02_geographic_crop_validation.png")
    plot_geographic_crop_comparison(multi_data, cropped_data, cropper, crop_vis_path)

    # 5. Extract 128x128 patches
    print("\n[Step 5] Extracting 128x128 candidate patches from cropped region...")
    patches = extract_patches(
        multichannel_crop=cropped_data,
        valid_mask_crop=cropped_mask,
        patch_size=args.patch_size,
        stride=args.stride,
        channel_13_idx=1
    )
    conv_count = sum(1 for p in patches if p["is_convective"])
    print(f"  Extracted {len(patches)} total valid 128x128 patches (0 fill pixels).")
    print(f"  Convective Salience Filter: {conv_count} patches satisfy convective criteria (< 241 K).")
    
    patch_vis_path = os.path.join(args.output_dir, "03_patch_samples_128x128.png")
    plot_patch_gallery(patches, patch_vis_path)

    # 6. Temporal sequence generator
    print("\n[Step 6] Initializing temporal sequence generator (4 inputs -> 1 midpoint target)...")
    seq_gen = TemporalSequenceGenerator(
        catalog=catalog,
        input_offsets_min=(-60, -30, 0, 30),
        target_offsets_min=(15,),
        nominal_step_min=5.0,
        channel_order=(9, 13, 14, 15),
        cropper=cropper
    )
    seq_summary = seq_gen.get_summary()
    print(f"  Total Observations Available: {seq_summary['total_observations']}")
    print(f"  Input Offsets:                {seq_summary['input_offsets_min']} minutes")
    print(f"  Target Offsets:               {seq_summary['target_offsets_min']} minutes")
    print(f"  Total Valid Sliding Windows:  {seq_summary['total_valid_windows']}")

    # Print table of discovered windows
    print("\n  --- Discovered Temporal Windows ---")
    for w_idx, win in enumerate(seq_gen.valid_windows):
        anchor = win["anchor_datetime"]
        in_times = [f['datetime'].strftime('%H:%M') for f in win['input_frames']]
        tgt_time = win['target_frames'][0]['datetime'].strftime('%H:%M')
        print(f"  Window {w_idx:02d}: Inputs: {in_times} | Target: {tgt_time} | Anchor: {anchor.strftime('%H:%M UTC')}")

    # 7. Load representative sequence window and visualize
    if seq_gen.valid_windows:
        print("\n[Step 7] Loading tensors and visualizing representative sequence window (Window 00)...")
        w0_data = seq_gen.load_window_tensors(0)
        seq_vis_path = os.path.join(args.output_dir, "04_sequence_evolution_5frames.png")
        plot_temporal_sequence_evolution(w0_data, args.patch_size, seq_vis_path)

        # 8. Save prototype processed sample in compressed format
        print("\n[Step 8] Exporting prototype processed sample to storage-efficient format...")
        os.makedirs(args.processed_dir, exist_ok=True)
        sample_path = os.path.join(args.processed_dir, "sample_sequence_window00.npz")
        
        # Extract patch sequences for Window 00
        patch_seqs = seq_gen.extract_sequence_patches(0, patch_size=args.patch_size, stride=args.stride)
        
        # Save sample tensors
        sample_in = w0_data["input_tensors"]   # (4, 4, H, W)
        sample_tgt = w0_data["target_tensors"] # (1, 4, H, W)
        
        np.savez_compressed(
            sample_path,
            inputs=sample_in,
            targets=sample_tgt,
            valid_mask=w0_data["combined_valid_mask"],
            channels=np.array([9, 13, 14, 15]),
            input_timestamps=[dt.isoformat() for dt in w0_data["input_timestamps"]],
            target_timestamps=[dt.isoformat() for dt in w0_data["target_timestamps"]],
            anchor_timestamp=w0_data["anchor_datetime"].isoformat(),
            crop_bounds=np.array(crop_info["row_bounds"] + crop_info["col_bounds"])
        )
        sample_size_kb = os.path.getsize(sample_path) / 1024
        print(f"  [Sample Saved]: {sample_path} ({sample_size_kb:.1f} KB)")

        # Save metadata catalog
        catalog_path = os.path.join(args.processed_dir, "sequence_metadata.json")
        meta_dict = {
            "dataset_info": {
                "satellite": "GOES-19",
                "instrument": "ABI",
                "product": "ABI-L2-CMIPC",
                "channels": [9, 13, 14, 15],
                "spatial_resolution": "2 km (native)",
                "crop_region": "Gulf / Southeast US",
                "lat_bounds": crop_info["lat_bounds"],
                "lon_bounds": crop_info["lon_bounds"],
                "crop_pixel_shape": crop_info["crop_shape"],
            },
            "temporal_design": {
                "input_offsets_minutes": seq_gen.input_offsets_min,
                "target_offsets_minutes": seq_gen.target_offsets_min,
                "temporal_super_resolution_factor": "2x (30 min to 15 min)",
            },
            "windows_count": len(seq_gen.valid_windows),
            "patch_sequences_in_window00": len(patch_seqs),
            "storage_estimate": {
                "current_sample_files": 96,
                "current_raw_size_mb": 282.9,
                "projected_7days_raw_gb": 29.7,
                "projected_10days_raw_gb": 42.4,
                "projected_10days_processed_patches_gb": 3.8
            }
        }
        with open(catalog_path, "w") as jf:
            json.dump(meta_dict, jf, indent=2)
        print(f"  [Metadata Saved]: {catalog_path}")

    print("\n" + "=" * 80)
    print("PHASE 2B PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

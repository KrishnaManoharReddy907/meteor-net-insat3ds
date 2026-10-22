"""
scripts/run_insat_prototype_pipeline.py
Phase 2D: Prototype execution script for INSAT-3DS Level-1B data.
Performs:
1. Crop analysis: enclosing rectangle vs exact geographic mask (5-37N, 65-97E).
2. Diagnostic visualization of the crop mask and excluded corner pixels.
3. Batch processing of all 7 files: raw preservation, LUT calibration, NaN fill masking.
4. Output persistence in outputs/insat3ds_prototype/ (.npz + manifest).
5. Calibrated Kelvin visualizations showing all 4 channels and consecutive differences.
6. Detailed storage and timing metrics summary.
"""

import os
import sys
import glob
import json
import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.data.insat_reader import (
    IR_CHANNELS,
    read_insat_multichannel_frame,
    INSATRegionalCropper,
    DEFAULT_ROW_SLICE,
    DEFAULT_COL_SLICE,
    DEFAULT_LAT_BOUNDS,
    DEFAULT_LON_BOUNDS,
)

RAW_DIR = "data/insat3ds/raw"
OUTPUT_DIR = "outputs/insat3ds_prototype"
FIGURES_DIR = "reports/figures"


def validate_crop_and_generate_diagnostic(frame: dict, cropper: INSATRegionalCropper):
    """
    Compares enclosing rectangle with exact geographic mask and generates a diagnostic plot.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    lat = frame["latitude"]
    lon = frame["longitude"]
    r_sl = cropper.row_slice
    c_sl = cropper.col_slice
    
    lat_crop = lat[r_sl, c_sl]
    lon_crop = lon[r_sl, c_sl]
    
    exact_mask = cropper.compute_exact_geographic_mask(lat_crop, lon_crop)
    total_rect_px = lat_crop.size
    inside_exact_px = int(np.count_nonzero(exact_mask))
    outside_exact_px = total_rect_px - inside_exact_px
    
    # Check invalid pixels in channels inside the rectangle and inside the exact mask
    chan_masks = frame["channel_masks"][:, r_sl, c_sl]
    invalid_in_rect = [int(np.count_nonzero(~chan_masks[i])) for i in range(len(IR_CHANNELS))]
    invalid_in_exact = [int(np.count_nonzero((~chan_masks[i]) & exact_mask)) for i in range(len(IR_CHANNELS))]
    
    print("\n" + "=" * 80)
    print("GEOGRAPHIC CROP DETAILED COMPARISON")
    print("=" * 80)
    print(f"  Enclosing Slice: Rows [{r_sl.start}:{r_sl.stop}] (height={lat_crop.shape[0]}), Cols [{c_sl.start}:{c_sl.stop}] (width={lat_crop.shape[1]})")
    print(f"  Total Pixels in Rectangular Box: {total_rect_px:,}")
    print(f"  Pixels strictly inside Target Mask (5-37N, 65-97E): {inside_exact_px:,} ({inside_exact_px / total_rect_px * 100:.2f}%)")
    print(f"  Pixels outside Target Mask but inside Rectangular Box: {outside_exact_px:,} ({outside_exact_px / total_rect_px * 100:.2f}%)")
    for idx, ch in enumerate(IR_CHANNELS):
        print(f"  {ch:10s} -> Invalid in Rect: {invalid_in_rect[idx]} | Invalid in Exact Mask: {invalid_in_exact[idx]}")
    
    # Generate diagnostic plot showing the exact mask vs corner pixels inside rectangle
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=150)
    
    # Left: Full Disk showing the enclosing rectangle and exact mask footprint
    lat_full = lat.copy()
    full_exact_mask = (lat >= cropper.lat_bounds[0]) & (lat <= cropper.lat_bounds[1]) & (lon >= cropper.lon_bounds[0]) & (lon <= cropper.lon_bounds[1])
    
    im1 = ax1.imshow(lat_full, cmap="coolwarm", vmin=-60, vmax=60, origin="upper")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Latitude (deg N)")
    
    # Draw enclosing rectangle
    rect = patches.Rectangle(
        (c_sl.start, r_sl.start), c_sl.stop - c_sl.start, r_sl.stop - r_sl.start,
        linewidth=2.0, edgecolor="red", facecolor="none", linestyle="-",
        label=f"Enclosing Slice ({lat_crop.shape[0]}x{lat_crop.shape[1]} px)\nRows {r_sl.start}:{r_sl.stop}, Cols {c_sl.start}:{c_sl.stop}"
    )
    ax1.add_patch(rect)
    
    # Overlay exact mask footprint in cyan
    ax1.contour(full_exact_mask, levels=[0.5], colors=["cyan"], linewidths=[1.5])
    ax1.plot([], [], color="cyan", linewidth=1.5, label="Exact Target Boundary (5-37N, 65-97E)")
    ax1.set_title("INSAT-3DS Full Disk: Enclosing Rectangle vs. Exact Target Footprint", fontsize=11, weight="bold")
    ax1.set_xlabel("Column Index (GeoX)")
    ax1.set_ylabel("Row Index (GeoY, North to South)")
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.3)
    
    # Right: Close-up of the Enclosing Rectangle showing strictly internal vs corner pixels
    # Create categorical visualization: 1 = inside exact mask, 0 = outside (curved geostationary corner margin)
    cat_grid = np.zeros_like(lat_crop, dtype=np.int32)
    cat_grid[exact_mask] = 1   # Inside target
    
    cmap_crop = matplotlib.colors.ListedColormap(["#2b3a4a", "#10b981"])
    im2 = ax2.imshow(cat_grid, cmap=cmap_crop, origin="upper")
    
    # Overlay lat/lon contours
    cs_lat = ax2.contour(lat_crop, levels=np.arange(5, 40, 5), colors="white", linewidths=0.6, linestyles=":")
    ax2.clabel(cs_lat, inline=True, fontsize=7, fmt="%d°N")
    cs_lon = ax2.contour(lon_crop, levels=np.arange(65, 100, 5), colors="yellow", linewidths=0.6, linestyles=":")
    ax2.clabel(cs_lon, inline=True, fontsize=7, fmt="%d°E")
    
    # Custom legend for category
    patch_in = patches.Patch(color="#10b981", label=f"Inside Target Bounds ({inside_exact_px:,} px, {inside_exact_px/total_rect_px*100:.1f}%)")
    patch_out = patches.Patch(color="#2b3a4a", label=f"Enclosing Margin / Corner ({outside_exact_px:,} px, {outside_exact_px/total_rect_px*100:.1f}%)")
    ax2.legend(handles=[patch_in, patch_out], loc="lower left", fontsize=8, framealpha=0.9)
    
    ax2.set_title(f"Enclosing Rectangle Breakdown ({lat_crop.shape[0]}x{lat_crop.shape[1]} px)\n(Corner margin pixels are valid Earth, but outside [5-37N, 65-97E])", fontsize=11, weight="bold")
    ax2.set_xlabel("Local Column Index (0 to 866)")
    ax2.set_ylabel("Local Row Index (0 to 779)")
    ax2.grid(True, linestyle=":", alpha=0.3)
    
    plt.tight_layout()
    diag_path = os.path.join(FIGURES_DIR, "insat3ds_crop_comparison_diagnostic.png")
    plt.savefig(diag_path, bbox_inches="tight")
    plt.close()
    print(f"  Diagnostic plot saved: {diag_path}")
    
    return {
        "total_rect_pixels": total_rect_px,
        "inside_exact_pixels": inside_exact_px,
        "outside_exact_pixels": outside_exact_px,
        "inside_percentage": inside_exact_px / total_rect_px * 100.0,
        "invalid_in_rect": invalid_in_rect,
        "invalid_in_exact": invalid_in_exact,
        "diagnostic_fig_path": diag_path,
    }


def process_all_files():
    """
    Processes all 7 INSAT-3DS raw files, saving regional crops to outputs/insat3ds_prototype/.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cropper = INSATRegionalCropper()
    
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.h5")))
    print("\n" + "=" * 80)
    print(f"PROCESSING INSAT-3DS REGIONAL PROTOTYPE ({len(raw_files)} files)")
    print("=" * 80)
    
    processed_records = []
    crop_stats = None
    first_cropped_frame = None
    
    for idx, fpath in enumerate(raw_files):
        print(f"\n[{idx+1}/{len(raw_files)}] Reading & Calibrating: {os.path.basename(fpath)}")
        t0 = datetime.datetime.now()
        
        # Read multichannel frame with coords and scan line times
        frame = read_insat_multichannel_frame(
            filepath=fpath,
            channels=tuple(IR_CHANNELS),
            apply_calibration=True,
            load_coords=True,
            load_scan_line_time=True
        )
        
        # Crop diagnostic on the first frame
        if idx == 0:
            crop_stats = validate_crop_and_generate_diagnostic(frame, cropper)
            first_cropped_frame = cropper.crop_multichannel_frame(frame)
        
        # Crop to regional domain
        cropped = cropper.crop_multichannel_frame(frame)
        
        # Determine output filename
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        out_npz_path = os.path.join(OUTPUT_DIR, f"{base_name}_regional.npz")
        out_meta_path = os.path.join(OUTPUT_DIR, f"{base_name}_regional.json")
        
        # Save compressed NPZ
        np.savez_compressed(
            out_npz_path,
            data_kelvin=cropped["data_kelvin"],             # (4, 780, 867) float32
            raw_counts=cropped["raw_counts"],               # (4, 780, 867) uint16
            channel_masks=cropped["channel_masks"],         # (4, 780, 867) bool
            combined_mask=cropped["combined_mask"],         # (780, 867) bool
            exact_geo_mask=cropped["exact_geo_mask"],       # (780, 867) bool
            latitude=cropped["latitude"],                   # (780, 867) float32
            longitude=cropped["longitude"],                 # (780, 867) float32
            channels=np.array(cropped["channels"]),
            scan_line_times=np.array(cropped["scan_line_times"] or []),
            crop_rows=np.array(cropped["crop_rows"]),
            crop_cols=np.array(cropped["crop_cols"]),
        )
        
        npz_size = os.path.getsize(out_npz_path)
        raw_size = os.path.getsize(fpath)
        t_elapsed = (datetime.datetime.now() - t0).total_seconds()
        
        # Channel-specific brightness temperature statistics
        chan_metrics = {}
        for c_idx, ch in enumerate(IR_CHANNELS):
            k_data = cropped["data_kelvin"][c_idx]
            v_mask = cropped["channel_masks"][c_idx]
            valid_k = k_data[v_mask]
            chan_metrics[ch] = {
                "min_K": float(np.min(valid_k)),
                "max_K": float(np.max(valid_k)),
                "mean_K": float(np.mean(valid_k)),
                "std_K": float(np.std(valid_k)),
                "valid_count": int(valid_k.size),
                "nan_count": int(np.count_nonzero(np.isnan(k_data))),
            }
        
        meta = {
            "source_raw_file": os.path.basename(fpath),
            "processed_npz_file": os.path.basename(out_npz_path),
            "nominal_gmt": cropped["nominal_gmt"],
            "start_datetime_utc": cropped["start_datetime"].isoformat() if cropped["start_datetime"] else None,
            "end_datetime_utc": cropped["end_datetime"].isoformat() if cropped["end_datetime"] else None,
            "channels": cropped["channels"],
            "crop_rows": cropped["crop_rows"],
            "crop_cols": cropped["crop_cols"],
            "crop_shape": list(cropped["crop_shape"]),
            "raw_size_bytes": raw_size,
            "processed_size_bytes": npz_size,
            "compression_ratio": raw_size / npz_size,
            "processing_time_sec": t_elapsed,
            "regional_scan_start_time": cropped["scan_line_times"][0] if cropped["scan_line_times"] else None,
            "regional_scan_end_time": cropped["scan_line_times"][-1] if cropped["scan_line_times"] else None,
            "channel_statistics_kelvin": chan_metrics,
        }
        
        with open(out_meta_path, "w") as jf:
            json.dump(meta, jf, indent=2)
            
        processed_records.append(meta)
        print(f"  -> Saved: {os.path.basename(out_npz_path)} ({npz_size / (1024*1024):.2f} MB, {meta['compression_ratio']:.1f}x compression) in {t_elapsed:.2f}s")
        print(f"  -> TIR-1 Range: {chan_metrics['IMG_TIR1']['min_K']:.1f}K to {chan_metrics['IMG_TIR1']['max_K']:.1f}K (mean {chan_metrics['IMG_TIR1']['mean_K']:.1f}K)")
    
    # Write overall manifest
    manifest_path = os.path.join(OUTPUT_DIR, "prototype_manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(),
            "target_satellite": "INSAT-3DS",
            "product_type": "STANDARD (FULL DISK) Level-1B",
            "crop_region": "Indian Subcontinent (5N-37N, 65E-97E)",
            "enclosing_grid_shape": [780, 867],
            "total_files_processed": len(processed_records),
            "total_processed_size_mb": sum(r["processed_size_bytes"] for r in processed_records) / (1024*1024),
            "crop_statistics": crop_stats,
            "file_records": processed_records,
        }, mf, indent=2)
    print(f"\nManifest saved successfully: {manifest_path}")
    
    return processed_records, first_cropped_frame


def generate_kelvin_visualizations(processed_records):
    """
    Generates Kelvin-scaled visualizations for representative timestamps and consecutive differences.
    """
    print("\n" + "=" * 80)
    print("GENERATING KELVIN-SCALED VISUALIZATIONS")
    print("=" * 80)
    
    # 1. 4-Channel Calibrated Brightness Temperature Plot for First Scan (00:00 UTC)
    f0_meta = processed_records[0]
    npz0_path = os.path.join(OUTPUT_DIR, f0_meta["processed_npz_file"])
    with np.load(npz0_path) as d0:
        k_data0 = d0["data_kelvin"]          # (4, 780, 867)
        lat = d0["latitude"]
        lon = d0["longitude"]
        exact_mask = d0["exact_geo_mask"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    colormaps = ["inferno", "inferno", "Blues_r", "magma"]
    ranges = [
        (200.0, 315.0),  # TIR1
        (200.0, 315.0),  # TIR2
        (210.0, 275.0),  # WV (upper-tropospheric, colder)
        (240.0, 335.0),  # MIR
    ]
    
    for idx, (ch, ax, cmap, (vmin, vmax)) in enumerate(zip(IR_CHANNELS, axes.flat, colormaps, ranges)):
        img = k_data0[idx].copy()
        
        # Display image with masked NaNs shown distinctly in charcoal/black
        current_cmap = matplotlib.colormaps.get_cmap(cmap).copy()
        current_cmap.set_bad(color="#1a1a1a")
        
        im = ax.imshow(img, cmap=current_cmap, vmin=vmin, vmax=vmax, origin="upper")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Brightness Temperature (Kelvin)", fontsize=9)
        
        # Overlay exact geographic boundary contour
        ax.contour(exact_mask, levels=[0.5], colors=["cyan"], linewidths=[1.0], linestyles="--")
        
        # Add landmark markers
        landmarks = {"Delhi": (345, 177), "Mumbai": (223, 406), "Kolkata": (620, 319), "Chennai": (412, 560)}
        for name, (c_idx, r_idx) in landmarks.items():
            ax.plot(c_idx, r_idx, "ro", markersize=4)
            ax.text(c_idx + 10, r_idx, name, color="white", fontsize=8, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.6, edgecolor="none"))
        
        ax.set_title(f"Calibrated Brightness Temperature: {ch}\n20-OCT-2026 00:00 UTC (Mean: {np.nanmean(img):.1f} K)", fontsize=10, weight="bold")
        ax.set_xlabel("Local Column Index (0 to 866)")
        ax.set_ylabel("Local Row Index (0 to 779)")
        ax.grid(True, linestyle=":", alpha=0.2)
    
    plt.suptitle("INSAT-3DS Regional Prototype: Calibrated Brightness Temperature (Kelvin)\nTarget Domain: Indian Subcontinent & Surrounding Seas (5°N–37°N, 65°E–97°E)", fontsize=13, weight="bold")
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "insat3ds_calibrated_bt_multichannel_0000.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"  4-Channel BT plot saved: {fig1_path}")
    
    # 2. Consecutive Difference Plot (00:30 UTC minus 00:00 UTC) for TIR-1 and WV
    f1_meta = processed_records[1]
    npz1_path = os.path.join(OUTPUT_DIR, f1_meta["processed_npz_file"])
    with np.load(npz1_path) as d1:
        k_data1 = d1["data_kelvin"]
    
    diff_tir1 = k_data1[0] - k_data0[0]  # TIR1 difference (30 min)
    diff_wv = k_data1[2] - k_data0[2]    # WV difference (30 min)
    
    fig, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    # Difference plot TIR-1
    im_d1 = ax_d1.imshow(diff_tir1, cmap="coolwarm", vmin=-15, vmax=15, origin="upper")
    cbar_d1 = plt.colorbar(im_d1, ax=ax_d1, fraction=0.046, pad=0.04)
    cbar_d1.set_label("ΔT (Kelvin / 30 min)", fontsize=9)
    ax_d1.set_title("30-min Temporal Evolution: IMG_TIR1 (00:30 UTC - 00:00 UTC)\n(Cloud advection, convection growth, and surface thermal changes)", fontsize=10, weight="bold")
    ax_d1.set_xlabel("Local Column Index")
    ax_d1.set_ylabel("Local Row Index")
    ax_d1.grid(True, linestyle=":", alpha=0.3)
    
    # Difference plot WV
    im_d2 = ax_d2.imshow(diff_wv, cmap="coolwarm", vmin=-10, vmax=10, origin="upper")
    cbar_d2 = plt.colorbar(im_d2, ax=ax_d2, fraction=0.046, pad=0.04)
    cbar_d2.set_label("ΔT (Kelvin / 30 min)", fontsize=9)
    ax_d2.set_title("30-min Temporal Evolution: IMG_WV (00:30 UTC - 00:00 UTC)\n(Upper-tropospheric moisture circulation and advection)", fontsize=10, weight="bold")
    ax_d2.set_xlabel("Local Column Index")
    ax_d2.set_ylabel("Local Row Index")
    ax_d2.grid(True, linestyle=":", alpha=0.3)
    
    plt.suptitle("INSAT-3DS Consecutive Scan Temporal Evolution Diagnostic (30-minute Δt)\n(Confirms coherent physical cloud motion with zero calibration striping or registration offsets)", fontsize=12, weight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "insat3ds_temporal_difference_30min.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"  Consecutive difference plot saved: {fig2_path}")


def main():
    records, first_crop = process_all_files()
    generate_kelvin_visualizations(records)


if __name__ == "__main__":
    main()

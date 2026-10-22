import os
import sys
import json
import time
import datetime
import numpy as np
import torch

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
from src.data.spatial_crop import GeographicCropper
from src.models.normalization import ChannelStandardizer

def main():
    print("=" * 80)
    print("PHASE 6A: COMPREHENSIVE MULTI-EVENT VALIDATION AND CHARACTERIZATION")
    print("=" * 80)
    
    data_dir = os.path.join(PROJECT_ROOT, "data")
    print(f"Scanning data catalog in {data_dir}...")
    catalog = build_timestamp_catalog(data_dir, channels=(9, 13, 14, 15))
    print(f"Total synchronized 4-channel observations discovered: {len(catalog)}")
    
    # 1. Cluster contiguous episodes (gap > 2 hours)
    sorted_cat = sorted(catalog, key=lambda x: x["datetime"])
    episodes = []
    curr_start = sorted_cat[0]["datetime"]
    curr_prev = sorted_cat[0]["datetime"]
    curr_scans = [sorted_cat[0]]
    for item in sorted_cat[1:]:
        dt = item["datetime"]
        if (dt - curr_prev) > datetime.timedelta(hours=2):
            ev_id = "event_" + curr_start.strftime("%Y-%m-%d")
            episodes.append({
                "event_id": ev_id,
                "start_dt": curr_start,
                "end_dt": curr_prev,
                "scans": curr_scans
            })
            curr_start = dt
            curr_scans = []
        curr_prev = dt
        curr_scans.append(item)
    ev_id = "event_" + curr_start.strftime("%Y-%m-%d")
    episodes.append({
        "event_id": ev_id,
        "start_dt": curr_start,
        "end_dt": curr_prev,
        "scans": curr_scans
    })
    
    print(f"\n[Step 1] Identified {len(episodes)} continuous meteorological episodes:")
    for ep in episodes:
        print(f"  - {ep['event_id']}: {ep['start_dt'].strftime('%Y-%m-%d %H:%M:%S')} to "
              f"{ep['end_dt'].strftime('%Y-%m-%d %H:%M:%S')} UTC ({len(ep['scans'])} scans)")

    # 2. Sequence windows discovery
    print("\n[Step 2] Constructing sequence windows per event...")
    event_windows = {}
    event_stats = {}
    cropper = GeographicCropper()
    
    for ep in episodes:
        ev_id = ep["event_id"]
        ev_catalog = ep["scans"]
        seq_gen = TemporalSequenceGenerator(
            catalog=ev_catalog,
            input_offsets_min=(-60, -30, 0, 30),
            target_offsets_min=(15,),
            channel_order=(9, 13, 14, 15),
            cropper=cropper
        )
        windows = seq_gen.valid_windows
        event_windows[ev_id] = {
            "seq_gen": seq_gen,
            "windows": windows
        }
        print(f"  {ev_id}: {len(windows)} valid sequence windows constructed.")

        # Measure temperature distribution across sample target frames
        # Analyze midpoint target of the central window
        mid_idx = len(windows) // 2
        mid_win = windows[mid_idx]
        tgt_frame = mid_win["target_frames"][0]
        mc_data = read_multichannel_timestamp(tgt_frame["channels"], channel_order=(9, 13, 14, 15))
        
        # Band 13 is index 1
        b13 = mc_data["data_kelvin"][1]
        vmask = mc_data["valid_mask"]
        b13_valid = b13[vmask]
        
        min_k = float(np.nanmin(b13_valid))
        mean_k = float(np.nanmean(b13_valid))
        max_k = float(np.nanmax(b13_valid))
        p01_k = float(np.percentile(b13_valid, 1))
        p05_k = float(np.percentile(b13_valid, 5))
        p50_k = float(np.percentile(b13_valid, 50))
        frac_below_241 = float(np.mean(b13_valid < 241.0))
        frac_below_221 = float(np.mean(b13_valid < 221.0))
        count_below_221 = int(np.sum(b13_valid < 221.0))

        # Patch extraction audit on central window
        patches = seq_gen.extract_sequence_patches(mid_idx, patch_size=128, stride=64, require_full_valid=True)
        conv_patch_count = 0
        for p in patches:
            p_b13 = p["target_patches"][0, 1]
            p_min = float(np.min(p_b13))
            p_frac241 = float(np.mean(p_b13 < 241.0))
            p_count221 = int(np.sum(p_b13 < 221.0))
            if p_min < 241.0 and (p_frac241 >= 0.02 or p_count221 >= 5):
                conv_patch_count += 1
                
        event_stats[ev_id] = {
            "total_scans": len(ev_catalog),
            "total_windows": len(windows),
            "sample_target_time": tgt_frame["datetime"].strftime("%Y-%m-%d %H:%M:%S UTC"),
            "b13_min_kelvin": min_k,
            "b13_mean_kelvin": mean_k,
            "b13_max_kelvin": max_k,
            "b13_p01_kelvin": p01_k,
            "b13_p05_kelvin": p05_k,
            "b13_p50_kelvin": p50_k,
            "frac_below_241k": frac_below_241,
            "frac_below_221k": frac_below_221,
            "count_below_221k": count_below_221,
            "sample_128x128_patches": len(patches),
            "sample_convective_patches": conv_patch_count,
            "convective_patch_pct": round((conv_patch_count / max(1, len(patches))) * 100.0, 1)
        }
        print(f"    -> Sample Target ({tgt_frame['datetime'].strftime('%H:%M UTC')}): "
              f"T_min={min_k:.1f}K, T_mean={mean_k:.1f}K, <241K: {frac_below_241*100:.2f}%, "
              f"<221K: {count_below_221} px | Patches: {conv_patch_count}/{len(patches)} convective ({event_stats[ev_id]['convective_patch_pct']}%)")

    # 3. Fit ChannelStandardizer strictly on Training Events
    print("\n[Step 3] Fitting ChannelStandardizer strictly on Training partition (Event 1 + Event 2)...")
    train_ev_ids = ["event_2026-01-01", "event_2026-05-20"]
    train_arrays = []
    train_masks = []
    
    # Sample uniformly from training windows to fit robust global moments
    for ev_id in train_ev_ids:
        sg = event_windows[ev_id]["seq_gen"]
        wins = event_windows[ev_id]["windows"]
        # Sample every 5th window across the 6-hour period
        for w_i in range(0, len(wins), 5):
            w_tensors = sg.load_window_tensors(w_i)
            # w_tensors['input_tensors'] is [4, 4, H, W]
            train_arrays.append(w_tensors["input_tensors"])
            # w_tensors['combined_valid_mask'] is [H, W]
            m_exp = np.broadcast_to(w_tensors["combined_valid_mask"][None, None, ...], (4, 4, *w_tensors["combined_valid_mask"].shape))
            train_masks.append(m_exp)
            
    train_x = np.concatenate(train_arrays, axis=0) # [N_total, 4, H, W]
    train_m = np.concatenate(train_masks, axis=0)
    print(f"  Training sample tensor shape: {train_x.shape}, Valid mask shape: {train_m.shape}")
    
    standardizer = ChannelStandardizer.fit(
        x=train_x,
        mask=train_m,
        channel_dim=1,
        channel_order=(9, 13, 14, 15),
        provenance={
            "description": "Fitted on GOES-19 Training Partition (Event 1 + Event 2)",
            "training_events": train_ev_ids,
            "created_at": datetime.datetime.now().isoformat(),
            "channel_order": [9, 13, 14, 15]
        }
    )
    
    means = standardizer.mean.tolist()
    stds = standardizer.std.tolist()
    print(f"  Fitted Mean Kelvin per channel [9, 13, 14, 15]: {[round(m, 2) for m in means]}")
    print(f"  Fitted Std Kelvin per channel  [9, 13, 14, 15]: {[round(s, 2) for s in stds]}")
    
    # Save frozen normalization stats
    norm_dir = os.path.join(PROJECT_ROOT, "outputs", "normalization")
    os.makedirs(norm_dir, exist_ok=True)
    stats_path = os.path.join(norm_dir, "frozen_global_train_stats.json")
    standardizer.save_stats(stats_path)
    print(f"  [Saved] Frozen normalization stats serialized to {stats_path}")

    # 4. Save validation summary
    val_summary_path = os.path.join(PROJECT_ROOT, "outputs", "full_acquisition", "multi_event_validation.json")
    with open(val_summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "total_observations": len(catalog),
            "events": event_stats,
            "training_normalization": {
                "stats_file": stats_path,
                "mean_kelvin": means,
                "std_kelvin": stds,
                "channel_order": [9, 13, 14, 15]
            }
        }, f, indent=2)
    print(f"\n[Saved] Multi-event validation report saved to {val_summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

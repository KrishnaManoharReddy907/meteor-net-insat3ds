import os
import glob
import numpy as np

OUTPUT_DIR = "outputs/insat3ds_prototype"
npz_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_regional.npz")))

print("=" * 80)
print("AUDIT OF ALL 7 PROCESSED REGIONAL NPZ ARCHIVES")
print("=" * 80)

summary = {}
channels = ["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_MIR"]

for fpath in npz_files:
    fname = os.path.basename(fpath)
    data = np.load(fpath)
    k_data = data["data_kelvin"]          # (4, 780, 867)
    raw_counts = data["raw_counts"]       # (4, 780, 867)
    chan_masks = data["channel_masks"]   # (4, 780, 867)
    comb_mask = data["combined_mask"]     # (780, 867)
    exact_geo = data["exact_geo_mask"]    # (780, 867)
    
    file_info = {}
    print(f"\n--- {fname} ---")
    print(f"  Shape: {k_data.shape}, Total pixels per channel: {k_data[0].size:,}")
    
    for c_idx, ch in enumerate(channels):
        raw_c = raw_counts[c_idx]
        k_c = k_data[c_idx]
        m_c = chan_masks[c_idx]
        
        nan_count = int(np.count_nonzero(np.isnan(k_c)))
        inf_count = int(np.count_nonzero(np.isinf(k_c)))
        finite_count = int(np.count_nonzero(np.isfinite(k_c)))
        fill_count = int(np.count_nonzero(raw_c == 1023))
        valid_mask_count = int(np.count_nonzero(m_c))
        
        # Check inside exact geo mask
        nan_in_geo = int(np.count_nonzero(np.isnan(k_c[exact_geo])))
        fill_in_geo = int(np.count_nonzero(raw_c[exact_geo] == 1023))
        
        # Check consistency: does (raw == 1023) exactly match (np.isnan(k))?
        nan_eq_fill = np.array_equal(np.isnan(k_c), (raw_c == 1023))
        # does valid_mask exactly match (~np.isnan(k))?
        mask_eq_finite = np.array_equal(m_c, np.isfinite(k_c))
        
        file_info[ch] = {
            "finite_count": finite_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "fill_1023_count": fill_count,
            "valid_mask_count": valid_mask_count,
            "nan_inside_exact_geo": nan_in_geo,
            "fill_inside_exact_geo": fill_in_geo,
            "nan_strictly_matches_fill": nan_eq_fill,
            "mask_strictly_matches_finite": mask_eq_finite,
            "min_K": float(np.nanmin(k_c)),
            "max_K": float(np.nanmax(k_c)),
        }
        
        status = "ALL VALID" if nan_count == 0 else f"CONTAINS {nan_count} NaN(s)"
        print(f"  {ch:10s}: Finite={finite_count:,} | NaN={nan_count} | Fill1023={fill_count} | MaskValid={valid_mask_count:,} [{status}]")
        if nan_count > 0:
            nan_r, nan_c = np.where(np.isnan(k_c))
            for nr, nc in zip(nan_r, nan_c):
                print(f"    -> Location: local row {nr}, col {nc} (Global: {nr+492}, {nc+943}) | Lat={data['latitude'][nr, nc]:.4f}, Lon={data['longitude'][nr, nc]:.4f} | ExactGeo={exact_geo[nr, nc]}")
                
    summary[fname] = file_info

comb_nan_counts = [int(np.count_nonzero(~data["combined_mask"])) for data in [np.load(f) for f in npz_files]]
print(f"\nCombined mask invalid counts across 7 files: {comb_nan_counts}")

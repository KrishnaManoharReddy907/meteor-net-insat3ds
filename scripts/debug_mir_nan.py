import numpy as np
import h5py

npz_path = "outputs/insat3ds_prototype/3SIMG_20OCT2026_0100_L1B_STD_V01R00_regional.npz"
data = np.load(npz_path)
k_data = data["data_kelvin"]
raw_counts = data["raw_counts"]
chan_masks = data["channel_masks"]
comb_mask = data["combined_mask"]
geo_mask = data["exact_geo_mask"]
lat = data["latitude"]
lon = data["longitude"]

mir_k = k_data[3]
nan_indices = np.where(np.isnan(mir_k))
print("NaN indices in regional crop (row, col):", nan_indices)

if len(nan_indices[0]) > 0:
    r, c = int(nan_indices[0][0]), int(nan_indices[1][0])
    global_r = r + 492
    global_c = c + 943
    print(f"\nLocal Crop Pixel (r={r}, c={c}):")
    print(f"  Global Full-Disk Pixel: (r={global_r}, c={global_c})")
    print(f"  IMG_MIR Raw DN: {raw_counts[3, r, c]}")
    print(f"  IMG_MIR Calibrated Kelvin: {mir_k[r, c]}")
    print(f"  IMG_MIR Channel Mask: {chan_masks[3, r, c]}")
    print(f"  Combined Mask: {comb_mask[r, c]}")
    print(f"  Exact Geo Mask: {geo_mask[r, c]}")
    print(f"  Decoded Latitude: {lat[r, c]:.4f} deg N")
    print(f"  Decoded Longitude: {lon[r, c]:.4f} deg E")
    print(f"  IMG_TIR1: Raw DN={raw_counts[0, r, c]}, Kelvin={k_data[0, r, c]:.2f}")
    print(f"  IMG_TIR2: Raw DN={raw_counts[1, r, c]}, Kelvin={k_data[1, r, c]:.2f}")
    print(f"  IMG_WV:   Raw DN={raw_counts[2, r, c]}, Kelvin={k_data[2, r, c]:.2f}")

    # Inspect the raw HDF5 file directly at that exact pixel
    raw_h5_path = "data/insat3ds/raw/3SIMG_20OCT2026_0100_L1B_STD_V01R00.h5"
    with h5py.File(raw_h5_path, "r") as h5:
        raw_val = h5["IMG_MIR"][0, global_r, global_c]
        lut_val = h5["IMG_MIR_TEMP"][raw_val]
        fill_attr = h5["IMG_MIR"].attrs.get("_FillValue")
        print("\nDirect Raw HDF5 Inspection:")
        print(f"  h5['IMG_MIR'][0, {global_r}, {global_c}]: {raw_val}")
        print(f"  h5['IMG_MIR'].attrs['_FillValue']: {fill_attr}")
        print(f"  h5['IMG_MIR_TEMP'][{raw_val}]: {lut_val}")

        # Check surrounding 3x3 window in raw HDF5
        window = h5["IMG_MIR"][0, global_r-1:global_r+2, global_c-1:global_c+2]
        print(f"  Surrounding 3x3 raw DN window:\n{window}")

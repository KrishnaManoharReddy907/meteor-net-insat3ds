"""
scripts/verify_insat_calibration_geo.py
Phase 2C: Scientific verification of INSAT-3DS calibration LUTs, geolocation,
masking, and temporal scan line metadata.
Read-only analysis tool.
"""

import os
import json
import datetime
import numpy as np
import h5py

RAW_DIR = "data/insat3ds/raw"
SAMPLE_FILE = os.path.join(RAW_DIR, "3SIMG_20OCT2026_0000_L1B_STD_V01R00.h5")


def verify_calibration_luts(fpath: str):
    print("=" * 80)
    print(f"1. CALIBRATION LOOKUP TABLES VERIFICATION: {os.path.basename(fpath)}")
    print("=" * 80)

    channels = ['IMG_TIR1', 'IMG_TIR2', 'IMG_WV', 'IMG_MIR']
    results = {}

    with h5py.File(fpath, 'r') as h5:
        for ch in channels:
            dset = h5[ch]
            attrs = {k: dset.attrs[k] for k in dset.attrs.keys()}
            temp_name = f"{ch}_TEMP"
            rad_name = f"{ch}_RADIANCE"

            temp_lut = h5[temp_name][:]
            rad_lut = h5[rad_name][:]
            temp_attrs = {k: h5[temp_name].attrs[k] for k in h5[temp_name].attrs.keys()}
            rad_attrs = {k: h5[rad_name].attrs[k] for k in h5[rad_name].attrs.keys()}

            # Monotonicity check
            # For Temp: invert=True means as count increases, temp decreases
            # Check diffs for count 0..1022
            temp_diffs = np.diff(temp_lut[:1023])
            is_temp_nonincreasing = np.all(temp_diffs <= 0)
            is_temp_strictly_decreasing = np.all(temp_diffs < 0)

            rad_diffs = np.diff(rad_lut[:1023])
            is_rad_nonincreasing = np.all(rad_diffs <= 0)
            is_rad_strictly_decreasing = np.all(rad_diffs < 0)

            # Specific check counts
            check_counts = [0, 1, 100, 200, 418, 500, 647, 800, 969, 1021, 1022, 1023]
            count_table = []
            for c in check_counts:
                count_table.append({
                    'count': c,
                    'temp_K': float(temp_lut[c]),
                    'rad': float(rad_lut[c]),
                })

            # Check flat regions (clipping)
            # Find where temp is max (340K) and min (180K)
            max_t = np.max(temp_lut[:1023])
            min_t = np.min(temp_lut[:1023])
            count_at_max = np.where(temp_lut[:1023] == max_t)[0]
            count_at_min = np.where(temp_lut[:1023] == min_t)[0]

            results[ch] = {
                'channel': ch,
                'attrs': {k: (attrs[k].decode('utf-8') if isinstance(attrs[k], bytes) else (attrs[k].tolist() if hasattr(attrs[k], 'tolist') else attrs[k])) for k in attrs},
                'temp_attrs': {k: (temp_attrs[k].decode('utf-8') if isinstance(temp_attrs[k], bytes) else str(temp_attrs[k])) for k in temp_attrs},
                'rad_attrs': {k: (rad_attrs[k].decode('utf-8') if isinstance(rad_attrs[k], bytes) else str(rad_attrs[k])) for k in rad_attrs},
                'temp_lut_shape': list(temp_lut.shape),
                'temp_lut_dtype': str(temp_lut.dtype),
                'is_temp_nonincreasing': bool(is_temp_nonincreasing),
                'is_temp_strictly_decreasing': bool(is_temp_strictly_decreasing),
                'temp_min': float(min_t),
                'temp_max': float(max_t),
                'count_range_at_max_temp': [int(count_at_max[0]), int(count_at_max[-1])] if len(count_at_max) > 0 else [],
                'count_range_at_min_temp': [int(count_at_min[0]), int(count_at_min[-1])] if len(count_at_min) > 0 else [],
                'is_rad_nonincreasing': bool(is_rad_nonincreasing),
                'rad_min': float(np.min(rad_lut[:1023])),
                'rad_max': float(np.max(rad_lut[:1023])),
                'count_samples': count_table,
            }

            print(f"\n--- {ch} ---")
            print(f"  Temp LUT shape: {temp_lut.shape}, dtype: {temp_lut.dtype}, units: {temp_attrs.get('units')}")
            print(f"  Rad LUT shape:  {rad_lut.shape}, dtype: {rad_lut.dtype}, units: {rad_attrs.get('units')}")
            print(f"  Temp min/max:   {min_t:.2f} K to {max_t:.2f} K")
            print(f"  Count 0:        Temp = {temp_lut[0]:.2f} K, Rad = {rad_lut[0]:.6f}")
            print(f"  Count 418:      Temp = {temp_lut[418]:.2f} K, Rad = {rad_lut[418]:.6f}")
            print(f"  Count 647:      Temp = {temp_lut[647]:.2f} K, Rad = {rad_lut[647]:.6f}")
            print(f"  Count 969:      Temp = {temp_lut[969]:.2f} K, Rad = {rad_lut[969]:.6f}")
            print(f"  Count 1022:     Temp = {temp_lut[1022]:.2f} K, Rad = {rad_lut[1022]:.6f}")
            print(f"  Count 1023:     Temp = {temp_lut[1023]:.2f} K, Rad = {rad_lut[1023]:.6f} (Fill entry)")
            print(f"  Max Temp ({max_t:.1f} K) holds for counts: {count_at_max[0]} to {count_at_max[-1]}")
            print(f"  Min Temp ({min_t:.1f} K) holds for counts: {count_at_min[0]} to {count_at_min[-1]}")
            print(f"  Radiance Scale: {attrs.get('online_radiance_scale_factor')}, Offset: {attrs.get('online_radiance_add_offset')}")
            print(f"  GSICS Scale:    {attrs.get('online_radiance_scale_factor_gsics')}, Offset: {attrs.get('online_radiance_add_offset_gsics')}")

    return results


def verify_geolocation_and_crop(fpath: str):
    print("\n" + "=" * 80)
    print(f"2. GEOGRAPHIC NAVIGATION AND REGIONAL CROP: {os.path.basename(fpath)}")
    print("=" * 80)

    with h5py.File(fpath, 'r') as h5:
        lat_raw = h5['Latitude'][:]
        lon_raw = h5['Longitude'][:]
        lat_attrs = {k: h5['Latitude'].attrs[k] for k in h5['Latitude'].attrs.keys()}
        lon_attrs = {k: h5['Longitude'].attrs[k] for k in h5['Longitude'].attrs.keys()}

        def get_scalar(attrs, key, default):
            v = attrs.get(key, default)
            if hasattr(v, 'item'):
                return v.item()
            if isinstance(v, (list, np.ndarray)) and len(v) == 1:
                return v[0]
            return v

        lat_scale = float(get_scalar(lat_attrs, 'scale_factor', 0.01))
        lat_offset = float(get_scalar(lat_attrs, 'add_offset', 0.0))
        lat_fill = int(get_scalar(lat_attrs, '_FillValue', 32767))

        lon_scale = float(get_scalar(lon_attrs, 'scale_factor', 0.01))
        lon_offset = float(get_scalar(lon_attrs, 'add_offset', 0.0))
        lon_fill = int(get_scalar(lon_attrs, '_FillValue', 32767))

        # Valid mask
        valid_geo = (lat_raw != lat_fill) & (lon_raw != lon_fill)
        valid_geo_count = int(np.count_nonzero(valid_geo))
        total_pixels = lat_raw.size

        lat_deg = np.full_like(lat_raw, np.nan, dtype=np.float32)
        lon_deg = np.full_like(lon_raw, np.nan, dtype=np.float32)

        lat_deg[valid_geo] = lat_raw[valid_geo] * lat_scale + lat_offset
        lon_deg[valid_geo] = lon_raw[valid_geo] * lon_scale + lon_offset

        lat_min = float(np.nanmin(lat_deg))
        lat_max = float(np.nanmax(lat_deg))
        lon_min = float(np.nanmin(lon_deg))
        lon_max = float(np.nanmax(lon_deg))

        print(f"  Latitude Shape:  {lat_raw.shape}, dtype: {lat_raw.dtype}, scale: {lat_scale}, fill: {lat_fill}")
        print(f"  Longitude Shape: {lon_raw.shape}, dtype: {lon_raw.dtype}, scale: {lon_scale}, fill: {lon_fill}")
        print(f"  Valid Coord Pixels: {valid_geo_count} / {total_pixels} ({valid_geo_count/total_pixels*100.2:.2f}%)")
        print(f"  Decoded Latitude Range:  [{lat_min:.4f} deg, {lat_max:.4f} deg]")
        print(f"  Decoded Longitude Range: [{lon_min:.4f} deg, {lon_max:.4f} deg]")

        # Representative Landmark Coordinates
        landmarks = {
            'Sub-Satellite Point (SSP)': (0.0, 82.0),
            'New Delhi': (28.61, 77.21),
            'Mumbai': (19.08, 72.88),
            'Chennai': (13.08, 80.27),
            'Kolkata': (22.57, 88.36),
            'Arabian Sea Center': (15.0, 65.0),
            'Bay of Bengal Center': (15.0, 90.0),
            'Cape Comorin (Kanyakumari)': (8.08, 77.55),
            'Srinagar': (34.08, 74.80),
        }

        print("\n  Representative Landmarks Pixel Coordinates (Row, Col):")
        landmark_coords = {}
        for name, (t_lat, t_lon) in landmarks.items():
            dist_sq = (lat_deg - t_lat)**2 + (lon_deg - t_lon)**2
            # Set invalid geo to infinity
            dist_sq[~valid_geo] = np.inf
            min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
            r, c = int(min_idx[0]), int(min_idx[1])
            found_lat = float(lat_deg[r, c])
            found_lon = float(lon_deg[r, c])
            err_deg = np.sqrt((found_lat - t_lat)**2 + (found_lon - t_lon)**2)
            landmark_coords[name] = {
                'target_lat': t_lat, 'target_lon': t_lon,
                'row': r, 'col': c,
                'actual_lat': found_lat, 'actual_lon': found_lon,
                'angular_err_deg': float(err_deg),
            }
            print(f"    {name:26s} Target: ({t_lat:5.2f}, {t_lon:5.2f}) -> Grid (r={r:4d}, c={c:4d}) -> Found: ({found_lat:5.2f}, {found_lon:5.2f}) [Err: {err_deg:.4f} deg]")

        # Target Region Crop Verification: Lat 5N - 37N, Lon 65E - 97E
        # Find boolean mask of pixels in target region
        in_target = valid_geo & (lat_deg >= 5.0) & (lat_deg <= 37.0) & (lon_deg >= 65.0) & (lon_deg <= 97.0)
        target_rows, target_cols = np.where(in_target)
        
        row_min, row_max = int(np.min(target_rows)), int(np.max(target_rows))
        col_min, col_max = int(np.min(target_cols)), int(np.max(target_cols))
        crop_height = row_max - row_min + 1
        crop_width = col_max - col_min + 1

        print("\n  Target Regional Bounding Box (Lat 5-37N, Lon 65-97E):")
        print(f"    Bounding Row Range: [{row_min}, {row_max}] (height = {crop_height} pixels)")
        print(f"    Bounding Col Range: [{col_min}, {col_max}] (width = {crop_width} pixels)")
        print(f"    Total Pixels in Crop Box: {crop_height * crop_width}")
        print(f"    Pixels strictly inside (5-37N, 65-97E): {int(np.count_nonzero(in_target))}")

        # Check if any pixel within [row_min:row_max+1, col_min:col_max+1] has fill/invalid coordinate
        crop_geo_valid = valid_geo[row_min:row_max+1, col_min:col_max+1]
        invalid_in_box = int(np.count_nonzero(~crop_geo_valid))
        print(f"    Invalid/Space Pixels inside rectangular crop box: {invalid_in_box} ({invalid_in_box/crop_geo_valid.size*100:.2f}%)")

        return {
            'lat_range': [lat_min, lat_max],
            'lon_range': [lon_min, lon_max],
            'valid_geo_count': valid_geo_count,
            'landmarks': landmark_coords,
            'target_crop': {
                'row_min': row_min, 'row_max': row_max,
                'col_min': col_min, 'col_max': col_max,
                'height': crop_height, 'width': crop_width,
                'invalid_pixels_in_box': invalid_in_box,
            }
        }


def verify_masks_and_cross_channel(fpath: str):
    print("\n" + "=" * 80)
    print(f"3. FULL-DISK MASK AND CROSS-CHANNEL COMPARISON: {os.path.basename(fpath)}")
    print("=" * 80)

    with h5py.File(fpath, 'r') as h5:
        lat_raw = h5['Latitude'][:]
        lat_fill = 32767
        geo_valid = (lat_raw != lat_fill)

        channels = ['IMG_TIR1', 'IMG_TIR2', 'IMG_WV', 'IMG_MIR']
        channel_masks = {}

        for ch in channels:
            raw = h5[ch][0, :, :]  # remove leading 1 dimension
            fill_v = h5[ch].attrs.get('_FillValue', 1023)
            fill_val = int(fill_v.item() if hasattr(fill_v, 'item') else (fill_v[0] if isinstance(fill_v, (list, np.ndarray)) else fill_v))
            is_fill = (raw == fill_val)
            is_valid = ~is_fill
            valid_cnt = int(np.count_nonzero(is_valid))
            fill_cnt = int(np.count_nonzero(is_fill))

            # Discrepancy with Geolocation Mask
            # 1. Geo valid but image is fill
            geo_valid_img_fill = int(np.count_nonzero(geo_valid & is_fill))
            # 2. Geo invalid (space) but image is valid
            geo_fill_img_valid = int(np.count_nonzero((~geo_valid) & is_valid))

            channel_masks[ch] = {
                'total_pixels': int(raw.size),
                'valid_pixels': valid_cnt,
                'fill_pixels': fill_cnt,
                'valid_pct': valid_cnt / raw.size * 100.0,
                'geo_valid_img_fill': geo_valid_img_fill,
                'geo_fill_img_valid': geo_fill_img_valid,
            }

            print(f"  {ch:10s} Valid Pixels: {valid_cnt:7d} ({valid_cnt/raw.size*100.0:.2f}%) | Fill Pixels: {fill_cnt:7d} ({fill_cnt/raw.size*100.0:.2f}%)")
            print(f"             Geo-Valid but Img-Fill: {geo_valid_img_fill} | Geo-Fill but Img-Valid: {geo_fill_img_valid}")

        # Check difference between TIR1 and WV
        tir1_raw = h5['IMG_TIR1'][0, :, :]
        wv_raw = h5['IMG_WV'][0, :, :]
        diff_mask = (tir1_raw != 1023) != (wv_raw != 1023)
        diff_count = int(np.count_nonzero(diff_mask))
        print(f"\n  TIR-1 vs WV Mask Discrepancy Pixels: {diff_count}")
        # Where are these differences located?
        if diff_count > 0:
            diff_rows, diff_cols = np.where(diff_mask)
            print(f"  Difference Row Range: [{np.min(diff_rows)}, {np.max(diff_rows)}]")
            print(f"  Difference Col Range: [{np.min(diff_cols)}, {np.max(diff_cols)}]")
            # Check if TIR1 is valid while WV is fill or vice versa
            tir1_valid_wv_fill = int(np.count_nonzero((tir1_raw != 1023) & (wv_raw == 1023)))
            wv_valid_tir1_fill = int(np.count_nonzero((wv_raw != 1023) & (tir1_raw == 1023)))
            print(f"  TIR1 Valid & WV Fill: {tir1_valid_wv_fill}")
            print(f"  WV Valid & TIR1 Fill: {wv_valid_tir1_fill}")

    return channel_masks


def verify_temporal_metadata(fpath: str):
    print("\n" + "=" * 80)
    print(f"4. TEMPORAL SCAN LINE METADATA: {os.path.basename(fpath)}")
    print("=" * 80)

    with h5py.File(fpath, 'r') as h5:
        attrs = h5.attrs
        acq_start = attrs.get('Acquisition_Start_Time')
        acq_end = attrs.get('Acquisition_End_Time')
        acq_gmt = attrs.get('Acquisition_Time_in_GMT')

        if isinstance(acq_start, bytes): acq_start = acq_start.decode('utf-8')
        if isinstance(acq_end, bytes): acq_end = acq_end.decode('utf-8')
        if isinstance(acq_gmt, bytes): acq_gmt = acq_gmt.decode('utf-8')

        print(f"  Nominal GMT:            {acq_gmt}")
        print(f"  Acquisition Start:      {acq_start}")
        print(f"  Acquisition End:        {acq_end}")

        if 'SCAN_LINE_TIME' in h5:
            slt = h5['SCAN_LINE_TIME'][:]
            print(f"  SCAN_LINE_TIME dataset present! Shape: {slt.shape}, dtype: {slt.dtype}")
            # Decode sample scan line times
            slt_decoded = [x.decode('utf-8').strip() if isinstance(x, bytes) else str(x) for x in slt]
            print(f"  Line 0 (North limb):    {slt_decoded[0]}")
            print(f"  Line 352 (Quarter):     {slt_decoded[352]}")
            print(f"  Line 704 (Equator/SSP): {slt_decoded[704]}")
            print(f"  Line 1056 (3-Quarter):  {slt_decoded[1056]}")
            print(f"  Line 1408 (South limb): {slt_decoded[-1]}")

            # Ratio of image rows (2816) to scan lines (1409):
            # 2816 / 1409 = 1.99858 ~ 2 lines per mirror scan sweep!
            print(f"  Scan Line Count = 1409 vs Image Rows = 2816 (Ratio = {2816/1409:.4f})")
            print("  INSAT Imager scans in two-detector swaths (each step/sweep acquires 2 image lines)!")

    return {
        'start_time': acq_start,
        'end_time': acq_end,
        'nominal_gmt': acq_gmt,
        'first_scan_line': slt_decoded[0] if 'SCAN_LINE_TIME' in h5 else None,
        'last_scan_line': slt_decoded[-1] if 'SCAN_LINE_TIME' in h5 else None,
    }


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.ndarray, list)):
            return [self.default(x) if not isinstance(x, (int, float, str, bool, type(None))) else x for x in obj]
        if isinstance(obj, (np.void, bytes)):
            return str(obj)
        try:
            return super().default(obj)
        except Exception:
            return str(obj)


def main():
    print("INSAT-3DS PHASE 2C SCIENTIFIC VERIFICATION RUNNING...")
    cal_res = verify_calibration_luts(SAMPLE_FILE)
    geo_res = verify_geolocation_and_crop(SAMPLE_FILE)
    mask_res = verify_masks_and_cross_channel(SAMPLE_FILE)
    time_res = verify_temporal_metadata(SAMPLE_FILE)

    summary_out = "reports/phase2c_verification_summary.json"
    os.makedirs(os.path.dirname(summary_out), exist_ok=True)
    with open(summary_out, "w") as f:
        json.dump({
            'calibration': cal_res,
            'geolocation': geo_res,
            'masks': mask_res,
            'temporal': time_res,
        }, f, indent=2, cls=CustomJSONEncoder)
    print(f"\nPhase 2C verification summary saved to: {summary_out}")


if __name__ == '__main__':
    main()

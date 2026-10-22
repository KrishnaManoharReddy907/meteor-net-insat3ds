"""
scripts/inspect_goes.py
Reusable inspection script for GOES-R / GOES-19 NetCDF-4 (CMIP) satellite files.
Designed for Python 3.10+ using h5py / netCDF4 / numpy.
"""

import os
import sys
import glob
import argparse
import datetime
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def parse_filename_metadata(filename: str) -> dict:
    """
    Parses standard GOES-R file naming convention:
    OR_ABI-L2-CMIPC-M6C13_G19_s20260010401193_e20260010403578_c20260010404062.nc
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    meta = {
        'filename': basename,
        'filepath': filename,
        'filesize_bytes': os.path.getsize(filename) if os.path.exists(filename) else 0,
        'filesize_mb': (os.path.getsize(filename) / (1024 * 1024)) if os.path.exists(filename) else 0,
    }
    
    if len(parts) >= 5:
        meta['system_env'] = parts[0]
        meta['product_mode_channel'] = parts[1]  # e.g. ABI-L2-CMIPC-M6C13
        meta['satellite_id'] = parts[2]           # e.g. G19
        meta['start_time_raw'] = parts[3]        # sYYYYJJJHHMMSSf
        meta['end_time_raw'] = parts[4]          # eYYYYJJJHHMMSSf
        if len(parts) >= 6:
            meta['creation_time_raw'] = parts[5].split('.')[0]
            
        # Parse channel ID
        if 'C' in meta['product_mode_channel']:
            try:
                c_idx = meta['product_mode_channel'].rfind('C')
                meta['channel_number'] = int(meta['product_mode_channel'][c_idx+1:])
            except Exception:
                meta['channel_number'] = None
                
        # Parse start timestamp
        try:
            s_str = meta['start_time_raw'][1:14]  # YYYYJJJHHMMSS
            dt_start = datetime.datetime.strptime(s_str, "%Y%j%H%M%S")
            meta['start_datetime_utc'] = dt_start
        except Exception:
            meta['start_datetime_utc'] = None
            
        # Parse end timestamp
        try:
            e_str = meta['end_time_raw'][1:14]
            dt_end = datetime.datetime.strptime(e_str, "%Y%j%H%M%S")
            meta['end_datetime_utc'] = dt_end
        except Exception:
            meta['end_datetime_utc'] = None

    return meta


def get_scalar(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, (np.ndarray, list)):
        arr = np.array(val)
        return arr.item() if arr.size == 1 else arr.tolist()
    return val


def goes_scan_to_latlon(x_rad, y_rad, h, r_eq, r_pol, lon_0_deg):
    """
    Transforms GOES ABI fixed grid scan angles (radians) into geographic Lat/Lon (degrees).
    Follows NOAA GOES-R Product Definition and User's Guide (PUG) Volume 5.
    """
    lon_0 = lon_0_deg * np.pi / 180.0
    H = h + r_eq
    a = np.sin(x_rad)**2 + np.cos(x_rad)**2 * (np.cos(y_rad)**2 + (r_eq**2 / r_pol**2) * np.sin(y_rad)**2)
    b = -2.0 * H * np.cos(x_rad) * np.cos(y_rad)
    c = H**2 - r_eq**2
    det = b**2 - 4 * a * c
    if det < 0:
        return None, None
    r_s = (-b - np.sqrt(det)) / (2 * a)
    s_x = r_s * np.cos(x_rad) * np.cos(y_rad)
    s_y = -r_s * np.sin(x_rad)
    s_z = r_s * np.cos(x_rad) * np.sin(y_rad)
    lat = np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((H - s_x)**2 + s_y**2))) * 180.0 / np.pi
    lon = (lon_0 - np.arctan(s_y / (H - s_x))) * 180.0 / np.pi
    return float(lat), float(lon)


def inspect_netcdf_file(filepath: str, compute_stats: bool = True) -> dict:
    """
    Deeply inspects a NetCDF-4 file using h5py.
    Extracts dimensions, variables, attributes, projection, and array statistics.
    """
    if h5py is None:
        raise RuntimeError("h5py is required for inspection. Please install h5py.")

    res = parse_filename_metadata(filepath)
    
    with h5py.File(filepath, 'r') as h5:
        # Global attributes
        res['global_attrs'] = {k: h5.attrs[k] for k in h5.attrs.keys()}
        
        # Dimensions / Coordinates
        dims = {}
        for coord_name in ['x', 'y', 't', 'band']:
            if coord_name in h5:
                d = h5[coord_name]
                dims[coord_name] = {
                    'shape': d.shape,
                    'dtype': str(d.dtype),
                    'attrs': {k: str(v) for k, v in d.attrs.items()}
                }
                if coord_name in ['x', 'y']:
                    scale = float(get_scalar(d.attrs.get('scale_factor', 1.0)))
                    offset = float(get_scalar(d.attrs.get('add_offset', 0.0)))
                    units = str(d.attrs.get('units', b'').decode('utf-8') if isinstance(d.attrs.get('units'), bytes) else d.attrs.get('units', ''))
                    raw_data = d[:]
                    decoded_min = float(raw_data.min()) * scale + offset
                    decoded_max = float(raw_data.max()) * scale + offset
                    dims[coord_name]['units'] = units
                    dims[coord_name]['scale_factor'] = scale
                    dims[coord_name]['add_offset'] = offset
                    dims[coord_name]['decoded_min'] = decoded_min
                    dims[coord_name]['decoded_max'] = decoded_max
                    dims[coord_name]['decoded_step'] = scale
        res['dimensions'] = dims
        
        # Band details
        if 'band_id' in h5:
            d = h5['band_id']
            res['band_id'] = int(d[()] if d.shape == () else d[0])
        if 'band_wavelength' in h5:
            d = h5['band_wavelength']
            res['band_wavelength'] = float(d[()] if d.shape == () else d[0])
            res['band_wavelength_units'] = str(h5['band_wavelength'].attrs.get('units', 'um'))
            
        # Projection information
        if 'goes_imager_projection' in h5:
            proj = h5['goes_imager_projection']
            proj_dict = {
                'grid_mapping_name': str(proj.attrs.get('grid_mapping_name', b'').decode('utf-8') if isinstance(proj.attrs.get('grid_mapping_name'), bytes) else proj.attrs.get('grid_mapping_name')),
                'perspective_point_height': float(get_scalar(proj.attrs.get('perspective_point_height', 0.0))),
                'semi_major_axis': float(get_scalar(proj.attrs.get('semi_major_axis', 0.0))),
                'semi_minor_axis': float(get_scalar(proj.attrs.get('semi_minor_axis', 0.0))),
                'inverse_flattening': float(get_scalar(proj.attrs.get('inverse_flattening', 0.0))),
                'latitude_of_projection_origin': float(get_scalar(proj.attrs.get('latitude_of_projection_origin', 0.0))),
                'longitude_of_projection_origin': float(get_scalar(proj.attrs.get('longitude_of_projection_origin', 0.0))),
            }
            res['projection'] = proj_dict
            
            # Compute center lat/lon
            if 'x' in dims and 'y' in dims:
                x_mid = 0.5 * (dims['x']['decoded_min'] + dims['x']['decoded_max'])
                y_mid = 0.5 * (dims['y']['decoded_min'] + dims['y']['decoded_max'])
                lat_c, lon_c = goes_scan_to_latlon(
                    x_mid, y_mid,
                    proj_dict['perspective_point_height'],
                    proj_dict['semi_major_axis'],
                    proj_dict['semi_minor_axis'],
                    proj_dict['longitude_of_projection_origin']
                )
                res['center_lat_lon'] = {'lat': lat_c, 'lon': lon_c}
            
        # CMI Variable
        if 'CMI' in h5:
            cmi_var = h5['CMI']

            scale_f = get_scalar(cmi_var.attrs.get('scale_factor', 1.0))
            add_o = get_scalar(cmi_var.attrs.get('add_offset', 0.0))
            fill_v = get_scalar(cmi_var.attrs.get('_FillValue', -1))
            val_rng = cmi_var.attrs.get('valid_range', [])
            if isinstance(val_rng, np.ndarray):
                val_rng = val_rng.tolist()

            cmi_meta = {
                'shape': cmi_var.shape,
                'raw_dtype': str(cmi_var.dtype),
                'units': str(cmi_var.attrs.get('units', b'').decode('utf-8') if isinstance(cmi_var.attrs.get('units'), bytes) else cmi_var.attrs.get('units')),
                'standard_name': str(cmi_var.attrs.get('standard_name', b'').decode('utf-8') if isinstance(cmi_var.attrs.get('standard_name'), bytes) else cmi_var.attrs.get('standard_name')),
                'long_name': str(cmi_var.attrs.get('long_name', b'').decode('utf-8') if isinstance(cmi_var.attrs.get('long_name'), bytes) else cmi_var.attrs.get('long_name')),
                'scale_factor': float(scale_f),
                'add_offset': float(add_o),
                '_FillValue': int(fill_v),
                'valid_range': val_rng,
            }
            
            # File precomputed stats if available
            for stat_key in ['min_brightness_temperature', 'max_brightness_temperature', 
                             'mean_brightness_temperature', 'std_dev_brightness_temperature',
                             'valid_pixel_count', 'outlier_pixel_count']:
                if stat_key in h5:
                    dset = h5[stat_key]
                    val = dset[()] if dset.shape == () else dset[0]
                    cmi_meta[f'meta_{stat_key}'] = float(val)
                    
            if compute_stats:
                raw_data = cmi_var[:]
                fill_val = cmi_meta['_FillValue']
                valid_mask = (raw_data != fill_val)
                total_pixels = raw_data.size
                valid_count = int(np.count_nonzero(valid_mask))
                fill_count = total_pixels - valid_count
                
                # Apply scale & offset to valid pixels
                calibrated_data = raw_data[valid_mask].astype(np.float32) * cmi_meta['scale_factor'] + cmi_meta['add_offset']
                
                cmi_meta['stats'] = {
                    'total_pixels': total_pixels,
                    'valid_pixels': valid_count,
                    'fill_pixels': fill_count,
                    'fill_percentage': (fill_count / total_pixels) * 100.0,
                    'valid_min_kelvin': float(np.min(calibrated_data)) if valid_count > 0 else None,
                    'valid_max_kelvin': float(np.max(calibrated_data)) if valid_count > 0 else None,
                    'valid_mean_kelvin': float(np.mean(calibrated_data)) if valid_count > 0 else None,
                    'valid_std_kelvin': float(np.std(calibrated_data)) if valid_count > 0 else None,
                    # Convective thresholds from METEOR-Net paper
                    'pixels_lt_241k': int(np.count_nonzero(calibrated_data < 241.0)) if valid_count > 0 else 0,
                    'pixels_lt_221k': int(np.count_nonzero(calibrated_data < 221.0)) if valid_count > 0 else 0,
                }
            res['CMI'] = cmi_meta
            
        # DQF Variable (Data Quality Flags)
        if 'DQF' in h5:
            dqf_var = h5['DQF']
            dqf_meta = {
                'shape': dqf_var.shape,
                'raw_dtype': str(dqf_var.dtype),
                'flag_meanings': str(dqf_var.attrs.get('flag_meanings', b'').decode('utf-8') if isinstance(dqf_var.attrs.get('flag_meanings'), bytes) else dqf_var.attrs.get('flag_meanings')),
                'flag_values': [int(v) for v in dqf_var.attrs.get('flag_values', [])] if 'flag_values' in dqf_var.attrs else [],
            }
            if compute_stats:
                dqf_data = dqf_var[:]
                unique_flags, counts = np.unique(dqf_data, return_counts=True)
                dqf_meta['flag_distribution'] = {int(k): int(v) for k, v in zip(unique_flags, counts)}
            res['DQF'] = dqf_meta

        # Summary flags
        if 'total_number_of_points' in h5:
            d = h5['total_number_of_points']
            res['total_number_of_points'] = int(d[()] if d.shape == () else d[0])
            
    return res


def scan_dataset_directory(data_dir: str):
    """
    Scans the directory structure, identifies NetCDF files,
    checks temporal ordering and cadences.
    """
    pattern = os.path.join(data_dir, "**", "*.nc")
    all_files = glob.glob(pattern, recursive=True)
    all_files.sort()
    
    print("=" * 80)
    print(f"DATASET DIRECTORY SCAN: {os.path.abspath(data_dir)}")
    print(f"Total NetCDF files discovered: {len(all_files)}")
    print("=" * 80)
    
    if not all_files:
        print("No .nc files found!")
        return []
    
    file_records = []
    by_channel = {}
    by_hour = {}
    
    for f in all_files:
        meta = parse_filename_metadata(f)
        file_records.append(meta)
        
        # Channel grouping
        chan = meta.get('channel_number', 'Unknown')
        by_channel.setdefault(chan, []).append(meta)
        
        # Hour grouping
        dt = meta.get('start_datetime_utc')
        hour_str = dt.strftime("%Y-%m-%d %H:00 UTC") if dt else "Unknown"
        by_hour.setdefault(hour_str, []).append(meta)
        
    print("\n[1] FILE COUNTS BY CHANNEL:")
    for chan, flist in sorted(by_channel.items(), key=lambda x: str(x[0])):
        print(f"  - Channel {chan}: {len(flist)} files")
        
    print("\n[2] FILE COUNTS BY UTC HOUR:")
    for h, flist in sorted(by_hour.items()):
        print(f"  - {h}: {len(flist)} files")
        
    # Check cadence for each channel
    print("\n[3] TEMPORAL CADENCE AND INTEGRITY ANALYSIS:")
    for chan, flist in sorted(by_channel.items(), key=lambda x: str(x[0])):
        # sort by start time
        flist.sort(key=lambda x: x['start_datetime_utc'] if x['start_datetime_utc'] else datetime.datetime.min)
        print(f"\n--- Channel {chan} (Total Scans: {len(flist)}) ---")
        prev_time = None
        intervals = []
        for idx, item in enumerate(flist):
            curr_time = item['start_datetime_utc']
            if prev_time and curr_time:
                delta_sec = (curr_time - prev_time).total_seconds()
                delta_min = delta_sec / 60.0
                intervals.append(delta_min)
                status = "OK (5.0m)" if abs(delta_min - 5.0) < 0.1 else f"GAP/DEVIATION ({delta_min:.1f}m)"
                print(f"  [{idx:02d}] {item['filename']} | Scan Start: {curr_time} UTC | Delta from prev: {delta_min:.1f} min [{status}]")
            else:
                print(f"  [{idx:02d}] {item['filename']} | Scan Start: {curr_time} UTC | [Initial Scan]")
            prev_time = curr_time
            
        if intervals:
            print(f"  Cadence Summary: Min interval={min(intervals):.1f}m, Max interval={max(intervals):.1f}m, Avg interval={np.mean(intervals):.1f}m")
            if all(abs(dt - 5.0) < 0.1 for dt in intervals):
                print("  Verification: PERFECT 5-MINUTE CADENCE throughout the entire continuous sequence.")
            else:
                print("  Verification: Cadence deviations or missing scans detected!")

    # Check cross-channel synchronization across timestamps
    print("\n[4] CROSS-CHANNEL SYNCHRONIZATION ANALYSIS:")
    timestamp_map = {}
    for r in file_records:
        dt = r.get('start_datetime_utc')
        chan = r.get('channel_number')
        if dt not in timestamp_map:
            timestamp_map[dt] = {}
        timestamp_map[dt][chan] = r

    expected_channels = sorted(list(by_channel.keys()))
    print(f"Expected channel set across all timestamps: {expected_channels}")
    print(f"Total distinct scan timestamps: {len(timestamp_map)}")
    
    sync_perfect = True
    for idx, (dt, chan_dict) in enumerate(sorted(timestamp_map.items())):
        present = sorted(list(chan_dict.keys()))
        missing = [c for c in expected_channels if c not in present]
        status = "SYNCHRONIZED" if not missing else f"MISSING CHANNELS: {missing}"
        if missing:
            sync_perfect = False
            print(f"  [{idx:02d}] {dt} UTC | Channels: {present} | STATUS: {status}")
            
    if sync_perfect:
        print(f"  --> VERIFIED: All {len(timestamp_map)} timestamps have 100% of channels {expected_channels} synchronized with zero missing channels.")
    else:
        print("  --> WARNING: One or more timestamps have missing or mismatched channels!")
                
    return file_records, by_channel


def print_representative_inspection(filepath: str):
    """Prints a detailed report of a representative file."""
    print("\n" + "=" * 80)
    print(f"REPRESENTATIVE FILE INSPECTION: {os.path.basename(filepath)}")
    print("=" * 80)
    
    info = inspect_netcdf_file(filepath, compute_stats=True)
    
    print(f"File Path: {info['filepath']}")
    print(f"File Size: {info['filesize_mb']:.2f} MB ({info['filesize_bytes']} bytes)")
    print(f"Satellite: {info.get('satellite_id', 'N/A')}")
    print(f"Product:   {info.get('product_mode_channel', 'N/A')}")
    print(f"Scan Start: {info.get('start_datetime_utc')} UTC (Raw: {info.get('start_time_raw')})")
    print(f"Scan End:   {info.get('end_datetime_utc')} UTC (Raw: {info.get('end_time_raw')})")
    
    print("\n--- Spectral Band Attributes ---")
    print(f"Band ID:         Channel {info.get('band_id')}")
    print(f"Central Lambda:  {info.get('band_wavelength')} {info.get('band_wavelength_units')}")
    
    print("\n--- Spatial Dimensions & Coordinate Grids ---")
    for dname, dinfo in info.get('dimensions', {}).items():
        if 'decoded_min' in dinfo:
            print(f"  {dname:5s}: shape={dinfo['shape']} dtype={dinfo['dtype']} scan_angle_rad=[{dinfo['decoded_min']:.6f}, {dinfo['decoded_max']:.6f}] {dinfo.get('units', '')} step={dinfo['decoded_step']:.6f}")
        else:
            print(f"  {dname:5s}: shape={dinfo['shape']} dtype={dinfo['dtype']}")
            
    print("\n--- Projection and Geonavigation Metadata ---")
    proj = info.get('projection', {})
    for pk, pv in proj.items():
        print(f"  {pk}: {pv}")
    if 'center_lat_lon' in info and info['center_lat_lon']['lat'] is not None:
        c_lat = info['center_lat_lon']['lat']
        c_lon = info['center_lat_lon']['lon']
        print(f"  Center Sub-satellite Point Coverage: Lat = {c_lat:.4f}° N, Lon = {c_lon:.4f}° E ({abs(c_lon):.4f}° W)")
    print("  Navigation Feasibility: FIXED GRID (ABI geostationary projection). Lat/Lon can be calculated")
    print("  analytically from scan angles (x, y) using PROJ / pyproj or standard GOES-R equations.")
    
    print("\n--- CMI Variable (Calibrated Clean IR Brightness Temperature) ---")
    cmi = info.get('CMI', {})
    print(f"  Standard Name:  {cmi.get('standard_name')}")
    print(f"  Long Name:      {cmi.get('long_name')}")
    print(f"  Units:          {cmi.get('units')}")
    print(f"  Raw Dtype:      {cmi.get('raw_dtype')} (2D Grid Shape: {cmi.get('shape')})")
    print(f"  Scale Factor:   {cmi.get('scale_factor')}")
    print(f"  Add Offset:     {cmi.get('add_offset')}")
    print(f"  _FillValue:     {cmi.get('_FillValue')}")
    print(f"  Valid Range:    {cmi.get('valid_range')}")
    
    if 'stats' in cmi:
        st = cmi['stats']
        print("\n--- Valid Pixel Statistics & Thermal Distribution ---")
        print(f"  Total Grid Pixels:   {st['total_pixels']:,}")
        print(f"  Valid Data Pixels:   {st['valid_pixels']:,} ({100.0 - st['fill_percentage']:.2f}%)")
        print(f"  Fill/Space Pixels:   {st['fill_pixels']:,} ({st['fill_percentage']:.2f}%)")
        print(f"  Min Brightness Temp: {st['valid_min_kelvin']:.2f} K")
        print(f"  Max Brightness Temp: {st['valid_max_kelvin']:.2f} K")
        print(f"  Mean Brightness Temp:{st['valid_mean_kelvin']:.2f} K")
        print(f"  Std Dev:             {st['valid_std_kelvin']:.2f} K")
        print(f"  Convective (< 241K): {st['pixels_lt_241k']:,} pixels (Moderate Convection)")
        print(f"  Deep Convective (< 221K): {st['pixels_lt_221k']:,} pixels (Vigorous Convection)")
        
    print("\n--- Data Quality Flags (DQF) ---")
    dqf = info.get('DQF', {})
    print(f"  Meanings: {dqf.get('flag_meanings')}")
    print(f"  Values:   {dqf.get('flag_values')}")
    print(f"  Distribution: {dqf.get('flag_distribution')}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect GOES-19 NetCDF satellite files.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory containing downloaded NetCDF files.")
    parser.add_argument("--file", type=str, default=None, help="Inspect a specific NetCDF file.")
    args = parser.parse_args()
    
    if args.file:
        print_representative_inspection(args.file)
    else:
        file_records, by_channel = scan_dataset_directory(args.data_dir)
        if by_channel:
            print("\n" + "#" * 80)
            print("REPRESENTATIVE FILE INSPECTION FOR EACH AVAILABLE CHANNEL")
            print("#" * 80)
            for chan in sorted(by_channel.keys()):
                rep_file = by_channel[chan][0]['filepath']
                print_representative_inspection(rep_file)


if __name__ == '__main__':
    main()

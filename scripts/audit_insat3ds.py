"""
scripts/audit_insat3ds.py
Standalone, read-only audit tool for INSAT-3DS Imager Level-1B (3SIMG_L1B_STD) HDF5 data.
Audits file integrity, HDF5 structure, metadata, temporal cadence, calibration tables,
and valid pixel statistics without modifying or converting raw files.
"""

import os
import sys
import glob
import datetime
import json
import numpy as np
import h5py

HDF5_MAGIC_BYTES = b'\x89HDF\r\n\x1a\n'


def decode_attr(val):
    """Safely decodes h5py attributes to Python primitives."""
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='ignore')
    if isinstance(val, (np.ndarray, list)):
        arr = np.array(val)
        if arr.size == 1:
            item = arr.item()
            if isinstance(item, bytes):
                return item.decode('utf-8', errors='ignore')
            return item
        return [decode_attr(x) for x in arr]
    return val


def audit_single_insat_file(filepath: str) -> dict:
    """
    Performs a strict read-only audit on a single INSAT-3DS HDF5 file.
    """
    basename = os.path.basename(filepath)
    size_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    size_mb = size_bytes / (1024 * 1024)
    ext = os.path.splitext(basename)[1].lower()

    record = {
        'filename': basename,
        'filepath': filepath,
        'filesize_bytes': size_bytes,
        'filesize_mb': size_mb,
        'extension': ext,
        'is_nonzero': size_bytes > 0,
        'magic_bytes_valid': False,
        'h5py_readable': False,
        'corruption_error': None,
        'root_attributes': {},
        'datasets': {},
        'cadence_info': {},
    }

    # 1. Check Magic Bytes
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(8)
            record['magic_bytes_valid'] = (magic == HDF5_MAGIC_BYTES)
            record['magic_bytes_hex'] = magic.hex()
    except Exception as e:
        record['corruption_error'] = f"Failed to read file header: {str(e)}"
        return record

    if not record['magic_bytes_valid']:
        record['corruption_error'] = f"Invalid magic bytes: {record.get('magic_bytes_hex')}"
        return record

    # 2. Open with h5py
    try:
        with h5py.File(filepath, 'r') as h5:
            record['h5py_readable'] = True
            
            # Root attributes
            attrs = {}
            for k in h5.attrs.keys():
                attrs[k] = decode_attr(h5.attrs[k])
            record['root_attributes'] = attrs
            
            # Key acquisition timestamps
            record['satellite_name'] = attrs.get('Satellite_Name')
            record['sensor_name'] = attrs.get('Sensor_Name')
            record['sensor_id'] = attrs.get('Sensor_Id')
            record['product_type'] = attrs.get('Product_Type')
            record['processing_level'] = attrs.get('Processing_Level')
            record['product_file_name_attr'] = attrs.get('HDF_Product_File_Name')
            record['unique_id'] = attrs.get('Unique_Id')
            record['nominal_time_gmt'] = attrs.get('Acquisition_Time_in_GMT')
            record['acquisition_date'] = attrs.get('Acquisition_Date')
            record['start_time_str'] = attrs.get('Acquisition_Start_Time')
            record['end_time_str'] = attrs.get('Acquisition_End_Time')
            record['product_creation_time'] = attrs.get('Product_Creation_Time')
            
            # Parse start and end timestamps into datetime objects
            # Format in INSAT: '20-OCT-2026T00:00:26.414'
            if record['start_time_str']:
                try:
                    s_clean = record['start_time_str'].split('.')[0]
                    record['start_datetime'] = datetime.datetime.strptime(s_clean, '%d-%b-%YT%H:%M:%S')
                except Exception:
                    record['start_datetime'] = None
            if record['end_time_str']:
                try:
                    e_clean = record['end_time_str'].split('.')[0]
                    record['end_datetime'] = datetime.datetime.strptime(e_clean, '%d-%b-%YT%H:%M:%S')
                except Exception:
                    record['end_datetime'] = None

            # Sub-satellite point coordinates
            record['nominal_coordinates_lat_lon'] = attrs.get('Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude')
            
            # 3. Recursive dataset inventory
            datasets_info = {}
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    d_meta = {
                        'path': f"/{name}",
                        'shape': list(obj.shape),
                        'dtype': str(obj.dtype),
                        'size': int(obj.size),
                        'ndim': int(obj.ndim),
                        'attrs': {ak: decode_attr(obj.attrs[ak]) for ak in obj.attrs.keys()}
                    }
                    datasets_info[name] = d_meta

            h5.visititems(visitor)
            record['datasets'] = datasets_info
            
            # 4. Specific checks on image datasets (TIR1, TIR2, WV, MIR, VIS, SWIR)
            channel_stats = {}
            for ch_name in ['IMG_TIR1', 'IMG_TIR2', 'IMG_WV', 'IMG_MIR', 'IMG_VIS', 'IMG_SWIR']:
                if ch_name in h5:
                    dset = h5[ch_name]
                    fill_val = decode_attr(dset.attrs.get('_FillValue'))
                    total_px = int(dset.size)
                    
                    # For TIR and WV (primary IR bands, ~7.9M pixels), load full array
                    # For VIS and SWIR (huge optical bands, 126M pixels), sample 1/16 pixels to be responsive and memory-safe
                    if ch_name in ['IMG_VIS', 'IMG_SWIR']:
                        raw_data = dset[0, ::4, ::4]
                        sampled = True
                    else:
                        raw_data = dset[:]
                        sampled = False
                    
                    if fill_val is not None:
                        fill_count_sample = int(np.count_nonzero(raw_data == fill_val))
                    else:
                        fill_count_sample = 0
                    
                    sample_size = raw_data.size
                    valid_count_sample = sample_size - fill_count_sample
                    valid_pct = (valid_count_sample / sample_size * 100.0) if sample_size > 0 else 0.0
                    
                    if valid_count_sample > 0:
                        valid_mask = (raw_data != fill_val) if fill_val is not None else np.ones_like(raw_data, dtype=bool)
                        v_min = int(np.min(raw_data[valid_mask]))
                        v_max = int(np.max(raw_data[valid_mask]))
                        v_mean = float(np.mean(raw_data[valid_mask]))
                    else:
                        v_min, v_max, v_mean = None, None, None

                    channel_stats[ch_name] = {
                        'shape': list(dset.shape),
                        'dtype': str(dset.dtype),
                        'central_wavelength': decode_attr(dset.attrs.get('central_wavelength')),
                        'bandwidth': decode_attr(dset.attrs.get('bandwidth')),
                        'resolution_km': decode_attr(dset.attrs.get('resolution')),
                        'fill_value': fill_val,
                        'total_pixels': total_px,
                        'is_sampled_for_stats': sampled,
                        'sample_size': sample_size,
                        'valid_percentage': valid_pct,
                        'raw_min_dn': v_min,
                        'raw_max_dn': v_max,
                        'raw_mean_dn': v_mean,
                    }
            record['channel_stats'] = channel_stats

            # 5. Calibration Lookup Tables
            lut_info = {}
            for lut_name in ['IMG_TIR1_TEMP', 'IMG_TIR1_RADIANCE', 'IMG_TIR2_TEMP', 'IMG_TIR2_RADIANCE', 
                            'IMG_WV_TEMP', 'IMG_WV_RADIANCE', 'IMG_MIR_TEMP', 'IMG_MIR_RADIANCE',
                            'IMG_VIS_ALBEDO', 'IMG_SWIR_ALBEDO', 'GreyCount']:
                if lut_name in h5:
                    lut_dset = h5[lut_name]
                    lut_data = lut_dset[:]
                    lut_info[lut_name] = {
                        'shape': list(lut_dset.shape),
                        'dtype': str(lut_dset.dtype),
                        'units': decode_attr(lut_dset.attrs.get('units')),
                        'min_value': float(np.nanmin(lut_data)),
                        'max_value': float(np.nanmax(lut_data)),
                        'first_5': [float(x) for x in lut_data[:5]],
                        'last_5': [float(x) for x in lut_data[-5:]],
                    }
            record['lut_info'] = lut_info

            # 6. Geolocation Datasets
            geo_info = {}
            for g_name in ['Latitude', 'Longitude', 'Latitude_VIS', 'Longitude_VIS', 'GeoX', 'GeoY']:
                if g_name in h5:
                    gdset = h5[g_name]
                    g_scale = decode_attr(gdset.attrs.get('scale_factor', 1.0))
                    g_offset = decode_attr(gdset.attrs.get('add_offset', 0.0))
                    g_fill = decode_attr(gdset.attrs.get('_FillValue'))
                    geo_info[g_name] = {
                        'shape': list(gdset.shape),
                        'dtype': str(gdset.dtype),
                        'scale_factor': g_scale,
                        'add_offset': g_offset,
                        'fill_value': g_fill,
                        'units': decode_attr(gdset.attrs.get('units')),
                    }
                    if g_name in ['Latitude', 'Longitude']:
                        g_raw = gdset[:]
                        g_valid = (g_raw != g_fill) if g_fill is not None else np.ones_like(g_raw, dtype=bool)
                        v_cnt = int(np.count_nonzero(g_valid))
                        if v_cnt > 0:
                            g_deg = g_raw[g_valid].astype(np.float32) * float(g_scale) + float(g_offset)
                            geo_info[g_name]['valid_min_deg'] = float(np.min(g_deg))
                            geo_info[g_name]['valid_max_deg'] = float(np.max(g_deg))
                            geo_info[g_name]['valid_pixels'] = v_cnt
            record['geo_info'] = geo_info

    except Exception as e:
        record['corruption_error'] = f"Failed to read HDF5 contents: {str(e)}"

    return record


def run_insat_sample_audit(raw_dir: str):
    """
    Scans, audits, and computes cadence for all INSAT-3DS HDF5 files in directory.
    """
    pattern = os.path.join(raw_dir, "*.h5")
    file_list = sorted(glob.glob(pattern))
    
    print("=" * 80, flush=True)
    print(f"INSAT-3DS RAW DATA SAMPLE AUDIT: {os.path.abspath(raw_dir)}", flush=True)
    print(f"Total .h5 files found: {len(file_list)}", flush=True)
    print("=" * 80, flush=True)

    if not file_list:
        print("No .h5 files found in directory!", flush=True)
        return None

    audit_records = []
    for f in file_list:
        rec = audit_single_insat_file(f)
        audit_records.append(rec)
        print(f"  [{'PASS' if rec['h5py_readable'] else 'FAIL'}] {rec['filename']} ({rec['filesize_mb']:.2f} MB)", flush=True)

    # Sort records by verified internal start_datetime
    audit_records.sort(key=lambda x: x.get('start_datetime') or datetime.datetime.min)

    # Compute consecutive cadences
    print("\n" + "=" * 80, flush=True)
    print("ACQUISITION TIMESTAMPS AND CADENCE VERIFICATION", flush=True)
    print("=" * 80, flush=True)

    cadence_table = []
    prev_dt = None
    for idx, rec in enumerate(audit_records):
        curr_dt = rec.get('start_datetime')
        end_dt = rec.get('end_datetime')
        gmt_nominal = rec.get('nominal_time_gmt')
        
        delta_min = None
        duration_min = None
        
        if curr_dt and end_dt:
            duration_sec = (end_dt - curr_dt).total_seconds()
            duration_min = duration_sec / 60.0
            
        if prev_dt and curr_dt:
            diff_sec = (curr_dt - prev_dt).total_seconds()
            delta_min = diff_sec / 60.0
            status = "NOMINAL (~30m)" if abs(delta_min - 30.0) < 1.0 else f"IRREGULAR ({delta_min:.1f}m)"
        else:
            status = "FIRST SCAN"

        row = {
            'index': idx,
            'filename': rec['filename'],
            'nominal_gmt': gmt_nominal,
            'start_utc': str(curr_dt),
            'end_utc': str(end_dt),
            'scan_duration_min': duration_min,
            'interval_from_prev_min': delta_min,
            'status': status,
        }
        cadence_table.append(row)
        
        print(f"[{idx:02d}] {rec['filename']} | Start: {curr_dt} UTC | End: {end_dt} UTC | Duration: {duration_min:.1f}m | Delta: {delta_min}m | Status: {status}", flush=True)
        prev_dt = curr_dt

    return {
        'file_records': audit_records,
        'cadence_table': cadence_table,
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
    raw_dir = "data/insat3ds/raw"
    res = run_insat_sample_audit(raw_dir)
    if res:
        out_json = "reports/insat3ds_audit_summary.json"
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2, cls=CustomJSONEncoder)
        print(f"\nAudit summary successfully saved to: {out_json}", flush=True)


if __name__ == '__main__':
    main()

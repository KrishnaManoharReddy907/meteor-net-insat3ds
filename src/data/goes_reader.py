"""
src/data/goes_reader.py
Module for reading, calibrating, and geolocating GOES-R / GOES-19 NetCDF-4 (CMIP) satellite files.
Supports single-file and multi-channel synchronized timestamp reading.
"""

import os
import glob
import json
import hashlib
import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import h5py


def get_scalar(val, default=0.0):
    """Safely extracts a Python scalar from h5py datasets or attributes."""
    if val is None:
        return default
    if isinstance(val, (np.ndarray, list)):
        arr = np.array(val)
        return arr.item() if arr.size == 1 else arr.tolist()
    return val


def decode_string(val, default=""):
    """Safely decodes bytes or string attributes."""
    if val is None:
        return default
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    return str(val)


class GOESProjection:
    """
    Handles forward and inverse geostationary projection transformations
    following NOAA GOES-R Product Definition and User's Guide (PUG) Volume 5.
    """
    def __init__(
        self,
        perspective_point_height=35786023.0,
        semi_major_axis=6378137.0,
        semi_minor_axis=6356752.31414,
        lon_0_deg=-75.0,
        x_scale=5.6000000768108293e-05,
        x_offset=-0.10133200138807297,
        y_scale=-5.6000000768108293e-05,
        y_offset=0.12821200489997864
    ):
        self.h = float(perspective_point_height)
        self.r_eq = float(semi_major_axis)
        self.r_pol = float(semi_minor_axis)
        self.lon_0_deg = float(lon_0_deg)
        self.lon_0 = self.lon_0_deg * np.pi / 180.0
        self.H = self.h + self.r_eq
        self.e2 = (self.r_eq**2 - self.r_pol**2) / (self.r_eq**2)
        
        self.x_scale = float(x_scale)
        self.x_offset = float(x_offset)
        self.y_scale = float(y_scale)
        self.y_offset = float(y_offset)

    @classmethod
    def from_h5(cls, h5_file):
        """Constructs projection directly from NetCDF/H5 dataset attributes."""
        proj = h5_file["goes_imager_projection"].attrs
        h = get_scalar(proj.get("perspective_point_height", 35786023.0))
        r_eq = get_scalar(proj.get("semi_major_axis", 6378137.0))
        r_pol = get_scalar(proj.get("semi_minor_axis", 6356752.31414))
        lon_0 = get_scalar(proj.get("longitude_of_projection_origin", -75.0))
        
        x_attrs = h5_file["x"].attrs
        x_scale = get_scalar(x_attrs.get("scale_factor", 5.6e-5))
        x_offset = get_scalar(x_attrs.get("add_offset", -0.101332))
        
        y_attrs = h5_file["y"].attrs
        y_scale = get_scalar(y_attrs.get("scale_factor", -5.6e-5))
        y_offset = get_scalar(y_attrs.get("add_offset", 0.128212))
        
        return cls(
            perspective_point_height=h,
            semi_major_axis=r_eq,
            semi_minor_axis=r_pol,
            lon_0_deg=lon_0,
            x_scale=x_scale,
            x_offset=x_offset,
            y_scale=y_scale,
            y_offset=y_offset
        )

    def scan_to_latlon(self, x_rad, y_rad):
        """Converts fixed grid scan angles (radians) to Latitude and Longitude (degrees)."""
        a = np.sin(x_rad)**2 + np.cos(x_rad)**2 * (np.cos(y_rad)**2 + (self.r_eq**2 / self.r_pol**2) * np.sin(y_rad)**2)
        b = -2.0 * self.H * np.cos(x_rad) * np.cos(y_rad)
        c = self.H**2 - self.r_eq**2
        det = b**2 - 4 * a * c
        
        if isinstance(det, np.ndarray):
            valid = (det >= 0)
            r_s = np.zeros_like(det, dtype=np.float64)
            r_s[valid] = (-b[valid] - np.sqrt(det[valid])) / (2 * a[valid])
            
            s_x = r_s * np.cos(x_rad) * np.cos(y_rad)
            s_y = -r_s * np.sin(x_rad)
            s_z = r_s * np.cos(x_rad) * np.sin(y_rad)
            
            lat = np.full_like(det, np.nan, dtype=np.float64)
            lon = np.full_like(det, np.nan, dtype=np.float64)
            
            lat[valid] = np.arctan((self.r_eq**2 / self.r_pol**2) * (s_z[valid] / np.sqrt((self.H - s_x[valid])**2 + s_y[valid]**2))) * 180.0 / np.pi
            lon[valid] = (self.lon_0 - np.arctan(s_y[valid] / (self.H - s_x[valid]))) * 180.0 / np.pi
            return lat, lon
        else:
            if det < 0:
                return None, None
            r_s = (-b - np.sqrt(det)) / (2 * a)
            s_x = r_s * np.cos(x_rad) * np.cos(y_rad)
            s_y = -r_s * np.sin(x_rad)
            s_z = r_s * np.cos(x_rad) * np.sin(y_rad)
            lat = np.arctan((self.r_eq**2 / self.r_pol**2) * (s_z / np.sqrt((self.H - s_x)**2 + s_y**2))) * 180.0 / np.pi
            lon = (self.lon_0 - np.arctan(s_y / (self.H - s_x))) * 180.0 / np.pi
            return float(lat), float(lon)

    def latlon_to_scan(self, lat_deg, lon_deg):
        """Converts geographic Latitude and Longitude (degrees) to fixed grid scan angles (radians)."""
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        phi_geo = np.arctan((self.r_pol**2 / self.r_eq**2) * np.tan(lat))
        r_c = self.r_pol / np.sqrt(1.0 - self.e2 * (np.cos(phi_geo)**2))
        s_x = self.H - r_c * np.cos(phi_geo) * np.cos(lon - self.lon_0)
        s_y = -r_c * np.cos(phi_geo) * np.sin(lon - self.lon_0)
        s_z = r_c * np.sin(phi_geo)
        
        x = np.arcsin(-s_y / np.sqrt(s_x**2 + s_y**2 + s_z**2))
        y = np.arctan(s_z / s_x)
        return float(x), float(y)

    def latlon_to_rowcol(self, lat_deg, lon_deg):
        """Converts Lat/Lon directly to floating-point (row, col) grid coordinates."""
        x, y = self.latlon_to_scan(lat_deg, lon_deg)
        col = (x - self.x_offset) / self.x_scale
        row = (y - self.y_offset) / self.y_scale
        return float(row), float(col)

    def rowcol_to_latlon(self, row, col):
        """Converts (row, col) pixel indices to Lat/Lon."""
        x = col * self.x_scale + self.x_offset
        y = row * self.y_scale + self.y_offset
        return self.scan_to_latlon(x, y)


def read_goes_single_file(filepath: str, return_quality_mask: bool = True):
    """
    Reads a single GOES-19 Level 2+ CMIP file.
    Decodes raw packed int16 values into physical Brightness Temperature (Kelvin).
    
    Returns:
        dict containing:
            - 'data_kelvin': 2D float32 array of shape (1500, 2500)
            - 'valid_mask': 2D bool array (True for valid Earth observations)
            - 'channel': int channel number (e.g. 13)
            - 'wavelength': float central wavelength in um
            - 'start_datetime': datetime object
            - 'end_datetime': datetime object
            - 'scale_factor': float
            - 'add_offset': float
            - 'fill_value': int
            - 'dqf': 2D int8 array
            - 'projection': GOESProjection object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, "r") as h5:
        # Verify product and variable
        if "CMI" not in h5:
            raise ValueError(f"Expected 'CMI' dataset not found in {filepath}")
        
        cmi_dset = h5["CMI"]
        raw_cmi = cmi_dset[:]
        
        # Calibration attributes
        scale = float(get_scalar(cmi_dset.attrs.get("scale_factor", 1.0)))
        offset = float(get_scalar(cmi_dset.attrs.get("add_offset", 0.0)))
        fill_val = int(get_scalar(cmi_dset.attrs.get("_FillValue", -1)))
        
        # Decode to physical Kelvin
        valid_mask = (raw_cmi != fill_val)
        
        # Allocate float32 array; fill invalid with NaN
        data_k = np.full(raw_cmi.shape, np.nan, dtype=np.float32)
        data_k[valid_mask] = raw_cmi[valid_mask].astype(np.float32) * scale + offset
        
        # Read DQF
        dqf = h5["DQF"][:] if "DQF" in h5 else None
        if dqf is not None and return_quality_mask:
            # DQF == 0 is good quality
            valid_mask = valid_mask & (dqf == 0)
            # Update data_k NaN mask to reflect strict DQF
            data_k[~valid_mask] = np.nan
            
        # Band details
        chan_id = int(get_scalar(h5["band_id"][()])) if "band_id" in h5 else None
        wavelength = float(get_scalar(h5["band_wavelength"][()])) if "band_wavelength" in h5 else None
        
        # Parse timestamps from filename
        basename = os.path.basename(filepath)
        parts = basename.split("_")
        s_dt = datetime.datetime.strptime(parts[3][1:14], "%Y%j%H%M%S")
        e_dt = datetime.datetime.strptime(parts[4][1:14], "%Y%j%H%M%S")
        
        proj = GOESProjection.from_h5(h5)
        
        return {
            "data_kelvin": data_k,
            "valid_mask": valid_mask,
            "channel": chan_id,
            "wavelength": wavelength,
            "start_datetime": s_dt,
            "end_datetime": e_dt,
            "scale_factor": scale,
            "add_offset": offset,
            "fill_value": fill_val,
            "dqf": dqf,
            "projection": proj,
            "filepath": filepath,
        }


def read_multichannel_timestamp(file_dict: dict, channel_order=(9, 13, 14, 15)):
    """
    Reads and stacks synchronized channel files for a single timestamp.
    
    Args:
        file_dict: dict mapping channel number (int) -> filepath (str)
        channel_order: tuple/list of channel numbers in desired output order
        
    Returns:
        dict containing:
            - 'data_kelvin': 3D float32 array of shape (C, H, W)
            - 'valid_mask': 2D bool array (intersection of valid masks across all channels)
            - 'channels': list of channel numbers
            - 'start_datetime': datetime object
            - 'projection': GOESProjection object
    """
    missing = [c for c in channel_order if c not in file_dict]
    if missing:
        raise ValueError(f"Missing required channels for timestamp: {missing}")

    chan_data = []
    combined_mask = None
    common_dt = None
    common_proj = None
    
    for c in channel_order:
        res = read_goes_single_file(file_dict[c], return_quality_mask=True)
        chan_data.append(res["data_kelvin"])
        
        if combined_mask is None:
            combined_mask = res["valid_mask"]
            common_dt = res["start_datetime"]
            common_proj = res["projection"]
        else:
            combined_mask = combined_mask & res["valid_mask"]
            if res["start_datetime"] != common_dt:
                raise ValueError(f"Timestamp mismatch: Channel {c} has {res['start_datetime']}, expected {common_dt}")

    # Stack along channel axis: shape (C, H, W)
    stacked = np.stack(chan_data, axis=0)
    
    return {
        "data_kelvin": stacked,
        "valid_mask": combined_mask,
        "channels": list(channel_order),
        "start_datetime": common_dt,
        "projection": common_proj,
    }


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 cryptographic hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_timestamp_catalog(
    data_dir: str,
    channels=(9, 13, 14, 15),
    on_conflict: str = "raise",
    audit_report_path: Optional[str] = None,
    return_audit: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Scans a directory recursively and groups verified files into synchronized timestamps.
    Performs explicit duplicate detection, hash verification, and conflict handling.

    Args:
        data_dir: Root directory containing GOES NetCDF files.
        channels: Tuple of required channel numbers.
        on_conflict: Action when conflicting files exist for same timestamp/channel:
                     'raise' raises ValueError;
                     'exclude' drops the conflicting channel/timestamp from catalog.
        audit_report_path: Optional path to save a JSON audit report of duplicates and conflicts.
        return_audit: If True, returns (catalog, audit_summary). If False, returns catalog.

    Returns:
        Sorted list of dicts: [{'datetime': dt, 'channels': {9: path, 13: path, ...}}, ...]
        (or tuple with audit_summary if return_audit=True).
    """
    pattern = os.path.join(data_dir, "**", "*.nc")
    all_files = glob.glob(pattern, recursive=True)

    # Collect all paths mapped to (s_dt, chan_num)
    raw_candidates: Dict[Tuple[datetime.datetime, int], List[str]] = {}
    for f in all_files:
        base = os.path.basename(f)
        parts = base.split("_")
        if len(parts) < 5:
            continue
        try:
            prod = parts[1]
            chan_str = prod.split("-")[-1]
            if not chan_str.startswith("M6C"):
                continue
            chan_num = int(chan_str[3:])
            if chan_num not in channels:
                continue
            s_dt = datetime.datetime.strptime(parts[3][1:14], "%Y%j%H%M%S")
        except Exception:
            continue

        key = (s_dt, chan_num)
        if key not in raw_candidates:
            raw_candidates[key] = []
        raw_candidates[key].append(f)

    # Audit records
    exact_duplicates = []
    conflicting_files = []
    resolved_catalog: Dict[datetime.datetime, Dict[int, str]] = {}

    for (s_dt, chan_num), path_list in raw_candidates.items():
        if len(path_list) == 1:
            if s_dt not in resolved_catalog:
                resolved_catalog[s_dt] = {}
            resolved_catalog[s_dt][chan_num] = path_list[0]
            continue

        # Multiple files for same (s_dt, chan_num) -> inspect sizes and hashes
        file_info = []
        for p in path_list:
            size_b = os.path.getsize(p)
            sha = compute_file_sha256(p)
            file_info.append({"path": p, "size_bytes": size_b, "sha256": sha})

        # Check if all files have identical size and SHA-256
        first_sha = file_info[0]["sha256"]
        first_size = file_info[0]["size_bytes"]
        is_exact_dup = all(item["sha256"] == first_sha and item["size_bytes"] == first_size for item in file_info)

        if is_exact_dup:
            # Deterministically choose lexicographically first path
            chosen_path = sorted(path_list)[0]
            if s_dt not in resolved_catalog:
                resolved_catalog[s_dt] = {}
            resolved_catalog[s_dt][chan_num] = chosen_path
            exact_duplicates.append({
                "datetime": s_dt.isoformat(),
                "channel": chan_num,
                "paths": path_list,
                "size_bytes": first_size,
                "sha256": first_sha,
                "chosen_path": chosen_path,
                "action": "deduplicated"
            })
        else:
            # Files have different sizes or hashes -> genuine conflict
            conflict_record = {
                "datetime": s_dt.isoformat(),
                "channel": chan_num,
                "files": file_info,
                "action": "raised" if on_conflict == "raise" else "excluded"
            }
            conflicting_files.append(conflict_record)

            if on_conflict == "raise":
                msg = (
                    f"Conflicting files detected for timestamp {s_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} "
                    f"and channel {chan_num}:\n"
                    + "\n".join(f"  - {f['path']} (size={f['size_bytes']}, sha256={f['sha256'][:12]}...)" for f in file_info)
                )
                raise ValueError(msg)
            elif on_conflict == "exclude":
                # Exclude this channel, preventing incomplete timestamp from entering valid catalog
                pass
            else:
                raise ValueError(f"Unknown on_conflict policy: '{on_conflict}'. Must be 'raise' or 'exclude'.")

    # Filter for timestamps having all required channels
    valid_catalog = []
    for dt in sorted(resolved_catalog.keys()):
        flist = resolved_catalog[dt]
        if all(c in flist for c in channels):
            valid_catalog.append({
                "datetime": dt,
                "channels": flist,
            })

    audit_summary = {
        "timestamp_scan": datetime.datetime.now().isoformat(),
        "data_dir": os.path.abspath(data_dir),
        "total_files_scanned": len(all_files),
        "unique_timestamp_channel_slots": len(raw_candidates),
        "exact_duplicate_slots": len(exact_duplicates),
        "conflicting_slots": len(conflicting_files),
        "valid_synchronized_timestamps": len(valid_catalog),
        "exact_duplicates": exact_duplicates,
        "conflicts": conflicting_files,
    }

    if audit_report_path:
        os.makedirs(os.path.dirname(os.path.abspath(audit_report_path)), exist_ok=True)
        with open(audit_report_path, "w") as jf:
            json.dump(audit_summary, jf, indent=2)

    if return_audit:
        return valid_catalog, audit_summary
    return valid_catalog


def audit_catalog_duplicates(
    data_dir: str,
    channels=(9, 13, 14, 15),
    output_json: Optional[str] = None
) -> Dict[str, Any]:
    """
    Dedicated audit utility to scan a data directory for exact duplicates and conflicts
    without crashing, returning the audit summary dict and optionally saving JSON.
    """
    _, audit_summary = build_timestamp_catalog(
        data_dir=data_dir,
        channels=channels,
        on_conflict="exclude",
        audit_report_path=output_json,
        return_audit=True
    )
    return audit_summary

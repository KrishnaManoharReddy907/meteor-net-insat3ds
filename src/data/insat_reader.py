"""
src/data/insat_reader.py
Modular reader, radiometric calibrator, and geographic cropper for INSAT-3DS
Level-1B Standard Full Disk (3SIMG_L1B_STD) HDF5 satellite imagery.

Strictly preserves raw counts, performs LUT-based calibration to Kelvin,
explicitly masks space background (DN 1023), and handles scan-line timing.
"""

import os
import datetime
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import h5py

IR_CHANNELS = ["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_MIR"]
OPTICAL_CHANNELS = ["IMG_VIS", "IMG_SWIR"]

# Standard enclosing rectangular bounds for Indian subcontinent and surrounding seas
DEFAULT_ROW_SLICE = slice(492, 1272)   # 780 rows (indices 492 to 1271)
DEFAULT_COL_SLICE = slice(943, 1810)   # 867 cols (indices 943 to 1809)
DEFAULT_LAT_BOUNDS = (5.0, 37.0)
DEFAULT_LON_BOUNDS = (65.0, 97.0)


def get_scalar(val, default=None):
    """Safely extracts a Python scalar from h5py datasets or attributes."""
    if val is None:
        return default
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (np.ndarray, list)):
        arr = np.array(val)
        return arr.item() if arr.size == 1 else arr.tolist()
    return val


def decode_string(val, default="") -> str:
    """Safely decodes bytes or string attributes."""
    if val is None:
        return default
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip()
    return str(val).strip()


def calibrate_counts_to_kelvin(
    raw_counts: np.ndarray,
    temp_lut: np.ndarray,
    fill_value: int = 1023,
    min_valid_dn: int = 0,
    max_valid_dn: int = 1022
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrates raw 10-bit integer digital numbers to brightness temperature (Kelvin)
    using the embedded ISRO Level-1B lookup table.

    CRITICAL SAFEGUARD:
    DN 1023 is the declared space/fill value. In the raw ISRO LUT, index 1023 contains
    180.0 K. This function treats DN 1023 (and any counts outside [min_valid_dn, max_valid_dn])
    as strictly invalid, assigning NaN to calibrated pixels and False to the valid mask.

    Args:
        raw_counts: 2D or 3D numpy array of integer digital numbers (uint16).
        temp_lut: 1D numpy array of length 1024 with brightness temperatures in Kelvin.
        fill_value: Declared fill count (default 1023).
        min_valid_dn: Minimum valid digital count (default 0).
        max_valid_dn: Maximum valid digital count (default 1022).

    Returns:
        calibrated_kelvin: float32 array with physical temperatures in Kelvin, NaN for invalid.
        valid_mask: boolean array (True where measurement is valid Earth observation).
    """
    if temp_lut.shape != (1024,):
        raise ValueError(f"Expected 1024-element temperature LUT, got shape {temp_lut.shape}")

    # Explicit validity mask: count must be within [0, 1022] and not equal to fill_value
    valid_mask = (raw_counts >= min_valid_dn) & (raw_counts <= max_valid_dn) & (raw_counts != fill_value)

    # Allocate float32 output array initialized with NaN
    calibrated_kelvin = np.full(raw_counts.shape, np.nan, dtype=np.float32)

    # Safe indexed lookup only for strictly valid pixels
    if np.any(valid_mask):
        valid_counts = raw_counts[valid_mask]
        calibrated_kelvin[valid_mask] = temp_lut[valid_counts]

    return calibrated_kelvin, valid_mask


def read_insat_channel(
    h5_file: h5py.File,
    channel_name: str,
    apply_calibration: bool = True
) -> Dict:
    """
    Reads a single image dataset, its raw counts, calibration table, and metadata.

    Args:
        h5_file: Open h5py.File handle.
        channel_name: Channel name (e.g., 'IMG_TIR1', 'IMG_WV').
        apply_calibration: Whether to apply LUT calibration to Kelvin.

    Returns:
        Dictionary containing raw counts, calibrated kelvin (if requested),
        valid mask, lookup table, and metadata.
    """
    if channel_name not in h5_file:
        raise KeyError(f"Channel {channel_name} not found in HDF5 file.")

    dset = h5_file[channel_name]
    # Squeeze leading dimensions if shape is (1, H, W)
    raw_counts = np.squeeze(dset[:])
    if raw_counts.dtype != np.uint16:
        raw_counts = raw_counts.astype(np.uint16)

    attrs = dset.attrs
    fill_val = int(get_scalar(attrs.get("_FillValue", 1023)))
    wavelength = float(get_scalar(attrs.get("central_wavelength", np.nan)))
    bandwidth = float(get_scalar(attrs.get("bandwidth", np.nan)))
    resolution = float(get_scalar(attrs.get("resolution", np.nan)))

    # Lookup table verification
    lut_name = f"{channel_name}_TEMP"
    temp_lut = None
    calibrated_kelvin = None
    valid_mask = None

    if lut_name in h5_file:
        temp_lut = h5_file[lut_name][:].astype(np.float32)
        if apply_calibration:
            calibrated_kelvin, valid_mask = calibrate_counts_to_kelvin(
                raw_counts=raw_counts,
                temp_lut=temp_lut,
                fill_value=fill_val
            )
    else:
        # If no temperature LUT (e.g. VIS channel), build mask from fill value
        valid_mask = (raw_counts != fill_val)

    return {
        "channel_name": channel_name,
        "raw_counts": raw_counts,
        "data_kelvin": calibrated_kelvin,
        "valid_mask": valid_mask,
        "temp_lut": temp_lut,
        "fill_value": fill_val,
        "central_wavelength_um": wavelength,
        "bandwidth_um": bandwidth,
        "resolution_km": resolution,
        "units": "K" if calibrated_kelvin is not None else "counts",
        "calibration_method": "embedded_lut" if calibrated_kelvin is not None else "raw_counts",
        "fill_policy": "mask_1023_as_nan",
    }


def read_insat_multichannel_frame(
    filepath: str,
    channels: Tuple[str, ...] = tuple(IR_CHANNELS),
    apply_calibration: bool = True,
    load_coords: bool = False,
    load_scan_line_time: bool = True
) -> Dict:
    """
    Reads all requested infrared channels for a single INSAT-3DS Level-1B file.

    Args:
        filepath: Path to INSAT-3DS .h5 file.
        channels: Tuple or list of channel names (default: TIR1, TIR2, WV, MIR).
        apply_calibration: If True, calibrates each channel to Kelvin.
        load_coords: If True, decodes and includes Latitude and Longitude grids.
        load_scan_line_time: If True, reads and includes /SCAN_LINE_TIME array.

    Returns:
        Dictionary containing multichannel stacked arrays and metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"INSAT-3DS file not found: {filepath}")

    with h5py.File(filepath, "r") as h5:
        attrs = h5.attrs
        satellite_name = decode_string(attrs.get("Satellite_Name", "INSAT-3DS"))
        sensor_name = decode_string(attrs.get("Sensor_Name", "IMAGER"))
        product_type = decode_string(attrs.get("Product_Type", "STANDARD (FULL DISK)"))
        processing_level = decode_string(attrs.get("Processing_Level", "L1B"))
        unique_id = decode_string(attrs.get("Unique_Id", ""))
        acq_gmt = decode_string(attrs.get("Acquisition_Time_in_GMT", ""))
        start_time_str = decode_string(attrs.get("Acquisition_Start_Time", ""))
        end_time_str = decode_string(attrs.get("Acquisition_End_Time", ""))

        start_dt = None
        end_dt = None
        if start_time_str:
            try:
                s_clean = start_time_str.split(".")[0]
                start_dt = datetime.datetime.strptime(s_clean, "%d-%b-%YT%H:%M:%S")
            except Exception:
                start_dt = None
        if end_time_str:
            try:
                e_clean = end_time_str.split(".")[0]
                end_dt = datetime.datetime.strptime(e_clean, "%d-%b-%YT%H:%M:%S")
            except Exception:
                end_dt = None

        # Read each requested channel
        raw_list = []
        kelvin_list = []
        mask_list = []
        lut_dict = {}

        for ch in channels:
            ch_res = read_insat_channel(h5, ch, apply_calibration=apply_calibration)
            raw_list.append(ch_res["raw_counts"])
            if apply_calibration:
                kelvin_list.append(ch_res["data_kelvin"])
            mask_list.append(ch_res["valid_mask"])
            lut_dict[ch] = ch_res["temp_lut"]

        # Stack into (C, H, W)
        raw_stacked = np.stack(raw_list, axis=0)
        kelvin_stacked = np.stack(kelvin_list, axis=0) if apply_calibration else None
        masks_stacked = np.stack(mask_list, axis=0)
        # Combined mask: True only where ALL channels have valid data
        combined_mask = np.all(masks_stacked, axis=0)

        # Geodetic coordinates
        lat_deg = None
        lon_deg = None
        geo_valid = None
        if load_coords:
            lat_raw = h5["Latitude"][:]
            lon_raw = h5["Longitude"][:]
            lat_scale = float(get_scalar(h5["Latitude"].attrs.get("scale_factor", 0.01)))
            lon_scale = float(get_scalar(h5["Longitude"].attrs.get("scale_factor", 0.01)))
            fill_lat = int(get_scalar(h5["Latitude"].attrs.get("_FillValue", 32767)))

            geo_valid = (lat_raw != fill_lat) & (lon_raw != fill_lat)
            lat_deg = np.full(lat_raw.shape, np.nan, dtype=np.float32)
            lon_deg = np.full(lon_raw.shape, np.nan, dtype=np.float32)
            lat_deg[geo_valid] = lat_raw[geo_valid] * lat_scale
            lon_deg[geo_valid] = lon_raw[geo_valid] * lon_scale

        # Scan line time
        scan_line_times = None
        if load_scan_line_time and "SCAN_LINE_TIME" in h5:
            slt_raw = h5["SCAN_LINE_TIME"][:]
            scan_line_times = [decode_string(x) for x in slt_raw]

        return {
            "source_file": os.path.basename(filepath),
            "filepath": filepath,
            "satellite_name": satellite_name,
            "sensor_name": sensor_name,
            "product_type": product_type,
            "processing_level": processing_level,
            "unique_id": unique_id,
            "nominal_gmt": acq_gmt,
            "start_datetime": start_dt,
            "end_datetime": end_dt,
            "channels": list(channels),
            "raw_counts": raw_stacked,            # (C, H, W), uint16
            "data_kelvin": kelvin_stacked,        # (C, H, W), float32
            "channel_masks": masks_stacked,       # (C, H, W), bool
            "combined_mask": combined_mask,       # (H, W), bool
            "luts": lut_dict,
            "latitude": lat_deg,                  # (H, W), float32 or None
            "longitude": lon_deg,                 # (H, W), float32 or None
            "geo_valid": geo_valid,               # (H, W), bool or None
            "scan_line_times": scan_line_times,   # list of str or None
        }


class INSATRegionalCropper:
    """
    Handles cropping of the Indian subcontinent regional target domain from
    INSAT-3DS Full Disk images.
    """
    def __init__(
        self,
        row_slice: slice = DEFAULT_ROW_SLICE,
        col_slice: slice = DEFAULT_COL_SLICE,
        lat_bounds: Tuple[float, float] = DEFAULT_LAT_BOUNDS,
        lon_bounds: Tuple[float, float] = DEFAULT_LON_BOUNDS
    ):
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds

    def compute_exact_geographic_mask(
        self,
        lat_crop: np.ndarray,
        lon_crop: np.ndarray
    ) -> np.ndarray:
        """
        Computes boolean mask of pixels strictly inside [lat_min, lat_max] and [lon_min, lon_max].
        """
        valid_coords = (~np.isnan(lat_crop)) & (~np.isnan(lon_crop))
        in_bounds = (
            valid_coords &
            (lat_crop >= self.lat_bounds[0]) &
            (lat_crop <= self.lat_bounds[1]) &
            (lon_crop >= self.lon_bounds[0]) &
            (lon_crop <= self.lon_bounds[1])
        )
        return in_bounds

    def crop_multichannel_frame(self, frame: Dict) -> Dict:
        """
        Extracts the regional bounding crop from a multichannel frame dictionary.
        """
        r_sl = self.row_slice
        c_sl = self.col_slice

        raw_crop = frame["raw_counts"][:, r_sl, c_sl]
        kelvin_crop = frame["data_kelvin"][:, r_sl, c_sl] if frame["data_kelvin"] is not None else None
        masks_crop = frame["channel_masks"][:, r_sl, c_sl]
        combined_mask_crop = frame["combined_mask"][r_sl, c_sl]

        lat_crop = frame["latitude"][r_sl, c_sl] if frame["latitude"] is not None else None
        lon_crop = frame["longitude"][r_sl, c_sl] if frame["longitude"] is not None else None

        exact_geo_mask = None
        if lat_crop is not None and lon_crop is not None:
            exact_geo_mask = self.compute_exact_geographic_mask(lat_crop, lon_crop)

        # Slice relevant scan line times
        # 1409 sweeps for 2816 rows: sweep index = row // 2
        crop_slt = None
        if frame["scan_line_times"] is not None:
            start_sweep = r_sl.start // 2
            end_sweep = (r_sl.stop - 1) // 2
            crop_slt = frame["scan_line_times"][start_sweep:end_sweep + 1]

        return {
            "source_file": frame["source_file"],
            "unique_id": frame["unique_id"],
            "nominal_gmt": frame["nominal_gmt"],
            "start_datetime": frame["start_datetime"],
            "end_datetime": frame["end_datetime"],
            "channels": frame["channels"],
            "raw_counts": raw_crop,                      # (C, H_crop, W_crop)
            "data_kelvin": kelvin_crop,                  # (C, H_crop, W_crop)
            "channel_masks": masks_crop,                 # (C, H_crop, W_crop)
            "combined_mask": combined_mask_crop,         # (H_crop, W_crop)
            "latitude": lat_crop,                        # (H_crop, W_crop)
            "longitude": lon_crop,                       # (H_crop, W_crop)
            "exact_geo_mask": exact_geo_mask,            # (H_crop, W_crop)
            "scan_line_times": crop_slt,
            "crop_rows": [r_sl.start, r_sl.stop - 1],
            "crop_cols": [c_sl.start, c_sl.stop - 1],
            "crop_shape": (r_sl.stop - r_sl.start, c_sl.stop - c_sl.start),
        }


def combine_preprocessing_masks(
    radiometric_mask: np.ndarray,
    exact_geo_mask: np.ndarray,
    mode: str = "intersection"
) -> np.ndarray:
    """
    Explicitly combines radiometric validity and geographic masks for downstream model use.

    Modes:
      - 'intersection': Pixel must have both valid radiometric measurement AND be inside exact geographic bounds.
      - 'radiometric_only': Keeps all valid radiometric Earth pixels (including corner margin pixels).
      - 'geographic_only': Strict geographic bounding regardless of radiometric measurement.

    Args:
        radiometric_mask: Boolean array indicating valid radiometric pixels (e.g. non-fill, finite).
        exact_geo_mask: Boolean array indicating pixels strictly within geographic target bounds.
        mode: Combination rule ('intersection', 'radiometric_only', 'geographic_only').

    Returns:
        Boolean mask array of matching spatial dimensions.
    """
    if mode == "intersection":
        return radiometric_mask & exact_geo_mask
    elif mode == "radiometric_only":
        return radiometric_mask.copy()
    elif mode == "geographic_only":
        return exact_geo_mask.copy()
    else:
        raise ValueError(
            f"Unknown mask combination mode: '{mode}'. "
            "Must be 'intersection', 'radiometric_only', or 'geographic_only'."
        )


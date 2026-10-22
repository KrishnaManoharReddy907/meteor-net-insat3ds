"""
src/data/sequence_generator.py
Module for building, validating, and extracting temporal sequences for METEOR-Net.
Implements the approved 30-to-15-minute sequence design with configurable temporal offsets.
"""

import datetime
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.data.goes_reader import read_multichannel_timestamp
from src.data.spatial_crop import GeographicCropper, extract_patches


class TemporalSequenceGenerator:
    """
    Builds temporal sequence windows from a catalog of synchronized observation timestamps.
    Enforces exact timestamp delta checks and zero-tolerance for missing frames.
    """
    def __init__(
        self,
        catalog: List[Dict],
        input_offsets_min: Tuple[int, ...] = (-60, -30, 0, 30),
        target_offsets_min: Tuple[int, ...] = (15,),
        nominal_step_min: float = 5.0,
        channel_order: Tuple[int, ...] = (9, 13, 14, 15),
        cropper: Optional[GeographicCropper] = None
    ):
        """
        Args:
            catalog: sorted list of timestamp dicts: [{'datetime': dt, 'channels': {9: path, ...}}, ...]
            input_offsets_min: offsets in minutes from anchor observation
            target_offsets_min: offsets in minutes for intermediate ground truth target(s)
            nominal_step_min: expected sensor observation cadence in minutes (5.0 for GOES CONUS)
            channel_order: ordered channels for multichannel tensors
            cropper: optional GeographicCropper to crop frames spatially
        """
        self.catalog = sorted(catalog, key=lambda x: x["datetime"])
        self.input_offsets_min = sorted(list(input_offsets_min))
        self.target_offsets_min = sorted(list(target_offsets_min))
        self.nominal_step_min = float(nominal_step_min)
        self.channel_order = tuple(channel_order)
        self.cropper = cropper or GeographicCropper()
        
        # Build lookup table by datetime
        self.dt_map = {item["datetime"]: item for item in self.catalog}
        
        # Discover all valid sequence windows
        self.valid_windows = self._find_valid_windows()

    def _find_valid_windows(self) -> List[Dict]:
        """
        Scans all catalog observations as potential anchors and checks if all
        required input and target timestamps exist with exact temporal deltas.
        """
        windows = []
        all_required_offsets = sorted(self.input_offsets_min + self.target_offsets_min)
        
        for item in self.catalog:
            anchor_dt = item["datetime"]
            window_record = {
                "anchor_datetime": anchor_dt,
                "input_frames": [],
                "target_frames": [],
                "all_timestamps": {},
                "is_valid": True,
            }
            
            # Check inputs
            for off in self.input_offsets_min:
                expected_dt = anchor_dt + datetime.timedelta(minutes=off)
                if expected_dt in self.dt_map:
                    window_record["input_frames"].append({
                        "offset_min": off,
                        "datetime": expected_dt,
                        "channels": self.dt_map[expected_dt]["channels"],
                    })
                    window_record["all_timestamps"][off] = expected_dt
                else:
                    window_record["is_valid"] = False
                    break
                    
            if not window_record["is_valid"]:
                continue
                
            # Check targets
            for off in self.target_offsets_min:
                expected_dt = anchor_dt + datetime.timedelta(minutes=off)
                if expected_dt in self.dt_map:
                    window_record["target_frames"].append({
                        "offset_min": off,
                        "datetime": expected_dt,
                        "channels": self.dt_map[expected_dt]["channels"],
                    })
                    window_record["all_timestamps"][off] = expected_dt
                else:
                    window_record["is_valid"] = False
                    break
                    
            if window_record["is_valid"]:
                windows.append(window_record)
                
        return windows

    def get_summary(self) -> Dict:
        """Returns statistical summary of catalog and discovered sequence windows."""
        total_scans = len(self.catalog)
        return {
            "total_observations": total_scans,
            "first_timestamp": self.catalog[0]["datetime"] if self.catalog else None,
            "last_timestamp": self.catalog[-1]["datetime"] if self.catalog else None,
            "input_offsets_min": self.input_offsets_min,
            "target_offsets_min": self.target_offsets_min,
            "total_valid_windows": len(self.valid_windows),
            "channel_order": list(self.channel_order),
        }

    def load_window_tensors(self, window_idx: int) -> Dict:
        """
        Loads the multichannel arrays for a specific window, applies geographic crop,
        and packages into input and target tensors.
        
        Returns:
            dict containing:
                - 'input_tensors': (N_in, C, H, W) float32 array in Kelvin
                - 'target_tensors': (N_tgt, C, H, W) float32 array in Kelvin
                - 'input_timestamps': list of datetimes
                - 'target_timestamps': list of datetimes
                - 'combined_valid_mask': (H, W) bool array
        """
        if window_idx < 0 or window_idx >= len(self.valid_windows):
            raise IndexError(f"Window index {window_idx} out of range (0 to {len(self.valid_windows)-1})")
            
        win = self.valid_windows[window_idx]
        
        # Load inputs
        in_arrays = []
        in_dts = []
        combined_mask = None
        
        for item in win["input_frames"]:
            res = read_multichannel_timestamp(item["channels"], self.channel_order)
            data_crop = self.cropper.crop_array(res["data_kelvin"])
            mask_crop = self.cropper.crop_array(res["valid_mask"])
            
            in_arrays.append(data_crop)
            in_dts.append(item["datetime"])
            combined_mask = mask_crop if combined_mask is None else (combined_mask & mask_crop)
            
        # Load targets
        tgt_arrays = []
        tgt_dts = []
        for item in win["target_frames"]:
            res = read_multichannel_timestamp(item["channels"], self.channel_order)
            data_crop = self.cropper.crop_array(res["data_kelvin"])
            mask_crop = self.cropper.crop_array(res["valid_mask"])
            
            tgt_arrays.append(data_crop)
            tgt_dts.append(item["datetime"])
            combined_mask = combined_mask & mask_crop
            
        # Stack: (N_in, C, H, W) and (N_tgt, C, H, W)
        in_tensor = np.stack(in_arrays, axis=0)
        tgt_tensor = np.stack(tgt_arrays, axis=0)
        
        return {
            "input_tensors": in_tensor,
            "target_tensors": tgt_tensor,
            "input_timestamps": in_dts,
            "target_timestamps": tgt_dts,
            "combined_valid_mask": combined_mask,
            "anchor_datetime": win["anchor_datetime"],
        }

    def extract_sequence_patches(
        self,
        window_idx: int,
        patch_size: int = 128,
        stride: int = 64,
        require_full_valid: bool = True
    ) -> List[Dict]:
        """
        Extracts synchronized (128 x 128) patch sequences across all temporal frames of a window.
        Every patch extracted is aligned in spatial position across all input and target steps!
        
        Returns:
            List of dicts:
                - 'input_patches': (N_in, C, patch_size, patch_size) float32
                - 'target_patches': (N_tgt, C, patch_size, patch_size) float32
                - 'row_slice': (r_start, r_end)
                - 'col_slice': (c_start, c_end)
                - 'anchor_datetime': datetime
                - 'is_convective': bool
                - 'convective_ratio_241': float
        """
        win_data = self.load_window_tensors(window_idx)
        in_tensors = win_data["input_tensors"]    # (4, C, H, W)
        tgt_tensors = win_data["target_tensors"]  # (1, C, H, W)
        mask = win_data["combined_valid_mask"]     # (H, W)
        
        N_in, C, H, W = in_tensors.shape
        N_tgt = tgt_tensors.shape[0]
        
        # Channel 13 index (typically index 1 in [9, 13, 14, 15])
        ch13_idx = self.channel_order.index(13) if 13 in self.channel_order else 0
        
        patch_sequences = []
        
        for r in range(0, H - patch_size + 1, stride):
            for c in range(0, W - patch_size + 1, stride):
                r_end = r + patch_size
                c_end = c + patch_size
                
                # Check spatial mask validity across all frames
                p_mask = mask[r:r_end, c:c_end]
                if require_full_valid and not np.all(p_mask):
                    continue
                    
                p_in = in_tensors[:, :, r:r_end, c:c_end]
                p_tgt = tgt_tensors[:, :, r:r_end, c:c_end]
                
                # Check for NaNs
                if np.any(np.isnan(p_in)) or np.any(np.isnan(p_tgt)):
                    continue
                    
                # Evaluate convective activity on the target midpoint frame (Channel 13)
                target_ch13 = p_tgt[0, ch13_idx]
                conv_pixels = np.count_nonzero(target_ch13 < 241.0)
                deep_pixels = np.count_nonzero(target_ch13 < 221.0)
                conv_ratio = conv_pixels / (patch_size * patch_size)
                
                is_convective = (conv_ratio >= 0.02) or (deep_pixels >= 5)
                
                patch_sequences.append({
                    "input_patches": p_in,
                    "target_patches": p_tgt,
                    "row_slice": (r, r_end),
                    "col_slice": (c, c_end),
                    "anchor_datetime": win_data["anchor_datetime"],
                    "is_convective": is_convective,
                    "convective_ratio_241": float(conv_ratio),
                    "deep_convective_count_221": int(deep_pixels),
                    "target_min_temp_k": float(np.min(target_ch13)),
                    "target_mean_temp_k": float(np.mean(target_ch13)),
                })
                
        return patch_sequences

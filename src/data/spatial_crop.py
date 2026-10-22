"""
src/data/spatial_crop.py
Module for geographic bounding box cropping, valid-pixel filtering,
and 128x128 patch extraction on multichannel satellite grids.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from src.data.goes_reader import GOESProjection


class GeographicCropper:
    """
    Crops geographic latitude/longitude bounding boxes from GOES-R fixed grid arrays.
    """
    def __init__(
        self,
        lat_min: float = 25.0,
        lat_max: float = 35.0,
        lon_min: float = -95.0,
        lon_max: float = -80.0,
        projection: Optional[GOESProjection] = None
    ):
        self.lat_min = float(lat_min)
        self.lat_max = float(lat_max)
        self.lon_min = float(lon_min)
        self.lon_max = float(lon_max)
        self.proj = projection or GOESProjection()
        
        # Compute row and col bounds
        self.r_min, self.r_max, self.c_min, self.c_max = self._compute_pixel_bounds()

    def _compute_pixel_bounds(self) -> Tuple[int, int, int, int]:
        """Calculates bounding pixel indices enclosing the geographic bounding box."""
        corners = [
            (self.lat_min, self.lon_min),  # SW
            (self.lat_min, self.lon_max),  # SE
            (self.lat_max, self.lon_min),  # NW
            (self.lat_max, self.lon_max),  # NE
        ]
        
        rows = []
        cols = []
        for lat, lon in corners:
            r, c = self.proj.latlon_to_rowcol(lat, lon)
            rows.append(r)
            cols.append(c)
            
        r_min = int(np.floor(min(rows)))
        r_max = int(np.ceil(max(rows)))
        c_min = int(np.floor(min(cols)))
        c_max = int(np.ceil(max(cols)))
        
        # Clamp to CONUS bounds (1500, 2500)
        r_min = max(0, r_min)
        r_max = min(1500, r_max)
        c_min = max(0, c_min)
        c_max = min(2500, c_max)
        
        return r_min, r_max, c_min, c_max

    def crop_array(self, data: np.ndarray) -> np.ndarray:
        """
        Crops an array of shape (H, W) or (C, H, W).
        """
        if data.ndim == 2:
            return data[self.r_min:self.r_max, self.c_min:self.c_max]
        elif data.ndim == 3:
            return data[:, self.r_min:self.r_max, self.c_min:self.c_max]
        elif data.ndim == 4:
            return data[:, :, self.r_min:self.r_max, self.c_min:self.c_max]
        else:
            raise ValueError(f"Unsupported array dimensionality: {data.ndim}")

    def get_crop_info(self) -> Dict:
        """Returns metadata regarding the geographic crop."""
        h = self.r_max - self.r_min
        w = self.c_max - self.c_min
        return {
            "lat_bounds": [self.lat_min, self.lat_max],
            "lon_bounds": [self.lon_min, self.lon_max],
            "row_bounds": [self.r_min, self.r_max],
            "col_bounds": [self.c_min, self.c_max],
            "crop_shape": (h, w),
            "total_pixels": h * w,
        }


def extract_patches(
    multichannel_crop: np.ndarray,
    valid_mask_crop: np.ndarray,
    patch_size: int = 128,
    stride: int = 64,
    channel_13_idx: int = 1,
    convective_threshold: float = 241.0,
    min_convective_ratio: float = 0.02,
    min_deep_pixels: int = 5,
    deep_threshold: float = 221.0,
    require_full_valid: bool = True
) -> List[Dict]:
    """
    Extracts fixed (patch_size x patch_size) patches from a cropped multichannel array.
    Filters out patches with space fill pixels and ranks/tags convective activity.
    
    Args:
        multichannel_crop: array of shape (C, H, W)
        valid_mask_crop: bool array of shape (H, W)
        patch_size: default 128
        stride: default 64 (50% overlap for comprehensive coverage)
        channel_13_idx: index of Channel 13 in multichannel_crop (default 1 for [9, 13, 14, 15])
        convective_threshold: Kelvin threshold for moderate convection (241.0 K)
        min_convective_ratio: minimum fraction of pixels < 241 K to qualify as convective
        min_deep_pixels: minimum pixels < 221 K to qualify as deep convective
        require_full_valid: if True, rejects patches containing any invalid/NaN/fill pixels
        
    Returns:
        List of dicts for each valid patch:
            - 'patch_data': (C, patch_size, patch_size) float32 array
            - 'patch_mask': (patch_size, patch_size) bool array
            - 'row_slice': (r_start, r_end)
            - 'col_slice': (c_start, c_end)
            - 'convective_ratio_241': float
            - 'deep_convective_count_221': int
            - 'is_convective': bool
    """
    C, H, W = multichannel_crop.shape
    patches = []
    
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            r_end = r + patch_size
            c_end = c + patch_size
            
            p_data = multichannel_crop[:, r:r_end, c:c_end]
            p_mask = valid_mask_crop[r:r_end, c:c_end]
            
            # Check valid mask completeness
            if require_full_valid:
                if not np.all(p_mask) or np.any(np.isnan(p_data)):
                    continue
            else:
                valid_ratio = np.mean(p_mask)
                if valid_ratio < 0.95:
                    continue
                    
            # Convective analysis on Channel 13
            ch13 = p_data[channel_13_idx]
            conv_pixels = np.count_nonzero(ch13 < convective_threshold)
            deep_pixels = np.count_nonzero(ch13 < deep_threshold)
            conv_ratio = conv_pixels / (patch_size * patch_size)
            
            is_convective = (conv_ratio >= min_convective_ratio) or (deep_pixels >= min_deep_pixels)
            
            patches.append({
                "patch_data": p_data,
                "patch_mask": p_mask,
                "row_slice": (r, r_end),
                "col_slice": (c, c_end),
                "convective_ratio_241": float(conv_ratio),
                "deep_convective_count_221": int(deep_pixels),
                "is_convective": is_convective,
                "min_temp_k": float(np.nanmin(ch13)),
                "max_temp_k": float(np.nanmax(ch13)),
                "mean_temp_k": float(np.nanmean(ch13)),
            })
            
    return patches

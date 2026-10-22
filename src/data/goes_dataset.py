"""
src/data/goes_dataset.py
Production-grade PyTorch Dataset for METEOR-Net temporal super-resolution.

Loads sequences defined in outputs/dataset_splits/splits_manifest.json on-demand:
- Enforces strict event-level split isolation (Train, Val, Test).
- Performs lazy, on-the-fly NetCDF reading with a bounded LRU cache.
- Prevents accidental loading of the 4 GB full dataset into RAM.
- Extracts synchronized [4, 4, 128, 128] input tensors, [4, 128, 128] targets,
  and [128, 128] validity masks.
- Preserves physical Kelvin ground-truth for mask-aware convective loss.
"""

import os
import json
from collections import OrderedDict
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.goes_reader import read_multichannel_timestamp
from src.data.spatial_crop import GeographicCropper
from src.models.normalization import ChannelStandardizer


class GOESSequencePatchDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for METEOR-Net sequence patches.
    
    Loads multi-channel GOES-19 observations lazily per sequence window,
    extracts spatial patches (128x128), and delivers standardized or physical tensors.
    """
    def __init__(
        self,
        manifest_path: str = "outputs/dataset_splits/splits_manifest.json",
        split: str = "train",
        patch_size: int = 128,
        stride: int = 64,
        channel_order: Tuple[int, ...] = (9, 13, 14, 15),
        cropper: Optional[GeographicCropper] = None,
        standardizer: Optional[ChannelStandardizer] = None,
        normalize_inputs: bool = False,
        require_full_valid: bool = True,
        convective_only: bool = False,
        cache_size: int = 2
    ):
        """
        Args:
            manifest_path: Path to splits_manifest.json.
            split: Partition to load ('train', 'val', or 'test').
            patch_size: Height/width of square patches (128).
            stride: Step size between adjacent spatial patches (e.g. 64 or 128).
            channel_order: Order of ABI channels (default: 9, 13, 14, 15).
            cropper: Optional GeographicCropper instance.
            standardizer: Optional ChannelStandardizer for input normalization.
            normalize_inputs: If True and standardizer is provided, normalizes inputs to ~N(0, 1).
            require_full_valid: If True, rejects patches with invalid/space/limb pixels.
            convective_only: If True, only returns patches meeting convective criteria.
            cache_size: Number of decoded sequence windows to retain in RAM (LRU).
        """
        super().__init__()
        self.manifest_path = manifest_path
        self.split = split.lower()
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.channel_order = tuple(channel_order)
        self.cropper = cropper or GeographicCropper()
        self.standardizer = standardizer
        self.normalize_inputs = bool(normalize_inputs and standardizer is not None)
        self.require_full_valid = bool(require_full_valid)
        self.convective_only = bool(convective_only)
        self.cache_size = max(1, int(cache_size))

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        if self.split not in self.manifest["splits"]:
            raise ValueError(f"Split '{self.split}' not found in manifest splits: {list(self.manifest['splits'].keys())}")

        self.sequence_entries: List[Dict[str, Any]] = self.manifest["splits"][self.split]
        
        # Bounded LRU cache for decoded sequence windows: {seq_idx: window_tensors_dict}
        self._window_cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()

        # Compute grid dimensions from cropper
        crop_h = self.cropper.r_max - self.cropper.r_min
        crop_w = self.cropper.c_max - self.cropper.c_min
        
        # Pre-compute spatial grid offsets
        self.row_offsets = list(range(0, crop_h - self.patch_size + 1, self.stride))
        self.col_offsets = list(range(0, crop_w - self.patch_size + 1, self.stride))
        self.patches_per_window = len(self.row_offsets) * len(self.col_offsets)

        # Index all items: (seq_idx, r_start, c_start)
        self.items: List[Tuple[int, int, int]] = []
        for s_idx in range(len(self.sequence_entries)):
            for r in self.row_offsets:
                for c in self.col_offsets:
                    self.items.append((s_idx, r, c))

    def __len__(self) -> int:
        return len(self.items)

    def _load_window_tensors(self, seq_idx: int) -> Dict[str, Any]:
        """
        Loads and decodes the 5 observation timestamps for a sequence window.
        Uses an LRU cache so adjacent patches reuse in-memory arrays.
        """
        if seq_idx in self._window_cache:
            self._window_cache.move_to_end(seq_idx)
            return self._window_cache[seq_idx]

        seq_info = self.sequence_entries[seq_idx]
        
        # 1. Load 4 input frames
        in_arrays = []
        in_masks = []
        for f_info in seq_info["source_files"]["input"]:
            # f_info['channels'] maps str(chan) -> relpath
            ch_dict = {int(c): os.path.join(".", p) for c, p in f_info["channels"].items()}
            res = read_multichannel_timestamp(ch_dict, self.channel_order)
            in_arrays.append(self.cropper.crop_array(res["data_kelvin"]))
            in_masks.append(self.cropper.crop_array(res["valid_mask"]))

        # 2. Load 1 target frame (actual observed +15 min scan)
        tgt_f_info = seq_info["source_files"]["target"][0]
        tgt_ch_dict = {int(c): os.path.join(".", p) for c, p in tgt_f_info["channels"].items()}
        tgt_res = read_multichannel_timestamp(tgt_ch_dict, self.channel_order)
        tgt_data = self.cropper.crop_array(tgt_res["data_kelvin"])
        tgt_mask = self.cropper.crop_array(tgt_res["valid_mask"])

        # Stack inputs: [4, 4, H, W]
        in_tensors = np.stack(in_arrays, axis=0).astype(np.float32)
        in_masks_arr = np.stack(in_masks, axis=0).astype(bool)
        
        # Combined mask: intersection across all 4 input frames and target frame
        comb_mask = tgt_mask.copy()
        for m in in_masks:
            comb_mask = comb_mask & m

        window_dict = {
            "input_tensors": in_tensors,        # [4, 4, H, W]
            "input_masks": in_masks_arr,        # [4, H, W]
            "target_tensor": tgt_data,          # [4, H, W]
            "target_mask": tgt_mask,            # [H, W]
            "combined_mask": comb_mask,         # [H, W]
            "sequence_id": seq_info["sequence_id"],
            "group_id": seq_info["group_id"]
        }

        # Store in LRU cache
        if len(self._window_cache) >= self.cache_size:
            self._window_cache.popitem(last=False)
        self._window_cache[seq_idx] = window_dict
        return window_dict

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a single synchronized patch item.
        
        Returns:
            dict containing:
                - 'input_kelvin': torch.FloatTensor [4, 4, 128, 128]
                - 'input_norm': torch.FloatTensor [4, 4, 128, 128] (if normalize_inputs is True)
                - 'input_mask': torch.BoolTensor [4, 128, 128]
                - 'target_kelvin': torch.FloatTensor [4, 128, 128]
                - 'target_mask': torch.BoolTensor [128, 128]
                - 'combined_mask': torch.BoolTensor [128, 128]
                - 'is_fully_valid': bool
                - 'is_convective': bool
                - 'metadata': dict with sequence_id, event_id, slice coords
        """
        seq_idx, r_start, c_start = self.items[index]
        r_end = r_start + self.patch_size
        c_end = c_start + self.patch_size

        win_dict = self._load_window_tensors(seq_idx)
        
        in_patch = win_dict["input_tensors"][:, :, r_start:r_end, c_start:c_end]  # [4, 4, 128, 128]
        in_mask_patch = win_dict["input_masks"][:, r_start:r_end, c_start:c_end]  # [4, 128, 128]
        tgt_patch = win_dict["target_tensor"][:, r_start:r_end, c_start:c_end]    # [4, 128, 128]
        tgt_mask_patch = win_dict["target_mask"][r_start:r_end, c_start:c_end]    # [128, 128]
        comb_mask_patch = win_dict["combined_mask"][r_start:r_end, c_start:c_end]# [128, 128]

        in_t = torch.from_numpy(in_patch).float()
        in_m = torch.from_numpy(in_mask_patch).bool()
        tgt_t = torch.from_numpy(tgt_patch).float()
        tgt_m = torch.from_numpy(tgt_mask_patch).bool()
        comb_m = torch.from_numpy(comb_mask_patch).bool()

        is_fully_valid = bool(comb_m.all().item())
        
        # Convective assessment on Band 13 target
        b13_tgt = tgt_t[1] # Band 13 is index 1
        t_min = float(b13_tgt[tgt_m].min().item()) if tgt_m.any() else 300.0
        frac_241 = float((b13_tgt[tgt_m] < 241.0).float().mean().item()) if tgt_m.any() else 0.0
        count_221 = int((b13_tgt[tgt_m] < 221.0).sum().item()) if tgt_m.any() else 0
        is_convective = (t_min < 241.0) and (frac_241 >= 0.02 or count_221 >= 5)

        item = {
            "input_kelvin": in_t,
            "input_mask": in_m,
            "target_kelvin": tgt_t,
            "target_mask": tgt_m,
            "combined_mask": comb_m,
            "is_fully_valid": is_fully_valid,
            "is_convective": is_convective,
            "metadata": {
                "sequence_id": win_dict["sequence_id"],
                "event_id": win_dict["group_id"],
                "row_slice": (r_start, r_end),
                "col_slice": (c_start, c_end),
                "b13_min_kelvin": t_min
            }
        }

        if self.normalize_inputs and self.standardizer is not None:
            # Mask expansion for standardizer: [4, 4, 128, 128]
            m_exp = in_m.unsqueeze(1).expand(-1, 4, -1, -1)
            item["input_norm"] = self.standardizer.normalize_input(in_t, m_exp)

        return item

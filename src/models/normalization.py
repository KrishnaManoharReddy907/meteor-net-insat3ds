"""
src/models/normalization.py
Isolated ChannelStandardizer utility for METEOR-Net temporal super-resolution.

Preserves physical Kelvin values on disk and in loss evaluation while providing
zero-mean, unit-variance standardized tensors to the neural network.
Safely handles invalid pixels, non-finite values, and zero standard deviations.
"""

import os
import json
import datetime
from typing import Optional, Tuple, Union, Dict, Any, List
import torch
import torch.nn as nn
import numpy as np


class ChannelStandardizer(nn.Module):
    """
    Channel-wise standardizer module.
    
    Transforms physical Kelvin inputs [B, T, C, H, W] to normalized space ~ N(0, 1)
    and de-normalizes network predictions [B, C, H, W] back to physical Kelvin.
    
    Guarantees:
      1. Physical Kelvin preservation: Raw data, target observations, and loss
         thresholds (e.g. 241 K, 221 K) remain strictly in Kelvin.
      2. Invalid pixel sanitization: Masked invalid pixels (space/limb/bad detectors)
         are filled with channel mean, mapping to exactly 0.0 in normalized space,
         preventing NaN/Inf propagation into convolutions.
      3. Zero/non-finite variance safety: If std is zero or non-finite, it is replaced
         with 1.0 (plus eps) to avoid division by zero.
      4. Autograd differentiability: De-normalization is purely linear, ensuring exact,
         well-conditioned gradient backpropagation: dL/dy_norm = dL/dy_phys * std.
      5. Flexible broadcasting: Supports 5D [B, T, C, H, W] and 4D [T, C, H, W] inputs,
         and 4D [B, C, H, W] and 3D [C, H, W] outputs.
    """
    def __init__(
        self,
        mean: Union[torch.Tensor, np.ndarray, list],
        std: Union[torch.Tensor, np.ndarray, list],
        eps: float = 1e-6,
        channel_order: Optional[Tuple[int, ...]] = None,
        provenance: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std_t = torch.as_tensor(std, dtype=torch.float32).flatten()

        if mean_t.numel() != std_t.numel():
            raise ValueError(
                f"Channel count mismatch: mean has {mean_t.numel()} elements, "
                f"std has {std_t.numel()} elements."
            )

        # Sanitize statistics: replace NaN/Inf/<=0 std with 1.0; replace NaN/Inf mean with 0.0
        safe_std = torch.where(
            torch.isfinite(std_t) & (std_t > 0),
            std_t,
            torch.ones_like(std_t)
        )
        safe_mean = torch.where(
            torch.isfinite(mean_t),
            mean_t,
            torch.zeros_like(mean_t)
        )

        self.num_channels = mean_t.numel()
        self.eps = float(eps)

        if channel_order is not None:
            if len(channel_order) != self.num_channels:
                raise ValueError(
                    f"channel_order length ({len(channel_order)}) does not match num_channels ({self.num_channels})"
                )
            self.channel_order = tuple(channel_order)
        else:
            self.channel_order = (9, 13, 14, 15) if self.num_channels == 4 else tuple(range(self.num_channels))

        self.provenance = dict(provenance) if provenance is not None else {}

        # Register buffers so they track device (CPU/CUDA) and are included in state_dict
        self.register_buffer("mean", safe_mean)
        self.register_buffer("std", safe_std)

    @classmethod
    def fit(
        cls,
        x: Union[torch.Tensor, np.ndarray],
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        channel_dim: Optional[int] = None,
        eps: float = 1e-6,
        channel_order: Optional[Tuple[int, ...]] = None,
        provenance: Optional[Dict[str, Any]] = None
    ) -> "ChannelStandardizer":
        """
        Computes per-channel mean and standard deviation from training data over valid pixels only.
        
        Args:
            x: Input array/tensor containing physical Kelvin values.
               Typically [B, T, C, H, W] (dim=5, C at 2), [T, C, H, W] (dim=4, C at 1),
               or [C, H, W] (dim=3, C at 0).
            mask: Optional boolean or binary mask of same spatial/temporal layout.
                  Valid pixels are True / 1.
            channel_dim: Explicit index of channel dimension. If None, inferred automatically.
            eps: Epsilon parameter for standardizer.
            
        Returns:
            Fitted ChannelStandardizer instance.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = x.float()

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).bool()
            else:
                mask = mask.bool()

        # Infer channel dimension if not given
        if channel_dim is None:
            if x.dim() == 5:
                channel_dim = 2  # [B, T, C, H, W]
            elif x.dim() == 4:
                channel_dim = 1  # [T, C, H, W]
            elif x.dim() == 3:
                channel_dim = 0  # [C, H, W]
            else:
                raise ValueError(f"Cannot infer channel dimension for tensor of shape {list(x.shape)}")

        num_channels = x.shape[channel_dim]
        means = []
        stds = []

        # Broadcast mask to match x if needed
        if mask is not None and mask.shape != x.shape:
            # If mask is [T, H, W] and x is [T, C, H, W]
            if x.dim() == 4 and mask.dim() == 3 and mask.shape == (x.shape[0], x.shape[2], x.shape[3]):
                mask = mask.unsqueeze(1).expand(-1, num_channels, -1, -1)
            # If mask is [B, T, H, W] and x is [B, T, C, H, W]
            elif x.dim() == 5 and mask.dim() == 4 and mask.shape == (x.shape[0], x.shape[1], x.shape[3], x.shape[4]):
                mask = mask.unsqueeze(2).expand(-1, -1, num_channels, -1, -1)
            # If mask is [H, W] and x is [T, C, H, W]
            elif x.dim() == 4 and mask.dim() == 2 and mask.shape == (x.shape[2], x.shape[3]):
                mask = mask.view(1, 1, mask.shape[0], mask.shape[1]).expand_as(x)
            # If mask is [H, W] and x is [B, T, C, H, W]
            elif x.dim() == 5 and mask.dim() == 2 and mask.shape == (x.shape[3], x.shape[4]):
                mask = mask.view(1, 1, 1, mask.shape[0], mask.shape[1]).expand_as(x)

        for c in range(num_channels):
            # Slice channel c
            slc = [slice(None)] * x.dim()
            slc[channel_dim] = c
            x_c = x[tuple(slc)]

            if mask is not None:
                m_c = mask[tuple(slc)]
                valid_mask = m_c & torch.isfinite(x_c)
            else:
                valid_mask = torch.isfinite(x_c)

            valid_values = x_c[valid_mask]
            if valid_values.numel() > 1:
                ch_mean = valid_values.mean().item()
                ch_std = valid_values.std(unbiased=False).item()
            elif valid_values.numel() == 1:
                ch_mean = valid_values.item()
                ch_std = 1.0
            else:
                # No valid pixels in this channel
                ch_mean = 0.0
                ch_std = 1.0

            if not np.isfinite(ch_std) or ch_std <= 0:
                ch_std = 1.0
            if not np.isfinite(ch_mean):
                ch_mean = 0.0

            means.append(ch_mean)
            stds.append(ch_std)

        return cls(means, stds, eps=eps, channel_order=channel_order, provenance=provenance)

    def normalize_input(
        self,
        x_phys: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Normalizes physical Kelvin inputs to zero-mean, unit-variance standardized space.
        
        Args:
            x_phys: Physical Kelvin input tensor of shape [B, T, C, H, W] or [T, C, H, W].
            mask: Optional validity mask (True/1 = valid, False/0 = invalid).
            
        Returns:
            Normalized tensor where invalid pixels are sanitized and set to 0.0.
        """
        dim = x_phys.dim()
        if dim == 5:
            # [B, T, C, H, W]
            mean_view = self.mean.view(1, 1, -1, 1, 1)
            std_view = self.std.view(1, 1, -1, 1, 1)
            ch_dim = 2
        elif dim == 4:
            # [T, C, H, W]
            mean_view = self.mean.view(1, -1, 1, 1)
            std_view = self.std.view(1, -1, 1, 1)
            ch_dim = 1
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {list(x_phys.shape)}")

        if mask is not None:
            mask_bool = mask.bool()
            # Align mask dimensions
            if dim == 5 and mask_bool.dim() == 4:
                # [B, T, H, W] -> [B, T, 1, H, W]
                mask_bool = mask_bool.unsqueeze(2)
            elif dim == 4 and mask_bool.dim() == 3:
                # [T, H, W] -> [T, 1, H, W]
                mask_bool = mask_bool.unsqueeze(1)
            elif mask_bool.dim() == 2:
                # [H, W] -> broadcast
                if dim == 5:
                    mask_bool = mask_bool.view(1, 1, 1, mask_bool.shape[0], mask_bool.shape[1])
                else:
                    mask_bool = mask_bool.view(1, 1, mask_bool.shape[0], mask_bool.shape[1])

            # Replace invalid / non-finite pixels with mean so safe_x - mean = 0.0
            safe_x = torch.where(
                mask_bool & torch.isfinite(x_phys),
                x_phys,
                mean_view
            )
            x_norm = (safe_x - mean_view) / (std_view + self.eps)
            # Enforce exact 0.0 at invalid locations
            x_norm = torch.where(mask_bool, x_norm, torch.zeros_like(x_norm))
        else:
            safe_x = torch.nan_to_num(x_phys, nan=0.0, posinf=0.0, neginf=0.0)
            x_norm = (safe_x - mean_view) / (std_view + self.eps)

        return x_norm

    def denormalize_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        De-normalizes standardized network output back to physical Kelvin.
        
        Args:
            y_norm: Normalized tensor of shape [B, C, H, W] or [C, H, W].
            
        Returns:
            Tensor in physical Kelvin space with active gradient history:
            y_phys = y_norm * (std + eps) + mean.
        """
        dim = y_norm.dim()
        if dim == 4:
            # [B, C, H, W]
            mean_view = self.mean.view(1, -1, 1, 1)
            std_view = self.std.view(1, -1, 1, 1)
        elif dim == 3:
            # [C, H, W]
            mean_view = self.mean.view(-1, 1, 1)
            std_view = self.std.view(-1, 1, 1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor for de-normalization, got shape {list(y_norm.shape)}")

        return y_norm * (std_view + self.eps) + mean_view

    def save_stats(
        self,
        filepath: str,
        provenance: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Saves frozen normalization statistics with complete provenance metadata to a JSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        merged_provenance = dict(self.provenance)
        if provenance:
            merged_provenance.update(provenance)
        if "saved_at" not in merged_provenance:
            merged_provenance["saved_at"] = datetime.datetime.now().isoformat()

        payload = {
            "num_channels": self.num_channels,
            "channel_order": list(self.channel_order),
            "means": [float(m) for m in self.mean.tolist()],
            "stds": [float(s) for s in self.std.tolist()],
            "eps": self.eps,
            "provenance": merged_provenance
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_stats_file(
        cls,
        filepath: str,
        expected_channel_order: Optional[Tuple[int, ...]] = (9, 13, 14, 15),
        eps: Optional[float] = None
    ) -> "ChannelStandardizer":
        """
        Loads frozen normalization statistics from a JSON file and returns an initialized standardizer.
        Validates channel order and length against expected configuration.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Normalization statistics file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        means = data["means"]
        stds = data["stds"]
        file_channel_order = tuple(data.get("channel_order", []))
        eps_val = eps if eps is not None else float(data.get("eps", 1e-6))
        prov = data.get("provenance", {})

        if expected_channel_order is not None and file_channel_order:
            if file_channel_order != tuple(expected_channel_order):
                raise ValueError(
                    f"Channel order mismatch in stats file {filepath}: "
                    f"file has {file_channel_order}, expected {expected_channel_order}"
                )

        return cls(
            mean=means,
            std=stds,
            eps=eps_val,
            channel_order=file_channel_order or expected_channel_order,
            provenance=prov
        )

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, channel_order={list(self.channel_order)}, mean={self.mean.tolist()}, std={self.std.tolist()}, eps={self.eps}"

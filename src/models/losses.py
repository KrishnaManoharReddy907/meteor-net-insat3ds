"""
src/models/losses.py
Mask-aware, convective-weighted training objective for METEOR-Net temporal super-resolution.
Fully protected against NaNs, infinities, and all-masked zero-gradient edge cases.
"""

import torch
import torch.nn as nn


class MaskAwareConvectiveMAELoss(nn.Module):
    """
    Mask-Aware Convective-Weighted Mean Absolute Error (MAE) Loss.
    
    Adheres strictly to Lin et al. (IEEE TGRS 2026), Eq. 22-24:
      l_r = (1 / n) * sum_{i=1}^n w(H_i) * |H_i - ^H_i|
    where n is the count of valid pixels across the batch.
    
    Convective weighting references Band 13 (10.33 um clean infrared window):
      - alpha = 1.0 for T >= 241 K (ambient surface, low cloud, clear sky)
      - beta  = 2.0 for 221 K <= T < 241 K (developing convective cloud)
      - gamma = 5.0 for T < 221 K (deep convective core, cold anvil)
      
    Safety guarantees:
      1. Target sanitization: Invalid targets (NaN, Inf) are sanitized prior to subtraction
         so that invalid values cannot contaminate the loss or gradients.
      2. Zero dilution: Empty patches in a batch contribute 0 to the error and 0 to the valid
         pixel count, preventing artificial loss attenuation.
      3. All-masked batch safety: If all pixels in a batch are masked, evaluates to a finite
         scalar 0.0 with valid zero autograd gradients, preserving computational graph continuity.
    """
    def __init__(
        self,
        ref_channel_idx: int = 1,
        t1_threshold: float = 241.0,
        t2_threshold: float = 221.0,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 5.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.ref_channel_idx = ref_channel_idx
        self.t1 = t1_threshold
        self.t2 = t2_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates mask-aware convective MAE.
        
        Args:
            pred: [B, C, H, W] float32 tensor of predicted midpoint Kelvin temperatures
            target: [B, C, H, W] float32 tensor of observed midpoint Kelvin temperatures
            mask: [B, C, H, W] or [B, 1, H, W] or [B, H, W] bool/binary validity mask (1=valid, 0=invalid)
            
        Returns:
            Scalar float32 loss tensor with active gradient graph.
        """
        # Ensure mask is broadcastable to [B, C, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.size(1) == 1 and pred.size(1) > 1:
            mask = mask.expand(-1, pred.size(1), -1, -1)
            
        mask_bool = mask.bool()
        valid_pixel_count = mask_bool.sum().float()

        # Edge Case: Entire batch is completely masked (0 valid pixels)
        if valid_pixel_count == 0.0:
            # Clean pred to ensure scalar is strictly finite (0.0) even if pred has NaNs
            clean_pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            return (clean_pred * 0.0).sum()

        # 1. Sanitize target: replace NaNs, Infs, or invalid values at masked locations with 0.0
        # This guarantees that (pred - target) cannot evaluate to NaN on masked pixels
        safe_target = torch.where(
            mask_bool,
            torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(target)
        )
        
        # 2. Sanitize pred: mask out any potential non-finite values at masked locations
        safe_pred = torch.where(mask_bool, pred, torch.zeros_like(pred))

        # 3. Compute absolute error
        diff = torch.abs(safe_pred - safe_target)

        # 4. Compute convective weights based on reference channel (Band 13)
        if safe_target.size(1) > self.ref_channel_idx:
            ref_target = safe_target[:, self.ref_channel_idx : self.ref_channel_idx + 1, :, :]
        else:
            ref_target = safe_target[:, 0:1, :, :]
        
        # Base weight alpha (1.0)
        weights = torch.full_like(pred, self.alpha)
        # Moderate convective weight beta (2.0) for 221 <= T < 241
        weights = torch.where(
            (ref_target < self.t1) & (ref_target >= self.t2),
            torch.full_like(pred, self.beta),
            weights
        )
        # Deep convective core weight gamma (5.0) for T < 221
        weights = torch.where(
            ref_target < self.t2,
            torch.full_like(pred, self.gamma),
            weights
        )

        # 5. Mask weighted difference so invalid pixels contribute strictly 0.0
        weighted_diff = torch.where(mask_bool, diff * weights, torch.zeros_like(diff))

        # 6. Global batch reduction normalized strictly by valid pixel count
        loss = weighted_diff.sum() / (valid_pixel_count + self.eps)
        return loss

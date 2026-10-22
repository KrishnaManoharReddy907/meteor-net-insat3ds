"""
tests/test_meteor_net.py
Automated unit tests validating technical acceptance criteria for METEOR-Net prototype.
Covers tensor shapes, finite forward pass, gradient flow across all modules, zero dead branches,
parameter count audit, and mask-aware loss behavior across all edge cases (NaNs, zero-valid).
"""

import unittest
import torch
from src.models.meteor_net import (
    METEORNet,
    MultichannelCollaborativeEncoding,
    TemporalFlowEstimationAndEvolution,
    EvolutionAwareFeatureFusionAndReconstruction
)
from src.models.losses import MaskAwareConvectiveMAELoss


class TestMETEORNetArchitecture(unittest.TestCase):
    """Unit tests for METEOR-Net module tensor dimensions and forward-backward flow."""

    @classmethod
    def setUpClass(cls):
        # Set deterministic seed for reproducibility
        torch.manual_seed(42)
        cls.model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4)

    def test_01_parameter_count_audit(self):
        """Criterion 7: Parameter audit matches analytical calculation 22,086,943 +- 5%."""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        expected_params = 22086943
        lower_bound = int(expected_params * 0.95)
        upper_bound = int(expected_params * 1.05)
        
        self.assertEqual(total_params, expected_params, f"Exact parameter mismatch: {total_params} vs {expected_params}")
        self.assertTrue(lower_bound <= total_params <= upper_bound, f"Parameter count {total_params} out of range [{lower_bound}, {upper_bound}]")
        
        # Verify submodules
        mce_p = sum(p.numel() for p in self.model.mce.parameters() if p.requires_grad)
        tfee_p = sum(p.numel() for p in self.model.tfee.parameters() if p.requires_grad)
        effr_p = sum(p.numel() for p in self.model.effr.parameters() if p.requires_grad)
        self.assertEqual(mce_p, 158888)
        self.assertEqual(tfee_p, 21412928)
        self.assertEqual(effr_p, 515127)

    def test_02_forward_shape_and_finite_outputs(self):
        """Criterion 1, 2, 3: Input [B, 4, 4, 128, 128] produces strictly [B, 4, 128, 128] with finite values."""
        B, T, C, H, W = 2, 4, 4, 128, 128
        dummy_input = torch.randn(B, T, C, H, W)
        
        out = self.model(dummy_input)
        
        self.assertEqual(out.shape, (B, C, H, W), f"Output shape mismatch: {out.shape}")
        self.assertTrue(torch.isfinite(out).all().item(), "Forward pass produced non-finite (NaN/Inf) values")

    def test_03_variable_batch_sizes(self):
        """Verify model handles batch size B=1 cleanly."""
        dummy_input = torch.randn(1, 4, 4, 128, 128)
        out = self.model(dummy_input)
        self.assertEqual(out.shape, (1, 4, 128, 128))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_04_backward_pass_and_zero_dead_branches(self):
        """Criterion 5, 6: Backward pass completes with non-zero finite gradients for EVERY trainable parameter."""
        model = METEORNet()
        dummy_input = torch.randn(2, 4, 4, 128, 128, requires_grad=True)
        out = model(dummy_input)
        
        # Synthetic loss
        loss = out.sum()
        loss.backward()
        
        # Check every single parameter in the model
        dead_parameters = []
        nan_parameters = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient (disconnected branch)")
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_parameters.append(name)
                # Check for strictly non-zero gradient (active in computation graph)
                if (param.grad == 0.0).all():
                    dead_parameters.append(name)
                    
        self.assertEqual(len(nan_parameters), 0, f"Parameters with NaN/Inf gradients: {nan_parameters}")
        self.assertEqual(len(dead_parameters), 0, f"Dead/unused parameter branches detected: {dead_parameters}")


class TestMaskAwareConvectiveMAELoss(unittest.TestCase):
    """Unit tests for MaskAwareConvectiveMAELoss across normal and extreme edge cases."""

    def setUp(self):
        self.loss_fn = MaskAwareConvectiveMAELoss(ref_channel_idx=1)

    def test_05_all_valid_batch(self):
        """Verify loss computation when all pixels are valid."""
        B, C, H, W = 2, 4, 16, 16
        pred = torch.full((B, C, H, W), 280.0, requires_grad=True)
        target = torch.full((B, C, H, W), 282.0)
        mask = torch.ones((B, C, H, W), dtype=torch.bool)
        
        loss = self.loss_fn(pred, target, mask)
        # Difference is 2.0 K, T >= 241 K so w=1.0, mean loss = 2.0 K
        self.assertAlmostEqual(loss.item(), 2.0, places=5)
        
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all().item())
        self.assertFalse((pred.grad == 0.0).all().item())

    def test_06_convective_weighting_thresholds(self):
        """Verify that weights alpha=1.0, beta=2.0, gamma=5.0 activate correctly based on Band 13."""
        B, C, H, W = 1, 4, 1, 3
        # Reference channel is channel index 1
        # Pixel 0: T=250 K (>= 241 K -> alpha=1.0)
        # Pixel 1: T=230 K (221 <= T < 241 -> beta=2.0)
        # Pixel 2: T=210 K (< 221 K -> gamma=5.0)
        pred = torch.zeros(B, C, H, W, requires_grad=True)
        target = torch.zeros(B, C, H, W)
        target[:, 1, 0, 0] = 250.0
        target[:, 1, 0, 1] = 230.0
        target[:, 1, 0, 2] = 210.0
        
        # Set absolute difference = 1.0 K everywhere across all 4 channels
        pred_val = target + 1.0
        mask = torch.ones((B, C, H, W), dtype=torch.bool)
        
        loss = self.loss_fn(pred_val, target, mask)
        # For each channel, pixel 0 has w=1, pixel 1 has w=2, pixel 2 has w=5
        # Total weighted error per channel: 1*1 + 1*2 + 1*5 = 8.0
        # Across 4 channels: 4 * 8.0 = 32.0
        # Valid pixel count: 4 * 3 = 12
        # Loss = 32.0 / 12 = 2.6666667 K
        expected = 32.0 / 12.0
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_07_partially_masked_batch(self):
        """Verify that masked pixels do not contribute error and do not dilute valid pixels."""
        B, C, H, W = 1, 4, 2, 2
        pred = torch.full((B, C, H, W), 250.0, requires_grad=True)
        target = torch.full((B, C, H, W), 252.0)
        
        # Mask top row valid, bottom row invalid
        mask = torch.zeros((B, C, H, W), dtype=torch.bool)
        mask[:, :, 0, :] = True  # 4 * 2 = 8 valid pixels, 8 invalid pixels
        
        loss = self.loss_fn(pred, target, mask)
        # Difference on valid pixels is 2.0 K, w=1.0, denominator is exactly 8 valid pixels
        self.assertAlmostEqual(loss.item(), 2.0, places=5)
        
        loss.backward()
        # Gradient should be non-zero on top row, and exactly zero on bottom row
        self.assertTrue((pred.grad[:, :, 0, :] != 0.0).all().item())
        self.assertTrue((pred.grad[:, :, 1, :] == 0.0).all().item())

    def test_08_completely_masked_batch(self):
        """Criterion: Fully masked batch evaluates to finite scalar 0.0 with valid zero gradients."""
        B, C, H, W = 2, 4, 8, 8
        pred = torch.randn(B, C, H, W, requires_grad=True)
        target = torch.randn(B, C, H, W)
        mask = torch.zeros((B, C, H, W), dtype=torch.bool)
        
        loss = self.loss_fn(pred, target, mask)
        
        self.assertTrue(torch.isfinite(loss).item(), "All-masked loss is not finite")
        self.assertEqual(loss.item(), 0.0, f"All-masked loss expected 0.0, got {loss.item()}")
        
        loss.backward()
        self.assertIsNotNone(pred.grad, "Gradient is None for all-masked batch")
        self.assertTrue(torch.isfinite(pred.grad).all().item(), "All-masked gradient contains NaN/Inf")
        self.assertTrue((pred.grad == 0.0).all().item(), "All-masked gradient is not strictly zero")

    def test_09_masked_targets_containing_nan_and_inf(self):
        """Criterion: Masked targets with NaNs/Infs do not contaminate loss or produce NaN gradients."""
        B, C, H, W = 2, 4, 8, 8
        pred = torch.full((B, C, H, W), 280.0, requires_grad=True)
        target = torch.full((B, C, H, W), 282.0)
        mask = torch.ones((B, C, H, W), dtype=torch.bool)
        
        # Introduce NaNs, +Inf, -Inf at masked pixels
        mask[:, :, 4:, :] = False
        target[:, :, 4:, :] = float('nan')
        target[0, 0, 5, 5] = float('inf')
        target[1, 1, 6, 6] = float('-inf')
        
        loss = self.loss_fn(pred, target, mask)
        
        self.assertTrue(torch.isfinite(loss).item(), "Loss contaminated by NaN/Inf in masked target")
        self.assertAlmostEqual(loss.item(), 2.0, places=5)
        
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all().item(), "Gradients contaminated by NaN/Inf in masked target")
        # Gradient at masked locations should be exactly zero
        self.assertTrue((pred.grad[:, :, 4:, :] == 0.0).all().item())

    def test_10_all_masked_batch_with_nan_predictions(self):
        """Criterion: All-masked batch returns finite 0.0 and finite zero grad even if pred has NaNs."""
        pred = torch.tensor([float('nan'), float('inf'), -float('inf')], requires_grad=True)
        target = torch.tensor([float('nan'), 280.0, 300.0])
        mask = torch.tensor([False, False, False])
        
        loss = self.loss_fn(pred.view(1, 1, 1, 3), target.view(1, 1, 1, 3), mask.view(1, 1, 1, 3))
        
        self.assertTrue(torch.isfinite(loss).item(), "Loss is not finite for NaN predictions")
        self.assertEqual(loss.item(), 0.0)
        
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all().item(), "Gradient contains NaN for NaN predictions")
        self.assertTrue((pred.grad == 0.0).all().item())

    def test_11_unmasked_nan_prediction_evaluates_to_nan(self):
        """Verify that non-finite predictions at VALID pixels evaluate to NaN loss (intentionally treated as training error)."""
        pred = torch.tensor([float('nan'), 280.0, 300.0], requires_grad=True)
        target = torch.tensor([280.0, 280.0, 300.0])
        mask = torch.tensor([True, True, True], dtype=torch.bool)  # All pixels valid
        
        loss = self.loss_fn(pred.view(1, 1, 1, 3), target.view(1, 1, 1, 3), mask.view(1, 1, 1, 3))
        self.assertTrue(torch.isnan(loss).item(), "Unmasked NaN prediction must evaluate to NaN loss to signal numerical failure")


if __name__ == '__main__':
    unittest.main()


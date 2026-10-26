"""
tests/test_normalization.py
Focused unit tests for ChannelStandardizer, invalid-pixel sanitization,
differentiability through de-normalization, and integration with MaskAwareConvectiveMAELoss.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.models.normalization import ChannelStandardizer
from src.models.losses import MaskAwareConvectiveMAELoss


class TestChannelStandardizer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.num_channels = 4
        self.means = [217.61, 221.84, 219.71, 218.60]
        self.stds = [6.85, 9.42, 9.87, 9.94]
        self.standardizer = ChannelStandardizer(self.means, self.stds)

    def test_initialization_and_buffers(self):
        """Test proper initialization, buffer registration, and float32 types."""
        self.assertEqual(self.standardizer.num_channels, 4)
        self.assertTrue(torch.is_tensor(self.standardizer.mean))
        self.assertTrue(torch.is_tensor(self.standardizer.std))
        self.assertEqual(self.standardizer.mean.shape, torch.Size([4]))
        self.assertEqual(self.standardizer.std.shape, torch.Size([4]))
        self.assertAlmostEqual(self.standardizer.mean[0].item(), 217.61, places=2)
        self.assertAlmostEqual(self.standardizer.std[1].item(), 9.42, places=2)

    def test_sanitize_zero_and_nonfinite_std(self):
        """Test that zero or non-finite std values are safely handled without crash."""
        bad_means = [220.0, float('nan'), 210.0, float('inf')]
        bad_stds = [0.0, -5.0, float('nan'), float('inf')]
        std_safe = ChannelStandardizer(bad_means, bad_stds)
        
        self.assertTrue(torch.isfinite(std_safe.mean).all())
        self.assertTrue(torch.isfinite(std_safe.std).all())
        self.assertTrue((std_safe.std > 0).all())
        # Check specific fallbacks
        self.assertEqual(std_safe.std[0].item(), 1.0)
        self.assertEqual(std_safe.std[1].item(), 1.0)
        self.assertEqual(std_safe.std[2].item(), 1.0)
        self.assertEqual(std_safe.std[3].item(), 1.0)
        self.assertEqual(std_safe.mean[1].item(), 0.0)

    def test_fit_from_tensor_and_numpy(self):
        """Test fitting statistics from training arrays with valid mask."""
        B, T, C, H, W = 1, 4, 4, 32, 32
        data = torch.zeros(B, T, C, H, W)
        mask = torch.ones(B, T, H, W, dtype=torch.bool)
        
        # Populate each channel with known Gaussian distribution
        target_means = [215.0, 225.0, 235.0, 245.0]
        target_stds = [5.0, 8.0, 10.0, 12.0]
        for c in range(C):
            data[:, :, c, :, :] = torch.randn(B, T, H, W) * target_stds[c] + target_means[c]
            
        # Add invalid pixels (space fill) with NaNs and Infs
        mask[:, :, 0:5, 0:5] = False
        data[:, :, :, 0:5, 0:5] = float('nan')

        fitted = ChannelStandardizer.fit(data, mask=mask)
        self.assertEqual(fitted.num_channels, C)
        for c in range(C):
            self.assertAlmostEqual(fitted.mean[c].item(), target_means[c], delta=0.5)
            self.assertAlmostEqual(fitted.std[c].item(), target_stds[c], delta=0.5)

    def test_fit_empty_or_single_pixel_channel(self):
        """Test fitting when a channel has 0 or 1 valid pixel."""
        x = torch.zeros(1, 4, 10, 10)
        mask = torch.zeros(1, 10, 10, dtype=torch.bool) # completely masked
        fitted = ChannelStandardizer.fit(x, mask=mask)
        self.assertTrue(torch.isfinite(fitted.mean).all())
        self.assertTrue(torch.isfinite(fitted.std).all())
        self.assertTrue((fitted.std > 0).all())

    def test_normalize_5d_and_4d_broadcasting(self):
        """Test normalization broadcasting for [B, T, C, H, W] and [T, C, H, W]."""
        # 5D input: [2, 4, 4, 64, 64]
        x_5d = torch.full((2, 4, 4, 64, 64), 220.0)
        norm_5d = self.standardizer.normalize_input(x_5d)
        self.assertEqual(norm_5d.shape, torch.Size([2, 4, 4, 64, 64]))
        self.assertTrue(torch.isfinite(norm_5d).all())

        # Check channel 0 value: (220.0 - 217.61) / (6.85 + 1e-6)
        expected_ch0 = (220.0 - 217.61) / (6.85 + 1e-6)
        self.assertAlmostEqual(norm_5d[0, 0, 0, 0, 0].item(), expected_ch0, places=3)

        # 4D input: [4, 4, 64, 64]
        x_4d = torch.full((4, 4, 64, 64), 220.0)
        norm_4d = self.standardizer.normalize_input(x_4d)
        self.assertEqual(norm_4d.shape, torch.Size([4, 4, 64, 64]))
        self.assertTrue(torch.isfinite(norm_4d).all())
        self.assertAlmostEqual(norm_4d[0, 0, 0, 0].item(), expected_ch0, places=3)

    def test_denormalize_4d_and_3d_broadcasting(self):
        """Test de-normalization broadcasting for [B, C, H, W] and [C, H, W]."""
        y_norm_4d = torch.zeros(2, 4, 64, 64)
        y_phys_4d = self.standardizer.denormalize_output(y_norm_4d)
        self.assertEqual(y_phys_4d.shape, torch.Size([2, 4, 64, 64]))
        # Normalized zero should map back to exact mean
        for c in range(4):
            self.assertAlmostEqual(y_phys_4d[0, c, 10, 10].item(), self.means[c], places=4)

        y_norm_3d = torch.zeros(4, 32, 32)
        y_phys_3d = self.standardizer.denormalize_output(y_norm_3d)
        self.assertEqual(y_phys_3d.shape, torch.Size([4, 32, 32]))
        for c in range(4):
            self.assertAlmostEqual(y_phys_3d[c, 5, 5].item(), self.means[c], places=4)

    def test_mask_sanitization_zero_filling(self):
        """Test that masked invalid pixels map to exactly 0.0 with zero NaN leakage."""
        x = torch.full((1, 4, 4, 32, 32), 220.0)
        mask = torch.ones(1, 4, 32, 32, dtype=torch.bool)
        
        # Invalidate corner with NaNs, infinities, and extreme values
        mask[:, :, 0:10, 0:10] = False
        x[:, :, :, 0:5, 0:5] = float('nan')
        x[:, :, :, 5:10, 0:5] = float('inf')
        x[:, :, :, 0:10, 5:10] = -9999.0

        x_norm = self.standardizer.normalize_input(x, mask=mask)
        
        # Valid region must be finite and normalized
        self.assertTrue(torch.isfinite(x_norm[:, :, :, 10:, 10:]).all())
        # Masked region must be strictly 0.0, zero NaNs or Infs
        masked_region = x_norm[:, :, :, 0:10, 0:10]
        self.assertTrue(torch.isfinite(masked_region).all())
        self.assertTrue((masked_region == 0.0).all())

    def test_invertibility_roundtrip(self):
        """Test that denormalize(normalize(x)) recovers original x on valid pixels."""
        x = torch.randn(1, 4, 4, 16, 16) * 10.0 + 220.0
        mask = torch.ones(1, 4, 16, 16, dtype=torch.bool)
        mask[:, :, 0:4, 0:4] = False
        
        x_norm = self.standardizer.normalize_input(x, mask=mask)
        # Squeeze time dim to test 4D prediction roundtrip
        target_frame = x[:, 2, :, :, :] # [1, 4, 16, 16]
        target_norm = x_norm[:, 2, :, :, :]
        recovered = self.standardizer.denormalize_output(target_norm)
        
        # Only compare valid pixels
        valid_mask_4d = mask[:, 2:3, :, :].expand(-1, 4, -1, -1)
        err = torch.abs(recovered[valid_mask_4d] - target_frame[valid_mask_4d])
        self.assertLess(err.max().item(), 1e-4)

    def test_autograd_differentiability_and_gradient_flow(self):
        """Test that autograd flows through de-normalization with exact analytical gradients."""
        # Simulated raw network prediction (normalized space)
        y_norm = torch.randn(1, 4, 16, 16, requires_grad=True)
        y_phys = self.standardizer.denormalize_output(y_norm)
        
        # Target in physical Kelvin
        target = torch.full((1, 4, 16, 16), 220.0)
        mask = torch.ones(1, 4, 16, 16, dtype=torch.bool)
        
        loss_fn = MaskAwareConvectiveMAELoss(ref_channel_idx=1)
        loss = loss_fn(y_phys, target, mask)
        loss.backward()

        self.assertIsNotNone(y_norm.grad)
        self.assertTrue(torch.isfinite(y_norm.grad).all())
        self.assertTrue((y_norm.grad.abs() > 0).any())

        # Check analytical gradient relation:
        # Since y_phys = y_norm * (std + eps) + mean,
        # dL/dy_norm = dL/dy_phys * (std + eps).
        # Gradients on channel c should be scaled proportional to std[c].
        grad_c0 = y_norm.grad[0, 0].abs().mean().item()
        grad_c3 = y_norm.grad[0, 3].abs().mean().item()
        # std[3] = 9.94 is larger than std[0] = 6.85
        self.assertGreater(grad_c3, grad_c0)

    def test_loss_convective_weighting_with_denormalized_prediction(self):
        """
        Verify that physical Kelvin predictions correctly trigger beta=2.0 and gamma=5.0
        weights in MaskAwareConvectiveMAELoss based on target Band 13 temperature.
        """
        loss_fn = MaskAwareConvectiveMAELoss(ref_channel_idx=1, alpha=1.0, beta=2.0, gamma=5.0)
        
        # Construct prediction differing from target by exactly 1.0 Kelvin everywhere
        pred_phys = torch.full((1, 4, 8, 8), 230.0)
        
        # Test Case 1: Target ambient (T = 260 K >= 241 K) -> weight = alpha = 1.0
        target_ambient = torch.full((1, 4, 8, 8), 260.0)
        mask = torch.ones(1, 4, 8, 8, dtype=torch.bool)
        loss_amb = loss_fn(target_ambient + 1.0, target_ambient, mask)
        self.assertAlmostEqual(loss_amb.item(), 1.0, places=4)

        # Test Case 2: Target moderate convective (T = 230 K: 221 <= T < 241) -> weight = beta = 2.0
        target_conv = torch.full((1, 4, 8, 8), 230.0)
        loss_conv = loss_fn(target_conv + 1.0, target_conv, mask)
        self.assertAlmostEqual(loss_conv.item(), 2.0, places=4)

        # Test Case 3: Target deep convective core (T = 215 K < 221) -> weight = gamma = 5.0
        target_deep = torch.full((1, 4, 8, 8), 215.0)
        loss_deep = loss_fn(target_deep + 1.0, target_deep, mask)
        self.assertAlmostEqual(loss_deep.item(), 5.0, places=4)

    def test_save_and_load_stats_with_provenance(self):
        """Test saving frozen normalization statistics with provenance and reloading."""
        import tempfile
        import os

        provenance = {
            "training_sequence_ids": ["seq_2026-001_w00", "seq_2026-001_w01"],
            "split": "train",
            "dataset": "GOES-19 CONUS",
            "description": "Training partition statistics"
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "norm_stats.json")
            self.standardizer.save_stats(json_path, provenance=provenance)
            self.assertTrue(os.path.exists(json_path))

            # Reload
            loaded = ChannelStandardizer.from_stats_file(json_path, expected_channel_order=(9, 13, 14, 15))
            self.assertEqual(loaded.channel_order, (9, 13, 14, 15))
            self.assertEqual(loaded.num_channels, 4)
            for c in range(4):
                self.assertAlmostEqual(loaded.mean[c].item(), self.means[c], places=2)
                self.assertAlmostEqual(loaded.std[c].item(), self.stds[c], places=2)
            self.assertEqual(loaded.provenance["training_sequence_ids"], provenance["training_sequence_ids"])

    def test_channel_order_consistency_and_validation(self):
        """Test that channel order is verified and mismatched order raises ValueError."""
        import tempfile
        import os

        # Verify default channel order
        self.assertEqual(self.standardizer.channel_order, (9, 13, 14, 15))

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "norm_stats.json")
            self.standardizer.save_stats(json_path)

            # Attempt to load expecting a different channel order -> must raise ValueError
            with self.assertRaises(ValueError) as ctx:
                ChannelStandardizer.from_stats_file(json_path, expected_channel_order=(13, 9, 14, 15))
            self.assertIn("Channel order mismatch", str(ctx.exception))

    def test_frozen_statistics_unaffected_by_val_test(self):
        """Confirm that normalizing validation or test tensors does NOT alter frozen statistics."""
        initial_mean = self.standardizer.mean.clone()
        initial_std = self.standardizer.std.clone()

        # Simulate radically different validation distribution (e.g. warm desert at 310 K)
        val_input = torch.full((2, 4, 4, 32, 32), 310.0)
        norm_val = self.standardizer.normalize_input(val_input)
        denorm_val = self.standardizer.denormalize_output(norm_val[:, 0])

        # Buffers must remain strictly unchanged
        self.assertTrue(torch.equal(self.standardizer.mean, initial_mean))
        self.assertTrue(torch.equal(self.standardizer.std, initial_std))

    def test_training_only_fitting(self):
        """Confirm that fitting is performed on training data only, isolating validation data."""
        # Train data: mean = 220 K, std = 10 K
        train_data = torch.randn(1, 4, 4, 64, 64) * 10.0 + 220.0
        # Val data: mean = 290 K, std = 25 K
        val_data = torch.randn(1, 4, 4, 64, 64) * 25.0 + 290.0

        # Fit strictly on train data
        train_standardizer = ChannelStandardizer.fit(
            train_data,
            channel_order=(9, 13, 14, 15),
            provenance={"split": "train"}
        )

        for c in range(4):
            self.assertAlmostEqual(train_standardizer.mean[c].item(), 220.0, delta=1.5)
            self.assertAlmostEqual(train_standardizer.std[c].item(), 10.0, delta=1.5)

        # Normalize validation data using frozen training statistics
        val_norm = train_standardizer.normalize_input(val_data)
        # In normalized space, val data should have mean ~ (290 - 220) / 10 = +7.0
        self.assertAlmostEqual(val_norm[:, :, 0].mean().item(), 7.0, delta=1.0)

    def test_device_and_dtype_dynamic_alignment_cpu(self):
        """
        Regression test: Verify that normalize_input and denormalize_output
        dynamically align buffers to the input tensor's device and dtype,
        preserving the input device and dtype without mutating registered buffers.
        """
        std = ChannelStandardizer(self.means, self.stds)
        init_mean = std.mean.clone()
        init_std = std.std.clone()

        # 1. Float32 CPU tensor
        x_f32 = torch.full((1, 4, 4, 16, 16), 220.0, dtype=torch.float32, device="cpu")
        m_f32 = torch.ones((1, 4, 16, 16), dtype=torch.bool, device="cpu")
        out_f32 = std.normalize_input(x_f32, mask=m_f32)
        self.assertEqual(out_f32.device, x_f32.device)
        self.assertEqual(out_f32.dtype, torch.float32)

        # 2. Float64 CPU tensor
        x_f64 = torch.full((1, 4, 4, 16, 16), 220.0, dtype=torch.float64, device="cpu")
        out_f64 = std.normalize_input(x_f64)
        self.assertEqual(out_f64.device, x_f64.device)
        self.assertEqual(out_f64.dtype, torch.float64)

        y_f64 = torch.zeros((1, 4, 16, 16), dtype=torch.float64, device="cpu")
        denorm_f64 = std.denormalize_output(y_f64)
        self.assertEqual(denorm_f64.device, y_f64.device)
        self.assertEqual(denorm_f64.dtype, torch.float64)

        # Registered buffers must remain strictly unmutated
        self.assertTrue(torch.equal(std.mean, init_mean))
        self.assertTrue(torch.equal(std.std, init_std))

    def test_device_canonicalization_and_comparison_contract(self):
        """
        Regression test: Verify device representation and comparison semantics.
        torch.device('cuda') without ordinal (index=None) vs torch.device('cuda:0') (index=0).
        Tensors transferred via .to(torch.device('cuda')) always have index=0 ('cuda:0').
        Canonicalizing unindexed CUDA device to explicit ordinal ensures exact equality.
        """
        # 1. CPU device comparison
        dev_cpu = torch.device("cpu")
        t_cpu = torch.zeros(2, 2, device=dev_cpu)
        self.assertEqual(t_cpu.device, dev_cpu)
        self.assertEqual(t_cpu.device.type, "cpu")

        # 2. CUDA device comparison semantics check (unindexed vs indexed)
        dev_unindexed = torch.device("cuda")
        self.assertIsNone(dev_unindexed.index)
        self.assertEqual(dev_unindexed.type, "cuda")

        dev_indexed = torch.device("cuda:0")
        self.assertEqual(dev_indexed.index, 0)
        self.assertEqual(dev_indexed.type, "cuda")

        # Crucial PyTorch behavior: torch.device("cuda") != torch.device("cuda:0")
        self.assertNotEqual(dev_unindexed, dev_indexed)

        # Canonicalization logic check
        dev_canonical = torch.device("cuda", 0)
        self.assertEqual(dev_canonical, dev_indexed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for hardware cross-device test")
    def test_cuda_cpu_cross_device_dataset_contract(self):
        """
        Regression test for Google Colab T4 smoke test Stage 3 failure:
        Standardizer is placed on CUDA:0, while dataset loading yields CPU tensors.
        normalize_input must dynamically evaluate on CPU without device mismatch,
        and denormalize_output must dynamically evaluate on CUDA:0 for model outputs.
        Exercises both unindexed torch.device('cuda') and canonicalized torch.device('cuda:0').
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda" and device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        std_cuda = ChannelStandardizer(self.means, self.stds).to(device)
        self.assertEqual(std_cuda.mean.device.type, "cuda")
        self.assertEqual(std_cuda.std.device.type, "cuda")

        # 1. Dataset patch loading on CPU (simulates GOESSequencePatchDataset.__getitem__)
        in_t = torch.full((4, 4, 128, 128), 220.0, dtype=torch.float32, device="cpu")
        in_m = torch.ones((4, 4, 128, 128), dtype=torch.bool, device="cpu")

        # Dataset item assertions (verifying CPU contract)
        self.assertEqual(in_t.device.type, "cpu")
        self.assertEqual(in_m.device.type, "cpu")

        # normalize_input must NOT raise: "Expected all tensors to be on the same device, but found cuda:0 and cpu!"
        norm_cpu = std_cuda.normalize_input(in_t, mask=in_m)
        self.assertEqual(norm_cpu.device.type, "cpu")
        self.assertEqual(norm_cpu.shape, in_t.shape)
        self.assertTrue(torch.isfinite(norm_cpu).all())

        # 2. Forward pass boundary: caller transfers model inputs to CUDA
        x_cuda = norm_cpu.unsqueeze(0).to(device)
        self.assertEqual(x_cuda.device, device)
        self.assertEqual(x_cuda.device.type, "cuda")

        # 3. Model prediction on CUDA: denormalize_output produces physical Kelvin on CUDA
        y_cuda = torch.zeros((1, 4, 128, 128), dtype=torch.float32, device=device)
        y_phys = std_cuda.denormalize_output(y_cuda)
        self.assertEqual(y_phys.device, device)
        self.assertEqual(y_phys.device.type, "cuda")
        self.assertTrue(torch.isfinite(y_phys).all())

        # 4. Standardizer on CPU with CUDA prediction input
        std_cpu = ChannelStandardizer(self.means, self.stds)
        y_phys_from_cpu = std_cpu.denormalize_output(y_cuda)
        self.assertEqual(y_phys_from_cpu.device, device)


if __name__ == "__main__":
    unittest.main()

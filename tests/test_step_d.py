"""Tests for Step D: train_v2.py scheduler, val_ids builder, residual prediction, and CLI."""
import math
import os
import sys
import unittest
import torch
import numpy as np

# On Windows, open memory-mapped numpy arrays prevent tempfile rmtree cleanup
if os.name == "nt":
    import tempfile
    _orig_td = tempfile.TemporaryDirectory
    class _SafeTD(_orig_td):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("ignore_cleanup_errors", True)
            super().__init__(*args, **kwargs)
    tempfile.TemporaryDirectory = _SafeTD

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import train_v2 as tv2  # noqa: E402
from src.models.normalization import ChannelStandardizer  # noqa: E402


class TestTrainV2(unittest.TestCase):
    def test_lr_schedule(self):
        # Warmup phase: step 0 to warmup (step 0 returns base * 1 / warm)
        lr0 = tv2.lr_at(0, total=100, base=1e-3, warm=10)
        self.assertAlmostEqual(lr0, 1e-4)

        lr_peak = tv2.lr_at(9, total=100, base=1e-3, warm=10)
        self.assertAlmostEqual(lr_peak, 1e-3)

        # Decay phase: at step 100, reaches min_frac * base
        lr_end = tv2.lr_at(100, total=100, base=1e-3, warm=10, min_frac=0.05)
        self.assertAlmostEqual(lr_end, 5e-5)

    def test_predict_standard(self):
        # Mock standardizer with known stats
        mean = torch.tensor([250.0, 280.0, 280.0, 280.0])
        std_val = torch.tensor([10.0, 20.0, 20.0, 20.0])
        std = ChannelStandardizer(mean=mean, std=std_val)

        # Dummy model returning zeros
        class ZeroModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 4, x.shape[3], x.shape[4])

        model = ZeroModel()
        b = {
            "input_norm": torch.zeros(2, 4, 4, 32, 32),
            "input_kelvin": torch.full((2, 4, 4, 32, 32), 260.0),
        }
        pred = tv2.predict(model, std, b, residual=False, amp=False, device=torch.device("cpu"))
        # Denormalized output for zero is mean: band 9 -> 250, band 13 -> 280
        self.assertAlmostEqual(float(pred[0, 0, 0, 0]), 250.0)
        self.assertAlmostEqual(float(pred[0, 1, 0, 0]), 280.0)

    def test_predict_residual(self):
        mean = torch.tensor([250.0, 280.0, 280.0, 280.0])
        std_val = torch.tensor([10.0, 20.0, 20.0, 20.0])
        std = ChannelStandardizer(mean=mean, std=std_val)

        class UnitModel(torch.nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], 4, x.shape[3], x.shape[4])

        model = UnitModel()
        # Frames t (index 2) and t+G (index 3) both at 260.0 K -> linear midpoint = 260.0 K
        inp_k = torch.full((2, 4, 4, 32, 32), 260.0)
        b = {
            "input_norm": torch.zeros(2, 4, 4, 32, 32),
            "input_kelvin": inp_k,
        }
        pred = tv2.predict(model, std, b, residual=True, amp=False, device=torch.device("cpu"))
        # Residual prediction: linear (260.0) + y_norm (1.0) * std (10.0 for band 9, 20.0 for band 13)
        self.assertAlmostEqual(float(pred[0, 0, 0, 0]), 270.0)
        self.assertAlmostEqual(float(pred[0, 1, 0, 0]), 280.0)


if __name__ == "__main__":
    unittest.main()

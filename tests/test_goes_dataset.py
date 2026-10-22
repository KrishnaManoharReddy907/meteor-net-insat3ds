"""
tests/test_goes_dataset.py
Unit tests for GOESSequencePatchDataset lazy-loading PyTorch Dataset.
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath("."))
from src.data.goes_dataset import GOESSequencePatchDataset
from src.models.normalization import ChannelStandardizer


class TestGOESSequencePatchDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = "outputs/dataset_splits/splits_manifest.json"
        cls.stats_path = "outputs/normalization/frozen_global_train_stats.json"
        cls.standardizer = ChannelStandardizer.from_stats_file(cls.stats_path)

    def test_partition_lengths(self):
        """Verify partition sizes match exact discrete event allocations."""
        ds_train = GOESSequencePatchDataset(manifest_path=self.manifest_path, split="train")
        ds_val = GOESSequencePatchDataset(manifest_path=self.manifest_path, split="val")
        ds_test = GOESSequencePatchDataset(manifest_path=self.manifest_path, split="test")

        # 108 sequences * 60 patches = 6480
        self.assertEqual(len(ds_train), 6480)
        # 54 sequences * 60 patches = 3240
        self.assertEqual(len(ds_val), 3240)
        # 54 sequences * 60 patches = 3240
        self.assertEqual(len(ds_test), 3240)

    def test_invalid_split_raises(self):
        """Verify requesting a non-existent split raises ValueError."""
        with self.assertRaises(ValueError):
            GOESSequencePatchDataset(manifest_path=self.manifest_path, split="nonexistent")

    def test_item_shapes_and_types(self):
        """Verify individual patch shapes and types conform to METEOR-Net requirements."""
        ds = GOESSequencePatchDataset(
            manifest_path=self.manifest_path,
            split="train",
            standardizer=self.standardizer,
            normalize_inputs=True,
            cache_size=2
        )
        item = ds[0]

        # Inputs
        self.assertIn("input_kelvin", item)
        self.assertIn("input_norm", item)
        self.assertIn("input_mask", item)
        self.assertEqual(item["input_kelvin"].shape, torch.Size([4, 4, 128, 128]))
        self.assertEqual(item["input_norm"].shape, torch.Size([4, 4, 128, 128]))
        self.assertEqual(item["input_mask"].shape, torch.Size([4, 128, 128]))
        self.assertEqual(item["input_kelvin"].dtype, torch.float32)
        self.assertEqual(item["input_norm"].dtype, torch.float32)
        self.assertEqual(item["input_mask"].dtype, torch.bool)

        # Targets
        self.assertIn("target_kelvin", item)
        self.assertIn("target_mask", item)
        self.assertIn("combined_mask", item)
        self.assertEqual(item["target_kelvin"].shape, torch.Size([4, 128, 128]))
        self.assertEqual(item["target_mask"].shape, torch.Size([128, 128]))
        self.assertEqual(item["combined_mask"].shape, torch.Size([128, 128]))
        self.assertEqual(item["target_kelvin"].dtype, torch.float32)
        self.assertEqual(item["target_mask"].dtype, torch.bool)
        self.assertEqual(item["combined_mask"].dtype, torch.bool)

        # Metadata
        self.assertIn("metadata", item)
        self.assertIn("sequence_id", item["metadata"])
        self.assertIn("event_id", item["metadata"])
        self.assertEqual(item["metadata"]["event_id"], "event_2026-01-01")

        # Finite checks
        self.assertTrue(torch.isfinite(item["input_norm"]).all())
        self.assertTrue(torch.isfinite(item["input_kelvin"]).all())
        self.assertTrue(torch.isfinite(item["target_kelvin"]).all())

    def test_lru_cache_reuse(self):
        """Verify that accessing subsequent patches within same window reuses cached tensors."""
        ds = GOESSequencePatchDataset(
            manifest_path=self.manifest_path,
            split="train",
            cache_size=2
        )
        # Patch 0 and Patch 1 are from sequence window 0
        self.assertEqual(ds.items[0][0], ds.items[1][0])
        
        # Access item 0: fills cache
        item0 = ds[0]
        self.assertIn(0, ds._window_cache)
        cache_ref = ds._window_cache[0]

        # Access item 1: cache hit
        item1 = ds[1]
        self.assertIs(ds._window_cache[0], cache_ref)


if __name__ == "__main__":
    unittest.main()

"""
tests/test_preprocessing.py
Unit tests for the Phase 2B GOES-19 multichannel preprocessing prototype.
Tests channel reading, calibration decoding, geonavigation, cropping,
patch extraction, and temporal sequence generation.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.goes_reader import (
    read_goes_single_file,
    read_multichannel_timestamp,
    build_timestamp_catalog,
    GOESProjection,
)
from src.data.spatial_crop import (
    GeographicCropper,
    extract_patches,
)
from src.data.sequence_generator import (
    TemporalSequenceGenerator,
)


class TestGOESPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.abspath("data")
        cls.sample_ch13 = os.path.join(
            cls.data_dir,
            "goes19", "2026", "001", "04",
            "OR_ABI-L2-CMIPC-M6C13_G19_s20260010401193_e20260010403578_c20260010404062.nc"
        )
        cls.day1_dir = os.path.join(cls.data_dir, "goes19", "2026", "001")
        cls.catalog = build_timestamp_catalog(cls.day1_dir, channels=(9, 13, 14, 15))

    def test_01_single_file_reader_and_calibration(self):
        """Validates NetCDF reading, packing decoding, scale/offset calibration to Kelvin."""
        self.assertTrue(os.path.exists(self.sample_ch13), "Sample file does not exist")
        res = read_goes_single_file(self.sample_ch13, return_quality_mask=True)
        
        # Dimensions & types
        self.assertEqual(res["data_kelvin"].shape, (1500, 2500))
        self.assertEqual(res["data_kelvin"].dtype, np.float32)
        self.assertEqual(res["channel"], 13)
        self.assertAlmostEqual(res["wavelength"], 10.33, places=2)
        
        # Physical temperature limits in Kelvin
        valid_pixels = res["data_kelvin"][res["valid_mask"]]
        self.assertTrue(len(valid_pixels) > 0)
        self.assertGreater(float(np.min(valid_pixels)), 180.0, "Min temp below physical cloud limit")
        self.assertLess(float(np.max(valid_pixels)), 330.0, "Max temp above terrestrial surface limit")
        
        # Invalid pixel handling
        self.assertFalse(np.any(np.isnan(valid_pixels)), "NaNs detected in valid mask")
        self.assertEqual(res["fill_value"], -1)

    def test_02_multichannel_alignment_and_synchronization(self):
        """Validates 4-channel stacking, ordering, and synchronization."""
        self.assertGreater(len(self.catalog), 0, "Catalog is empty")
        first_ts = self.catalog[0]
        
        multi = read_multichannel_timestamp(first_ts["channels"], channel_order=(9, 13, 14, 15))
        self.assertEqual(multi["data_kelvin"].shape, (4, 1500, 2500))
        self.assertEqual(multi["channels"], [9, 13, 14, 15])
        
        # Check that valid mask has identical shape and high coverage
        self.assertEqual(multi["valid_mask"].shape, (1500, 2500))
        valid_ratio = np.mean(multi["valid_mask"])
        self.assertGreater(valid_ratio, 0.98, "Valid ratio unexpectedly low")

    def test_03_geonavigation_projection_inversion(self):
        """Validates analytical forward and inverse fixed grid transformations."""
        proj = GOESProjection()
        
        # Test coordinates in CONUS (Florida/Gulf coast)
        orig_lat, orig_lon = 30.083, -87.097
        x_rad, y_rad = proj.latlon_to_scan(orig_lat, orig_lon)
        rec_lat, rec_lon = proj.scan_to_latlon(x_rad, y_rad)
        
        self.assertAlmostEqual(orig_lat, rec_lat, places=3)
        self.assertAlmostEqual(orig_lon, rec_lon, places=3)
        
        # Test row/col conversion
        row, col = proj.latlon_to_rowcol(orig_lat, orig_lon)
        rec_lat2, rec_lon2 = proj.rowcol_to_latlon(row, col)
        self.assertAlmostEqual(orig_lat, rec_lat2, places=3)
        self.assertAlmostEqual(orig_lon, rec_lon2, places=3)

    def test_04_geographic_crop_and_patch_extraction(self):
        """Validates regional bounding box cropping and 128x128 patch extraction."""
        cropper = GeographicCropper(lat_min=25.0, lat_max=35.0, lon_min=-95.0, lon_max=-80.0)
        crop_info = cropper.get_crop_info()
        
        # Check reasonable dimensions inside CONUS
        r_min, r_max = crop_info["row_bounds"]
        c_min, c_max = crop_info["col_bounds"]
        self.assertTrue(0 <= r_min < r_max <= 1500)
        self.assertTrue(0 <= c_min < c_max <= 2500)
        self.assertGreater(r_max - r_min, 128)
        self.assertGreater(c_max - c_min, 128)
        
        # Crop representative data
        multi = read_multichannel_timestamp(self.catalog[0]["channels"])
        crop_data = cropper.crop_array(multi["data_kelvin"])
        crop_mask = cropper.crop_array(multi["valid_mask"])
        
        self.assertEqual(crop_data.shape, (4, r_max - r_min, c_max - c_min))
        self.assertTrue(np.all(crop_mask), "Cropped region contains unexpected fill pixels")
        
        # Extract patches
        patches = extract_patches(crop_data, crop_mask, patch_size=128, stride=64)
        self.assertGreater(len(patches), 0, "No patches extracted")
        p0 = patches[0]
        self.assertEqual(p0["patch_data"].shape, (4, 128, 128))
        self.assertFalse(np.any(np.isnan(p0["patch_data"])))

    def test_05_temporal_sequence_generator(self):
        """Validates sequence window discovery, offsets check, and tensor construction."""
        cropper = GeographicCropper(lat_min=25.0, lat_max=35.0, lon_min=-95.0, lon_max=-80.0)
        seq_gen = TemporalSequenceGenerator(
            catalog=self.catalog,
            input_offsets_min=(-60, -30, 0, 30),
            target_offsets_min=(15,),
            nominal_step_min=5.0,
            cropper=cropper
        )
        
        # Exactly 54 sliding windows on 72 continuous 5-minute scans
        self.assertEqual(len(seq_gen.valid_windows), 54)
        
        # Check Window 00 offsets and timestamps
        w0 = seq_gen.valid_windows[0]
        in_offsets = [f["offset_min"] for f in w0["input_frames"]]
        tgt_offsets = [f["offset_min"] for f in w0["target_frames"]]
        self.assertEqual(in_offsets, [-60, -30, 0, 30])
        self.assertEqual(tgt_offsets, [15])
        
        # Load window tensors
        tensors = seq_gen.load_window_tensors(0)
        self.assertEqual(tensors["input_tensors"].shape[0], 4, "Expected 4 input frames")
        self.assertEqual(tensors["input_tensors"].shape[1], 4, "Expected 4 channels")
        self.assertEqual(tensors["target_tensors"].shape[0], 1, "Expected 1 target frame")
        self.assertEqual(tensors["target_tensors"].shape[1], 4, "Expected 4 channels")
        
        # Extract aligned sequence patches
        patch_seqs = seq_gen.extract_sequence_patches(0, patch_size=128, stride=64)
        self.assertGreater(len(patch_seqs), 0)
        ps0 = patch_seqs[0]
        self.assertEqual(ps0["input_patches"].shape, (4, 4, 128, 128))
        self.assertEqual(ps0["target_patches"].shape, (1, 4, 128, 128))


if __name__ == "__main__":
    unittest.main()

"""
tests/test_insat_preprocessing.py
Automated tests for Phase 2D INSAT-3DS reader, LUT calibration,
fill masking, geographic cropping, and metadata preservation.
"""

import os
import glob
import datetime
import unittest
import numpy as np
import h5py

from src.data.insat_reader import (
    IR_CHANNELS,
    calibrate_counts_to_kelvin,
    read_insat_channel,
    read_insat_multichannel_frame,
    INSATRegionalCropper,
    combine_preprocessing_masks,
)

RAW_DIR = os.path.join("data", "insat3ds", "raw")
OUTPUT_DIR = os.path.join("outputs", "insat3ds_prototype")

# Original baseline raw file sizes from Phase 2A/B/C audit
EXPECTED_RAW_SIZES = {
    "3SIMG_20OCT2026_0000_L1B_STD_V01R00.h5": 429881910,
    "3SIMG_20OCT2026_0030_L1B_STD_V01R00.h5": 433901688,
    "3SIMG_20OCT2026_0100_L1B_STD_V01R00.h5": 438332589,
    "3SIMG_20OCT2026_0130_L1B_STD_V01R00.h5": 443028756,
    "3SIMG_20OCT2026_0200_L1B_STD_V01R00.h5": 448208094,
    "3SIMG_20OCT2026_0230_L1B_STD_V01R00.h5": 453503534,
    "3SIMG_20OCT2026_0300_L1B_STD_V01R00.h5": 459223982,
}


class TestINSATPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pattern = os.path.join(RAW_DIR, "*.h5")
        cls.raw_files = sorted(glob.glob(pattern))
        cls.sample_path = cls.raw_files[0]

    def test_01_lut_indexing_and_valid_dn(self):
        """Validates that valid digital counts map accurately to Kelvin temperatures."""
        with h5py.File(self.sample_path, "r") as h5:
            temp_lut = h5["IMG_TIR1_TEMP"][:]
            
            # Synthetic array of digital numbers
            raw_test = np.array([0, 100, 418, 647, 969, 1022], dtype=np.uint16)
            cal_k, v_mask = calibrate_counts_to_kelvin(raw_test, temp_lut, fill_value=1023)
            
            self.assertTrue(np.all(v_mask))
            self.assertEqual(cal_k[0], 340.0)
            self.assertAlmostEqual(float(cal_k[2]), 309.41, delta=0.05)
            self.assertAlmostEqual(float(cal_k[3]), 277.29, delta=0.05)
            self.assertAlmostEqual(float(cal_k[4]), 180.50, delta=0.05)
            self.assertEqual(cal_k[5], 180.0)

    def test_02_fill_value_1023_masking(self):
        """Validates that DN 1023 is explicitly masked as NaN and NEVER assigned 180 K."""
        with h5py.File(self.sample_path, "r") as h5:
            temp_lut = h5["IMG_TIR1_TEMP"][:]
            # Confirm raw LUT entry 1023 has 180.0 K (the trap)
            self.assertEqual(temp_lut[1023], 180.0)
            
            # Test array with fill values and out-of-bounds counts
            raw_test = np.array([1023, 1023, 418, 1024, 65535], dtype=np.uint16)
            cal_k, v_mask = calibrate_counts_to_kelvin(raw_test, temp_lut, fill_value=1023)
            
            # Index 0 and 1 must be invalid and NaN
            self.assertFalse(v_mask[0])
            self.assertFalse(v_mask[1])
            self.assertTrue(np.isnan(cal_k[0]))
            self.assertTrue(np.isnan(cal_k[1]))
            self.assertNotEqual(cal_k[0], 180.0)
            
            # Index 2 (DN 418) must be valid
            self.assertTrue(v_mask[2])
            self.assertAlmostEqual(float(cal_k[2]), 309.41, delta=0.05)
            
            # Out of bounds counts must also be invalid
            self.assertFalse(v_mask[3])
            self.assertFalse(v_mask[4])
            self.assertTrue(np.isnan(cal_k[3]))

    def test_03_calibration_output_shape_units_and_finite_values(self):
        """Validates multichannel frame reading, output dimensions, and physical finite ranges."""
        frame = read_insat_multichannel_frame(
            self.sample_path,
            channels=tuple(IR_CHANNELS),
            apply_calibration=True,
            load_coords=False,
            load_scan_line_time=False
        )
        
        # Shapes
        self.assertEqual(frame["raw_counts"].shape, (4, 2816, 2805))
        self.assertEqual(frame["data_kelvin"].shape, (4, 2816, 2805))
        self.assertEqual(frame["channel_masks"].shape, (4, 2816, 2805))
        self.assertEqual(frame["combined_mask"].shape, (2816, 2805))
        
        # Data types
        self.assertEqual(frame["raw_counts"].dtype, np.uint16)
        self.assertEqual(frame["data_kelvin"].dtype, np.float32)
        self.assertEqual(frame["combined_mask"].dtype, bool)
        
        # Valid Earth pixels must be strictly between 180.0 K and 340.0 K
        valid_k = frame["data_kelvin"][0][frame["combined_mask"]]
        self.assertGreaterEqual(float(np.min(valid_k)), 180.0)
        self.assertLessEqual(float(np.max(valid_k)), 340.0)
        self.assertTrue(np.all(np.isfinite(valid_k)))

    def test_04_exact_geographic_mask_vs_enclosing_crop(self):
        """Validates crop bounding box dimensions and compares exact mask vs enclosing rectangle."""
        frame = read_insat_multichannel_frame(
            self.sample_path,
            channels=tuple(IR_CHANNELS),
            apply_calibration=True,
            load_coords=True,
            load_scan_line_time=True
        )
        
        cropper = INSATRegionalCropper()
        cropped = cropper.crop_multichannel_frame(frame)
        
        # Enclosing slice dimensions: 780 rows x 867 cols
        self.assertEqual(cropped["crop_shape"], (780, 867))
        self.assertEqual(cropped["data_kelvin"].shape, (4, 780, 867))
        
        # Compare exact geographic mask (5-37N, 65-97E)
        exact_mask = cropped["exact_geo_mask"]
        total_rect = 780 * 867  # 676,260
        inside_exact = int(np.count_nonzero(exact_mask))
        outside_exact = total_rect - inside_exact
        
        self.assertEqual(total_rect, 676260)
        self.assertEqual(inside_exact, 620052)
        self.assertEqual(outside_exact, 56208)
        
        # Ensure that every pixel in exact mask has valid latitude & longitude
        lat_in = cropped["latitude"][exact_mask]
        lon_in = cropped["longitude"][exact_mask]
        self.assertTrue(np.all(lat_in >= 5.0))
        self.assertTrue(np.all(lat_in <= 37.0))
        self.assertTrue(np.all(lon_in >= 65.0))
        self.assertTrue(np.all(lon_in <= 97.0))

    def test_05_image_and_coordinate_shapes_consistency(self):
        """Validates that coordinate grids and image channels have identical shapes."""
        cropper = INSATRegionalCropper()
        with h5py.File(self.sample_path, "r") as h5:
            lat = h5["Latitude"][:]
            lon = h5["Longitude"][:]
            t1 = h5["IMG_TIR1"][0]
            
            self.assertEqual(lat.shape, t1.shape)
            self.assertEqual(lon.shape, t1.shape)
            
            lat_c = lat[cropper.row_slice, cropper.col_slice]
            lon_c = lon[cropper.row_slice, cropper.col_slice]
            t1_c = t1[cropper.row_slice, cropper.col_slice]
            
            self.assertEqual(lat_c.shape, (780, 867))
            self.assertEqual(lon_c.shape, (780, 867))
            self.assertEqual(t1_c.shape, (780, 867))

    def test_06_timestamp_ordering_and_scan_line_metadata(self):
        """Validates preserved scan line times and nominal 30-minute pacing across files."""
        # Check scan line times in first file
        frame0 = read_insat_multichannel_frame(self.raw_files[0], load_scan_line_time=True)
        slt0 = frame0["scan_line_times"]
        self.assertEqual(len(slt0), 1409)
        
        dt_start = datetime.datetime.strptime(slt0[0].split(".")[0], "%d-%b-%Y %H:%M:%S")
        dt_end = datetime.datetime.strptime(slt0[-1].split(".")[0], "%d-%b-%Y %H:%M:%S")
        duration = (dt_end - dt_start).total_seconds() / 60.0
        self.assertAlmostEqual(duration, 25.8, delta=1.5)
        
        # Check recurrence across consecutive files
        frame1 = read_insat_multichannel_frame(self.raw_files[1], load_scan_line_time=True)
        slt1 = frame1["scan_line_times"]
        
        dt0_delhi = datetime.datetime.strptime(slt0[334].split(".")[0], "%d-%b-%Y %H:%M:%S")
        dt1_delhi = datetime.datetime.strptime(slt1[334].split(".")[0], "%d-%b-%Y %H:%M:%S")
        interval = (dt1_delhi - dt0_delhi).total_seconds() / 60.0
        self.assertAlmostEqual(interval, 30.0, delta=0.5)

    def test_07_preservation_of_raw_source_files(self):
        """Validates that all raw source files are completely untouched and unmodified."""
        for fname, expected_bytes in EXPECTED_RAW_SIZES.items():
            fpath = os.path.join(RAW_DIR, fname)
            self.assertTrue(os.path.exists(fpath), f"Raw file missing: {fname}")
            actual_bytes = os.path.getsize(fpath)
            self.assertEqual(actual_bytes, expected_bytes,
                             f"Raw file {fname} size altered! Expected {expected_bytes}, found {actual_bytes}")

    def test_08_combine_preprocessing_masks_failure_modes(self):
        """Validates mask combination modes and checks failure modes (radiometric dropout inside geo vs valid Earth outside geo)."""
        fpath_0100 = os.path.join(OUTPUT_DIR, "3SIMG_20OCT2026_0100_L1B_STD_V01R00_regional.npz")
        self.assertTrue(os.path.exists(fpath_0100), "01:00 UTC regional npz must exist")
        data = np.load(fpath_0100)
        
        mir_mask = data["channel_masks"][3]  # IMG_MIR is index 3
        exact_geo = data["exact_geo_mask"]
        
        # Test mode 'intersection'
        inter = combine_preprocessing_masks(mir_mask, exact_geo, mode="intersection")
        self.assertEqual(inter.shape, exact_geo.shape)
        
        # Failure mode A: Radiometric dropout inside exact geo bounds (pixel 57, 587)
        self.assertTrue(exact_geo[57, 587], "Pixel (57, 587) should be inside exact geographic bounds")
        self.assertFalse(mir_mask[57, 587], "Pixel (57, 587) in 01:00 UTC MIR should be radiometric False (DN 1023 dropout)")
        self.assertFalse(inter[57, 587], "Intersection mask must reject radiometric dropout even though inside geo bounds")
        
        # Failure mode B: Valid Earth radiometric pixel outside exact geo bounds (corner margins)
        outside_geo_valid_rad = (~exact_geo) & mir_mask
        self.assertTrue(np.any(outside_geo_valid_rad), "There must be valid Earth pixels in the crop bounding box outside exact geo polygon")
        # Under intersection, these must be False
        self.assertTrue(np.all(~inter[outside_geo_valid_rad]), "Intersection must exclude pixels outside exact geo")
        
        # Under 'radiometric_only', these outside-geo valid pixels remain True
        rad_only = combine_preprocessing_masks(mir_mask, exact_geo, mode="radiometric_only")
        self.assertTrue(np.all(rad_only[outside_geo_valid_rad]))
        
        # Under 'geographic_only', pixel (57, 587) is True despite radiometric dropout
        geo_only = combine_preprocessing_masks(mir_mask, exact_geo, mode="geographic_only")
        self.assertTrue(geo_only[57, 587])
        
        # Error mode
        with self.assertRaises(ValueError):
            combine_preprocessing_masks(mir_mask, exact_geo, mode="invalid_mode")

    def test_09_mir_nan_source_verification(self):
        """Verifies the raw root cause of the 01:00 UTC IMG_MIR NaN: raw DN 1023 at Tibet pixel."""
        raw_0100 = os.path.join(RAW_DIR, "3SIMG_20OCT2026_0100_L1B_STD_V01R00.h5")
        cropper = INSATRegionalCropper()
        
        with h5py.File(raw_0100, "r") as h5:
            mir_raw = h5["IMG_MIR"][0]
            # Check global pixel coordinates: row 549, col 1530 (cropper offset: 492 + 57 = 549, 943 + 587 = 1530)
            global_r = cropper.row_slice.start + 57
            global_c = cropper.col_slice.start + 587
            self.assertEqual(global_r, 549)
            self.assertEqual(global_c, 1530)
            self.assertEqual(mir_raw[global_r, global_c], 1023, "Raw DN at (549, 1530) in 01:00 UTC must be fill value 1023")
            
            # Verify other channels at this identical coordinate are valid
            self.assertNotEqual(h5["IMG_TIR1"][0, global_r, global_c], 1023)
            self.assertNotEqual(h5["IMG_TIR2"][0, global_r, global_c], 1023)
            self.assertNotEqual(h5["IMG_WV"][0, global_r, global_c], 1023)
            
        # Verify in processed npz: calibrated is NaN and valid_mask is False
        npz_0100 = os.path.join(OUTPUT_DIR, "3SIMG_20OCT2026_0100_L1B_STD_V01R00_regional.npz")
        data = np.load(npz_0100)
        self.assertTrue(np.isnan(data["data_kelvin"][3, 57, 587]))
        self.assertFalse(data["channel_masks"][3, 57, 587])
        
        # Verify total NaNs across all channels in 01:00 UTC
        self.assertEqual(np.isnan(data["data_kelvin"][0]).sum(), 0)
        self.assertEqual(np.isnan(data["data_kelvin"][1]).sum(), 0)
        self.assertEqual(np.isnan(data["data_kelvin"][2]).sum(), 0)
        self.assertEqual(np.isnan(data["data_kelvin"][3]).sum(), 1)

    def test_10_scan_timing_sweep_mapping_and_recurrence(self):
        """Verifies 1409 sweeps to 2816 rows mapping (2 rows per sweep) and micro-jitter recurrence."""
        with h5py.File(self.sample_path, "r") as h5:
            slt = h5["SCAN_LINE_TIME"][:]
            img_shape = h5["IMG_TIR1"].shape
            self.assertEqual(len(slt), 1409)
            self.assertEqual(img_shape[1], 2816)
            self.assertEqual(round(img_shape[1] / len(slt)), 2)
            
        cropper = INSATRegionalCropper()
        # Regional crop: row 492 to 1271 -> sweep 246 to 635
        sweep_start = cropper.row_slice.start // 2
        sweep_end = (cropper.row_slice.stop - 1) // 2
        self.assertEqual(sweep_start, 246)
        self.assertEqual(sweep_end, 635)
        self.assertEqual(sweep_end - sweep_start + 1, 390)
        
        # Check recurrence intervals between consecutive scans in raw_files
        intervals = []
        for i in range(len(self.raw_files) - 1):
            with h5py.File(self.raw_files[i], "r") as h0, h5py.File(self.raw_files[i+1], "r") as h1:
                t0_str = h0["SCAN_LINE_TIME"][sweep_start].decode("utf-8")
                t1_str = h1["SCAN_LINE_TIME"][sweep_start].decode("utf-8")
                d0 = datetime.datetime.strptime(t0_str, "%d-%b-%Y %H:%M:%S.%f")
                d1 = datetime.datetime.strptime(t1_str, "%d-%b-%Y %H:%M:%S.%f")
                sec = (d1 - d0).total_seconds()
                intervals.append(sec)
                
        # All intervals should be ~1800 s but exhibit empirical micro-jitter (1799.5 s to 1800.7 s)
        for sec in intervals:
            self.assertTrue(1799.0 <= sec <= 1801.0, f"Interval {sec} outside expected 1800s pacing")
        self.assertTrue(any(abs(sec - 1800.0) > 0.3 for sec in intervals),
                        "Empirical pacing should reflect actual sensor micro-jitter rather than artificial 1800.0s")


if __name__ == "__main__":
    unittest.main()

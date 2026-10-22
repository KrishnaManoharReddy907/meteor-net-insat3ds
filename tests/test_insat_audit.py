"""
tests/test_insat_audit.py
Automated read-only integrity and validation tests for INSAT-3DS sample data files.
"""

import os
import glob
import datetime
import unittest
import numpy as np
import h5py

HDF5_MAGIC_BYTES = b'\x89HDF\r\n\x1a\n'
RAW_DIR = os.path.join("data", "insat3ds", "raw")


class TestINSAT3DSSampleAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pattern = os.path.join(RAW_DIR, "*.h5")
        cls.file_list = sorted(glob.glob(pattern))
        cls.expected_count = 7

    def test_01_files_exist_and_count(self):
        """Confirm all 7 expected INSAT-3DS sample files exist."""
        self.assertEqual(len(self.file_list), self.expected_count,
                         f"Expected {self.expected_count} files, found {len(self.file_list)}")

    def test_02_magic_bytes_and_file_sizes(self):
        """Confirm each file has nonzero size (> 300 MB) and valid HDF5 magic header."""
        for fpath in self.file_list:
            size_bytes = os.path.getsize(fpath)
            size_mb = size_bytes / (1024 * 1024)
            self.assertGreater(size_mb, 300.0, f"File {fpath} is unexpectedly small: {size_mb:.2f} MB")
            
            with open(fpath, "rb") as f:
                header = f.read(8)
                self.assertEqual(header, HDF5_MAGIC_BYTES,
                                 f"File {fpath} does not match HDF5 signature. Found: {header.hex()}")

    def test_03_h5py_readable_and_root_metadata(self):
        """Verify each file opens with h5py and has expected root attributes."""
        for fpath in self.file_list:
            with h5py.File(fpath, "r") as h5:
                # Check root attributes exist
                attrs = h5.attrs
                sat = attrs.get("Satellite_Name")
                sensor = attrs.get("Sensor_Name")
                prod = attrs.get("Product_Type")
                level = attrs.get("Processing_Level")
                
                # Satellite and Sensor
                sat_str = sat.decode('utf-8') if isinstance(sat, bytes) else str(sat)
                sensor_str = sensor.decode('utf-8') if isinstance(sensor, bytes) else str(sensor)
                prod_str = prod.decode('utf-8') if isinstance(prod, bytes) else str(prod)
                level_str = level.decode('utf-8') if isinstance(level, bytes) else str(level)
                
                self.assertIn("INSAT-3DS", sat_str)
                self.assertIn("IMAGER", sensor_str)
                self.assertIn("STANDARD", prod_str)
                self.assertIn("L1B", level_str)

    def test_04_timestamps_and_filename_agreement(self):
        """Verify internal timestamps agree with the filename acquisition time."""
        for fpath in self.file_list:
            fname = os.path.basename(fpath)
            # Example filename: 3SIMG_20OCT2026_0030_L1B_STD_V01R00.h5
            parts = fname.split("_")
            expected_date = parts[1]  # 20OCT2026
            expected_time = parts[2]  # 0030
            
            with h5py.File(fpath, "r") as h5:
                acq_date = h5.attrs.get("Acquisition_Date")
                acq_gmt = h5.attrs.get("Acquisition_Time_in_GMT")
                acq_start = h5.attrs.get("Acquisition_Start_Time")
                
                date_str = acq_date.decode('utf-8') if isinstance(acq_date, bytes) else str(acq_date)
                gmt_str = acq_gmt.decode('utf-8') if isinstance(acq_gmt, bytes) else str(acq_gmt)
                start_str = acq_start.decode('utf-8') if isinstance(acq_start, bytes) else str(acq_start)
                
                self.assertEqual(date_str, expected_date, f"Date mismatch for {fname}")
                self.assertEqual(gmt_str, expected_time, f"GMT time mismatch for {fname}")
                
                # Start time should have date and hour/min
                exp_hhmm = f"{expected_time[:2]}:{expected_time[2:]}"
                self.assertIn(exp_hhmm, start_str, f"Start time {start_str} does not contain {exp_hhmm}")

    def test_05_temporal_cadence(self):
        """Verify that consecutive files have a 30-minute interval."""
        start_datetimes = []
        for fpath in self.file_list:
            with h5py.File(fpath, "r") as h5:
                raw_start = h5.attrs.get("Acquisition_Start_Time")
                s_str = raw_start.decode('utf-8') if isinstance(raw_start, bytes) else str(raw_start)
                s_clean = s_str.split(".")[0]
                dt = datetime.datetime.strptime(s_clean, "%d-%b-%YT%H:%M:%S")
                start_datetimes.append(dt)
        
        start_datetimes.sort()
        for i in range(1, len(start_datetimes)):
            delta = (start_datetimes[i] - start_datetimes[i-1]).total_seconds() / 60.0
            self.assertAlmostEqual(delta, 30.0, delta=1.0,
                                   msg=f"Cadence anomaly between scan {i-1} and {i}: {delta} min")

    def test_06_infrared_dataset_shapes_and_types(self):
        """Verify the 4 infrared bands (TIR1, TIR2, WV, MIR) exist with shape (1, 2816, 2805)."""
        expected_bands = ["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_MIR"]
        expected_shape = (1, 2816, 2805)
        
        for fpath in self.file_list:
            with h5py.File(fpath, "r") as h5:
                for band in expected_bands:
                    self.assertIn(band, h5, f"Missing band {band} in {os.path.basename(fpath)}")
                    dset = h5[band]
                    self.assertEqual(dset.shape, expected_shape,
                                     f"Shape mismatch for {band} in {os.path.basename(fpath)}")
                    self.assertEqual(dset.dtype, np.uint16,
                                     f"Dtype mismatch for {band} in {os.path.basename(fpath)}")

    def test_07_calibration_lookup_tables(self):
        """Verify temperature and radiance lookup tables exist for all IR channels."""
        lut_names = [
            "IMG_TIR1_TEMP", "IMG_TIR1_RADIANCE",
            "IMG_TIR2_TEMP", "IMG_TIR2_RADIANCE",
            "IMG_WV_TEMP", "IMG_WV_RADIANCE",
            "IMG_MIR_TEMP", "IMG_MIR_RADIANCE"
        ]
        
        for fpath in self.file_list:
            with h5py.File(fpath, "r") as h5:
                for lut in lut_names:
                    self.assertIn(lut, h5, f"Missing LUT {lut} in {os.path.basename(fpath)}")
                    dset = h5[lut]
                    self.assertEqual(dset.shape, (1024,), f"Shape of {lut} must be (1024,)")
                    self.assertEqual(dset.dtype, np.float32, f"Dtype of {lut} must be float32")


if __name__ == '__main__':
    unittest.main()

"""
tests/test_insat_calibration_geo.py
Phase 2C automated scientific tests for INSAT-3DS calibration LUTs,
geolocation decoding, regional crop bounding, and scan line timing.
"""

import os
import glob
import datetime
import unittest
import numpy as np
import h5py

RAW_DIR = os.path.join("data", "insat3ds", "raw")


class TestINSAT3DSCalibrationAndGeolocation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pattern = os.path.join(RAW_DIR, "*.h5")
        cls.file_list = sorted(glob.glob(pattern))
        cls.sample_path = cls.file_list[0]

    def test_01_lookup_table_dimensions_and_types(self):
        """Verify brightness temperature and radiance LUTs exist for all 4 IR channels."""
        ir_channels = ["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_MIR"]
        with h5py.File(self.sample_path, "r") as h5:
            for ch in ir_channels:
                temp_lut = h5[f"{ch}_TEMP"]
                rad_lut = h5[f"{ch}_RADIANCE"]
                
                self.assertEqual(temp_lut.shape, (1024,))
                self.assertEqual(temp_lut.dtype, np.float32)
                self.assertEqual(rad_lut.shape, (1024,))
                self.assertEqual(rad_lut.dtype, np.float32)

    def test_02_temperature_lut_values_and_clamping(self):
        """Verify TIR-1 LUT values at key digital counts and check saturation clamping."""
        with h5py.File(self.sample_path, "r") as h5:
            temp_lut = h5["IMG_TIR1_TEMP"][:]
            
            # Count 0: maximum saturation clamp
            self.assertEqual(temp_lut[0], 340.0)
            
            # Count 418: verify actual value is 309.41 K, disproving the linear 312 K estimate
            self.assertAlmostEqual(float(temp_lut[418]), 309.41, delta=0.05,
                                   msg="TIR-1 count 418 must be ~309.41 K, not 312 K")
            
            # Count 647 (sample mean)
            self.assertAlmostEqual(float(temp_lut[647]), 277.29, delta=0.05)
            
            # Count 969
            self.assertAlmostEqual(float(temp_lut[969]), 180.50, delta=0.05)
            
            # Count 1022 and 1023 (fill)
            self.assertEqual(temp_lut[1022], 180.0)
            self.assertEqual(temp_lut[1023], 180.0)
            
            # Saturation plateaus
            # Max 340 K holds for count 0..138
            self.assertTrue(np.all(temp_lut[:139] == 340.0))
            # Min 180 K holds for count 970..1023
            self.assertTrue(np.all(temp_lut[970:1024] == 180.0))

    def test_03_gsics_radiance_agreement(self):
        """Verify radiance LUT closely matches inverted GSICS linear formula."""
        with h5py.File(self.sample_path, "r") as h5:
            for ch in ["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_MIR"]:
                dset = h5[ch]
                rad_lut = h5[f"{ch}_RADIANCE"][:]
                
                s_gs = float(np.squeeze(dset.attrs["online_radiance_scale_factor_gsics"]))
                o_gs = float(np.squeeze(dset.attrs["online_radiance_add_offset_gsics"]))
                
                counts = np.arange(1024)
                rad_gs_inv = s_gs * (1023 - counts) + o_gs
                
                # Exclude bottom counts where LUT is clamped to 0.0
                unclamped = rad_lut[:950] > 0.05
                mae = np.mean(np.abs(rad_lut[:950][unclamped] - rad_gs_inv[:950][unclamped]))
                self.assertLess(mae, 0.001, f"GSICS radiance formula does not match LUT for {ch}: MAE={mae}")

    def test_04_coordinate_decoding_and_ranges(self):
        """Verify latitude/longitude decoding and sub-satellite point center."""
        with h5py.File(self.sample_path, "r") as h5:
            lat_raw = h5["Latitude"][:]
            lon_raw = h5["Longitude"][:]
            
            lat_scale = float(np.squeeze(h5["Latitude"].attrs.get("scale_factor", 0.01)))
            lon_scale = float(np.squeeze(h5["Longitude"].attrs.get("scale_factor", 0.01)))
            fill_val = int(np.squeeze(h5["Latitude"].attrs.get("_FillValue", 32767)))
            
            valid = (lat_raw != fill_val) & (lon_raw != fill_val)
            lat_deg = lat_raw[valid] * lat_scale
            lon_deg = lon_raw[valid] * lon_scale
            
            self.assertAlmostEqual(float(np.min(lat_deg)), -81.04, delta=0.01)
            self.assertAlmostEqual(float(np.max(lat_deg)), 81.04, delta=0.01)
            self.assertAlmostEqual(float(np.min(lon_deg)), 0.84, delta=0.01)
            self.assertAlmostEqual(float(np.max(lon_deg)), 163.15, delta=0.01)
            
            # Sub-satellite point (0N, 82E) at row 1408, col 1402
            ssp_lat = lat_raw[1408, 1402] * lat_scale
            ssp_lon = lon_raw[1408, 1402] * lon_scale
            self.assertAlmostEqual(ssp_lat, 0.0, delta=0.05)
            self.assertAlmostEqual(ssp_lon, 82.0, delta=0.05)

    def test_05_target_crop_bounds_and_validity(self):
        """Verify target crop bounding box for India (5N-37N, 65E-97E) and 100% validity."""
        with h5py.File(self.sample_path, "r") as h5:
            lat_raw = h5["Latitude"][:]
            lon_raw = h5["Longitude"][:]
            lat_scale = float(np.squeeze(h5["Latitude"].attrs.get("scale_factor", 0.01)))
            lon_scale = float(np.squeeze(h5["Longitude"].attrs.get("scale_factor", 0.01)))
            fill_val = int(np.squeeze(h5["Latitude"].attrs.get("_FillValue", 32767)))
            
            valid = (lat_raw != fill_val) & (lon_raw != fill_val)
            lat_deg = np.full_like(lat_raw, np.nan, dtype=np.float32)
            lon_deg = np.full_like(lon_raw, np.nan, dtype=np.float32)
            lat_deg[valid] = lat_raw[valid] * lat_scale
            lon_deg[valid] = lon_raw[valid] * lon_scale
            
            in_target = valid & (lat_deg >= 5.0) & (lat_deg <= 37.0) & (lon_deg >= 65.0) & (lon_deg <= 97.0)
            rows, cols = np.where(in_target)
            
            r_min, r_max = int(np.min(rows)), int(np.max(rows))
            c_min, c_max = int(np.min(cols)), int(np.max(cols))
            
            self.assertEqual(r_min, 492)
            self.assertEqual(r_max, 1271)
            self.assertEqual(c_min, 943)
            self.assertEqual(c_max, 1809)
            
            # Check crop contains zero space/fill pixels
            crop_valid = valid[r_min:r_max+1, c_min:c_max+1]
            self.assertEqual(int(np.count_nonzero(~crop_valid)), 0,
                             "Target crop box contains space/fill pixels!")

    def test_06_temporal_scan_line_metadata(self):
        """Verify SCAN_LINE_TIME dataset structure, 1409 sweeps, and North-to-South direction."""
        with h5py.File(self.sample_path, "r") as h5:
            self.assertIn("SCAN_LINE_TIME", h5)
            slt = h5["SCAN_LINE_TIME"][:]
            self.assertEqual(slt.shape, (1409,))
            
            slt_dec = [x.decode('utf-8').strip() for x in slt]
            # Verify North-to-South scan time monotonicity
            dts = [datetime.datetime.strptime(s.split('.')[0], "%d-%b-%Y %H:%M:%S") for s in slt_dec]
            for i in range(1, len(dts)):
                self.assertGreaterEqual(dts[i], dts[i-1], f"Scan time decreased at line {i}")
            
            # Verify total sweep duration is ~25.8 to 26.9 minutes
            sweep_dur = (dts[-1] - dts[0]).total_seconds() / 60.0
            self.assertAlmostEqual(sweep_dur, 25.8, delta=1.5)


if __name__ == '__main__':
    unittest.main()

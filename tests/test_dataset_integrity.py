"""
tests/test_dataset_integrity.py
Unit tests for duplicate-file detection, conflict resolution policies,
and event-level dataset split manifest generation.
"""

import os
import sys
import shutil
import tempfile
import unittest
import json
import datetime

sys.path.insert(0, os.path.abspath("."))
from src.data.goes_reader import (
    build_timestamp_catalog,
    compute_file_sha256,
    audit_catalog_duplicates,
)
from scripts.create_dataset_split import create_dataset_splits


class TestDatasetIntegrityAndSplits(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.real_data_dir = os.path.abspath("data")
        self.real_day1_dir = os.path.join(self.real_data_dir, "goes19", "2026", "001")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_compute_file_sha256(self):
        """Test SHA-256 computation utility."""
        test_file = os.path.join(self.tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("satellite_test_content_12345")
        
        sha = compute_file_sha256(test_file)
        self.assertEqual(len(sha), 64)
        # Verify deterministic hash
        self.assertEqual(sha, compute_file_sha256(test_file))

    def test_normal_catalog_construction_unaffected(self):
        """Test that normal catalog construction on real data continues to function identically."""
        catalog = build_timestamp_catalog(self.real_day1_dir, channels=(9, 13, 14, 15))
        self.assertEqual(len(catalog), 72)
        for entry in catalog:
            self.assertEqual(set(entry["channels"].keys()), {9, 13, 14, 15})

    def test_exact_duplicates_detection_and_resolution(self):
        """Test that exact file duplicates (same size & SHA-256) are deduplicated without error."""
        # Find one real file to replicate
        sample_files = [f for f in os.listdir(os.path.join(self.real_data_dir, "goes19", "2026", "001", "04")) if f.endswith(".nc")]
        sample_name = sample_files[0]
        src_path = os.path.join(self.real_data_dir, "goes19", "2026", "001", "04", sample_name)

        # Create two subdirectories in tmpdir with exact duplicate files
        dir1 = os.path.join(self.tmpdir, "dir1")
        dir2 = os.path.join(self.tmpdir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        shutil.copyfile(src_path, os.path.join(dir1, sample_name))
        shutil.copyfile(src_path, os.path.join(dir2, sample_name))

        # Scan with audit
        catalog, audit = build_timestamp_catalog(
            self.tmpdir,
            channels=(int(sample_name.split("_")[1].split("-")[-1][3:]),),
            on_conflict="raise",
            return_audit=True
        )

        self.assertEqual(audit["exact_duplicate_slots"], 1)
        self.assertEqual(audit["conflicting_slots"], 0)
        self.assertEqual(len(catalog), 1)
        # Resolved path should be one of the duplicates
        resolved_path = catalog[0]["channels"][int(sample_name.split("_")[1].split("-")[-1][3:])]
        self.assertIn(resolved_path, [os.path.join(dir1, sample_name), os.path.join(dir2, sample_name)])

    def test_conflicting_files_raise_error(self):
        """Test that conflicting files (different size/hash) raise a clear ValueError when on_conflict='raise'."""
        sample_files = [f for f in os.listdir(os.path.join(self.real_data_dir, "goes19", "2026", "001", "04")) if f.endswith(".nc")]
        sample_name = sample_files[0]
        src_path = os.path.join(self.real_data_dir, "goes19", "2026", "001", "04", sample_name)

        dir1 = os.path.join(self.tmpdir, "dir1")
        dir2 = os.path.join(self.tmpdir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        shutil.copyfile(src_path, os.path.join(dir1, sample_name))
        # Create a modified/corrupted file in dir2
        with open(os.path.join(dir2, sample_name), "wb") as f:
            f.write(b"CORRUPTED_OR_DIFFERENT_RETRANSMISSION_DATA")

        ch_num = int(sample_name.split("_")[1].split("-")[-1][3:])

        with self.assertRaises(ValueError) as ctx:
            build_timestamp_catalog(self.tmpdir, channels=(ch_num,), on_conflict="raise")
        self.assertIn("Conflicting files detected", str(ctx.exception))

    def test_conflicting_files_excluded_safely(self):
        """Test that conflicting files are excluded safely when on_conflict='exclude'."""
        sample_files = [f for f in os.listdir(os.path.join(self.real_data_dir, "goes19", "2026", "001", "04")) if f.endswith(".nc")]
        sample_name = sample_files[0]
        src_path = os.path.join(self.real_data_dir, "goes19", "2026", "001", "04", sample_name)

        dir1 = os.path.join(self.tmpdir, "dir1")
        dir2 = os.path.join(self.tmpdir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        shutil.copyfile(src_path, os.path.join(dir1, sample_name))
        with open(os.path.join(dir2, sample_name), "wb") as f:
            f.write(b"DIFFERENT_BYTES_IN_SECONDARY_FILE")

        ch_num = int(sample_name.split("_")[1].split("-")[-1][3:])

        catalog, audit = build_timestamp_catalog(
            self.tmpdir,
            channels=(ch_num,),
            on_conflict="exclude",
            return_audit=True
        )

        self.assertEqual(audit["conflicting_slots"], 1)
        self.assertEqual(audit["exact_duplicate_slots"], 0)
        # Incomplete timestamp must be excluded from valid catalog
        self.assertEqual(len(catalog), 0)

    def test_audit_catalog_duplicates_utility(self):
        """Test the standalone audit_catalog_duplicates function on real data."""
        audit_json = os.path.join(self.tmpdir, "audit_report.json")
        summary = audit_catalog_duplicates(self.real_day1_dir, output_json=audit_json)
        
        self.assertTrue(os.path.exists(audit_json))
        self.assertEqual(summary["exact_duplicate_slots"], 0)
        self.assertEqual(summary["conflicting_slots"], 0)
        self.assertEqual(summary["valid_synchronized_timestamps"], 72)

    def test_split_manifest_insufficient_groups(self):
        """Test that split generator refuses to manufacture 3-way splits when only 1 group exists."""
        manifest_path = os.path.join(self.tmpdir, "test_manifest.json")
        manifest = create_dataset_splits(
            data_dir=self.real_day1_dir,
            output_manifest=manifest_path,
            seed=42,
            compute_hashes=False
        )

        self.assertTrue(os.path.exists(manifest_path))
        self.assertFalse(manifest["evaluation_validity"]["scientifically_valid_three_way_split_possible"])
        self.assertEqual(manifest["evaluation_validity"]["split_status"], "INSUFFICIENT_INDEPENDENT_GROUPS")
        self.assertEqual(manifest["summary"]["total_sequence_windows"], 54)
        # All windows assigned to train development split
        self.assertEqual(manifest["summary"]["train_sequences"], 54)
        self.assertEqual(manifest["summary"]["val_sequences"], 0)
        self.assertEqual(manifest["summary"]["test_sequences"], 0)
        self.assertIn("Scientifically valid 3-way evaluation is NOT possible", manifest["evaluation_validity"]["warning"])

    def test_multievent_four_group_split_success(self):
        """Test that full 4-event dataset produces scientifically valid 3-way event-level split."""
        catalog = build_timestamp_catalog(self.real_data_dir, channels=(9, 13, 14, 15))
        self.assertEqual(len(catalog), 288)

        manifest_path = os.path.join(self.tmpdir, "four_event_manifest.json")
        manifest = create_dataset_splits(
            data_dir=self.real_data_dir,
            output_manifest=manifest_path,
            seed=42,
            compute_hashes=False
        )
        self.assertTrue(manifest["evaluation_validity"]["scientifically_valid_three_way_split_possible"])
        self.assertEqual(manifest["evaluation_validity"]["split_status"], "VALID_EVENT_SPLIT")
        self.assertEqual(manifest["summary"]["total_sequence_windows"], 216)
        self.assertEqual(manifest["summary"]["train_sequences"], 108)
        self.assertEqual(manifest["summary"]["val_sequences"], 54)
        self.assertEqual(manifest["summary"]["test_sequences"], 54)
        self.assertIsNotNone(manifest.get("normalization_provenance"))


if __name__ == "__main__":
    unittest.main()

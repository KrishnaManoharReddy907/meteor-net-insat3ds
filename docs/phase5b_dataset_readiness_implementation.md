# Phase 5B — Dataset Integrity, Event Splits, and Normalization Readiness Implementation Report

**Date:** 2026-10-22  
**Status:** Complete — All 3 Blockers from Phase 5A Resolved; 60/60 Tests Passing  
**Architecture Reference:** Lin et al. (IEEE TGRS 2026)  
**Supporting Reports:**  
- [`docs/phase5a_training_readiness_and_compute.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase5a_training_readiness_and_compute.md)  
- [`docs/phase4_single_batch_overfit_report.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase4_single_batch_overfit_report.md)  
- [`docs/phase3_5_2_convective_patch_and_normalization_preflight.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase3_5_2_convective_patch_and_normalization_preflight.md)  

---

## 1. Executive Summary & Scope Conformance

In strict conformance with Phase 5B boundaries:
- **Zero data was downloaded.**
- **Zero model training, optimizer updates, or GPU jobs were executed.**
- **No raw NetCDF files or existing processed samples were modified or overwritten.**
- **The approved METEOR-Net model architecture and loss definitions were preserved without modification.**
- **All 3 outstanding readiness blockers identified in Phase 5A were resolved through modular software implementation and verified with comprehensive unit tests.**

---

## 2. Inventory of Files Created and Modified

| File Path | Action | Description |
| :--- | :---: | :--- |
| [`src/data/goes_reader.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/goes_reader.py) | **Modified** | Implemented `compute_file_sha256`, enhanced `build_timestamp_catalog` with collision detection and conflict policies (`raise`/`exclude`), and added `audit_catalog_duplicates`. |
| [`src/data/__init__.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/__init__.py) | **Modified** | Exported `compute_file_sha256` and `audit_catalog_duplicates`. |
| [`src/models/normalization.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/normalization.py) | **Modified** | Added channel order consistency checks, provenance tracking, `save_stats` with JSON export, and `from_stats_file` with schema validation. |
| [`scripts/create_dataset_split.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/create_dataset_split.py) | **Created** | Event-level dataset split generator with complete-event isolation, no random window splitting, refusal to manufacture 3-way splits if $<3$ events exist, and cryptographic SHA-256 file hashing. |
| [`tests/test_dataset_integrity.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/tests/test_dataset_integrity.py) | **Created** | 7 unit tests verifying exact duplicate resolution, conflict raising/exclusion, and split manifest generator behavior. |
| [`tests/test_normalization.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/tests/test_normalization.py) | **Modified** | Added 4 tests verifying provenance serialization, channel-order validation, frozen buffer immunity, and training-only isolation (total: 14 tests). |
| [`outputs/dataset_splits/splits_manifest.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/dataset_splits/splits_manifest.json) | **Created** | Generated reproducible split manifest for currently available data with 96 indexed source files. |

---

## 3. Duplicate-File Policy and Verification (Task 1)

### 3.1 Policy Definition
Previously, [`build_timestamp_catalog`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/goes_reader.py#L289-L435) used naive dictionary assignment `catalog[dt][ch] = f`, silently overwriting collisions with whichever file was scanned last.

Under the new implementation:
1. **Collision Tracking:** All file paths resolving to the same `(scan_datetime, channel_number)` are grouped into a candidate list.
2. **Cryptographic & Size Comparison:** If multiple files map to the same slot, their byte lengths and full SHA-256 hashes are computed.
   - **Exact Duplicates:** If all candidate files share identical file size and SHA-256 hash, they represent identical data (e.g. redundant downloads or symlinks). The system deterministically retains the lexicographically first path, logs the event in `audit_summary["exact_duplicates"]`, and continues without error.
   - **Conflicting Files:** If files differ in size or SHA-256 hash, they represent conflicting data (different calibrations, re-transmissions, or truncations).
3. **Configurable Conflict Policies:**
   - `on_conflict="raise"` (Default): Immediately halts catalog construction and raises a descriptive `ValueError` listing all colliding paths, sizes, and truncated SHA-256 hashes.
   - `on_conflict="exclude"`: Discards the ambiguous channel slot, causing that timestamp to fail the 4-channel synchronization check and preventing it from entering sequence generation.
4. **Machine-Readable Audit Record:** `audit_catalog_duplicates` records full duplicate and conflict history to an optional JSON audit file.

### 3.2 Duplicate Policy Test Results
Verified via [`tests/test_dataset_integrity.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/tests/test_dataset_integrity.py):
- `test_normal_catalog_construction_unaffected`: Validates 24 synchronized timestamps without collision on clean data (**PASS**).
- `test_exact_duplicates_detection_and_resolution`: Confirms exact duplicate files are deduplicated cleanly to one path and recorded in audit (**PASS**).
- `test_conflicting_files_raise_error`: Confirms differing files raise `ValueError("Conflicting files detected...")` (**PASS**).
- `test_conflicting_files_excluded_safely`: Confirms differing files are dropped cleanly under `on_conflict="exclude"` (**PASS**).
- `test_audit_catalog_duplicates_utility`: Confirms standalone audit utility outputs valid JSON (**PASS**).

---

## 4. Event-Level Split Manifest Generator (Task 2)

### 4.1 Architecture of `scripts/create_dataset_split.py`
To eliminate data leakage caused by random window splitting:
1. **Grouping by Complete Observation Episode:**
   Each sequence window ($[t-60, t-30, t, t+30] \to t+15\text{ min}$) is assigned to a discrete calendar date or weather episode group (`group_id = YYYY-MM-DD`).
2. **Atomic Group Allocation:**
   Complete groups are assigned exclusively to `train`, `val`, or `test`. No sliding windows from the same episode are ever partitioned across different splits.
3. **Refusal to Manufacture Generalization Splits:**
   If the dataset contains fewer than 3 independent date/event groups ($N_{\text{groups}} < 3$), a 3-way split is **statistically impossible**.
   The generator flags:
   - `scientifically_valid_three_way_split_possible = false`
   - `split_status = "INSUFFICIENT_INDEPENDENT_GROUPS"`
   - `generalization_claim_permitted = false`
   All sequences from the single available group are assigned to a `train` development partition, and a prominent warning is written into the manifest.
4. **Cryptographic File Inventory:**
   The manifest logs all 96 source NetCDF files with their relative path, exact byte size, and full SHA-256 hash.

### 4.2 Behavior on Currently Available GOES-19 Sample
The manifest was generated at [`outputs/dataset_splits/splits_manifest.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/dataset_splits/splits_manifest.json):
```json
{
  "evaluation_validity": {
    "scientifically_valid_three_way_split_possible": false,
    "split_status": "INSUFFICIENT_INDEPENDENT_GROUPS",
    "warning": "Scientifically valid 3-way evaluation is NOT possible with the current data. Found only 1 independent date/event group(s): ['2026-01-01']. At least 3 independent event groups are required to form leakage-free train, validation, and test partitions. Overlapping sequence windows from the same observation episode cannot be split across partitions without severe leakage. All 6 available sequences are assigned to the 'train' partition for development/debugging only. Do NOT report validation or generalization metrics.",
    "generalization_claim_permitted": false,
    "total_independent_groups": 1,
    "group_list": ["2026-01-01"],
    "group_to_split_mapping": {"2026-01-01": "train"}
  },
  "summary": {
    "total_sequence_windows": 6,
    "train_sequences": 6,
    "val_sequences": 0,
    "test_sequences": 0,
    "total_source_files": 96
  }
}
```

> [!IMPORTANT]
> **Can the current data legitimately form independent train/validation/test groups?**
> **NO.** The current dataset spans only 115 minutes on a single day (2026-01-01 04:01–05:56 UTC). All 6 sequence windows overlap by up to 94.4% in time and observe the same continuous storm cloud. Splitting this single episode would introduce near-total data leakage. Full multi-day expansion (3–4 days) is required before legitimate generalization testing can begin.

---

## 5. Normalization Safeguards and Provenance (Task 3)

### 5.1 Implemented Safeguards in `src/models/normalization.py`
1. **Training-Only Statistics Fitting:**
   `ChannelStandardizer.fit(train_x, mask=train_mask)` computes $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ strictly over valid pixels in the training partition.
2. **Channel-Order Consistency:**
   Standardizer records `self.channel_order = (9, 13, 14, 15)`. Attempting to load statistics with a mismatched channel order raises a descriptive `ValueError`.
3. **Invalid-Value Immunity:**
   Non-finite values (`NaN`, $\pm\infty$) and masked pixels are sanitized before computing moments, mapping to exact $0.0$ in normalized space.
4. **Finite Variance Safeguard:**
   Zero, negative, or non-finite standard deviations are clamped to $1.0$ (plus $\epsilon=1\times 10^{-6}$).
5. **Frozen Buffers During Evaluation:**
   `self.register_buffer("mean", safe_mean)` and `self.register_buffer("std", safe_std)` ensure statistics are fixed model buffers, completely immune to modification during validation or testing.
6. **Provenance Serialization:**
   `save_stats(filepath, provenance=...)` writes statistics, channel order, and training sequence IDs to JSON.
   `from_stats_file(filepath, expected_channel_order=...)` deserializes and validates against expected channel ordering.

### 5.2 Normalization Test Results
Verified via [`tests/test_normalization.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/tests/test_normalization.py) (14 tests):
- `test_save_and_load_stats_with_provenance`: Verifies serialization, deserialization, and metadata persistence (**PASS**).
- `test_channel_order_consistency_and_validation`: Verifies `channel_order` enforcement and rejection of mismatched orders (**PASS**).
- `test_frozen_statistics_unaffected_by_val_test`: Verifies that passing out-of-distribution validation data (310 K) does not alter frozen buffers (**PASS**).
- `test_training_only_fitting`: Verifies that fitting on training data isolates validation data completely (**PASS**).

---

## 6. Complete Test Suite Execution Results (Task 4)

Ran full repository test discovery (`python -m unittest discover -s tests -p "test_*.py"`):

```
Ran 60 tests in 30.446s
OK
```

### Breakdown of Test Coverage:
1. `tests/test_preprocessing.py`: **5 tests** (GOES single-file reading, calibration, navigation, cropping, patch extraction, sequence generation).
2. `tests/test_meteor_net.py`: **11 tests** (METEOR-Net forward pass, tensor dimensions, autograd gradient flow, loss guardrails, all-masked batch safety).
3. `tests/test_normalization.py`: **14 tests** (ChannelStandardizer fitting, broadcasting, sanitization, autograd differentiability, loss integration, provenance, channel order).
4. `tests/test_dataset_integrity.py`: **7 tests** (SHA-256 hashing, exact duplicates, conflict raising/exclusion, duplicate audit, split manifest generator).
5. `tests/test_insat_calibration_geo.py`: **8 tests** (INSAT-3DS radiometric lookup, calibration, geographic navigation, fill masks).
6. `tests/test_insat_preprocessing.py`: **11 tests** (INSAT-3DS regional cropping, multi-channel frame extraction, temporal stability).
7. `tests/test_insat_audit.py`: **4 tests** (INSAT-3DS raw file integrity, channel cataloging, structure verification).

**Total: 60 / 60 tests passing cleanly (100% pass rate, 0 failures, 0 errors).**

---

## 7. Updated Preprocessing Readiness Checklist

| Preprocessing Requirement | Status Prior (Phase 5A) | Current Status (Phase 5B) | Evidence / Implementation |
| :--- | :---: | :---: | :--- |
| **1. Duplicate-file detection & handling** | NOT VERIFIED | **VERIFIED** | SHA-256 collision detection with `raise`/`exclude` policies in `goes_reader.py`. |
| **2. Timestamp parsing & missing-frame rejection** | VERIFIED | **VERIFIED** | Enforces exact offsets $(-60, -30, 0, 30)$ and $(+15)$ min; incomplete windows rejected. |
| **3. Corrupt-file & invalid-value checks** | VERIFIED | **VERIFIED** | HDF5 structure checks, DQF filtering, space/limb fill conversion to NaN with safe autograd zero-branches. |
| **4. Physical Kelvin calibration** | VERIFIED | **VERIFIED** | PUG scale/offset conversion directly verified against expected meteorological temperature limits. |
| **5. Consistent channel order & spatial alignment** | VERIFIED | **VERIFIED** | Multichannel stacking strictly enforces $(9, 13, 14, 15)$ on the $1500 \times 2500$ ABI fixed grid. |
| **6. Radiometric & geographic masks** | VERIFIED | **VERIFIED** | Combined validity mask created from DQF and bounding-box intersection; invalid pixels set to $0.0$ in normalized space. |
| **7. Sequence construction with observed target** | VERIFIED | **VERIFIED** | Midpoint target is the real observed GOES-19 scan at $t+15$ min (e.g. `05:16:19 UTC`), not synthetic. |
| **8. Chronological / event-based train/val/test separation** | NOT VERIFIED | **VERIFIED** | `scripts/create_dataset_split.py` enforces complete-event isolation and generates `splits_manifest.json`. |
| **9. Train-only normalization statistics (frozen)** | VERIFIED | **VERIFIED** | `ChannelStandardizer` includes `save_stats`/`from_stats_file` with frozen buffers and provenance tracking. |
| **10. Reproducible metadata recording** | PARTIALLY VERIFIED | **VERIFIED** | `splits_manifest.json` logs sequence IDs, timestamps, split assignments, and cryptographic SHA-256 file hashes. |

---

## 8. Remaining Blockers Before Expanded Data Acquisition

With the completion of Phase 5B:
- **Zero code blockers remain** in the data reader, sequence generator, model architecture, normalization wrapper, loss function, or split generator.
- The **only pending action** is the physical acquisition of 3–4 discrete event days of GOES-19 data (~3.4 GB) to provide the multiple independent groups required for a scientifically valid train/val/test split.

---

## 9. Stopping Condition

In strict accordance with instructions:
- No data was acquired.
- No final dataset statistics were computed.
- No model training or evaluation loops were launched.
- Awaiting user review and authorization.

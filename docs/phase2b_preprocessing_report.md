# Phase 2B Preprocessing Prototype Report: GOES-19 Ingestion, Spatial Cropping, and Sequence Generation

**Project Title**: Confidence-Aware Temporal Super-Resolution of INSAT-3DS Satellite Imagery using a METEOR-Net-based, flow-free temporal-evolution architecture  
**Date**: October 16, 2026  
**Status**: Phase 2B Complete — Preprocessing Pipeline Validated, Tested, and Benchmarked  

---

## 1. Executive Summary

In Phase 2B, we implemented and validated an end-to-end, modular, and reversible preprocessing prototype on the 24 synchronized GOES-19 observations (96 files total) spanning UTC hours `04` and `05` on January 1, 2026.

Following the approved Phase 2A decisions:
1. **Temporal Sequence Design**: Implemented Option 1 with 4 input frames at $[t-60\text{m}, t-30\text{m}, t, t+30\text{m}]$ and a single observed midpoint target at $t+15\text{m}$ ($k-12, k-6, k, k+6 \to k+3$).
2. **Native Resolution Retention**: Retained the native GOES-19 $2\text{ km}$ spatial grid during initial preprocessing and sequence construction.
3. **No Bulk Download Yet**: The pipeline was thoroughly tested and verified on the existing 2-hour sample, generating storage estimates before committing to larger downloads.

All 5 unit tests in our newly created test suite pass with 100% success. Four high-resolution scientific visualizations and a compressed prototype tensor dataset were produced and validated.

---

## 2. Preprocessing Architecture & Code Modules

The preprocessing pipeline is organized into modular components under `src/data/` and execution scripts under `scripts/`:

```text
MAJOR PROJECT/
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py               # Package exports
│       ├── goes_reader.py            # NetCDF-4 unpacking, calibration, projection & cataloging
│       ├── spatial_crop.py           # Geonavigation, geographic cropping & 128x128 patch extraction
│       └── sequence_generator.py     # Multi-frame sequence windowing & tensor assembly
├── scripts/
│   ├── inspect_goes.py               # Standalone inspection CLI tool
│   └── run_preprocessing.py          # Prototype execution CLI and visualization generator
├── tests/
│   └── test_preprocessing.py         # Automated unit test suite (5 passing tests)
├── outputs/
│   ├── preprocessing/                # High-resolution scientific PNG visualizations
│   │   ├── 01_full_conus_4channels.png
│   │   ├── 02_geographic_crop_validation.png
│   │   ├── 03_patch_samples_128x128.png
│   │   └── 04_sequence_evolution_5frames.png
│   └── processed_sample/             # Compressed sequence tensors & metadata
│       ├── sample_sequence_window00.npz
│       └── sequence_metadata.json
└── docs/
    ├── dataset_inspection.md         # Preserved Phase 1 report
    ├── phase2_design_review.md       # Preserved Phase 2A report
    └── phase2b_preprocessing_report.md# [This Document] Phase 2B report
```

### Module Responsibilities
- **`goes_reader.py`**:
  - Encapsulates `GOESProjection`: Performs forward and inverse geostationary projection transformations directly from file metadata ($x, y \text{ radians} \leftrightarrow \text{Latitude, Longitude}$).
  - `read_goes_single_file`: Unpacks raw `int16` NetCDF datasets using `scale_factor` and `add_offset` to physical Kelvin, avoiding double-scaling. Strictly handles `_FillValue = -1` and Data Quality Flags (`DQF == 0`).
  - `read_multichannel_timestamp`: Stacks synchronized Channels 9, 13, 14, and 15 into arrays of shape `(4, 1500, 2500)`.
  - `build_timestamp_catalog`: Scans folder hierarchies and groups verified files by timestamp.
- **`spatial_crop.py`**:
  - `GeographicCropper`: Converts geographic lat/lon bounding boxes (e.g. $25^\circ\text{N}-35^\circ\text{N}, 80^\circ\text{W}-95^\circ\text{W}$) into CONUS pixel bounding indices.
  - `extract_patches`: Extracts non-overlapping or sliding $(128 \times 128)$ patches, rejects any patch with fill pixels, and computes convective salience metrics ($T_{13} < 241\text{ K}$ and $T_{13} < 221\text{ K}$).
- **`sequence_generator.py`**:
  - `TemporalSequenceGenerator`: Identifies continuous sequence windows matching the approved relative offsets $[-60\text{m}, -30\text{m}, 0\text{m}, +30\text{m}] \to [+15\text{m}]$, rejects incomplete windows, and loads aligned multi-frame 4-channel tensors `(4, 4, H, W)` and `(1, 4, H, W)`.

---

## 3. Multichannel Calibration & Physical Ingestion Results

Every timestamp in the 2-hour window contains all 4 channels, with identical raster geometry and valid pixel masks:

| Channel Number & Spectral Role | Central $\lambda$ ($\mu\text{m}$) | Array Shape | Dynamic Range ($T_b$) | Mean $T_b$ | Std Dev $\sigma$ | Valid Pixels | Fill Pixels |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Channel 09 (Mid WV)** | $6.93$ | $(1500, 2500)$ | $202.27\text{ K} - 266.62\text{ K}$ | $245.24\text{ K}$ | $10.95\text{ K}$ | $3,702,837$ ($98.74\%$) | $47,163$ ($1.26\%$) |
| **Channel 13 (Clean IR Window)** | $10.33$ | $(1500, 2500)$ | $205.83\text{ K} - 298.19\text{ K}$ | $269.93\text{ K}$ | $19.61\text{ K}$ | $3,702,837$ ($98.74\%$) | $47,163$ ($1.26\%$) |
| **Channel 14 (Longwave IR Window)**| $11.19$ | $(1500, 2500)$ | $204.58\text{ K} - 297.95\text{ K}$ | $269.30\text{ K}$ | $19.99\text{ K}$ | $3,702,837$ ($98.74\%$) | $47,163$ ($1.26\%$) |
| **Channel 15 (Dirty IR Split Window)**| $12.27$ | $(1500, 2500)$ | $204.05\text{ K} - 294.82\text{ K}$ | $267.82\text{ K}$ | $19.41\text{ K}$ | $3,702,837$ ($98.74\%$) | $47,163$ ($1.26\%$) |

### Verification of Physical Calibration:
1. **No Double-Scaling**: Raw integers in the file are signed 16-bit values. Applying $T = \text{raw} \times \text{scale\_factor} + \text{add\_offset}$ yields values in the physical Kelvin interval $[202\text{ K}, 300\text{ K}]$. Reading libraries that auto-mask or auto-scale were carefully handled to ensure linear calibration is executed exactly once.
2. **Quality Flags (`DQF`)**: Valid pixels have `DQF == 0` (good quality) with zero focal plane temperature exceedances. Space/off-disk pixels have `DQF == -1` and raw value `-1`.
3. **Stacked Array**: For each timestamp, the stacked array has shape `(4, 1500, 2500)` in `float32` representation.

---

## 4. Geographic Cropping & Valid-Pixel Integrity

### 4.1 Bounding Box Derivation
The target region spans $25.0^\circ\text{N} - 35.0^\circ\text{N}$ latitude and $80.0^\circ\text{W} - 95.0^\circ\text{W}$ longitude (Gulf of Mexico and Southeast United States). 

Using analytical projection inversion equations from NOAA PUG Volume 5:
- **Latitude Extent**: $25.0^\circ\text{N}$ to $35.0^\circ\text{N}$
- **Longitude Extent**: $-95.0^\circ\text{E}$ to $-80.0^\circ\text{E}$
- **Calculated Pixel Row Bounds**: Rows $[537, 990]$ (Height: $453\text{ pixels}$)
- **Calculated Pixel Col Bounds**: Cols $[851, 1591]$ (Width: $740\text{ pixels}$)
- **Cropped Spatial Dimensions**: $(4, 453, 740)$

### 4.2 Valid-Pixel Ratio
- **Total Pixels in Crop**: $453 \times 740 = 335,220\text{ pixels}$
- **Space / Fill Pixels**: **0 pixels** ($0.00\%$)
- **Valid Earth Pixels**: **335,220 pixels** (**100.00% valid**)

> [!NOTE]
> Because the Southeast US / Gulf subgrid is situated safely inside the terrestrial CONUS footprint, it contains **zero outer-space fill pixels**, eliminating boundary contamination during 2D convolution and patch extraction.

### 4.3 Meteorological Findings on Convective Location (Jan 1, 2026)
- Evaluating the full CONUS disk on January 1, 2026 revealed **$350,934$ convective pixels ($T_{13} < 241\text{ K}$)** and **$23,754$ deep convective core pixels ($T_{13} < 221\text{ K}$)**.
- **Geographic Distribution**: The primary convective cloud system was concentrated in the **Western United States / Eastern Pacific** (centered at $33.6^\circ\text{N}, 113.2^\circ\text{W}$ over the Southwest, with deep cloud tops below $210\text{ K}$).
- **Southeast US Status**: The Gulf / Southeast US region on this particular winter day exhibited clear-sky terrestrial cooling and low-altitude marine stratocumulus ($270\text{ K} - 295\text{ K}$).
- **Validation Lesson**: As cautioned in the Phase 2B requirements, **not every geographic crop will contain convective activity on a given winter day**. The configurable cropper and patch extractor successfully handle both ambient and convective regimes without crashing or generating empty tensors.

---

## 5. Temporal Sequence Window Discovery

Using our approved primary temporal formulation:
- **Inputs**: $L_{-2} = t - 60\text{m}$, $L_{-1} = t - 30\text{m}$, $L_0 = t$, $L_1 = t + 30\text{m}$ (relative indices $k-12, k-6, k, k+6$).
- **Midpoint Ground Truth Target**: $H_{15\text{m}} = t + 15\text{m}$ (relative index $k+3$).

Across the 24 continuous 5-minute scans spanning $04:01\text{ UTC}$ to $05:56\text{ UTC}$ (115 minutes elapsed), a sequence window requires a minimum time span of $90\text{ minutes}$ ($t - 60\text{m}$ to $t + 30\text{m}$).

The sequence generator discovered **exactly 6 valid, continuous sliding windows**:

| Window Index | Input Frame 1 ($t-60\text{m}$) | Input Frame 2 ($t-30\text{m}$) | Input Frame 3 (Anchor $t$) | Target Frame ($t+15\text{m}$) | Input Frame 4 ($t+30\text{m}$) | Window Span | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Window 00** | 04:01:19 UTC | 04:31:19 UTC | 05:01:19 UTC | **05:16:19 UTC** | 05:31:19 UTC | 90 min | **Complete (0 missing)** |
| **Window 01** | 04:06:19 UTC | 04:36:19 UTC | 05:06:19 UTC | **05:21:19 UTC** | 05:36:19 UTC | 90 min | **Complete (0 missing)** |
| **Window 02** | 04:11:19 UTC | 04:41:19 UTC | 05:11:19 UTC | **05:26:19 UTC** | 05:41:19 UTC | 90 min | **Complete (0 missing)** |
| **Window 03** | 04:16:19 UTC | 04:46:19 UTC | 05:16:19 UTC | **05:31:19 UTC** | 05:46:19 UTC | 90 min | **Complete (0 missing)** |
| **Window 04** | 04:21:19 UTC | 04:51:19 UTC | 05:21:19 UTC | **05:36:19 UTC** | 05:51:19 UTC | 90 min | **Complete (0 missing)** |
| **Window 05** | 04:26:19 UTC | 04:56:19 UTC | 05:26:19 UTC | **05:41:19 UTC** | 05:56:19 UTC | 90 min | **Complete (0 missing)** |

Each window produces:
- `input_tensors`: Shape `(4, 4, 453, 740)` in float32.
- `target_tensors`: Shape `(1, 4, 453, 740)` in float32.
- `patch_sequences`: 60 spatially synchronized $(128 \times 128)$ patches extracted across all 5 temporal frames simultaneously.

---

## 6. Generated Visualizations & Artifacts

All visual deliverables were rendered and saved under `outputs/preprocessing/`:

1. **`01_full_conus_4channels.png`**:
   - 4-panel meteorological comparison of Channels 9, 13, 14, and 15 across the full CONUS disk at `04:01:19 UTC`.
   - Displays calibrated Kelvin color bars, temperature distribution statistics, and valid/masked pixel counts.
2. **`02_geographic_crop_validation.png`**:
   - Shows the CONUS reference disk with the Southeast US / Gulf bounding box marked in green dashed lines.
   - Panels showing the cropped Channel 13 subgrid ($453 \times 740$) and the Split-Window Brightness Temperature Difference ($\text{BTD} = T_{13} - T_{15}$) for cloud and moisture discrimination.
3. **`03_patch_samples_128x128.png`**:
   - Gallery of candidate $128 \times 128$ patches with row/col slice bounds and local temperature stats.
4. **`04_sequence_evolution_5frames.png`**:
   - Displays the METEOR-Net sequence-flow formulation across 5 frames for a representative $128 \times 128$ patch:
     Input 1 ($04:01$), Input 2 ($04:31$), Input 3 ($05:01$), **Target Midpoint ($05:16$)**, and Input 4 ($05:31$).

### Processed Prototype Tensor Artifacts:
Saved under `outputs/processed_sample/`:
- **`sample_sequence_window00.npz`** ($6.38\text{ MB}$ compressed):
  Contains the full cropped arrays for Window 00 (`inputs` shape `(4, 4, 453, 740)`, `targets` shape `(1, 4, 453, 740)`, `valid_mask`, timestamps, and channel list).
- **`sequence_metadata.json`** ($955\text{ bytes}$):
  Machine-readable catalog detailing bounding boxes, temporal offsets, channel order, and storage projections.

---

## 7. Automated Test Suite Results

The unit test suite was implemented in `tests/test_preprocessing.py` and executed using standard `unittest`:

```text
Ran 5 tests in 3.717s

test_01_single_file_reader_and_calibration ... ok
test_02_multichannel_alignment_and_synchronization ... ok
test_03_geonavigation_projection_inversion ... ok
test_04_geographic_crop_and_patch_extraction ... ok
test_05_temporal_sequence_generator ... ok

OK (5/5 tests passed)
```

**Verification Details**:
- `test_01`: Confirmed scale factor and add offset yield realistic brightness temperatures ($200\text{ K} - 305\text{ K}$), valid masks are boolean, and no unmasked NaNs exist.
- `test_02`: Confirmed 4-channel stacking has shape `(4, 1500, 2500)` and valid masks align.
- `test_03`: Confirmed forward and inverse projection transformations invert within $< 0.001^\circ$ error.
- `test_04`: Confirmed crop dimensions are non-empty and 100% valid, and extracted patches have shape `(4, 128, 128)`.
- `test_05`: Confirmed exactly 6 windows are discovered and tensor loading preserves shape `(4, 4, H, W)` and `(1, 4, H, W)`.

---

## 8. Storage Projections for Bulk Data Acquisition

Before downloading additional data from the NOAA AWS archive (`noaa-goes19`), we evaluated the storage requirements based on our prototype measurements:

| Dataset Scope | Scan Count (Timestamps) | File Count (4 Channels) | Raw NetCDF Storage | Cropped Region Storage (`.npz` float32) | Filtered $128 \times 128$ Patches (`.npz`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Current Prototype** | 24 (2 hours) | 96 files | $282.9\text{ MB}$ | $\sim 38\text{ MB}$ | $\sim 5\text{ MB}$ |
| **1 Full Day (24 h)** | 288 | 1,152 files | $\sim 4.25\text{ GB}$ | $\sim 570\text{ MB}$ | $\sim 75\text{ MB}$ |
| **7 Days (1 Week)** | 2,016 | 8,064 files | $\sim 29.7\text{ GB}$ | $\sim 3.9\text{ GB}$ | $\sim 525\text{ MB}$ |
| **10 Days (Target Batch)**| 2,880 | 11,520 files | $\sim 42.4\text{ GB}$ | $\sim 5.6\text{ GB}$ | $\sim 750\text{ MB}$ |

### Storage Strategy Recommendation:
If local disk space is constrained, we recommend a **streaming crop pattern**:
1. Fetch 1 day of raw NetCDF files ($4.25\text{ GB}$).
2. Extract the cropped regional subgrid or filtered $128 \times 128$ patches directly into `.npz` tensors ($\approx 75\text{ MB}$).
3. Purge the raw NetCDF files for that day, retaining only the processed tensors.
4. Over a 10-day period, total retained storage will be **under 1 GB** for patch datasets, instead of accumulating $42.4\text{ GB}$ of raw files.

---

## 9. Summary of Decisions Requiring Your Approval

Before we proceed to **Phase 3 (METEOR-Net Architecture Implementation & Sequence Pipeline Integration)**, please confirm:

1. **Storage Handling for Bulk Data Fetch**:
   - **Option A (Streaming Crop — Recommended)**: Download raw files in daily batches, extract cropped $128 \times 128$ sequence tensors into `outputs/processed/`, and delete the raw files to keep disk footprint under 2 GB.
   - **Option B (Accumulate Raw)**: Retain all raw NetCDF files locally (requires $\sim 42.4\text{ GB}$ of free disk space for 10 days).
2. **Crop Region for Convective Training**:
   - Should we keep the Gulf/Southeast US region ($25^\circ\text{N}-35^\circ\text{N}, 80^\circ\text{W}-95^\circ\text{W}$), or expand/shift the bounding box to capture the active storm belt observed over the Western US / Pacific ($30^\circ\text{N}-40^\circ\text{N}, 105^\circ\text{W}-120^\circ\text{W}$) on winter days?
3. **INSAT-3DS Readiness**:
   - Confirm that we will keep the cross-satellite domain adaptation interface modular and wait for your MOSDAC account approval and data sample before executing INSAT-specific calibration.

---
*Report prepared and validated via automated prototype runs and unit tests on October 16, 2026.*

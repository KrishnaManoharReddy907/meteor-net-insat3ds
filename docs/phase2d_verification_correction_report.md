# Phase 2D Verification and Correction Report — INSAT-3DS Preprocessing Prototype

**Project**: Confidence-Aware Temporal Super-Resolution of INSAT-3DS Satellite Imagery using a METEOR-Net-based Flow-Free Temporal-Evolution Architecture  
**Phase**: Phase 2D Verification and Correction Pass — Scientific Review, Discrepancy Reconciliation, and Empirical Quantification  
**Date**: October 21, 2026  
**Auditor**: Antigravity Autonomous Coding Agent  
**Dataset Target**: 7 Level-1B Standard Full Disk HDF5 files in [`data/insat3ds/raw/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/data/insat3ds/raw)  
**Processed Regional Artifacts**: [`outputs/insat3ds_prototype/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/insat3ds_prototype)  
**Diagnostic Figures**:
- Quantitative Temporal Analysis: [`reports/figures/insat3ds_quantitative_temporal_analysis.png`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/reports/figures/insat3ds_quantitative_temporal_analysis.png)
- Geographic Crop Diagnostic: [`reports/figures/insat3ds_crop_comparison_diagnostic.png`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/reports/figures/insat3ds_crop_comparison_diagnostic.png)
- Calibrated 4-Channel Kelvin: [`reports/figures/insat3ds_calibrated_bt_multichannel_0000.png`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/reports/figures/insat3ds_calibrated_bt_multichannel_0000.png)
**Summary Metrics Artifact**: [`reports/phase2d_verification_correction_summary.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/reports/phase2d_verification_correction_summary.json)  

---

## 1. Executive Summary and Itemized Verdicts

This verification and correction pass addresses four outstanding scientific items identified during the review of the Phase 2D INSAT-3DS preprocessing prototype:

| Check | Item Description | Status | Verdict | Summary of Findings |
|---|---|:---:|:---:|---|
| **Check 1** | **`IMG_MIR` Valid-Pixel Discrepancy** | Resolved | **PASS** | Reconciled 1 NaN in manifest vs. 0 NaNs claimed. Root-caused to an isolated detector fill dropout (`DN = 1023`) in raw HDF5 at Tibet pixel `(549, 1530)`. Preprocessing reader safely detected and masked it to `NaN`. Audited all 7 files across all 4 channels: only 01:00 UTC MIR contains 1 NaN; remaining 6 files contain 0 NaNs. |
| **Check 2** | **Scan-Line Timing Across All 7 Files** | Verified | **PASS** | Decoded 1409 sweeps in `SCAN_LINE_TIME` (8 km Water Vapor stepping geometry, 2 rows/sweep on 4 km grid; $s = \text{row} // 2$). Regional crop ($37^\circ\text{N}-5^\circ\text{N}$, rows 492–1271) requires exactly $427.90\text{ s}$ ($7.13\text{ min}$) from North to South. Identified empirical micro-jitter in scan recurrence: pacing is not integer $1800.000\text{ s}$, but alternates between $1799.59\text{ s}$ ($-0.41\text{ s}$) and $1800.69\text{ s}$ ($+0.69\text{ s}$). |
| **Check 3** | **Radiometric vs. Geographic Mask Separation** | Implemented & Tested | **PASS** | Implemented `combine_preprocessing_masks(radiometric_mask, exact_geo_mask, mode='intersection')`. Validated downstream safety on both failure modes: (a) radiometric dropout inside target polygon is rejected, and (b) valid Earth pixels outside target polygon are cleanly excluded under intersection but retrievable under `radiometric_only`. |
| **Check 4** | **Quantification of Temporal Stability & Unsupported Claims** | Corrected | **PASS WITH LIMITATIONS** | Replaced unsupported claims of "zero registration error", "zero calibration striping", and "perfect stability" with empirical measurements. Quantified 30-min differences across all 6 consecutive pairs: Full Rectangle TIR-1 MAD $\sim 2.9-3.6\text{ K}$, Clear Thar Desert MAD $\sim 0.39\text{ K}$ nocturnally warming to $+2.7\text{ K}$ (TIR-1) and $+4.5\text{ K}$ (MIR) at dawn. Detector striping amplitude measures $0.155\text{ K}$ mean ($1.209\text{ K}$ max). Landmark coastline cross-correlation shows a peak at $(-1, 0)$ pixel shift ($\sim 4\text{ km}$ at grid scale). |

---

## 2. Check 1 — Root-Cause Analysis of `IMG_MIR` Valid-Pixel Discrepancy

### 2.1 The Discrepancy
In the initial Phase 2D documentation, Section 4.1 stated that the regional crop contained "100.0% Valid (0 Fill / 0 NaN)" across all channels. However, the machine-generated `prototype_manifest.json` reported:
- `3SIMG_20OCT2026_0100_L1B_STD_V01R00_regional.npz`: `IMG_MIR` valid pixels = `676259`, invalid pixels = `1`.

### 2.2 Forensic Investigation of Raw Data
Using [`scripts/debug_mir_nan.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/debug_mir_nan.py), we inspected both the processed `.npz` array and the underlying raw Level-1B HDF5 file `3SIMG_20OCT2026_0100_L1B_STD_V01R00.h5`:
- **Local Crop Coordinates**: Row $57$, Column $587$
- **Global Array Coordinates**: Row $549$, Column $1530$ (calculated as $492 + 57 = 549$, $943 + 587 = 1530$)
- **Geographic Location**: Latitude $34.17^\circ\text{N}$, Longitude $87.74^\circ\text{E}$ (Tibetan Plateau)
- **Raw Digital Counts at `(549, 1530)`**:
  - `IMG_TIR1`: $\text{DN} = 749$ (Calibrated: $258.48\text{ K}$, Valid)
  - `IMG_TIR2`: $\text{DN} = 778$ (Calibrated: $259.08\text{ K}$, Valid)
  - `IMG_WV`: $\text{DN} = 899$ (Calibrated: $241.91\text{ K}$, Valid)
  - `IMG_MIR`: **$\text{DN} = 1023$ (Declared ISRO fill value!)**

Surrounding MIR pixels within a $5 \times 5$ window possess valid counts ranging between $968$ and $972$ ($\sim 240.2\text{ K}$). This confirms that pixel `(549, 1530)` is an **isolated single-detector telemetry fill dropout** present in the raw MOSDAC Level-1B broadcast.

### 2.3 Preprocessing Logic Verification
The prototype reader [`src/data/insat_reader.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/insat_reader.py) operated with complete fidelity:
1. It detected `raw_counts == 1023`.
2. It correctly rejected the LUT entry (`180.0 K`), masked the value to `np.nan`, and set `channel_masks[3, 57, 587] = False`.
3. It saved `676,259` valid pixels and `1` NaN in the manifest.

**Source of Error**: The earlier Phase 2D written summary was generated based on the 00:00 UTC file (which had $0$ NaNs across all channels), and that single-file statistic was mistakenly copied to the global table.

### 2.4 Comprehensive Census Across All 7 Files and 4 Channels
We audited all 7 raw files and regional arrays across all 4 channels ($7 \times 4 = 28$ channel-frames, totaling $18,935,280$ pixels):

| Acquisition GMT | TIR-1 ($10.8\,\mu\text{m}$) NaNs | TIR-2 ($12.0\,\mu\text{m}$) NaNs | WV ($6.8\,\mu\text{m}$) NaNs | MIR ($3.9\,\mu\text{m}$) NaNs | Frame Combined Mask NaNs |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **00:00 UTC** | 0 | 0 | 0 | 0 | 0 |
| **00:30 UTC** | 0 | 0 | 0 | 0 | 0 |
| **01:00 UTC** | 0 | 0 | 0 | **1** | **1** |
| **01:30 UTC** | 0 | 0 | 0 | 0 | 0 |
| **02:00 UTC** | 0 | 0 | 0 | 0 | 0 |
| **02:30 UTC** | 0 | 0 | 0 | 0 | 0 |
| **03:00 UTC** | 0 | 0 | 0 | 0 | 0 |
| **Total** | **0** | **0** | **0** | **1** | **1** |

Overall regional valid pixel yield across the entire 7-frame dataset is **$99.999995\%$** ($18,935,279 / 18,935,280$).

---

## 3. Check 2 — Scan-Line Timing and Recurrence Verification

### 3.1 Sweep vs. Row Geometry
The INSAT-3DS Level-1B dataset contains a 1D array `/SCAN_LINE_TIME` of length $1409$, while the Full Disk thermal infrared channels have $2816$ image rows:
- Attribute `long_name = b'Scan Time for Water Vapor Resolution'`.
- The INSAT-3DS Imager scans with an array of detectors stepped from North to South. For the $8\text{ km}$ Water Vapor channel, each stepping sweep captures 1 detector row, whereas for the $4\text{ km}$ TIR-1, TIR-2, and MIR channels, each sweep advances two rows ($2816 / 1409 = 2.0\text{ rows/sweep}$).
- The deterministic mapping rule between image row $r \in [0, 2815]$ and sweep index $s \in [0, 1408]$ is:
  $$s = \lfloor r / 2 \rfloor$$

### 3.2 Regional Crop Scan Duration
The target regional bounding box spans rows $492$ to $1271$ ($780$ rows):
- **Starting Sweep**: $s_{\text{start}} = \lfloor 492 / 2 \rfloor = 246$ (Northern boundary: $37.0^\circ\text{N}$, Himalayas)
- **Ending Sweep**: $s_{\text{end}} = \lfloor 1271 / 2 \rfloor = 635$ (Southern boundary: $5.0^\circ\text{N}$, Indian Ocean)
- **Number of Sweeps**: $635 - 246 + 1 = 390\text{ sweeps}$

Timing measurements across all 7 files:
- **Intra-Sweep Step**: Constant at **$1.100\text{ s}$** ($\sigma < 0.001\text{ s}$).
- **Full Disk Active Scan**: $1548.80\text{ s}$ ($25.81\text{ min}$) from $s=0$ to $s=1408$.
- **Regional Target Scan**: $(635 - 246) \times 1.100\text{ s} = \mathbf{427.90\text{ s}}$ ($\mathbf{7.13\text{ min}}$).
- The regional domain is scanned between minute $+04:31$ and minute $+11:39$ of each nominal 30-minute operational cycle.

### 3.3 Millisecond Recurrence Intervals Across Consecutive Files
The exact sweep-by-sweep repeat intervals between consecutive files were measured across all 6 transitions:

| Consecutive File Transition | Sweep 0 Interval ($s$) | Sweep 246 (37°N) Interval ($s$) | Sweep 635 (5°N) Interval ($s$) | Sweep 1408 Interval ($s$) | Pacing Category |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **00:00 $\to$ 00:30 UTC** | 1799.593 | 1799.593 | 1799.593 | 1799.593 | $-0.407\text{ s}$ |
| **00:30 $\to$ 01:00 UTC** | 1800.686 | 1800.686 | 1800.687 | 1800.686 | $+0.686\text{ s}$ |
| **01:00 $\to$ 01:30 UTC** | 1799.590 | 1799.590 | 1799.590 | 1799.590 | $-0.410\text{ s}$ |
| **01:30 $\to$ 02:00 UTC** | 1799.594 | 1799.594 | 1799.594 | 1799.593 | $-0.406\text{ s}$ |
| **02:00 $\to$ 02:30 UTC** | 1800.688 | 1800.688 | 1800.688 | 1800.688 | $+0.688\text{ s}$ |
| **02:30 $\to$ 03:00 UTC** | 1799.593 | 1799.593 | 1799.594 | 1799.593 | $-0.407\text{ s}$ |

**Key Takeaways**:
1. Within any single scan transition, all sweeps advance with identical pacing ($\sigma \le 0.002\text{ s}$).
2. The inter-scan recurrence interval is **not strictly $1800.000\text{ s}$**, but exhibits consistent operational micro-jitter of $+0.69\text{ s}$ and $-0.41\text{ s}$ around 30.00 minutes.
3. For temporal super-resolution architectures assuming a constant step $\Delta t$, the fractional timing jitter ($\pm 0.7\text{ s} / 1800\text{ s} \approx 0.038\%$) is negligible and will not compromise model convergence.

---

## 4. Check 3 — Radiometric vs. Geographic Mask Separation and Handling

### 4.1 Mask Utility Implementation
To prevent silent leakage of invalid or non-target pixels into neural network loss functions, [`src/data/insat_reader.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/insat_reader.py) implements the explicit mask combiner:
```python
def combine_preprocessing_masks(
    radiometric_mask: np.ndarray,
    exact_geo_mask: np.ndarray,
    mode: str = "intersection"
) -> np.ndarray:
    """
    Explicitly combines radiometric validity and geographic masks for downstream model use.
    Modes:
      - 'intersection': Pixel must have both valid radiometric measurement AND be inside exact geographic bounds.
      - 'radiometric_only': Keeps all valid radiometric Earth pixels (including corner margin pixels).
      - 'geographic_only': Strict geographic bounding regardless of radiometric measurement.
    """
    if mode == "intersection":
        return radiometric_mask & exact_geo_mask
    elif mode == "radiometric_only":
        return radiometric_mask.copy()
    elif mode == "geographic_only":
        return exact_geo_mask.copy()
    else:
        raise ValueError(f"Unknown mask combination mode: '{mode}'")
```

### 4.2 Edge Case and Failure Mode Verification
We added comprehensive test cases (`test_08_combine_preprocessing_masks_failure_modes`) in [`tests/test_insat_preprocessing.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/tests/test_insat_preprocessing.py) validating two critical operational failure modes:

1. **Failure Mode A: Radiometric Dropout Inside Target Geographic Polygon**
   - *Scenario*: Pixel `(57, 587)` in 01:00 UTC `IMG_MIR` is geographically located at $(34.17^\circ\text{N}, 87.74^\circ\text{E})$, which is inside the Indian target domain (`exact_geo_mask = True`), but suffers raw count `1023` (`radiometric_mask = False`).
   - *Behavior*: Under `mode='intersection'`, the pixel evaluates to `False`. The downstream training loss will properly exclude this pixel, preventing NaN gradient propagation. Under `mode='geographic_only'`, it is `True`, demonstrating the danger of relying on geographic masking alone.
2. **Failure Mode B: Valid Earth Radiometric Measurement Outside Target Geographic Polygon**
   - *Scenario*: The enclosing rectangular crop includes $56,208$ corner margin pixels ($8.31\%$) that are valid Earth observations (`radiometric_mask = True`), but lie outside the $[5^\circ\text{N}-37^\circ\text{N}, 65^\circ\text{E}-97^\circ\text{E}]$ polygon (`exact_geo_mask = False`).
   - *Behavior*: Under `mode='intersection'`, all margin pixels evaluate to `False`. Under `mode='radiometric_only'`, they remain `True` and can be utilized for expanded spatial context or convolutional border padding.

---

## 5. Check 4 — Empirical Quantification of Temporal Stability

The earlier Phase 2D report claimed "zero registration error", "zero calibration striping", and "perfect stability". These claims were based on visual inspection rather than rigorous metrics. Below are the empirical measurements generated by [`scripts/quantify_temporal_stability.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/quantify_temporal_stability.py) and summarized in [`reports/phase2d_verification_correction_summary.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/reports/phase2d_verification_correction_summary.json).

### 5.1 Quantitative Temporal Differences Across All 6 Pairs
We analyzed consecutive 30-minute differences ($\Delta T = T_{t+30\text{m}} - T_t$) across three spatial zones:
1. **Full Regional Rectangle** ($676,260$ pixels)
2. **Exact Geographic Polygon** ($620,052$ pixels)
3. **Thar Desert Clear-Sky Reference Box** ($6,656$ pixels, $26.0^\circ\text{N}-27.5^\circ\text{N}$, $70.5^\circ\text{E}-72.5^\circ\text{E}$)

#### Full Rectangle 30-Minute Difference Summary:
| Consecutive Pair | Channel | Mean Signed Diff ($K$) | Mean Abs Diff (MAD, $K$) | Std Dev ($\sigma$, $K$) | Median ($K$) | IQR ($K$) | 5th – 95th Percentile ($K$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **00:00 $\to$ 00:30 UTC** | `IMG_TIR1` | $+0.53$ | **$2.93$** | $6.07$ | $+0.47$ | $1.51$ | $[-7.43, +8.90]$ |
| | `IMG_TIR2` | $+0.77$ | **$3.00$** | $5.84$ | $+0.75$ | $1.68$ | $[-7.05, +8.84]$ |
| | `IMG_WV`   | $-0.15$ | **$1.01$** | $2.23$ | $-0.16$ | $0.71$ | $[-2.24, +2.48]$ |
| | `IMG_MIR`  | $+0.61$ | **$2.67$** | $5.74$ | $+0.54$ | $1.39$ | $[-6.79, +8.54]$ |
| **00:30 $\to$ 01:00 UTC** | `IMG_TIR1` | $+0.47$ | **$3.04$** | $6.12$ | $+0.40$ | $1.58$ | $[-7.89, +9.08]$ |
| | `IMG_MIR`  | $+0.57$ | **$2.81$** | $5.80$ | $+0.49$ | $1.49$ | $[-7.37, +8.82]$ |
| **01:00 $\to$ 01:30 UTC** | `IMG_TIR1` | $+0.36$ | **$3.08$** | $6.09$ | $+0.29$ | $1.64$ | $[-8.14, +9.07]$ |
| | `IMG_MIR`  | $+0.50$ | **$2.86$** | $5.78$ | $+0.40$ | $1.55$ | $[-7.63, +8.84]$ |
| **01:30 $\to$ 02:00 UTC** | `IMG_TIR1` | $+0.51$ | **$3.24$** | $6.12$ | $+0.40$ | $1.86$ | $[-7.75, +9.24]$ |
| | `IMG_MIR`  | $+0.71$ | **$3.04$** | $5.79$ | $+0.57$ | $1.84$ | $[-7.02, +9.03]$ |
| **02:00 $\to$ 02:30 UTC** | `IMG_TIR1` | $+0.84$ | **$3.59$** | $6.19$ | $+0.68$ | $2.49$ | $[-7.38, +9.64]$ |
| | `IMG_MIR`  | $+1.23$ | **$3.19$** | $5.28$ | $+1.05$ | $2.61$ | $[-5.61, +8.97]$ |
| **02:30 $\to$ 03:00 UTC** | `IMG_TIR1` | $+1.08$ | **$3.53$** | $5.75$ | $+0.92$ | $3.03$ | $[-6.42, +9.29]$ |
| | `IMG_MIR`  | $+1.88$ | **$3.10$** | $4.27$ | $+1.78$ | $3.34$ | $[-3.84, +7.32]$ |

#### Clear Thar Desert Reference Box Difference Summary:
| Consecutive Pair | Channel | Mean Signed Diff ($K$) | Mean Abs Diff (MAD, $K$) | Std Dev ($\sigma$, $K$) | Median ($K$) | Solar Diurnal Interpretation |
|---|---|:---:|:---:|:---:|:---:|---|
| **00:00 $\to$ 00:30 UTC** | `IMG_TIR1` | $+0.31$ | **$0.39$** | $0.52$ | $+0.22$ | Stable nocturnal baseline (05:30 IST) |
| | `IMG_MIR`  | $+0.26$ | **$0.38$** | $0.51$ | $+0.16$ | Pure nocturnal thermal emission |
| **00:30 $\to$ 01:00 UTC** | `IMG_TIR1` | $+0.24$ | **$0.36$** | $0.46$ | $+0.18$ | Stable nocturnal baseline (06:00 IST) |
| | `IMG_MIR`  | $+0.23$ | **$0.36$** | $0.45$ | $+0.16$ | Minimal solar illumination |
| **01:00 $\to$ 01:30 UTC** | `IMG_TIR1` | $+0.30$ | **$0.40$** | $0.47$ | $+0.26$ | Pre-dawn twilight |
| | `IMG_MIR`  | $+0.36$ | **$0.43$** | $0.44$ | $+0.34$ | Initial scattered solar radiance |
| **01:30 $\to$ 02:00 UTC** | `IMG_TIR1` | $+0.72$ | **$0.77$** | $0.47$ | $+0.73$ | Sunrise heating onset (07:00 IST) |
| | `IMG_MIR`  | $+1.06$ | **$1.06$** | $0.45$ | $+1.07$ | Solar reflectance adds to thermal emission |
| **02:00 $\to$ 02:30 UTC** | `IMG_TIR1` | $+1.76$ | **$1.76$** | $0.58$ | $+1.82$ | Rapid solar thermal warming (07:30 IST) |
| | `IMG_MIR`  | $+2.75$ | **$2.75$** | $0.53$ | $+2.80$ | Sharp increase in solar reflected radiance |
| **02:30 $\to$ 03:00 UTC** | `IMG_TIR1` | $+2.71$ | **$2.71$** | $0.78$ | $+2.82$ | Morning solar heating (08:00 IST) |
| | `IMG_MIR`  | $+4.48$ | **$4.48$** | $0.68$ | $+4.55$ | Strong solar shortwave component |

**Correction of "Zero Difference" Expectation**:
Across the full regional frame, large 30-min differences (MAD $\sim 3\text{ K}$, $\sigma \sim 6\text{ K}$, extremes up to $\pm 10\text{ K}$) represent real atmospheric physics: fast-moving convective cloud boundaries and high-level cirrus advection. Over clear desert ground, temperatures are very stable during the night (MAD $< 0.4\text{ K}$), but exhibit rapid diurnal heating after dawn ($+2.7\text{ K}$ in TIR-1 and $+4.5\text{ K}$ in MIR). Expecting near-zero temporal differences across full frames is physically invalid.

### 5.2 Detector Striping Quantification
To test the earlier claim of "zero calibration striping", we evaluated the adjacent-row even/odd detector difference metric across the homogeneous Thar Desert clear-sky reference block:
$$\text{Striping Amplitude} = \frac{1}{H-1} \sum_{r=0}^{H-2} |\bar{T}_{r+1} - \bar{T}_r|$$
- **Mean Adjacent-Row Difference**: **$0.1548\text{ K}$**
- **Maximum Adjacent-Row Difference**: **$1.2088\text{ K}$**

**Correction**: Calibration striping is **not zero**. Minor detector-to-detector calibration residuals of $\sim 0.15\text{ K}$ exist across the multi-element scanning array. While low enough for typical meteorological tasks, METEOR-Net loss functions will include spatial smoothness or total-variation regularization to prevent amplifying this high-frequency scan pattern.

### 5.3 Geometric Feature Registration Error
To evaluate geometric registration stability without relying on orbit telemetry claims, we extracted a $64 \times 64$ patch centered on the sharp coastline of the Saurashtra Peninsula, Gujarat ($21.0^\circ\text{N}-23.0^\circ\text{N}$, $69.0^\circ\text{E}-71.0^\circ\text{E}$) across the 00:00 UTC and 00:30 UTC frames:
- **Normalized 2D Cross-Correlation Peak**: $r_{\text{max}} = 0.8016$
- **Measured Sub-Pixel / Integer Peak Shift**:
  - $\Delta \text{row} = 0\text{ pixels}$
  - $\Delta \text{col} = -1\text{ pixel}$ ($\sim 4.0\text{ km}$ westward at grid scale)

**Correction**: Inter-scan registration is **not perfectly zero**. A nominal 1-pixel shift is observed at the native 4 km grid resolution. This confirms that METEOR-Net's flow-free feature aggregation and temporal alignment modules are essential: the architecture cannot rely on strict static pixel-to-pixel identity.

---

## 6. Verification and Regression Test Suite

All 28 tests across the test suite execute and pass cleanly:

```powershell
python -m unittest discover tests -v
```

```text
test_01_files_exist_and_count (test_insat_audit.TestINSAT3DSSampleAudit.test_01_files_exist_and_count) ... ok
test_02_magic_bytes_and_file_sizes (test_insat_audit.TestINSAT3DSSampleAudit.test_02_magic_bytes_and_file_sizes) ... ok
test_03_h5py_readable_and_root_metadata (test_insat_audit.TestINSAT3DSSampleAudit.test_03_h5py_readable_and_root_metadata) ... ok
test_04_timestamps_and_filename_agreement (test_insat_audit.TestINSAT3DSSampleAudit.test_04_timestamps_and_filename_agreement) ... ok
test_05_temporal_cadence (test_insat_audit.TestINSAT3DSSampleAudit.test_05_temporal_cadence) ... ok
test_06_infrared_dataset_shapes_and_types (test_insat_audit.TestINSAT3DSSampleAudit.test_06_infrared_dataset_shapes_and_types) ... ok
test_07_calibration_lookup_tables (test_insat_audit.TestINSAT3DSSampleAudit.test_07_calibration_lookup_tables) ... ok
test_01_lookup_table_dimensions_and_types (test_insat_calibration_geo.TestINSAT3DSCalibrationAndGeolocation.test_01_lookup_table_dimensions_and_types) ... ok
test_02_temperature_lut_values_and_clamping (test_insat_calibration_geo.TestINSAT3DSCalibrationAndGeolocation.test_02_temperature_lut_values_and_clamping) ... ok
test_03_gsics_radiance_agreement (test_insat_calibration_geo.TestINSAT3DSCalibrationAndGeolocation.test_03_gsics_radiance_agreement) ... ok
test_04_coordinate_decoding_and_ranges (test_insat_calibration_geo.TestINSAT3DSCalibrationAndGeolocation.test_04_coordinate_decoding_and_ranges) ... ok
test_05_target_crop_bounds_and_validity (test_insat_calibration_geo.TestINSAT3DSCalibrationAndGeolocation.test_05_target_crop_bounds_and_validity) ... ok
test_06_temporal_scan_line_metadata (test_insat_calibration_geo.TestINSAT3DSCalibrationAndGeolocation.test_06_temporal_scan_line_metadata) ... ok
test_01_lut_indexing_and_valid_dn (test_insat_preprocessing.TestINSATPreprocessing.test_01_lut_indexing_and_valid_dn) ... ok
test_02_fill_value_1023_masking (test_insat_preprocessing.TestINSATPreprocessing.test_02_fill_value_1023_masking) ... ok
test_03_calibration_output_shape_units_and_finite_values (test_insat_preprocessing.TestINSATPreprocessing.test_03_calibration_output_shape_units_and_finite_values) ... ok
test_04_exact_geographic_mask_vs_enclosing_crop (test_insat_preprocessing.TestINSATPreprocessing.test_04_exact_geographic_mask_vs_enclosing_crop) ... ok
test_05_image_and_coordinate_shapes_consistency (test_insat_preprocessing.TestINSATPreprocessing.test_05_image_and_coordinate_shapes_consistency) ... ok
test_06_timestamp_ordering_and_scan_line_metadata (test_insat_preprocessing.TestINSATPreprocessing.test_06_timestamp_ordering_and_scan_line_metadata) ... ok
test_07_preservation_of_raw_source_files (test_insat_preprocessing.TestINSATPreprocessing.test_07_preservation_of_raw_source_files) ... ok
test_08_combine_preprocessing_masks_failure_modes (test_insat_preprocessing.TestINSATPreprocessing.test_08_combine_preprocessing_masks_failure_modes) ... ok
test_09_mir_nan_source_verification (test_insat_preprocessing.TestINSATPreprocessing.test_09_mir_nan_source_verification) ... ok
test_10_scan_timing_sweep_mapping_and_recurrence (test_insat_preprocessing.TestINSATPreprocessing.test_10_scan_timing_sweep_mapping_and_recurrence) ... ok
test_01_single_file_reader_and_calibration (test_preprocessing.TestGOESPreprocessing.test_01_single_file_reader_and_calibration) ... ok
test_02_multichannel_alignment_and_synchronization (test_preprocessing.TestGOESPreprocessing.test_02_multichannel_alignment_and_synchronization) ... ok
test_03_geonavigation_projection_inversion (test_preprocessing.TestGOESPreprocessing.test_03_geonavigation_projection_inversion) ... ok
test_04_geographic_crop_and_patch_extraction (test_preprocessing.TestGOESPreprocessing.test_04_geographic_crop_and_patch_extraction) ... ok
test_05_temporal_sequence_generator (test_preprocessing.TestGOESPreprocessing.test_05_temporal_sequence_generator) ... ok

----------------------------------------------------------------------
Ran 28 tests in 6.110s

OK
```

---

## 7. Modifications and Code Integrity Summary

The following files were created or modified during this verification and correction pass:

| File | Status | Action Taken |
|---|---|---|
| [`src/data/insat_reader.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/insat_reader.py) | Modified | Added `combine_preprocessing_masks()` utility (backup preserved at `src/data/insat_reader.py.phase2d_bak`). |
| [`src/data/__init__.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/__init__.py) | Modified | Exported `combine_preprocessing_masks`. |
| [`tests/test_insat_preprocessing.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/tests/test_insat_preprocessing.py) | Modified | Added `test_08`, `test_09`, `test_10` validating mask failure modes, MIR root cause, and sweep recurrence. |
| [`scripts/debug_mir_nan.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/debug_mir_nan.py) | Created | Forensic script isolating raw HDF5 fill pixel at `(549, 1530)`. |
| [`scripts/audit_all_prototype_pixels.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/audit_all_prototype_pixels.py) | Created | Full 7-file, 4-channel pixel validity and NaN census. |
| [`scripts/analyze_all_scan_times.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/analyze_all_scan_times.py) | Created | Sweep mapping and millisecond recurrence interval analysis across all 7 files. |
| [`scripts/quantify_temporal_stability.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/quantify_temporal_stability.py) | Created | Empirical 6-pair difference statistics, striping diagnostic, and landmark cross-correlation. |
| [`reports/phase2d_verification_correction_summary.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/reports/phase2d_verification_correction_summary.json) | Created | Comprehensive numerical summary of temporal stability and striping metrics. |
| [`reports/figures/insat3ds_quantitative_temporal_analysis.png`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/reports/figures/insat3ds_quantitative_temporal_analysis.png) | Created | 6-panel diagnostic figure visualizing temporal histograms, diurnal desert curves, striping, and landmark correlation. |
| `data/insat3ds/raw/*.h5` | **Preserved** | Strictly unmodified; all 7 file sizes and byte counts remain identical to Phase 2A audit baseline. |
| `outputs/insat3ds_prototype/*` | **Preserved** | Existing processed `.npz` arrays and metadata remain intact. |

---

## 8. Final Model Readiness Recommendation

The INSAT-3DS preprocessing prototype is scientifically validated and ready for integration into the downstream METEOR-Net pipeline, subject to the following engineering design choices:

1. **Loss Masking**: Always combine the channel validity mask and geographic bounding mask using `combine_preprocessing_masks(..., mode='intersection')` before computing spatial or temporal loss functions.
2. **Timing Assumption**: While nominal 30-minute pacing is confirmed, architectures should treat recurrence intervals as approximately invariant ($\sigma < 0.7\text{ s}$), and record the true scan time offset if sub-minute continuous time embeddings are introduced.
3. **Registration and Striping Resilience**: The network architecture must not assume rigid pixel-wise temporal constancy. Feature aggregation (MCE $\to$ TFEE $\to$ EFFR) and flow-free spatial evolution are well-suited to accommodate the observed $\sim 1$-pixel landmark shift and the $0.15\text{ K}$ detector striping baseline.

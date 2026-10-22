# Phase 6B — Dataset-to-Model Integration and Training Smoke-Test Preflight Report

**Timestamp:** 2026-10-22T16:56:00+05:30  
**Phase Status:** **PASS (PREFLIGHT AUDIT & SMOKE TEST VERIFIED)**  
**Dataset Version:** GOES-19 CONUS Multi-Event Dataset (4 Independent Storm Episodes, 1,152 NetCDF files, 216 sequence windows)  
**Model Architecture:** METEOR-Net (MCE $\to$ TFEE $\to$ EFFR, 22,086,943 parameters)  
**Test Suite Status:** **65 / 65 Passing** (100% test pass rate across entire repository)  

---

## Executive Summary

Phase 6B completed an end-to-end dataset-to-model integration audit and executed a strictly bounded single-step integration smoke test. The audit reconciled the discrete event partitioning policy, implemented a memory-bounded lazy PyTorch dataset loader, traced tensor shapes and physical unit conversions across the entire computational graph, executed one optimizer step on real training data, confirmed strict weight and normalization invariance during validation, and validated checkpoint serialization and reloading.

All integration criteria and strict preflight constraints are satisfied. Full training has **not** been started.

---

## 1. Split-Ratio Discrepancy Analysis and Resolution

### 1.1 The Discrepancy Explained

| Attribute | Nominal Configured Ratio | Realized Partitioning | Root Cause & Scientific Rationale |
| :--- | :---: | :---: | :--- |
| **Train** | 60% | 50% (2 events) | Meteorological event isolation prohibits splitting an episode across partitions. |
| **Validation** | 20% | 25% (1 event) | With exactly 4 discrete storm episodes, integer event partitioning allows only whole-event assignments ($2, 1, 1$). |
| **Test** | 20% | 25% (1 event) | Continuous storm episodes cannot be divided into fractional events ($2.4 / 0.8 / 0.8$) without causing severe temporal leakage. |

- **Strict Event Isolation**: Overlapping temporal windows (sampled at 10-minute cadence within a 6-hour sequence) and adjacent spatial patches from the same observation period share convective dynamics, synoptic atmospheric forcing, and sensor calibration states. Splitting patches or windows from the same date across train and test partitions would cause profound data leakage.
- **Resolution**: Both [`scripts/create_dataset_split.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/create_dataset_split.py) and [`outputs/dataset_splits/splits_manifest.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/dataset_splits/splits_manifest.json) have been updated to explicitly record both the nominal target ratios (`nominal_target_ratios: {"train": 0.6, "val": 0.2, "test": 0.2}`) and the realized discrete event assignments (`realized_event_counts: {"train": 2, "val": 1, "test": 1}`, `realized_event_ratios: {"train": 0.5, "val": 0.25, "test": 0.25}`). No artificial events were manufactured.

### 1.2 Partition Counts and Statistical Independence

Every sequence in the manifest was verified to belong to exactly one partition, with its associated raw NetCDF files cryptographically indexed by SHA-256.

> [!IMPORTANT]
> **Statistical Independence Distinction**: Spatial patches and sliding sequence windows are **not** independent weather events. While the dataset contains **12,960 patch instances**, they represent exactly **4 independent weather episodes**.

| Partition | Assigned Meteorological Event | Calendar Date & Window | Raw NetCDF Files | Synchronized Scans | Sequence Windows | Spatial Patches ($128 \times 128$) | Share of Dataset |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | `event_2026-01-01`<br>`event_2026-05-20` | Jan 1, 2026 (02:00–08:00 UTC)<br>May 20, 2026 (18:00–24:00 UTC) | 576 | 144 | 108 | 6,480 | 50.0% |
| **Val** | `event_2026-07-19` | Jul 19–20, 2026 (20:00–02:00 UTC) | 288 | 72 | 54 | 3,240 | 25.0% |
| **Test** | `event_2026-09-07` | Sep 7, 2026 (16:00–22:00 UTC) | 288 | 72 | 54 | 3,240 | 25.0% |
| **Total** | **4 Independent Episodes** | — | **1,152** | **288** | **216** | **12,960** | **100.0%** |

*Note: Spatial patch grid based on CONUS geographic crop ($453 \times 740$ pixels, stride 64) produces $6 \times 10 = 60$ patches per sequence window.*

---

## 2. Complete Data-to-Model Pipeline Trace

### 2.1 Component Architecture and Interface Contracts

The full data ingestion, normalization, inference, and loss pipeline was verified via [`src/data/goes_dataset.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/goes_dataset.py), [`src/models/normalization.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/normalization.py), [`src/models/meteor_net.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/meteor_net.py), and [`src/models/losses.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/losses.py).

```mermaid
flowchart TD
    subgraph Data Loading [On-Demand Lazy Loading (src/data/goes_dataset.py)]
        A[splits_manifest.json] --> B[GOESSequencePatchDataset]
        B -->|Read 5 Observation Scans| C[Raw NetCDF ABI Bands 09, 13, 14, 15]
        C -->|Bounded LRU Cache (size=2)| D[Window Arrays Kelvin]
        D -->|Extract 128x128 Patches| E[Input Kelvin [B, 4, 4, 128, 128]<br>Target Kelvin [B, 4, 128, 128]<br>Combined Mask [B, 128, 128]]
    end

    subgraph Pre-Inference Normalization [src/models/normalization.py]
        F[frozen_global_train_stats.json] --> G[ChannelStandardizer]
        E -->|Input Kelvin + Mask| G
        G -->|Normalize Valid Pixels / Zero Fill Masked| H[Input Norm [B, 4, 4, 128, 128] ~ N(0,1)]
    end

    subgraph Deep Model Forward [src/models/meteor_net.py]
        H --> I[METEOR-Net Architecture<br>MCE -> TFEE -> EFFR]
        I --> J[Predicted Output Norm [B, 4, 128, 128]]
    end

    subgraph Loss Evaluation [Physical Kelvin Space (src/models/losses.py)]
        J --> K[ChannelStandardizer.denormalize_output]
        K -->|Differentiable Linear Map| L[Predicted Kelvin [B, 4, 128, 128]]
        L --> M[MaskAwareConvectiveMAELoss]
        E -->|Target Kelvin + Combined Mask| M
        M -->|Band 13 Thresholds: 241K, 221K| N[Physical Convective MAE Loss (Scalar K)]
    end
```

### 2.2 Critical Invariants and Audit Findings

1. **Input Batch Shapes**: Model forward pass strictly expects and consumes `[B, 4, 4, 128, 128]` (Batch, Time=4, Channels=4, Height=128, Width=128).
2. **Channel Ordering**: Rigidly enforced across all modules as exactly `(9, 13, 14, 15)`:
   - Band 09 (6.95 µm Mid-level Water Vapor)
   - Band 13 (10.33 µm Clean Longwave Window — convective reference)
   - Band 14 (11.20 µm Traditional Longwave Window)
   - Band 15 (12.30 µm "Dirty" Longwave Window)
3. **Target Scan Authenticity**: Target observations are the **real observed scans at +15 minutes** ($t=15\text{ min}$) from actual NOAA GOES-19 archives. Synthetic or interpolated targets are never used.
4. **Frozen Normalization Statistics**: Fitted strictly on the Training partition (`event_2026-01-01` and `event_2026-05-20`):
   - Means (K): `[248.6812, 279.7365, 279.0453, 276.9893]`
   - Stds (K): `[10.2850, 17.9001, 18.2796, 17.6614]`
   - Zero validation or test data contributed to these statistics.
5. **Invalid Pixel Sanitization**: Invalid/limb/masked pixels are filled with the channel mean during normalization, mapping to exactly `0.0` in normalized space. This prevents `NaN`/`Inf` corruption in convolutional layers.
6. **Physical Loss Consistency**: Model predictions `y_norm` are de-normalized back to physical Kelvin via $y_{\text{phys}} = y_{\text{norm}} \times (\sigma + \epsilon) + \mu$ prior to computing loss. Convective thresholds ($\alpha=1.0$ for $T \ge 241\text{ K}$, $\beta=2.0$ for $221\text{ K} \le T < 241\text{ K}$, $\gamma=5.0$ for $T < 221\text{ K}$) evaluate exclusively against physical Band 13 temperatures.
7. **Bounded Memory Architecture**: `GOESSequencePatchDataset` implements on-demand NetCDF file opening with a 2-window LRU cache. Adjacent patches within a sequence window share decoded memory, while previous sequences are evicted. Python heap memory remains bounded ($<520\text{ MB}$) and does not load the 4 GB dataset into RAM.

---

## 3. Integration Smoke-Test Execution and Results

The bounded integration smoke test was executed via [`scripts/run_integration_smoke_test.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/run_integration_smoke_test.py).

### 3.1 Step-by-Step Test Results

| Step | Verification Check | Expected Outcome | Actual Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Normalization Statistics Loading | Exactly 4 channels `(9, 13, 14, 15)` matching frozen JSON | Means: `[248.68, 279.74, 279.05, 276.99]`<br>Stds: `[10.28, 17.90, 18.28, 17.66]` | **PASS** |
| **2** | Model & Loss Initialization | Clean graph with 22,086,943 trainable parameters | 22,086,943 parameters, AdamW lr=1e-4 | **PASS** |
| **3** | Training Patch Ingestion | 1 patch `[1, 4, 4, 128, 128]` from Event 1 (`event_2026-01-01`) | Input norm finite, target physical finite, 16,384 valid pixels | **PASS** |
| **4** | Training Forward & Backward | Finite loss in physical K, finite backward gradients | Loss: **5.6588 K**, Mean grad norm: **2.7417** (all 104 tensors finite) | **PASS** |
| **5** | Single Optimizer Step | Model parameters updated | **104 / 104** parameter tensors modified | **PASS** |
| **6** | Validation Patch Ingestion | 1 patch from independent Event 3 (`event_2026-07-19`) | Sequence `seq_event_2026-07-19_210115_w108` (convective core active) | **PASS** |
| **7** | Validation Forward Pass | Finite loss in physical K under `torch.no_grad()` | Loss: **15.6258 K** (convective penalty active) | **PASS** |
| **8** | Validation Invariance Check | Zero parameter mutation, zero normalization mutation | **0 / 104** model parameters modified<br>Mean/Std strictly invariant bit-for-bit | **PASS** |
| **9** | Checkpoint Serialization | Save state dict, optimizer, standardizer, step metadata | Saved `smoke_test_checkpoint.pt` (**252.89 MB**) in 0.309 s | **PASS** |
| **10** | Checkpoint Reload & Fidelity | Reloaded model reproduces identical predictions | State dict matches bit-for-bit, inference diff = **0.0** | **PASS** |

### 3.2 Resource Profile and Timing Measurements

Measurements captured via `time.perf_counter()` and `tracemalloc`:

| Operation | Elapsed Time (s) | Memory Impact |
| :--- | :---: | :--- |
| **Training Patch NetCDF Load & Crop** | 1.302 s | Initial window caching |
| **Training Forward Pass** | 1.875 s | Activation graph construction |
| **Training Backward Pass (Autograd)** | 3.710 s | Gradient backpropagation across 22.1M params |
| **AdamW Optimizer Step** | 0.111 s | Momentum and variance state updates |
| **Total Train Integration Step** | **7.074 s** | — |
| **Validation Patch NetCDF Load & Crop** | 1.672 s | Event 3 window caching |
| **Validation Forward Pass (`no_grad`)** | 1.945 s | Evaluation inference |
| **Total Val Integration Step** | **3.644 s** | — |
| **Checkpoint Save** | 0.309 s | Serialized 252.89 MB payload to disk |
| **Checkpoint Reload & Verify** | 0.456 s | Deserialization and bitwise parameter check |
| **Total Preflight Duration** | **18.437 s** | Clean, deterministic execution |
| **Peak Python Heap Memory** | — | **513.91 MB** (well within host headroom) |
| **Current Python Heap Memory** | — | **229.04 MB** |

Artifact generated: [`outputs/smoke_test/smoke_test_metrics.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/smoke_test/smoke_test_metrics.json).

---

## 4. Test Suite and Regression Status

A dedicated unit test suite for the lazy dataset loader was implemented in [`tests/test_goes_dataset.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/tests/test_goes_dataset.py).

The complete project test suite was executed:
```bash
python -m unittest discover tests
```

**Result: 65 / 65 tests passing (0 failures, 0 errors, 30.89s elapsed)**
- `tests/test_dataset_integrity.py`: 8 tests (SHA-256, deduplication, conflict resolution, split manifest generator)
- `tests/test_goes_dataset.py`: 4 tests (partition lengths, item shapes, types, LRU cache hit/reuse)
- `tests/test_insat_audit.py`: 6 tests (HDF5 structure, metadata extraction, corruption checks)
- `tests/test_insat_calibration_geo.py`: 5 tests (radiance calibration, projection, navigation)
- `tests/test_insat_preprocessing.py`: 12 tests (temporal alignment, multi-channel collation)
- `tests/test_meteor_net.py`: 13 tests (MCE, TFEE, EFFR, end-to-end forward pass, loss autograd)
- `tests/test_normalization.py`: 11 tests (ChannelStandardizer, physical de-normalization, invalid pixel masking)
- `tests/test_preprocessing.py`: 6 tests (GOES projection, geographic crop, spatial integrity)

---

## 5. Training Readiness Checklist & Blocker Status

| Readiness Item | Verification Status | Notes |
| :--- | :---: | :--- |
| **Raw NetCDF Dataset** | **VERIFIED** | 1,152 files across 4 events, 0 missing files, 0 hash conflicts |
| **Event Split Manifest** | **VERIFIED** | 216 sequence windows partitioned (50% Train, 25% Val, 25% Test) |
| **Normalization Statistics** | **FROZEN & VERIFIED** | Fitted strictly on Train; zero validation or test contamination |
| **Lazy Dataset Loader** | **IMPLEMENTED & TESTED** | `GOESSequencePatchDataset` with bounded LRU cache ($<520\text{ MB}$ RAM) |
| **Data-to-Model Pipeline** | **VERIFIED** | Batch shape `[B, 4, 4, 128, 128]`, channel order `(9, 13, 14, 15)` |
| **Physical Loss Interface** | **VERIFIED** | Model output de-normalized to Kelvin before convective loss |
| **Autograd & Optimization** | **VERIFIED** | 104/104 parameter gradients finite; 1 optimizer step executed |
| **Validation Leakage** | **VERIFIED NONE** | Zero weight updates or normalization changes during validation |
| **Checkpoint Fidelity** | **VERIFIED** | Saved and reloaded with exact parameter reproduction |
| **Remaining Blockers** | **NONE** | System is fully preflighted and ready for first controlled training run |

---

## 6. Strict Stopping Boundary

As specified by project instructions:
- **No full training or multi-epoch experiments have been started.**
- **No baseline evaluations have been run.**
- **No raw data or frozen statistics were modified.**
- Awaiting user authorization before launching Phase 7 (Controlled Baseline Training Run).

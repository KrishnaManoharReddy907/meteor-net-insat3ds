# Phase 3 — Architecture and GOES-19 Data-Readiness Audit Report (Final Revision)

**Project Title**: Confidence-Aware Temporal Super-Resolution of INSAT-3DS Satellite Imagery using a METEOR-Net-based, flow-free temporal-evolution architecture  
**Phase**: Phase 3 — Technical Prototype Architecture Specification & GOES-19 Data-Readiness Audit  
**Date**: October 22, 2026 (Final Consistency Verification Pass)  
**Auditor**: Antigravity Autonomous Coding Agent  
**Operational Scope**: Inspection, Technical Specification, and Empirical Data Audit. (No code implementation, model training, or data acquisition performed in this step).  

---

## 1. Executive Verdict: Measured Facts, Analytical Calculations, and Unmeasured Estimates

### **VERDICT: TECHNICAL PROTOTYPE SPECIFIED — CURRENT DATA SAMPLE SUPPORTS TECHNICAL DEBUGGING ONLY**

```
+-------------------------------------------------------------------------------------------------------------------+
|                                      PHASE 3 AUDIT STATUS & EVIDENCE TIERS                                         |
+-------------------------------------------------------------------------------------------------------------------+
|  1. Architecture Formulation   : SPECIFIED FOR PROTOTYPE (MCE -> TFEE -> EFFR; Stride-2 Latent; Deterministic)     |
|  2. Model Parameter Count      : 22,086,943 (~22.09M) [ANALYTICAL CALCULATION; NOT yet measured in PyTorch]       |
|  3. Peak GPU Memory (B=4)      : ~3.30 GB [ANALYTICAL ESTIMATE; NOT yet measured on RTX 3050 hardware]            |
|  4. Peak GPU Memory (B=8)      : ~5.45 GB [ANALYTICAL ESTIMATE; Approaches 6 GB ceiling; B=4 recommended]         |
|  5. Existing Raw Data (96 .nc) : EMPIRICALLY MEASURED (100% synchronized, uncorrupted, 5.00 min cadence)          |
|  6. Temporal Windows Formed    : 6 SLIDING WINDOWS (EMPIRICALLY MEASURED; 80% frame overlap across 115 minutes)    |
|  7. Independent Weather Events : EXACTLY 1 SYNOPTIC EVENT (Current sample represents 1 single morning storm)      |
|  8. Extracted Patches          : 360 (SE US) / 384 (West US) with 50% spatial overlap (EMPIRICALLY MEASURED)      |
|  9. Dataset Role               : TECHNICAL DEBUGGING ONLY (Cannot establish generalization; severe autocorrelation)|
| 10. Prototype Acceptance Scope : TECHNICAL VERIFICATION (Shape, finite grads, zero dead branches, peak VRAM metric)|
+-------------------------------------------------------------------------------------------------------------------+
```

### 1.1 Strict Separation of Measured Facts, Analytical Calculations, and Estimates

To ensure scientific defensibility, every claim in this report is strictly audited and classified into its precise evidentiary category:

| Claim / Metric | Value | Evidentiary Status | Methodology / Verification Source |
|---|:---:|:---:|---|
| **Raw File Inventory** | 96 NetCDF files ($282.9\text{ MB}$) | **Empirically Measured** | Directly verified via file system scan across `data/goes19/2026/001/{04,05}/`. |
| **Channel Synchronization** | 24 timestamps $\times$ 4 channels | **Empirically Measured** | Verified by parsing NetCDF global attributes; 100% matched to $0.1\text{s}$ precision. |
| **Scan Cadence & Span** | $5.00\text{ min}$ delta, $115.0\text{ min}$ span | **Empirically Measured** | Measured from `04:01:19 UTC` to `05:56:19 UTC` on January 1, 2026 ($\sigma = 0.00\text{ min}$). |
| **Usable Sequence Windows** | Exactly 6 continuous sliding windows | **Empirically Measured** | Executed `TemporalSequenceGenerator` with offsets $[-60, -30, 0, +30]\to +15\text{ min}$. |
| **Target Midpoint Authenticity**| Observed Level-2 scan at $t+15\text{ min}$ | **Empirically Measured** | Confirmed physical satellite file exists for each target midpoint; not interpolated. |
| **SE US Patch Convection** | $0 \text{ out of } 360$ patches convective | **Empirically Measured** | Executed patch extraction on SE US crop; min temp $263.2\text{ K}$, 0 pixels $< 241\text{ K}$. |
| **Western US Convection** | $36.79\%$ conv, $4.25\%$ deep core | **Empirically Measured** | Executed patch extraction on Western US crop; $135,604$ pixels $< 241\text{ K}$, $15,652 < 221\text{ K}$. |
| **Temporal Storm Evolution** | Core shrinks $17,463 \to 15,195$ px | **Empirically Measured** | Computed pixel counts at $t-60, t-30, t, t+15, t+30\text{ min}$ on Band 13. |
| **Model Parameter Count** | **22,086,943 weights ($88.35\text{ MB}$)** | **Analytical Calculation** | Recalculated algebraically across all layers (including recurrent $W_x$ and $W_h$); **NOT yet instantiated**. |
| **Activation Memory ($B=4$)** | $\sim 2.15\text{ GB}$ (latent $64\times 64$) | **Analytical Estimate** | Calculated algebraically from intermediate tensor dimensions and float32 bytes. |
| **Activation Memory ($B=8$)** | $\sim 4.29\text{ GB}$ (latent $64\times 64$) | **Analytical Estimate** | Calculated algebraically; indicates $B=8$ approaches 6 GB hardware limit. |
| **Peak GPU VRAM Usage ($B=4$)**| $\sim 3.30\text{ GB}$ on RTX 3050 | **Analytical Estimate** | **NOT measured on hardware yet**. Sum of driver, weights, optimizer, and activations. |
| **Peak GPU VRAM Usage ($B=8$)**| $\sim 5.45\text{ GB}$ on RTX 3050 | **Analytical Estimate** | **NOT measured on hardware yet**. Recommended default batch size is $B=4$. |

### 1.2 Role of the Existing Dataset: Technical Debugging Only
The existing 96-file GOES-19 sample is **strictly limited to technical pipeline debugging, shape validation, and gradient verification**. It cannot support scientific claims of generalization or model superiority:
- **Temporal Windows vs. Weather Events**: The dataset contains 6 sliding windows, but these 6 windows represent **exactly one weather event** observed over a duration of only 115 minutes.
- **Severe Temporal Overlap**: Sliding windows spaced 5 minutes apart share 4 out of 5 frames (**$80\%$ frame overlap**). Successive windows are near-duplicate observations of the exact same evolving cloud field.
- **Overlapping Patches Are Not Independent**: Extracting 360 patches (stride 64) yields spatial tiles that overlap by $50\%$ with their spatial neighbors and by $80\%$ across temporal windows. **Treating these 360 patches as independent statistical samples would artificially inflate the effective sample size and cause severe data leakage.**
- **Lack of Synoptic and Diurnal Diversity**: Spanning only 115 minutes during local pre-dawn night, the sample contains zero diurnal variation, no solar transition, and no alternative meteorological regimes.

---

## 2. Files Inspected and Evidence Sources

The following 11 project artifacts were directly inspected to establish the evidence base for this audit:

| Document / Code Artifact | File Path | Status | Verification Summary |
|---|---|:---:|---|
| **Base Paper Audit Report** | [`docs/meteor_net_base_paper_architecture_audit.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/meteor_net_base_paper_architecture_audit.md) | Inspected | Source-grounded breakdown of Lin et al. (2026), MCE/TFEE/EFFR modules, Eq. 1–27, and evidence tiers. |
| **Original Paper PDF** | [`docs/references/Self-Supervised_Temporal_Super-Resolution_Framework_With_METEOR-Net_for_Enhancing_Convective_Process_Reconstruction_in_Geostationary_Satellite_Cloud_Data (1).pdf`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/references/Self-Supervised_Temporal_Super-Resolution_Framework_With_METEOR-Net_for_Enhancing_Convective_Process_Reconstruction_in_Geostationary_Satellite_Cloud_Data%20(1).pdf) | Inspected | 19-page IEEE TGRS publication; confirmed figures 4, 5, 6, 7, equations 14–20, and experimental parameters. |
| **Phase 2A Design Review** | [`docs/phase2_design_review.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase2_design_review.md) | Inspected | 96-file synchronization matrix, 4-channel selection (Bands 9, 13, 14, 15), and unpaired adaptation rationale. |
| **Phase 2B Preprocessing Report** | [`docs/phase2b_preprocessing_report.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase2b_preprocessing_report.md) | Inspected | Ingestion pipeline, Southeast US crop ($453 \times 740$), native 2 km grid, and 6 sequence windows. |
| **Phase 2C Geonavigation Review** | [`docs/phase2c_insat_calibration_geolocation_review.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase2c_insat_calibration_geolocation_review.md) | Inspected | INSAT-3DS calibration LUT verification, GSICS radiance alignment, and Indian crop bounds. |
| **Phase 2D Verification Report** | [`docs/phase2d_verification_correction_report.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase2d_verification_correction_report.md) | Inspected | Empirical correction of striping ($0.155\text{ K}$), registration error ($\pm 1$ pixel), and mask combination rules. |
| **GOES Reader Module** | [`src/data/goes_reader.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/goes_reader.py) | Inspected | `read_goes_single_file`, `GOESProjection`, `read_multichannel_timestamp`, and `build_timestamp_catalog`. |
| **Spatial Crop & Patch Module** | [`src/data/spatial_crop.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/spatial_crop.py) | Inspected | `GeographicCropper` and `extract_patches` ($128 \times 128$ patch extraction, convective threshold tagging). |
| **Sequence Generator Module** | [`src/data/sequence_generator.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/sequence_generator.py) | Inspected | `TemporalSequenceGenerator`, offset validation, and synchronized window tensor construction. |
| **Processed Sample Artifact** | [`outputs/processed_sample/sample_sequence_window00.npz`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/processed_sample/sample_sequence_window00.npz) | Inspected | Verified keys (`inputs`, `targets`, `valid_mask`), input shape `(4, 4, 453, 740)`, target shape `(1, 4, 453, 740)`. |
| **Sequence Metadata Catalog** | [`outputs/processed_sample/sequence_metadata.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/processed_sample/sequence_metadata.json) | Inspected | Documents 6 windows, 60 patches per window, and storage estimates. |

---

## 3. TASK B — Empirical Audit of the Actual GOES-19 Sample

The sample dataset in `data/goes19/2026/001/` was audited via direct inspection and execution of [`scripts/audit_goes_sample_readiness.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/audit_goes_sample_readiness.py).

### 3.1 Measured Inventory and Alignment Matrix
- **Raw File Count**: Exactly **96 NetCDF-4 (`.nc`) files** ($282.9\text{ MB}$ total).
- **Hours Covered**: UTC hour folders `04` (48 files) and `05` (48 files) for Julian day `001` (January 1, 2026).
- **Channels**: 24 files each for Band 9 ($6.9\,\mu\text{m}$ WV), Band 13 ($10.3\,\mu\text{m}$ Clean IR), Band 14 ($11.2\,\mu\text{m}$ IR), and Band 15 ($12.3\,\mu\text{m}$ Dirty IR).
- **Temporal Alignment**: All 24 observation timestamps match across all 4 channels to the exact tenth of a second (0% missing files, 0 duplicate timestamps).
- **Cadence**: Pacing is strictly **$5.00\text{ minutes}$** ($\min = 5.00, \max = 5.00, \sigma = 0.00\text{ min}$) spanning from `04:01:19 UTC` to `05:56:19 UTC` ($115.0\text{ minutes}$ elapsed).
- **Grid Dimensions & Precision**: Native CONUS Level-2 grid is $1500 \times 2500$ pixels at $2.0\text{ km}$ nadir resolution. Data type is `float32` calibrated in absolute Kelvin ($K$).

### 3.2 Sequence Windows Formed Under Approved 4-Input Design
Using the approved 4-input configuration with offsets $[-60\text{m}, -30\text{m}, 0\text{m}, +30\text{m}]$ and midpoint target $[+15\text{m}]$:
- A single sequence window requires an elapsed span of $90\text{ minutes}$ ($t - 60\text{m}$ to $t + 30\text{m}$).
- From the 115-minute sample, **exactly 6 continuous sliding sequence windows** can be formed:

| Window Index | Input 1 ($t-60\text{m}$) | Input 2 ($t-30\text{m}$) | Input 3 (Anchor $t$) | Target Frame ($t+15\text{m}$) | Input 4 ($t+30\text{m}$) | Window Span | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Window 00** | `04:01:19 UTC` | `04:31:19 UTC` | `05:01:19 UTC` | **`05:16:19 UTC`** | `05:31:19 UTC` | 90 min | Observed ground truth |
| **Window 01** | `04:06:19 UTC` | `04:36:19 UTC` | `05:06:19 UTC` | **`05:21:19 UTC`** | `05:36:19 UTC` | 90 min | Observed ground truth |
| **Window 02** | `04:11:19 UTC` | `04:41:19 UTC` | `05:11:19 UTC` | **`05:26:19 UTC`** | `05:41:19 UTC` | 90 min | Observed ground truth |
| **Window 03** | `04:16:19 UTC` | `04:46:19 UTC` | `05:16:19 UTC` | **`05:31:19 UTC`** | `05:46:19 UTC` | 90 min | Observed ground truth |
| **Window 04** | `04:21:19 UTC` | `04:51:19 UTC` | `05:21:19 UTC` | **`05:36:19 UTC`** | `05:51:19 UTC` | 90 min | Observed ground truth |
| **Window 05** | `04:26:19 UTC` | `04:56:19 UTC` | `05:26:19 UTC` | **`05:41:19 UTC`** | `05:56:19 UTC` | 90 min | Observed ground truth |

Every target frame is an **actual observed GOES-19 Level-2 scan** recorded at the physical midpoint ($t+15\text{ min}$), not an interpolated frame.

### 3.3 Patch Extraction and Rejection Statistics

#### 3.3.1 Southeast US / Gulf Regional Crop ($25^\circ-35^\circ\text{N}, 80^\circ-95^\circ\text{W}$, shape $453 \times 740$)
- **Stride 64 (50% Spatial Overlap)**:
  - Candidate Patches: $6 \text{ rows} \times 10 \text{ cols} = 60 \text{ patches/window} \times 6 \text{ windows} = 360 \text{ candidate patches}$.
  - Accepted Patches: **360 accepted** (Rejection rate: $0.0\%$).
- **Stride 128 (Non-overlapping)**:
  - Candidate Patches: $3 \text{ rows} \times 5 \text{ cols} = 15 \text{ patches/window} \times 6 \text{ windows} = 90 \text{ candidate patches}$.
  - Accepted Patches: **90 accepted** (Rejection rate: $0.0\%$).
- **Valid Pixel Fraction**: $100.00\%$ ($0$ space fill pixels, $0$ NaN values, $0$ radiometric flags).
- **Target Band 13 Temperature Distribution**:
  - Min: $263.23\text{ K}$, Median: $271.09\text{ K}$, Mean: $279.75\text{ K}$, Max: $294.54\text{ K}$.
  - Fraction of patches with any pixel $< 273\text{ K}$ (Cold cloud / winter terrain): **$75.0\%$**.
  - Fraction of patches with any pixel $< 241\text{ K}$ (Convective cloud): **$0.0000$ ($0\%$)**.
  - Fraction of patches with any pixel $< 221\text{ K}$ (Deep convective core): **$0.0000$ ($0\%$)**.
  - **Convective Classification**: **$0 \text{ out of } 360$ patches qualify as convective**.
  - **Dominant Land Cover**: 100% dominated by ambient winter terrain, cloud-free ocean (Gulf of Mexico), and warm marine stratocumulus.

#### 3.3.2 Full CONUS Reference Disk Audit at Target Midpoint (`05:16:19 UTC`)
Auditing all $3,750,000$ pixels ($1500 \times 2500$) of the full CONUS grid:
- **Valid Earth Pixels**: $3,702,837$ ($98.74\%$).
- **Space Fill / Limb Pixels Rejected**: $47,163$ ($1.26\%$) rejected by geometric/ellipsoid mask.
- **Cold Cloud Pixels ($T < 273\text{ K}$)**: $1,844,029$ ($49.80\%$).
- **Convective Pixels ($T < 241\text{ K}$)**: $353,423$ ($9.54\%$).
- **Deep Convective Pixels ($T < 221\text{ K}$)**: $23,086$ ($0.62\%$).
- **Deep Core Centroid**: Concentrated over the Western US / Eastern Pacific ($25^\circ-38^\circ\text{N}, 105^\circ-120^\circ\text{W}$, mean longitude $-107.66^\circ\text{E}$, minimum temperature $206.93\text{ K}$).

#### 3.3.3 Western US Convective Sector Audit ($25^\circ-38^\circ\text{N}, 105^\circ-120^\circ\text{W}$, shape $585 \times 630$)
Testing patch extraction over the active Western storm region:
- **Candidate Patches (Stride 64)**: $8 \text{ rows} \times 8 \text{ cols} = 64 \text{ patches/window} \times 6 \text{ windows} = 384 \text{ candidate patches}$.
- **Accepted Patches**: **384 accepted** ($100\%$ valid Earth pixels, 0 space fill).
- **Target Band 13 Temperature Distribution**:
  - Min: **$210.74\text{ K}$**, Median: **$247.86\text{ K}$**, Max: $292.15\text{ K}$.
  - Fraction of pixels $< 241\text{ K}$ (Convective): **$36.79\%$** ($135,604\text{ pixels}$).
  - Fraction of pixels $< 221\text{ K}$ (Deep Core): **$4.25\%$** ($15,652\text{ pixels}$).
  - **Patches with Convective Pixels ($< 241\text{ K}$)**: **$55 \text{ out of } 64$ ($85.9\%$)**.
  - **Patches with Deep Convective Pixels ($< 221\text{ K}$)**: **$33 \text{ out of } 64$ ($51.6\%$)**.

### 3.4 Temporal Evolution Analysis: Measured Storm Decay
Tracking the Western US storm across the five sequence positions in Window 00:

| Sequence Frame | Nominal Timestamp | Min Temp ($K$) | Mean Temp ($K$) | Pixels $< 241\text{ K}$ (Conv) | Pixels $< 221\text{ K}$ (Deep Core) | Physical Evolution State |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Input 1 ($t-60\text{m}$)** | `04:01:19 UTC` | **$208.22\text{ K}$** | $250.93\text{ K}$ | $139,472$ ($37.84\%$) | $17,463$ ($4.74\%$) | Peak convective intensity |
| **Input 2 ($t-30\text{m}$)** | `04:31:19 UTC` | **$209.70\text{ K}$** | $251.35\text{ K}$ | $139,046$ ($37.73\%$) | $16,467$ ($4.47\%$) | Steady state, slight warming |
| **Input 3 ($t$)** | `05:01:19 UTC` | **$210.44\text{ K}$** | $251.74\text{ K}$ | $137,027$ ($37.18\%$) | $16,111$ ($4.37\%$) | Gradual core decay |
| **Target ($t+15\text{m}$)** | `05:16:19 UTC` | **$210.74\text{ K}$** | $251.90\text{ K}$ | $135,604$ ($36.79\%$) | $15,652$ ($4.25\%$) | **Physical midpoint observed** |
| **Input 4 ($t+30\text{m}$)** | `05:31:19 UTC` | **$210.74\text{ K}$** | $252.05\text{ K}$ | $132,604$ ($35.98\%$) | $15,195$ ($4.12\%$) | Continued dissipation & advection |

**Findings on Storm Dynamics**:
1. **Strong Persistence with Gradual Dissipation**: Convective structures persist throughout the entire 90-minute window, retaining $> 87\%$ of their deep core area ($17,463 \to 15,195\text{ pixels}$).
2. **Monotonic Core Warming**: The deep core area ($T < 221\text{ K}$) decreases monotonically by $13.0\%$, and the minimum cloud top temperature warms by $2.52\text{ K}$ ($208.22\text{ K} \to 210.74\text{ K}$).
3. **Advection Velocity**: Convective cores exhibit eastward advection of approximately $18-24\text{ km/h}$ ($\sim 4-6\text{ pixels}$ per 30 minutes).

---

## 4. Finalized Technical-Prototype Architecture Specification (MCE → TFEE → EFFR)

This specification defines the **smallest internally consistent implementation** necessary for a forward/backward feasibility test. It does not claim to be an exact reproduction of Lin et al. (2026), but rather an implementation-ready adaptation for our task and hardware constraints.

### 4.1 MCE (Multichannel Collaborative Encoding)
- **Input Channels**: $C_{\text{in}} = 4$ GOES-19 channels: Band 9 ($6.9\,\mu\text{m}$ WV), Band 13 ($10.3\,\mu\text{m}$ Clean IR), Band 14 ($11.2\,\mu\text{m}$ IR), Band 15 ($12.3\,\mu\text{m}$ Dirty IR).
- **Weight Sharing**: GCS and LCE weights are **strictly shared across all 4 input time steps** ($t \in \{-60, -30, 0, +30\}\text{ min}$). Physical radiative transfer relationships are stationary across the sequence.
- **Global Channel Squeeze (GCS)**:
  - Global average pooling across $H \times W \implies 4$-element channel descriptor vector.
  - Two $1 \times 1$ convs ($W_{s1}: 4 \to 4$, $W_{s2}: 4 \to 4$) with LeakyReLU and Sigmoid generate channel gating vector $s \in [0, 1]^4$ (40 parameters).
  - Multi-channel collaborative projection: $\tilde{L}_t = W_{xs} * (L_t \odot s)$, where $W_{xs}$ is a $3 \times 3$ conv projecting $4 \to 64$ feature channels ($2,368$ parameters).
- **Local Convolutional Encoding (LCE)**:
  - **Sequence Branch ($N = 3$)**: Three sequential Conv-GN-LeakyReLU blocks ($64 \to 64$, $3 \times 3$ kernel, GroupNorm with 32 groups).
    - *Spatial Resolution Choice `[Engineering Adaptation]`*: Layer 1 uses **stride 2**, downsampling spatial dimensions from $128 \times 128 \to 64 \times 64$ ($H'' = W'' = 64$). Layers 2 and 3 operate with stride 1.
    - *Engineering Justification*: On our 6 GB RTX 3050 GPU, maintaining $128 \times 128$ throughout causes immediate out-of-memory errors during ConvLSTM backpropagation. Stride 2 reduces recurrent activation memory by $\sim 75\%$, enabling local batch execution.
    - *Parameter Count*: $3 \times [64 \times 64 \times 9 + 64 + 128] = 3 \times 37,056 = \mathbf{111,168}$ parameters.
  - **Skip Branch ($N = 1$)**: Single-layer $3 \times 3$ Conv-GN-LeakyReLU (stride 1) operating on boundary frames $L_0$ ($t = 0$) and $L_1$ ($t = +30\text{ min}$) to preserve high-resolution spatial textures: $X'_0, X'_1 \in \mathbb{R}^{B \times 64 \times 128 \times 128}$ ($\mathbf{37,056}$ parameters).
  - **Skip Midpoint Combination `[Engineering Adaptation]`**:
    $$X'_{t_{\text{mid}}} = W_{\text{skip}} * [X'_0 \, ; \, X'_1] \in \mathbb{R}^{B \times 64 \times 128 \times 128}$$
    where $W_{\text{skip}}$ is a learned $1 \times 1$ convolution mapping $128 \to 64$ channels ($64 \times 128 \times 1 + 64 = \mathbf{8,256}$ parameters).
- **MCE Subtotal**: $\mathbf{158,888}$ parameters.

---

### 4.2 TFEE Tensor Interfaces and State Transitions

To resolve all interface ambiguities for implementation, the exact tensor shapes, recurrent state transitions, and flow connections are documented below.

> **CRITICAL CLARIFICATION ON TEMPORAL FLOWS**:  
> The model's learned temporal features $\{F^n_{\text{low}}\}$ are **NOT physical optical-flow displacement fields or motion vector grids $(u, v)$**. They are deep latent spatiotemporal feature representations learned by the ConvLSTM to encode multiscale cloud evolution, convective expansion, and vertical development without optical-flow warping.

#### 4.2.1 Stage 1: Low-Resolution Flow Estimation ($F_{\text{low}}$)
- **Input Sequence**: Four latent frames from MCE sequence branch: $\mathcal{X} = \{X_1, X_2, X_3, X_4\}$, corresponding to nominal times $t \in \{-60\text{m}, -30\text{m}, 0\text{m}, +30\text{m}\}$.
  - Shape of each $X_t$: `[B, 64, 64, 64]` (under stride-2 adaptation).
  - Sequence tensor shape: `[B, 4, 64, 64, 64]`.
- **Recurrent Architecture**: 4-layer stacked ConvLSTM with LayerNorm.
  - Hidden channels: $C_{\text{hid}} = 256$ for all 4 layers ($n \in \{1, 2, 3, 4\}$).
  - Kernel size: $3 \times 3$, Padding: $1$, Dilation: $1$.
  - Gate Implementation: For computational efficiency, all four internal gates (input $i$, forget $f$, cell candidate $\tilde{C}$, output $o$) share a **single combined convolution** producing $4 \times 256 = 1,024$ output channels per layer.
  - Initial states at $t=0$: $Z^n_0 = \mathbf{0} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$, $C^n_0 = \mathbf{0} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.
- **Step-by-Step Recurrence Across 4 Steps ($t = 1 \dots 4$)**:
  - **Layer 1 ($n=1$)**:
    - Input: Concatenation $[X_t \, ; \, Z^1_{t-1}]$ with $64 + 256 = 320$ input channels.
    - Gate Conv: $1024 \times 320 \times 3 \times 3 + 1024 = 2,950,144$ weights/biases.
    - LayerNorm: 4 sets of affine parameters over 256 channels $\implies 4 \times 2 \times 256 = 2,048$.
    - Layer 1 Parameters: $\mathbf{2,952,192}$.
  - **Layers 2, 3, 4 ($n \in \{2, 3, 4\}$)**:
    - Input: Concatenation $[Z^{n-1}_t \, ; \, Z^n_{t-1}]$ with $256 + 256 = 512$ input channels.
    - Gate Conv: $1024 \times 512 \times 3 \times 3 + 1024 = 4,719,616$ weights/biases.
    - LayerNorm: $2,048$ affine parameters.
    - Parameters per Layer: $\mathbf{4,721,664}$.
    - Three Layers Total: $3 \times 4,721,664 = \mathbf{14,164,992}$.
- **Horizontal Temporal-Flow Tensors Produced**:
  - After processing step $t=4$ ($t+30\text{ min}$), the accumulated sequence dynamics across the 4 layers form the multiscale temporal-flow tensors:
    $$F^n_{\text{low}} = Z^n_4 \in \mathbb{R}^{B \times 256 \times 64 \times 64} \quad \text{for } n \in \{1, 2, 3, 4\}$$
- **Stage 1 Subtotal**: $2,952,192 + 14,164,992 = \mathbf{17,117,184}$ parameters.

#### 4.2.2 Stage 2: Single-Target High-Resolution Evolution ($F_{\text{high}}$)
- **Adaptation Rationale `[Engineering Adaptation]`**: The paper performs a 2-step recurrent evolution ($X_0 \to Y_{t_1} \to Y_{t_2}$ for 15-to-5 min). Our task requires **exactly one intermediate midpoint target** ($30\text{ to } 15\text{ min}$). We adapt this to a single bottom-to-top vertical evolution pass.
- **Inputs to Evolution**:
  - Spatial anchor feature: $X_0 \in \mathbb{R}^{B \times 64 \times 64 \times 64}$ (frame 3 of the sequence, at nominal time $t = 0\text{ min}$).
  - Multiscale horizontal flows: $\{F^1_{\text{low}}, F^2_{\text{low}}, F^3_{\text{low}}, F^4_{\text{low}}\}$, each in $\mathbb{R}^{B \times 256 \times 64 \times 64}$.
- **Vertical Layer Transitions `[Engineering Adaptation]`**:
  - **Layer 1**:
    - Input: Concatenation $[X_0 \, ; \, F^1_{\text{low}}] \in \mathbb{R}^{B \times 320 \times 64 \times 64}$.
    - Transformation: $3 \times 3$ Conv2d ($320 \to 256$) + GroupNorm(32) + LeakyReLU(0.2).
    - Parameters: $256 \times 320 \times 9 + 256 + 512 = \mathbf{738,048}$.
    - Output: $E^1 \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.
  - **Layers 2, 3, 4**:
    - Input: Concatenation $[E^{n-1} \, ; \, F^n_{\text{low}}] \in \mathbb{R}^{B \times 512 \times 64 \times 64}$.
    - Transformation: $3 \times 3$ Conv2d ($512 \to 256$) + GroupNorm(32) + LeakyReLU(0.2).
    - Parameters per Layer: $256 \times 512 \times 9 + 256 + 512 = \mathbf{1,180,416}$.
    - Three Layers Total: $3 \times 1,180,416 = \mathbf{3,541,248}$.
    - Outputs: $E^2, E^3, E^4 \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.
  - **Top Midpoint Latent Projection**:
    - Projection: $1 \times 1$ Conv2d ($256 \to 64$).
    - Parameters: $64 \times 256 \times 1 + 64 = \mathbf{16,448}$.
    - Output: $Y_{t_{\text{mid}}} = W_{\text{evol\_out}} * E^4 \in \mathbb{R}^{B \times 64 \times 64 \times 64}$.
- **Stage 2 Subtotal**: $738,048 + 3,541,248 + 16,448 = \mathbf{4,295,744}$ parameters.
- **TFEE Module Total**: $17,117,184 + 4,295,744 = \mathbf{21,412,928}$ parameters.

---

### 4.3 EFFR (Evolution-Aware Feature Fusion & Reconstruction) and Attention Resolution

#### 4.3.1 Skip-Branch Consistency & Dimensional Alignment
- **Source Boundary Frames**:
  - Left boundary frame: $L_0$ at nominal time $t = 0\text{ min}$ (sequence index 2), shape `[B, 4, 128, 128]`.
  - Right boundary frame: $L_1$ at nominal time $t = +30\text{ min}$ (sequence index 3), shape `[B, 4, 128, 128]`.
- **Shallow Skip Feature Extraction**:
  - Boundary frames pass through GCS ($W_{xs}$, shared weights) $\to \tilde{L}_0, \tilde{L}_1 \in \mathbb{R}^{B \times 64 \times 128 \times 128}$.
  - Both pass through the 1-layer shallow LCE Skip convolution ($3 \times 3$ Conv-GN-LeakyReLU, stride 1) $\to X'_0, X'_1 \in \mathbb{R}^{B \times 64 \times 128 \times 128}$.
- **Midpoint Skip Fusion Operation `[Engineering Adaptation]`**:
  - Concatenation along channel axis: $[X'_0 \, ; \, X'_1] \in \mathbb{R}^{B \times 128 \times 128 \times 128}$.
  - $1 \times 1$ Conv projection $W_{\text{skip}}$ mapping $128 \to 64$ channels:
    $$X'_{t_{\text{mid}}} = W_{\text{skip}} * [X'_0 \, ; \, X'_1] \in \mathbb{R}^{B \times 64 \times 128 \times 128}$$
- **Reconstruction Submodule (Spatial Scale Restoration)**:
  - Input: $Y_{t_{\text{mid}}} \in \mathbb{R}^{B \times 64 \times 64 \times 64}$.
  - Layer 1: $3 \times 3$ Conv2d + GroupNorm(32) + LeakyReLU ($64 \to 64$, shape `[B, 64, 64, 64]`) $\implies \mathbf{37,056}$ parameters.
  - Layer 2: $3 \times 3$ ConvTranspose2d with **stride 2**, padding 1, output_padding 1 ($64 \to 64$, restoring $64 \times 64 \to 128 \times 128$) + GroupNorm(32) + LeakyReLU $\implies \mathbf{37,056}$ parameters.
  - Output: $\tilde{Y}_{t_{\text{mid}}} \in \mathbb{R}^{B \times 64 \times 128 \times 128}$.
- **Verification of Dimension Compatibility**:
  - Skip feature $X'_{t_{\text{mid}}}$ has shape `[B, 64, 128, 128]`.
  - Evolved feature $\tilde{Y}_{t_{\text{mid}}}$ has shape `[B, 64, 128, 128]`.
  - Both tensors have **strictly identical dimensions** (Batch $B$, Channels $64$, Height $128$, Width $128$).
  - Elementwise sum $I = X'_{t_{\text{mid}}} + \tilde{Y}_{t_{\text{mid}}} \in \mathbb{R}^{B \times 64 \times 128 \times 128}$ is dimensionally exact and free of broadcasting errors.

#### 4.3.2 Evolution-Map Resolution and Attention Channel Reconciliation
In the previous draft, the trace table erroneously listed `[B, 129, 128, 128]` for the input to $W_p$, creating a contradiction with the mathematical formulation. We explicitly reconcile the equations and tensor dimensions as follows:

1. **Channel Weight Map ($c_{\text{map}}$) [Eq. 16, p. 8]**:
   $$c_{\text{map}} = W_{c2} * \delta\left( W_{c1} * \text{GAP}(I) + b_{c1} \right) + b_{c2} \in \mathbb{R}^{B \times 64 \times 1 \times 1}$$
   - Global Average Pooling: $\text{GAP}(I) \in \mathbb{R}^{B \times 64 \times 1 \times 1}$.
   - $W_{c1}$: $1 \times 1$ Conv mapping $64 \to 16$ (reduction ratio $r=4$) $\implies 64 \times 16 \times 1 + 16 = \mathbf{1,040}$ parameters.
   - $W_{c2}$: $1 \times 1$ Conv mapping $16 \to 64 \implies 16 \times 64 \times 1 + 64 = \mathbf{1,088}$ parameters.
2. **Spatial Weight Map ($s_{\text{map}}$) [Eq. 17, p. 8]**:
   $$s_{\text{map}} = W_s * \text{concat}\left( \text{mean}_c(I), \, \max_c(I) \right) + b_s \in \mathbb{R}^{B \times 1 \times 128 \times 128}$$
   - Channel pooling concatenates channel-wise mean and max: shape `[B, 2, 128, 128]`.
   - $W_s$: $7 \times 7$ Conv mapping $2 \to 1$ (padding 3) $\implies 1 \times 2 \times 49 + 1 = \mathbf{99}$ parameters.
3. **Evolution-Aware Map ($e_{\text{map}}$) [Eq. 18, p. 8] — Reconciled Channel Definition**:
   $$e_{\text{map}} = \sigma\left( W_p * \text{concat}\left( c_{\text{map}} + s_{\text{map}}, \, I \right) + b_p \right) \in [0, 1]^{B \times 64 \times 128 \times 128}$$
   - **Broadcast Addition**: $c_{\text{map}} \in \mathbb{R}^{B \times 64 \times 1 \times 1}$ and $s_{\text{map}} \in \mathbb{R}^{B \times 1 \times 128 \times 128}$ are broadcast-added to yield:
     $$(c_{\text{map}} + s_{\text{map}}) \in \mathbb{R}^{B \times 64 \times 128 \times 128}$$
   - **Concatenation**: Concatenating $(c_{\text{map}} + s_{\text{map}})$ with $I \in \mathbb{R}^{B \times 64 \times 128 \times 128}$ produces an exact tensor of shape:
     $$\text{concat}\left( c_{\text{map}} + s_{\text{map}}, \, I \right) \in \mathbb{R}^{B \times 128 \times 128 \times 128}$$
   - **Convolution $W_p$**: Takes strictly **$128$ input channels** and produces **$64$ output channels** via a $7 \times 7$ kernel (padding 3):
     - Parameters: $64 \times 128 \times 49 + 64 = \mathbf{401,472}$ parameters.
     - Output shape: `[B, 64, 128, 128]`.
   - *Resolution of the 129 Typo*: The previous draft's mention of `129` channels arose from an alternative 3-way concatenation interpretation ($1 + 64 + 64 = 129$). We explicitly adopt the broadcast-addition formulation from Eq. 18, which strictly yields $64 + 64 = \mathbf{128}$ input channels to $W_p$.
4. **Dynamic Redistribution [Eq. 19, p. 8]**:
   $$\tilde{I} = I + e_{\text{map}} \odot \tilde{Y}_{t_{\text{mid}}} + (1 - e_{\text{map}}) \odot X'_{t_{\text{mid}}} \in \mathbb{R}^{B \times 64 \times 128 \times 128}$$
5. **Scale Restoration [Eq. 20, p. 8]**:
   $$\hat{H}_{t_{\text{mid}}} = W_{h1} * \delta\left( \text{GN}(W_{h2} * \tilde{I} + b_{h2}) \right) + b_{h1} \in \mathbb{R}^{B \times 4 \times 128 \times 128}$$
   - $W_{h2}$: $3 \times 3$ Conv ($64 \to 64$) + GroupNorm(32) $\implies 64 \times 64 \times 9 + 64 + 128 = \mathbf{37,056}$ parameters.
   - $W_{h1}$: $1 \times 1$ Conv ($64 \to 4$) mapping feature channels back to the 4 physical infrared observation bands $\implies 4 \times 64 \times 1 + 4 = \mathbf{260}$ parameters.
- **EFFR Module Total**: $37,056 + 37,056 + 1,040 + 1,088 + 99 + 401,472 + 37,056 + 260 = \mathbf{515,127}$ parameters.

---

### 4.4 End-to-End Trainable Parameter Audit & Shape Trace

Below is the verified layer-by-layer analytical breakdown of trainable parameters across the network:

| Pipeline Stage | Module / Component | Exact Layer Construction | Input Shape | Output Shape | Trainable Parameter Count | Status |
|---|---|---|:---:|:---:|:---:|:---:|
| **MCE** | GCS Gating | $1\times 1$ Conv ($4 \to 4$) + LReLU + $1\times 1$ Conv ($4 \to 4$) | `[B, 4, 1, 1]` | `[B, 4, 1, 1]` | **40** | Paper Spec |
| **MCE** | GCS Projection ($W_{xs}$) | $3\times 3$ Conv ($4 \to 64$, bias) | `[B, 4, 128, 128]` | `[B, 64, 128, 128]` | **2,368** | Paper Spec |
| **MCE (Seq)** | LCE Sequence L1 | $3\times 3$ Conv ($64 \to 64$, stride 2) + GN(32) + LReLU | `[B, 64, 128, 128]` | `[B, 64, 64, 64]` | **37,056** | `[Adaptation]` |
| **MCE (Seq)** | LCE Sequence L2 | $3\times 3$ Conv ($64 \to 64$, stride 1) + GN(32) + LReLU | `[B, 64, 64, 64]` | `[B, 64, 64, 64]` | **37,056** | Paper Spec |
| **MCE (Seq)** | LCE Sequence L3 | $3\times 3$ Conv ($64 \to 64$, stride 1) + GN(32) + LReLU | `[B, 64, 64, 64]` | `[B, 64, 64, 64]` | **37,056** | Paper Spec |
| **MCE (Skip)**| LCE Skip Layer | $3\times 3$ Conv ($64 \to 64$, stride 1) + GN(32) + LReLU | `[B, 64, 128, 128]` | `[B, 64, 128, 128]` | **37,056** | Paper Spec |
| **MCE (Skip)**| Skip Blend ($W_{\text{skip}}$)| $1\times 1$ Conv ($128 \to 64$, bias) | `[B, 128, 128, 128]` | `[B, 64, 128, 128]` | **8,256** | `[Adaptation]` |
| **MCE Total** | **Subtotal MCE** | — | — | — | **158,888** | Analytical |
| **TFEE (Flow)**| ConvLSTM Layer 1 | $3\times 3$ Conv ($320 \to 1024$) + LayerNorm | `[B, 64, 64, 64]` | `[B, 256, 64, 64]` | **2,952,192** | Paper Spec |
| **TFEE (Flow)**| ConvLSTM Layer 2 | $3\times 3$ Conv ($512 \to 1024$) + LayerNorm | `[B, 256, 64, 64]` | `[B, 256, 64, 64]` | **4,721,664** | Paper Spec |
| **TFEE (Flow)**| ConvLSTM Layer 3 | $3\times 3$ Conv ($512 \to 1024$) + LayerNorm | `[B, 256, 64, 64]` | `[B, 256, 64, 64]` | **4,721,664** | Paper Spec |
| **TFEE (Flow)**| ConvLSTM Layer 4 | $3\times 3$ Conv ($512 \to 1024$) + LayerNorm | `[B, 256, 64, 64]` | `[B, 256, 64, 64]` | **4,721,664** | Paper Spec |
| **TFEE (Evol)**| Evolution Layer 1 | $3\times 3$ Conv ($320 \to 256$) + GN(32) + LReLU | `[B, 320, 64, 64]` | `[B, 256, 64, 64]` | **738,048** | `[Adaptation]` |
| **TFEE (Evol)**| Evolution Layer 2 | $3\times 3$ Conv ($512 \to 256$) + GN(32) + LReLU | `[B, 512, 64, 64]` | `[B, 256, 64, 64]` | **1,180,416** | `[Adaptation]` |
| **TFEE (Evol)**| Evolution Layer 3 | $3\times 3$ Conv ($512 \to 256$) + GN(32) + LReLU | `[B, 512, 64, 64]` | `[B, 256, 64, 64]` | **1,180,416** | `[Adaptation]` |
| **TFEE (Evol)**| Evolution Layer 4 | $3\times 3$ Conv ($512 \to 256$) + GN(32) + LReLU | `[B, 512, 64, 64]` | `[B, 256, 64, 64]` | **1,180,416** | `[Adaptation]` |
| **TFEE (Evol)**| Evolution Projection | $1\times 1$ Conv ($256 \to 64$, bias) | `[B, 256, 64, 64]` | `[B, 64, 64, 64]` | **16,448** | `[Adaptation]` |
| **TFEE Total**| **Subtotal TFEE** | — | — | — | **21,412,928** | Analytical |
| **EFFR** | Reconstruction L1 | $3\times 3$ Conv ($64 \to 64$) + GN(32) + LReLU | `[B, 64, 64, 64]` | `[B, 64, 64, 64]` | **37,056** | Paper Spec |
| **EFFR** | Deconv L2 ($W^T_2$) | $3\times 3$ ConvTranspose ($64 \to 64$, s=2) + GN + LReLU | `[B, 64, 64, 64]` | `[B, 64, 128, 128]` | **37,056** | `[Adaptation]` |
| **EFFR** | Channel Attention | $1\times 1$ Conv ($64 \to 16$) + LReLU + $1\times 1$ Conv ($16 \to 64$) | `[B, 64, 1, 1]` | `[B, 64, 1, 1]` | **2,128** | Paper Spec |
| **EFFR** | Spatial Attention | $7\times 7$ Conv ($2 \to 1$, padding 3) | `[B, 2, 128, 128]` | `[B, 1, 128, 128]` | **99** | Paper Spec |
| **EFFR** | Evolution Map ($W_p$) | $7\times 7$ Conv ($128 \to 64$, padding 3) + Sigmoid | `[B, 128, 128, 128]` | `[B, 64, 128, 128]` | **401,472** | Reconciled |
| **EFFR** | Scale Restoration L1 | $3\times 3$ Conv ($64 \to 64$) + GN(32) + LReLU | `[B, 64, 128, 128]` | `[B, 64, 128, 128]` | **37,056** | Paper Spec |
| **EFFR** | Scale Restoration L2 | $1\times 1$ Conv ($64 \to 4$, bias) | `[B, 64, 128, 128]` | `[B, 4, 128, 128]` | **260** | Paper Spec |
| **EFFR Total**| **Subtotal EFFR** | — | — | — | **515,127** | Analytical |
| **GRAND TOTAL**| **Full METEOR-Net** | — | `[B, 4, 4, 128, 128]` | `[B, 4, 128, 128]` | **22,086,943** | **Analytical (~22.09M)** |

- **Weight Storage Footprint**: $22,086,943 \times 4\text{ bytes} \approx \mathbf{88.35\text{ MB}}$ (in `float32`).
- **Technical Acceptance Range ($\pm 5\%$)**: **$20,982,596$ to $23,191,290$ parameters** when instantiated in PyTorch.

---

## 5. Original-Paper Features, Proposed Adaptations, and Unresolved Paper Details

### 5.1 Detailed Side-by-Side Architectural Mapping

| Network Component | Original METEOR-Net Feature (Lin et al. 2026) | Proposed GOES-19 Midpoint Adaptation | Evidence Tier | Adaptation Rationale & Engineering Justification |
|---|---|---|:---:|---|
| **Satellite & Channels** | FY-4A AGRI (Bands 9, 10, 12, 14) | GOES-19 ABI (Bands 9, 13, 14, 15) | `[Category B]` | Repurposed to GOES-19 ABI; channel selection matches physical IR window and water vapor absorption. |
| **Temporal Task** | $15\text{ min} \to 5\text{ min}$ (2 intermediate targets $t_1, t_2$) | $30\text{ min} \to 15\text{ min}$ (1 intermediate midpoint $t_{\text{mid}}$) | `[Category B]` | Adapted to match operational satellite interval doubling ($30 \to 15\text{ min}$ prepares for INSAT-3DS cadence). |
| **Input Sequence Configuration** | 4 frames: $\{I_{t-30}, I_{t-15}, I_t, I_{t+15}\}$ | 4 frames: $\{I_{t-60}, I_{t-30}, I_t, I_{t+30}\}$ | `[Category B]` | Preserves the paper's 4-input sequence length while scaling temporal spacing from 15 min to 30 min. |
| **Target Frame Selection** | Real observed 5-minute FY-4A scans | Real observed 15-minute GOES-19 scans | `[Category A]` | Ground-truth verification uses actual physical satellite scans at $t+15\text{ min}$, never synthetic interpolations. |
| **Spatial Latent Stride** | Unspecified in paper text (implied stride 1) | **Stride 2 in LCE Layer 1 ($H''=64$)** | `[Category C]` | Essential adaptation for 6 GB RTX 3050 GPU; cuts ConvLSTM activation memory by $75\%$ to prevent OOM. |
| **Spatial Deconvolution** | 2-layer reconstruction conv (stride 1) | **ConvTranspose2d stride 2 in EFFR** | `[Category C]` | Restores spatial resolution from $64 \times 64$ back to $128 \times 128$ native observation grid. |
| **Skip Feature Blend** | Symbol $X'_{t_i}$ referenced without equation | **Learned $1 \times 1$ Conv**: $W_{\text{skip}} * [X'_0 ; X'_1]$ | `[Category C]` | Resolves paper ambiguity with a differentiable, learned spatial weighting between boundary skip tensors. |
| **Evolution Pass Count** | 2-step recurrent evolution ($Y_{t_1} \to Y_{t_2}$) | **Single bottom-to-top evolution pass** | `[Category B]` | Mathematical reduction from 2-target prediction to 1-target prediction; eliminates redundant unrolling. |
| **Evolution Attention Input** | Equation: $\text{concat}(c_{\text{map}} + s_{\text{map}}, I)$ | **Broadcast addition yielding 128 channels** | `[Category A]` | Mathematical reconciliation; broadcast-sum has 64 channels, concat with $I$ has 128 channels. |
| **Output Representation** | Deterministic point reconstruction | **Strictly deterministic 4-channel output** | `[Category A]` | Deterministic baseline established first; dual-head uncertainty explicitly excluded from initial prototype. |

*Evidence Tiers*: `[Category A]` = Explicitly stated in paper text/equations; `[Category B]` = Strongly supported adaptation for task definition; `[Category C]` = Engineering design choice to resolve ambiguities or hardware constraints.

### 5.2 Explicit Record of Unresolved Paper Details
The following architectural details remain unresolved in Lin et al. (2026) and have been bridged via explicit engineering decisions:
1. **High-Resolution Evolution Cell Internal Gating**: The paper illustrates $F_{\text{high}}$ in Fig. 5 as a recurrent stack coupling $X_{\text{bottom}}$ with horizontal flow $F^n_{\text{low}}$, but omits internal recurrent cell equations. Our prototype uses channel concatenation $[E^{n-1} \, ; \, F^n_{\text{low}}]$ followed by Conv-GN-LeakyReLU to pass state upward.
2. **Latent Spatial Resolution**: The paper never clarifies whether spatial dimensions are downsampled in LCE. Maintaining $128 \times 128$ throughout causes immediate GPU OOM on 6 GB hardware; our prototype adopts stride 2 ($64 \times 64$ latent) restored by ConvTranspose2d.
3. **Skip Feature Intermediate Indexing**: The paper denotes skip features for intermediate times as $X'_{t_i}$ without explaining how boundary features $X'_0$ and $X'_1$ produce $X'_{t_i}$. Our prototype implements a learned $1 \times 1$ conv blend $W_{\text{skip}} * [X'_0 \, ; \, X'_1]$.

---

## 6. TASK D — Validity Masks, Weighted Loss Semantics, and Zero-Valid Safeguard

### 6.1 Audit of Existing Validity Masks in Project
Inspection of `src/data/goes_reader.py`, `src/data/sequence_generator.py`, and `outputs/processed_sample/sample_sequence_window00.npz` establishes:
1. **Per-Channel Radiometric Validity**: Each GOES NetCDF file defines radiometric validity as `(raw_cmi != fill_val) & (dqf == 0)`.
2. **Empirical Channel Uniformity on CONUS**: Across all 24 sample scans on the full CONUS grid, all 4 channels have identical valid counts ($3,702,837$ valid Earth pixels, $47,163$ space fill pixels, $0$ differing pixels).
3. **Multi-Channel Intersected Mask**: In `read_multichannel_timestamp`, the reader computes a common spatial intersection across channels: `combined_mask = combined_mask & res["valid_mask"]`.
4. **Separation of Radiometric and Geographic Eligibility**:
   - $M_{\text{rad}, c, i} \in \{0, 1\}$: Instrument detector quality ($DQF == 0$, non-fill).
   - $M_{\text{geo}, i} \in \{0, 1\}$: Geographic domain eligibility (inside bounding box crop, on-disk).
   - Combined Target Mask: $M_{c, i} = M_{\text{rad}, c, i} \cdot M_{\text{geo}, i}$. When a common 2D spatial mask is supplied, it broadcasts across all channels: $M_{c, i} = M_{\text{common}, i}$.
   - **Invalid values are never treated as valid observations** and are never replaced with valid temperatures.

### 6.2 Weighted Loss Normalization Semantics
We document the normalization formulation for convective-weighted MAE:
- Adhering to Lin et al. (2026) Eq. 22 ($l_r = \frac{1}{n}\sum w_i |H_i - \hat{H}_i|$), loss is normalized by **valid-pixel count $N_{\text{valid}}$**:
  $$\mathcal{L} = \frac{\sum_{c=1}^C \sum_{i=1}^{H \times W} M_{c, i} \cdot w(H_{i, \text{ref}}) \cdot |\hat{H}_{c, i} - H_{c, i}|}{\sum_{c=1}^C \sum_{i=1}^{H \times W} M_{c, i} + \epsilon}$$
- *Implication*: Convective pixels ($w=2, 5$) scale up both loss magnitude and gradients, prioritizing convective storm features in the optimization landscape over clear-sky terrain.

### 6.3 Numerically Safe Loss Formulation and Zero-Valid Safeguard
In PyTorch and IEEE 754 arithmetic, operations such as `0.0 * float('nan')` or `0.0 * float('inf')` evaluate to `NaN`! If masked target pixels contain `NaN` or invalid sentinels, direct multiplication `mask * |pred - target|` causes `NaN` gradient contamination. 

To guarantee numerical safety and prevent loss dilution, the loss module implements the following **four-step differentiable calculation**:

```python
def mask_aware_convective_mae(pred, target, mask, ref_channel_idx=1):
    # 1. Clean target: Replace NaNs/invalid values in target with a finite dummy (0.0) where mask is False
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    
    # 2. Compute absolute difference on cleaned tensors
    diff = torch.abs(pred - safe_target)
    
    # 3. Calculate convective weights based on reference channel (Band 13)
    # Direction: Lower temperature indicates higher, deeper convective cloud tops
    ref_target = safe_target[:, ref_channel_idx:ref_channel_idx+1, :, :]
    weights = torch.ones_like(pred)
    weights = torch.where((ref_target < 241.0) & (ref_target >= 221.0), 2.0 * weights, weights)
    weights = torch.where(ref_target < 221.0, 5.0 * weights, weights)
    
    # 4. Apply mask to weighted difference so invalid pixels contribute strictly 0.0
    weighted_diff = torch.where(mask, diff * weights, torch.zeros_like(diff))
    
    # 5. Global batch reduction over valid pixels
    batch_valid_count = mask.sum().float()
    
    # Zero-valid safeguard: Differentiable zero without graph detachment
    if batch_valid_count == 0.0:
        return (pred * 0.0).sum()
    else:
        return weighted_diff.sum() / batch_valid_count
```

**Properties of this Specification**:
1. **Zero Dilution on Partially Masked Batches**: Fully masked patches contribute 0 to the numerator and 0 to the denominator. They have zero effect on the loss value and do not dilute the valid patch loss.
2. **All-Masked Batch Safety**: If $N_{\text{valid}} == 0$ across the entire batch, `(pred * 0.0).sum()` evaluates to a finite scalar `0.0` with valid zero gradients ($\nabla_\theta = 0$), preserving computational graph continuity without `NaN`s.
3. **NaN Immunity**: Target `NaN` values are sanitized before subtraction, completely preventing `NaN` gradient propagation.

### 6.4 Explicit Unit Tests for Loss Implementation
The following 4 test cases must pass in `tests/test_meteor_net.py`:
- **Test 1 (All-Valid Batch)**: Target has 100% valid pixels; verify that output matches analytical convective-weighted MAE.
- **Test 2 (Partially Masked Batch)**: 50% of pixels masked; verify that masked pixels contribute zero error and that denominator equals exactly the count of valid pixels.
- **Test 3 (Completely Masked Batch)**: $M=0$ everywhere; verify that loss returns finite scalar `0.0`, `loss.backward()` succeeds, and all parameter gradients exist and are strictly equal to zero (no `NaN`).
- **Test 4 (Masked Target with NaNs)**: Target tensor explicitly contains `float('nan')` and `float('inf')` at masked locations; verify that loss is finite and gradients contain zero `NaN` values.

---

## 7. TASK E — Baseline Suite and Leakage-Safe Evaluation Protocol

### 7.1 Baseline Suite
1. **Linear Interpolation Baseline**:
   $$\hat{H}_{\text{linear}} = 0.5 \cdot L_0 + 0.5 \cdot L_{+30\text{m}}$$
   Serves as the lower-bound reference for temporal continuity.
2. **Dense Farneback Optical Flow Baseline**:
   - Computes motion field $\vec{v} = (u, v)$ between $L_0$ and $L_{+30\text{m}}$ on Band 13 using OpenCV `calcOpticalFlowFarneback` ($\text{pyr\_scale}=0.5$, $\text{levels}=3$, $\text{winsize}=15$, $\text{iterations}=3$, $\text{poly\_n}=5$, $\text{poly\_sigma}=1.2$).
   - Warps $L_0$ forward by $0.5 \cdot \vec{v}$ and $L_{+30\text{m}}$ backward by $-0.5 \cdot \vec{v}$, then averages the valid warped arrays.
   - *Failure Handling*: Optical flow divergence at image boundaries or occlusions is clamped; invalid warped pixels fall back to linear interpolation.
3. **RIFE Baseline Status**:
   - Audited repository: **No RIFE implementation or pretrained checkpoint currently exists in the workspace**.
   - Per instructions, RIFE will not be downloaded. It is documented as an external reference baseline for future comparison.
4. **Adapted METEOR-Net Baseline**:
   - Full 4-channel, flow-free, sequence-based model proposed in Section 4.

### 7.2 Quantitative Metrics Protocol
All metrics are computed exclusively over valid target pixels ($M = 1$):
- **PSNR**: Computed with physical data range $\Delta R = 340.0\text{ K} - 180.0\text{ K} = 160.0\text{ K}$:
  $$\text{PSNR} = 10 \log_{10}\left( \frac{160.0^2}{\text{MSE}} \right)$$
- **SSIM**: Calculated with window size 11, data range 160.0 K. Local windows containing any invalid/masked pixel are zero-weighted in the spatial mean.
- **Meteorological Contingency Metrics (CSI, POD, FAR)**:
  Contingency table computed per frame on Band 13 for each threshold $T_{\text{thresh}} \in \{273\text{ K}, 241\text{ K}, 221\text{ K}\}$:
  - $\text{Hits} = \sum M_i \cdot \mathbb{I}(\hat{H}_i < T_{\text{thresh}} \land H_i < T_{\text{thresh}})$
  - $\text{Misses} = \sum M_i \cdot \mathbb{I}(\hat{H}_i \ge T_{\text{thresh}} \land H_i < T_{\text{thresh}})$
  - $\text{False Alarm} = \sum M_i \cdot \mathbb{I}(\hat{H}_i < T_{\text{thresh}} \land H_i \ge T_{\text{thresh}})$
  - $\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{FA}}$, $\text{POD} = \frac{\text{Hits}}{\text{Hits} + \text{Misses}}$, $\text{FAR} = \frac{\text{FA}}{\text{Hits} + \text{FA}}$.
  - If denominator is 0 (no observed or predicted events), metric evaluates to $\text{NaN}$ and is excluded from the class mean.

### 7.3 Leakage-Safe Chronological Partitioning Protocol
- **Strict Partitioning Rule**: Data must be split chronologically by **complete observation days**, never by random patch shuffling.
- **Why Random Shuffling Fails**: Patches extracted with stride 64 share $50\%$ spatial overlap, and sliding windows spaced 5 minutes apart share $80\%$ temporal frames. Randomly splitting patches would place identical cloud structures simultaneously in train and test sets, artificially inflating SSIM by $> 0.05$.
- **Sample Reality**: The current 2-hour sample spans only one morning (`04:01` to `05:56 UTC` on Jan 1, 2026). **It cannot support any independent validation or test split.**

---

## 8. TASK F — Recursive 7.5-Minute Reconstruction & Validation Plan

### 8.1 Recursive Model Invocation Architecture
The primary model $\mathcal{M}_{30\to 15}$ is trained to infer the midpoint of a 30-minute interval from 4 surrounding observations:
$$[t-60, t-30, t, t+30] \xrightarrow{\mathcal{M}} t+15$$
To produce a 7.5-minute reconstruction ($15 \to 7.5\text{ min}$):
1. **Pass 1 (Primary Reconstruction)**: Evaluate $\mathcal{M}_{30\to 15}$ to generate intermediate frames at $t+15\text{ min}$ and $t+45\text{ min}$.
2. **Pass 2 (Recursive Step)**: Construct a fine-scale sequence where observation cadence is nominally 15 minutes:
   $$[t-30, t, t+15, t+30] \xrightarrow{\mathcal{M}_{\text{recursive}}} t+7.5\text{ min}$$
   Here, $\mathcal{M}$ is invoked recursively over the newly densified 15-minute grid.

### 8.2 The Ground Truth Reality & Validation Limitations
- **GOES-19 CONUS Ground Truth Reality**: Operational CONUS scans occur every **5 minutes** ($00, 05, 10, 15, 20, 25, 30 \dots$).
- **The Missing 7.5-Minute Truth**: There is **no physical observation at 7.5 minutes** in standard CONUS data. An agent cannot claim that 5-minute CONUS data provides direct ground truth for a 7.5-minute target.
- **Defensible Validation Strategies**:
  1. **Synthetic Resolution Testing (Primary Quantitative Defense)**:
     Train a model on a 60-to-30-minute task ($[t-120, t-60, t, t+60] \to t+30$). Apply it recursively to predict 15-minute midpoints, and quantitatively evaluate the recursive 15-minute predictions against the real observed 15-minute CONUS frames. This rigorously quantifies recursive error accumulation where ground truth is known.
  2. **Mesoscale Sector Validation (Direct Physical Ground Truth)**:
     GOES-19 operational Mesoscale sectors scan at **1-minute or 30-second cadence**. By extracting Mesoscale data, real observations exist at $7.0\text{ min}$ and $8.0\text{ min}$ (or exact $7.5\text{ min}$ on 30-second scans), enabling direct physical validation of 7.5-minute recursive predictions.

---

## 9. TASK G — Hardware, Memory, and Runtime: Analytical Calculations vs. Unmeasured Estimates

### 9.1 Machine Specifications
- **CPU**: AMD Ryzen 5 8645HS (6 cores, 12 threads, up to 5.0 GHz).
- **RAM**: 16 GB DDR5.
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (**6 GB GDDR6 VRAM**, 2048 CUDA cores).

### 9.2 Analytical Memory and Parameter Audit

| Component | Trainable Parameters | Parameter Memory | Activation Memory ($B=4$) | Activation Memory ($B=8$) | Evidentiary Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **MCE Module** | 158,888 | $0.64\text{ MB}$ | $\sim 110\text{ MB}$ | $\sim 220\text{ MB}$ | Analytical Calculation |
| **TFEE Flow (ConvLSTM)** | 17,117,184 | $68.47\text{ MB}$ | $\sim 1,250\text{ MB}$ | $\sim 2,500\text{ MB}$ | Analytical Calculation |
| **TFEE Evolution (Pyramid)** | 4,295,744 | $17.18\text{ MB}$ | $\sim 420\text{ MB}$ | $\sim 840\text{ MB}$ | Analytical Calculation |
| **EFFR Module** | 515,127 | $2.06\text{ MB}$ | $\sim 370\text{ MB}$ | $\sim 730\text{ MB}$ | Analytical Calculation |
| **Total Model Footprint** | **22,086,943** | **$88.35\text{ MB}$** | **$\sim 2.15\text{ GB}$** | **$\sim 4.29\text{ GB}$** | **Analytical Calculation** |

### 9.3 Peak GPU Memory Projections (NOT Yet Measured on Hardware)
- **Model + Gradients + Adam Optimizer States**: $88.35\text{ MB} \times 4 \approx \mathbf{0.35\text{ GB}}$.
- **Driver Overhead + CUDA Runtime**: $\sim \mathbf{0.80\text{ GB}}$ baseline.
- **Peak VRAM Projection at Batch Size $B=4$**:
  $$V_{\text{peak}} \approx 0.80 + 0.35 + 2.15 \approx \mathbf{3.30\text{ GB}} \quad \text{[Comfortably fits within 6 GB VRAM]}$$
- **Peak VRAM Projection at Batch Size $B=8$**:
  $$V_{\text{peak}} \approx 0.80 + 0.35 + 4.29 \approx \mathbf{5.44\text{ GB}} \quad \text{[Approaches 6 GB VRAM limit; risk of OOM]}$$
- **Operational Recommendation**: Default prototype training at **$B=4$**. $B=8$ should be evaluated empirically or run with mixed precision (`torch.cuda.amp.autocast`).
- **CRITICAL STATUS NOTE**: These values are **analytical estimates**. Direct hardware measurement of peak allocated and reserved VRAM across $B \in \{1, 2, 4, 8\}$ on the physical RTX 3050 is an explicit acceptance criterion for prototype verification.

---

## 10. Technical Acceptance Criteria for Prototype Implementation

The initial prototype implementation will be validated exclusively against **technical correctness criteria**. Scientific performance metrics (such as convective MAE reduction or outperforming optical flow) are explicitly excluded from this technical step and reserved for subsequent data-driven experiments.

The prototype implementation must satisfy the following **9 Technical Acceptance Criteria**:

1. **Input Tensor Geometry**: Accepts input sequences with shape strictly equal to `[B, 4, 4, 128, 128]` (Batch, Time=4, Channels=4, Height=128, Width=128).
2. **Output Tensor Geometry**: Produces midpoint reconstruction with shape strictly equal to `[B, 4, 128, 128]`.
3. **Finite Forward Pass**: Forward pass produces finite outputs (no `NaN`, no `Inf`) on synthetic dummy tensors and valid sample patches.
4. **Finite Loss Computation**: Evaluates mask-aware loss without numerical underflow, division-by-zero, or `NaN` gradient propagation.
5. **Successful Backward Pass & Gradient Flow**: Backward pass completes successfully with finite gradients ($\|\nabla_\theta\| > 0$, no `NaN`, no `Inf`) across every trainable parameter tensor in MCE, TFEE, and EFFR.
6. **Zero Dead Branches**: Verification that no trainable branch is disconnected or unused during the forward-backward graph execution.
7. **Model Parameter Audit**: Instantiated PyTorch model parameter count matches analytical expectation:
   $$\text{Parameters} = 22,086,943 \pm 5\% \implies [20,982,596 \text{ to } 23,191,290]$$
8. **Empirical GPU Memory Measurement**: Direct hardware measurement and recording of `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()` on the RTX 3050 laptop GPU across batch sizes $B \in \{1, 2, 4, 8\}$ in `float32`.
9. **Automated Unit Test Suite**: Comprehensive unit tests covering tensor shapes, gradient flow, mask-aware loss behavior, target NaN handling, and all-masked zero-gradient behavior.

---

## 11. Precise Implementation Scope for Next Step & Later Optional Overfit Debugging

### 11.1 Scope of Next Authorized Step: Prototype Architecture Implementation
When authorized, the next development step will be strictly confined to technical prototype construction:
- **Target Files**:
  - `src/models/meteor_net.py`: Contains modular `MCE`, `TFEE`, and `EFFR` classes, and the parent `METEORNet` model.
  - `src/models/losses.py`: Contains the mask-aware convective-weighted MAE loss with global batch reduction.
  - `tests/test_meteor_net.py`: Automated unit tests validating all 9 technical acceptance criteria.
  - `scripts/profile_prototype_memory.py`: Profiling script measuring actual allocated and reserved VRAM on the RTX 3050.
- **Strict Boundary**: NO model training, NO data acquisition, NO pipeline refactoring, and NO modifications to raw files.

### 11.2 Later Optional Single-Window Overfit Debugging Test
Following successful technical verification, a single-window overfit test may be executed as an **optional debugging tool**:
- **Purpose**: Verify that the optimization loop (Adam optimizer, learning rate schedule, loss backpropagation) reduces training loss on a fixed set of 64 patches from Window 00.
- **Scientific Caveat**: Overfitting a single 90-minute window **is NOT an independent validation test** and cannot establish model generalization, physical robustness, or superiority over baselines. It serves strictly as a numerical optimization sanity check before any bulk dataset expansion is requested.
- **Execution Status**: **Deferred**. Not to be run in this step.

---

*Report updated autonomously for Phase 3 Final Consistency Verification. All source data, reports, and existing pipeline code preserved in read-only status.*

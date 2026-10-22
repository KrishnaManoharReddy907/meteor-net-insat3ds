# Phase 2A Design Review: Channel Expansion, Temporal Sequence Architecture, and INSAT-3DS Readiness

**Project Title**: Confidence-Aware Temporal Super-Resolution of INSAT-3DS Satellite Imagery using a METEOR-Net-based, flow-free temporal-evolution architecture  
**Date**: October 16, 2026  
**Status**: Phase 2A Complete — Architectural Verification & Ingestion Readiness Report (Awaiting Approval to begin Preprocessing Implementation)  

---

## 1. Executive Summary

This report establishes the findings for **Phase 2A** of our major project. In Phase 1, we audited the initial GOES-19 Channel 13 dataset. In Phase 2A, we have:
1. Located and verified the newly downloaded **Channels 9, 14, and 15** files alongside existing Channel 13 files for January 1, 2026 (UTC hours 04 and 05).
2. Performed a rigorous cross-channel synchronization and physical calibration audit across all **96 files** (24 timestamps $\times$ 4 channels).
3. Deconstructed the METEOR-Net base paper's temporal formulation (*Lin et al., IEEE TGRS 2026*) to resolve the discrepancy in the Phase 1 draft sequence, establishing a mathematically sound sampling strategy to simulate INSAT-3DS 30-to-15-minute and recursive 15-to-7.5-minute reconstruction using 5-minute GOES-19 observations.
4. Formulated a comprehensive ingestion checklist for the target ISRO **`3SIMG_L1B_STD`** INSAT-3DS Imager product.
5. Evaluated spatial resolution downsampling (4 km) and patch selection criteria, clarifying why cross-satellite learning between GOES-19 and INSAT-3DS operates as **unpaired domain adaptation** rather than pixel-registered paired super-resolution.

---

## 2. Part A — GOES-19 Expanded Sample & Cross-Channel Synchronization Audit

### 2.1 File Inventory and Channel Breakdown
The project dataset under `data/goes19/2026/001/` now contains **96 NetCDF-4 (`.nc`) files**, completely populating Channels 9, 13, 14, and 15 for both UTC hour folders `04` and `05`:

- **Hour 04 (`04:01:19 UTC` to `04:56:19 UTC`)**:
  - Channel 09 (`M6C09`): 12 files ($29.64\text{ MB}$ total)
  - Channel 13 (`M6C13`): 12 files ($44.80\text{ MB}$ total)
  - Channel 14 (`M6C14`): 12 files ($44.62\text{ MB}$ total)
  - Channel 15 (`M6C15`): 12 files ($44.34\text{ MB}$ total)
- **Hour 05 (`05:01:19 UTC` to `05:56:19 UTC`)**:
  - Channel 09 (`M6C09`): 12 files ($29.68\text{ MB}$ total)
  - Channel 13 (`M6C13`): 12 files ($44.82\text{ MB}$ total)
  - Channel 14 (`M6C14`): 12 files ($44.64\text{ MB}$ total)
  - Channel 15 (`M6C15`): 12 files ($44.38\text{ MB}$ total)

**Total Dataset Size**: $282.92\text{ MB}$ across 96 files. No files were missing, duplicated, or corrupted.

### 2.2 Cross-Channel Synchronization Matrix
Every one of the 24 distinct observation timestamps was audited across all four channels. The scan start timestamps match to the exact tenth of a second across all channels for every raster sweep:

| Scan Index | Start Timestamp (UTC) | End Timestamp (UTC) | Channel 09 File | Channel 13 File | Channel 14 File | Channel 15 File | Synchronization Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00** | `2026-01-01 04:01:19.3` | `04:03:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.69 MB) | **100% Synchronized** |
| **01** | `2026-01-01 04:06:19.3` | `04:08:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **02** | `2026-01-01 04:11:19.3` | `04:13:58` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.69 MB) | **100% Synchronized** |
| **03** | `2026-01-01 04:16:19.3` | `04:18:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.69 MB) | **100% Synchronized** |
| **04** | `2026-01-01 04:21:19.3` | `04:23:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.69 MB) | **100% Synchronized** |
| **05** | `2026-01-01 04:26:19.3` | `04:28:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **06** | `2026-01-01 04:31:19.3` | `04:33:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **07** | `2026-01-01 04:36:19.3` | `04:38:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **08** | `2026-01-01 04:41:19.3` | `04:43:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **09** | `2026-01-01 04:46:19.3` | `04:48:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **10** | `2026-01-01 04:51:19.3` | `04:53:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **11** | `2026-01-01 04:56:19.3` | `04:58:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **12** | `2026-01-01 05:01:19.3` | `05:03:57` | Present (2.47 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **13** | `2026-01-01 05:06:19.3` | `05:08:57` | Present (2.47 MB) | Present (3.73 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **14** | `2026-01-01 05:11:19.3` | `05:13:57` | Present (2.47 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **15** | `2026-01-01 05:16:19.3` | `05:18:57` | Present (2.47 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **16** | `2026-01-01 05:21:19.3` | `05:23:57` | Present (2.48 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **17** | `2026-01-01 05:26:19.3` | `05:28:57` | Present (2.47 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **18** | `2026-01-01 05:31:19.3` | `05:33:57` | Present (2.47 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **19** | `2026-01-01 05:36:19.4` | `05:38:58` | Present (2.47 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **20** | `2026-01-01 05:41:19.4` | `05:43:57` | Present (2.48 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **21** | `2026-01-01 05:46:19.4` | `05:48:57` | Present (2.47 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **22** | `2026-01-01 05:51:19.4` | `05:53:57` | Present (2.47 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |
| **23** | `2026-01-01 05:56:19.4` | `05:58:57` | Present (2.47 MB) | Present (3.74 MB) | Present (3.72 MB) | Present (3.70 MB) | **100% Synchronized** |

### 2.3 Physical and Calibration Comparison Across the 4 Channels
A comparative audit of the four channels for the initial scan (`2026-01-01 04:01:19 UTC`) confirms exact structural alignment:

| Property | Channel 09 (Mid WV) | Channel 13 (Clean IR) | Channel 14 (Longwave IR) | Channel 15 (Dirty IR) | Verification Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Central Wavelength** | $6.93\ \mu\text{m}$ | $10.33\ \mu\text{m}$ | $11.19\ \mu\text{m}$ | $12.27\ \mu\text{m}$ | Band attributes verified from NetCDF headers |
| **Spatial Dimensions** | $(1500, 2500)$ | $(1500, 2500)$ | $(1500, 2500)$ | $(1500, 2500)$ | Exactly identical raster grid |
| **Raw Data Type** | `int16` | `int16` | `int16` | `int16` | Compact scaled integer storage |
| **Scale Factor** | $0.04233911$ | $0.06145332$ | $0.05985075$ | $0.05956082$ | Specific to individual channel dynamic range |
| **Add Offset** | $137.7000\text{ K}$ | $89.6200\text{ K}$ | $96.1900\text{ K}$ | $97.3800\text{ K}$ | Derived linear calibration to Kelvin |
| **Fill Value / NaNs** | `-1` (0 NaNs) | `-1` (0 NaNs) | `-1` (0 NaNs) | `-1` (0 NaNs) | Consistent fill indicator |
| **Valid Pixel Count** | $3,702,837$ ($98.74\%$) | $3,702,837$ ($98.74\%$) | $3,702,837$ ($98.74\%$) | $3,702,837$ ($98.74\%$) | **Identical valid mask across all 4 channels** |
| **Space Fill Count** | $47,163$ ($1.26\%$) | $47,163$ ($1.26\%$) | $47,163$ ($1.26\%$) | $47,163$ ($1.26\%$) | Off-earth corner pixels identical across bands |
| **DQF Distribution** | `0`: 3,702,837; `-1`: 47,163 | `0`: 3,702,837; `-1`: 47,163 | `0`: 3,702,837; `-1`: 47,163 | `0`: 3,702,837; `-1`: 47,163 | 100% of valid pixels have `DQF = 0` (good quality) |
| **Brightness Temp Range** | $[202.27, 266.62]\text{ K}$ | $[205.83, 298.19]\text{ K}$ | $[204.58, 297.95]\text{ K}$ | $[204.05, 294.82]\text{ K}$ | Calibrated physical values ($T_b$) in Kelvin |
| **Mean Temperature** | $245.24\text{ K}$ | $269.93\text{ K}$ | $269.30\text{ K}$ | $267.82\text{ K}$ | WV is cooler due to mid-troposphere emission |
| **Standard Deviation** | $10.95\text{ K}$ | $19.61\text{ K}$ | $19.99\text{ K}$ | $19.41\text{ K}$ | Surface-penetrating window bands have higher $\sigma$ |
| **Pixels $< 241\text{ K}$ (Moderate)**| $1,372,475$ ($37.07\%$) | $350,934$ ($9.48\%$) | $393,541$ ($10.63\%$) | $427,429$ ($11.54\%$) | Water vapor channel detects broader cold vapor |
| **Pixels $< 221\text{ K}$ (Deep Conv)**| $59,768$ ($1.61\%$) | $23,754$ ($0.64\%$) | $34,084$ ($0.92\%$) | $41,298$ ($1.12\%$) | Deep overshooting convective core signatures |
| **Coordinate Grid `x`** | scale $5.6\times 10^{-5}$, offset $-0.101332$ | scale $5.6\times 10^{-5}$, offset $-0.101332$ | scale $5.6\times 10^{-5}$, offset $-0.101332$ | scale $5.6\times 10^{-5}$, offset $-0.101332$ | Identical fixed grid projection radians |
| **Coordinate Grid `y`** | scale $-5.6\times 10^{-5}$, offset $0.128212$ | scale $-5.6\times 10^{-5}$, offset $0.128212$ | scale $-5.6\times 10^{-5}$, offset $0.128212$ | scale $-5.6\times 10^{-5}$, offset $0.128212$ | Identical fixed grid projection radians |

> [!IMPORTANT]
> **Zero Spatial Registration Error**: Channels 9, 13, 14, and 15 share the exact identical projection origin, grid scale factors, offsets, and pixel dimensions. They can be stacked directly into an array of shape `(4, 1500, 2500)` without spatial resampling or interpolation.

---

## 3. Part B — Temporal Sequence Design & Alignment with METEOR-Net Architecture

### 3.1 Base Paper Architecture vs. Project Adaptation

In the METEOR-Net base paper (*Lin et al., Section II-B & III-A*):
- **Satellite & Observation Mode**: FengYun-4A (FY-4A) AGRI instrument.
- **Problem Statement**: Enhance full-disk temporal resolution from **15 minutes to 5 minutes** (a **3x temporal super-resolution factor**).
- **Base Paper Sequence Structure**:
  - Full-disk revisit interval is $\Delta T_{\text{LR}} = 15\text{ minutes}$.
  - The model reconstructs **two intermediate frames** ($T = 2$) inside the revisit interval:
    $$H_{t1} = t_0 + 5\text{ min},\quad H_{t2} = t_0 + 10\text{ min}$$
  - To model higher-order non-linear dynamics, cloud growth, and dissipation, the paper introduces two historical context frames, creating a **6-frame temporal window**:
    $$\{L_{-2}, L_{-1}, L_0, H_{t1}, H_{t2}, L_1\}$$
  - In this sequence:
    - $L_{-2} = t_0 - 30\text{ min}$ (historical context 2)
    - $L_{-1} = t_0 - 15\text{ min}$ (historical context 1)
    - $L_0 = t_0$ (current observation / start anchor)
    - $L_1 = t_0 + 15\text{ min}$ (future observation / end anchor)
    - $H_{t1} = t_0 + 5\text{ min}$ (ground truth target 1)
    - $H_{t2} = t_0 + 10\text{ min}$ (ground truth target 2)

> [!WARNING]
> **Resolution of Phase 1 Discrepancy**:
> The Phase 1 report casually proposed inputs at `t-30`, `t-15`, `t`, `t+15` and targets at `t+5`, `t+10`. While this directly mimicked the paper's FY-4A $15\text{m} \to 5\text{m}$ setup, **it was inconsistent with our target satellite (INSAT-3DS)**!
> INSAT-3DS has a nominal revisit interval of **30 minutes**, not 15 minutes. Our primary objective is **30-to-15-minute reconstruction**, which is a **2x temporal factor with a single midpoint target**, not a 3x factor with two targets!

### 3.2 Corrected Temporal Design for INSAT-3DS Super-Resolution

For INSAT-3DS, the nominal interval between consecutive operational full-disk scans is $\Delta T_{\text{INSAT}} = 30\text{ minutes}$.

#### Level 1: 30-to-15-Minute Reconstruction (Midpoint Interpolation, 2x)
- **Input Observation Interval**: Nominal 30 minutes ($0\text{ min}$ and $+30\text{ min}$).
- **Intermediate Target**: Midpoint at $+15\text{ minutes}$ ($T = 1$ intermediate frame).
- **Historical Temporal Context**:
  Following the sequence-flow paradigm of METEOR-Net, relying solely on two adjacent frames ($L_0$ and $L_1$) reduces the network to linear interpolation assumptions. To allow the ConvLSTM in the TFEE module to capture non-linear convective acceleration, the input sequence is expanded with uniform 30-minute historical context:
  - $L_{-2} = t_0 - 60\text{ min}$ (Scan $k - 12$ on GOES-19)
  - $L_{-1} = t_0 - 30\text{ min}$ (Scan $k - 6$ on GOES-19)
  - $L_0 = t_0$ (Scan $k$ on GOES-19)
  - $L_1 = t_0 + 30\text{ min}$ (Scan $k + 6$ on GOES-19)
- **Target Frame**:
  - $H_{15\text{m}} = t_0 + 15\text{ min}$ (Scan $k + 3$ on GOES-19)

#### How 5-Minute GOES-19 Data Simulates this Process:
Because GOES-19 CONUS data is sampled at $\Delta t = 5\text{ minutes}$, every relevant time offset in the 30-minute regime is an exact integer multiple of 5:
- $\Delta t = 15\text{ min} \implies 3\text{ scans}$
- $\Delta t = 30\text{ min} \implies 6\text{ scans}$
- $\Delta t = 60\text{ min} \implies 12\text{ scans}$

Thus, by striding over GOES-19 scans by indices:
$$\text{Inputs: } [k-12, k-6, k, k+6] \quad\longrightarrow\quad \text{Target: } [k+3]$$
we establish a mathematically exact physical simulation of INSAT-3DS 30-minute observations with a dense ground-truth target at the 15-minute midpoint!

### 3.3 Temporal Design Matrix

| Configuration | Task / Role | Input Frames ($\Delta t$) | Historical Context | Target Frames ($\Delta t$) | Super-Resolution Factor | Ground Truth Source | Operational Application |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **Lin et al. (Base Paper)** | FY-4A Regional Replication | $L_0, L_1$ ($15\text{m}$) | $L_{-2}$ ($-30\text{m}$), $L_{-1}$ ($-15\text{m}$) | $H_{t1}$ ($+5\text{m}$), $H_{t2}$ ($+10\text{m}$) | 3x ($15\text{m} \to 5\text{m}$) | FY-4A REGC Regional Mode | Baseline comparison |
| **Config A (Primary Project Design)** | **INSAT-3DS Level 1 Super-Resolution** | $L_0$ ($0\text{m}$), $L_1$ ($+30\text{m}$) | $L_{-2}$ ($-60\text{m}$), $L_{-1}$ ($-30\text{m}$) | $H_{15\text{m}}$ ($+15\text{m}$) | **2x ($30\text{m} \to 15\text{m}$)** | **GOES-19 CONUS (Scan $k+3$)** | **Target deployment on INSAT-3DS** |
| **Config B (Short Context Variant)** | Reduced-Latency Level 1 | $L_0$ ($0\text{m}$), $L_1$ ($+30\text{m}$) | $L_{-1}$ ($-30\text{m}$) *(3 inputs total)* | $H_{15\text{m}}$ ($+15\text{m}$) | 2x ($30\text{m} \to 15\text{m}$) | GOES-19 CONUS (Scan $k+3$) | Useful if only 1 prior scan available |
| **Config C (Level 2: Recursive)** | **Recursive Level 2 Super-Resolution** | $L_0$ ($0\text{m}$), $\hat{H}_{15\text{m}}$ ($+15\text{m}$) | $\hat{H}_{-15\text{m}}$ ($-15\text{m}$), $L_{-30\text{m}}$ | $\hat{H}_{7.5\text{m}}$ ($+7.5\text{m}$) | **2x ($15\text{m} \to 7.5\text{m}$)** | *No physical 7.5m scan on GOES CONUS or INSAT* | Recursive inference on generated Level 1 output |

### 3.4 Level 2: Recursive 15-to-7.5-Minute Reconstruction Analysis

Can the model trained on 30-to-15-minute intervals be applied recursively to generate 7.5-minute frames?
1. **Kinematic Scale Mismatch**:
   Cloud motion displacement is proportional to the time gap:
   $$\mathbf{d} \approx \mathbf{v} \cdot \Delta T$$
   Over 15 minutes, cloud displacement is roughly half that over 30 minutes. If a model trained strictly on 30-minute gaps ($T_{\text{gap}} = 30\text{m}$) is applied to a 15-minute gap without temporal scale conditioning, the network's recurrent feature evolution may overestimate displacement.
2. **Ground-Truth Availability on GOES-19 CONUS**:
   GOES-19 CONUS is sampled strictly every **5 minutes** ($0, 5, 10, 15, 20, \dots$). There is **no physical observation at $7.5\text{ minutes}$** in CONUS mode!
   - Ground truth at 7- or 8-minute offsets only exists on GOES-19 **Mesoscale sectors** (1-minute scan cadence).
3. **Recommended Recursive Implementation Strategy**:
   - **Architectural Adjustment**: Modify the TFEE module to take an explicit temporal offset anchor $t_{\text{rel}} \in (0, 1)$ or train a scale-conditioned temporal embedding.
   - **Level 1 ($30\text{m} \to 15\text{m}$)** is trained and validated rigorously against real 15-minute GOES-19 observations ($k+3$).
   - **Level 2 ($15\text{m} \to 7.5\text{m}$)** is applied recursively using the generated 15-minute frame as input. Validation for Level 2 will use:
     - Internal physical consistency (conservation of radiance, smooth cloud-top motion vectors).
     - Quantitative evaluation on GOES-19 Mesoscale 1-minute data sampled at 7.5-minute intervals.

---

## 4. Part C — INSAT-3DS Data-Ingestion Specification (`3SIMG_L1B_STD`)

The candidate target product for our cross-satellite adaptation is **`3SIMG_L1B_STD`**, the INSAT-3DS Imager Level-1B Standard full-disk HDF product distributed via ISRO's MOSDAC portal.

The following checklist specifies the exact criteria that must be validated upon receiving the initial MOSDAC sample files:

```markdown
### INSAT-3DS Ingestion & Technical Verification Checklist

#### 1. File Structure & HDF5 Hierarchy
- [ ] Verify container format: HDF5 (.h5).
- [ ] Inspect Root Groups & Datasets:
  - `/IMG_VIS`  (0.55 – 0.75 µm, visible, 1 km)
  - `/IMG_SWIR` (1.55 – 1.70 µm, shortwave IR, 1 km)
  - `/IMG_MIR`  (3.80 – 4.00 µm, midwave IR, 4 km)
  - `/IMG_WV`   (6.50 – 7.10 µm, water vapor, 8 km)
  - `/IMG_TIR1` (10.30 – 11.30 µm, thermal IR-1, 4 km)
  - `/IMG_TIR2` (11.50 – 12.50 µm, thermal IR-2, 4 km)
- [ ] Verify dataset shape for thermal channels: nominal full-disk array ~ (2812, 2812).

#### 2. Temporal & Cadence Verification
- [ ] Parse global attributes: `Acquisition_Start_Time`, `Acquisition_End_Time`, `Nominal_Date_Time`.
- [ ] Confirm observation interval: Nominal 30 minutes between consecutive full-disk scans.
- [ ] Verify scan direction: South-to-North or North-to-South raster timing.

#### 3. Radiometric Calibration & Conversion Pipeline
- [ ] Verify raw data values: Level-1B stores raw digital counts (DN / integer counts), NOT brightness temperature!
- [ ] Locate calibration tables / attributes:
  - Check for dataset attributes: `Slope`, `Intercept`, `Cal_Coeff`, or `/CALIBRATION_TABLE`.
  - Formula: $\text{Radiance } (L_\lambda) = \text{DN} \times \text{Slope} + \text{Intercept}$.
- [ ] Brightness Temperature ($T_b$) Conversion:
  - Apply the inverted Planck radiation function using channel-specific effective central wavenumber $\nu_c$ ($cm^{-1}$) or constants ($c_1, c_2, \alpha, \beta$):
    $$T_b = \frac{c_2 \cdot \nu_c}{\ln\left(1 + \frac{c_1 \cdot \nu_c^3}{L_\lambda}\right)}$$
  - Confirm calibrated $T_b$ range matches atmospheric limits (180 K – 330 K).

#### 4. Invalid Pixels, Space Mask, and Quality Flags
- [ ] Check `_FillValue` (commonly -999 or 65535).
- [ ] Inspect Data Quality Flag dataset (`/Quality_Flags` or `DQF`).
- [ ] Verify space mask to cleanly separate the Earth disk from cold outer-space background.

#### 5. Geolocation & Navigation Metadata
- [ ] Coordinate projection: Geostationary fixed grid projection centered at longitude 82.0° E.
- [ ] Parameters: `semi_major_axis` (6378137.0 m), `inverse_flattening` (298.257), `satellite_height` (~35786 km).
- [ ] Check whether pixel-level `Latitude` and `Longitude` datasets are pre-computed or if coordinates must be computed analytically from scan angles.
```

### 4.2 Verified Spectral Counterpart Matrix (GOES-19 vs. INSAT-3DS)

| Spectral Role | GOES-19 ABI Channel | ABI Central $\lambda$ | INSAT-3DS Imager Band | INSAT-3DS Band Range | INSAT-3DS Central $\lambda$ | Physical Comparability & Sensor Differences |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Water Vapor** | **Channel 9** | $6.93\ \mu\text{m}$ | **WV** | $6.50 - 7.10\ \mu\text{m}$ | $\sim 6.80\ \mu\text{m}$ | **Strong Overlap**: GOES Ch 9 ($6.93\ \mu\text{m}$) falls squarely within the INSAT WV bandpass ($6.5 - 7.1\ \mu\text{m}$). Both capture mid-tropospheric humidity dynamics. |
| **Clean IR Window** | **Channel 13** | $10.33\ \mu\text{m}$ | **TIR-1** | $10.20 - 11.20\ \mu\text{m}$ | $\sim 10.80\ \mu\text{m}$ | **High Comparability**: GOES Ch 13 is at the lower edge of INSAT TIR-1 ($10.2 - 11.2\ \mu\text{m}$). Strongest cloud-top thermal signal. |
| **Traditional IR Window**| **Channel 14** | $11.19\ \mu\text{m}$ | **TIR-1** | $10.20 - 11.20\ \mu\text{m}$ | $\sim 10.80\ \mu\text{m}$ | **Close Spectral Center**: GOES Ch 14 is adjacent to INSAT TIR-1 upper bound. Provides auxiliary window radiance. |
| **Dirty IR / Split Window**| **Channel 15** | $12.27\ \mu\text{m}$ | **TIR-2** | $11.50 - 12.50\ \mu\text{m}$ | $\sim 12.00\ \mu\text{m}$ | **Direct Match**: GOES Ch 15 matches INSAT TIR-2 ($11.5 - 12.5\ \mu\text{m}$). Enables split-window $(T_{\text{TIR1}} - T_{\text{TIR2}})$ differential absorption constraints. |

---

## 5. Part D — Spatial Resolution, Geographic Grids, and Patch Selection Review

### 5.1 Evaluation of the Common 4 km Target Grid
1. **Nadir Resolution Discrepancy**:
   - GOES-19 ABI IR channels have a nadir spatial resolution of **2 km** ($0.000056\text{ rad}$).
   - INSAT-3DS Imager TIR-1 and TIR-2 channels have a nadir spatial resolution of **4 km**.
   - INSAT-3DS Imager WV channel has a nadir spatial resolution of **8 km**.
2. **Downsampling GOES-19 to 4 km**:
   - **Recommended**: Downsampling GOES-19 IR channels by a factor of 2 (from 2 km to 4 km via area-average pooling) aligns the spatial frequency content with INSAT-3DS TIR-1/2!
   - This ensures the neural network does not learn high-frequency spatial textures that cannot physically exist in INSAT-3DS observations.
3. **Handling the 8 km Water Vapor Channel**:
   - INSAT-3DS WV is at 8 km resolution. In cross-satellite fusion, the 8 km WV grid is routinely upsampled to the common 4 km grid via bilinear interpolation, matching the TIR-1/2 spatial matrix.

### 5.2 Domain Adaptation vs. Paired Super-Resolution
> [!IMPORTANT]
> **Why US GOES-19 and Indian INSAT-3DS CANNOT be treated as paired observations**:
> - GOES-19 is positioned at **$75.0^\circ\text{ W}$** longitude (GOES-East), observing North America, the Caribbean, and the western Atlantic.
> - INSAT-3DS is positioned at **$82.0^\circ\text{ E}$** longitude, observing India, South Asia, and the Indian Ocean.
> - These two satellites observe completely disjoint geographic regions separated by over $150^\circ$ of longitude. They never observe the same cloud system, atmospheric column, or weather event at the same time.
> - Therefore, our task is **unpaired domain adaptation** (transfer learning):
>   1. The model is trained on high-cadence (5-minute) GOES-19 data to learn the general physical laws of convective cloud generation, non-linear advection, and dissipation.
>   2. The model is then adapted to INSAT-3DS by aligning feature representations and sensor noise characteristics, without requiring pixel-registered paired ground truth over India.

### 5.3 Valid 128 × 128 Patch Selection Algorithm
To build high-quality training tensors from the $1,500 \times 2,500$ GOES-19 grid:
1. **Zero Space/Fill Tolerance**:
   - As confirmed in our audit, $1.26\%$ of pixels in the CONUS product are outer-space fill pixels (`_FillValue = -1`).
   - Any $128 \times 128$ crop containing even a single `-1` or fill pixel must be strictly rejected.
2. **Convective Salience Filtering**:
   - If patches are sampled uniformly, the majority will consist of clear-sky oceans or flat continental surfaces where no temporal evolution occurs.
   - Following Lin et al., every candidate patch must satisfy a **convective density threshold**:
     $$\frac{\sum [T_{\text{Ch13}} < 241\text{ K}]}{128 \times 128} \ge 5\% \quad \text{or} \quad \sum [T_{\text{Ch13}} < 221\text{ K}] \ge 10\text{ pixels}$$
3. **Temporal Sequence Continuity**:
   - Strict 0-missing-frame tolerance: all frames in the sequence $[L_{-2}, L_{-1}, L_0, H_{15\text{m}}, L_1]$ must be valid, contiguous, and uncorrupted.

---

## 6. Summary of Unresolved Decisions & Next Steps

Before launching the Phase 2B preprocessing script, the following architectural choices require confirmation:

1. **Temporal Context Length for Initial Training**:
   - **Option 1 (Recommended — 4 Inputs)**: Use full 4-frame uniform context $[t-60\text{m}, t-30\text{m}, t, t+30\text{m}] \to t+15\text{m}$. Faithfully replicates METEOR-Net's sequence-flow capability.
   - **Option 2 (3 Inputs)**: Use 3-frame context $[t-30\text{m}, t, t+30\text{m}] \to t+15\text{m}$. Lower latency, fewer parameters.
2. **Spatial Resampling Timing**:
   - Shall we downsample GOES-19 to 4 km immediately during patch extraction, or retain native 2 km for GOES-19 baseline experiments and downsample only during the INSAT-3DS domain adaptation phase?
3. **Bulk Fetch Window**:
   - The current 2-hour sample (24 timestamps) validates pipeline correctness. For training, we will need to schedule a batch fetch of 7–10 days of convective season data from AWS S3 (`noaa-goes19`).

---
*Report prepared and validated on October 16, 2026.*

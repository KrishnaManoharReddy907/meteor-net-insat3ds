# Dataset Inspection & Technical Audit Report: Phase 1
**Project Title**: Confidence-Aware Temporal Super-Resolution of INSAT-3DS Satellite Imagery using a METEOR-Net-based, flow-free temporal-evolution architecture  
**Date of Audit**: October 13, 2026  
**Status**: Phase 1 Complete — Awaiting Approval for Phase 2 Preprocessing  

---

## 1. Executive Summary

This report establishes the baseline technical audit of our initial training and evaluation dataset for Phase 1 of the major project. 

The primary goals of this phase were:
1. Thoroughly inspect the existing repository structure, raw NetCDF files, and research documents without altering any user data.
2. Confirm the exact observation cadence, timestamps, calibration, spatial dimensions, and physical viability of the downloaded GOES-19 data.
3. Validate whether Channel 13 (`M6C13`) contains calibrated brightness temperatures suitable for convective process reconstruction (as evaluated in our base paper, Lin et al., IEEE TGRS 2026).
4. Evaluate the feasibility of integrating auxiliary infrared and water-vapor channels (Channels 9, 14, and 15).
5. Cross-reference GOES-19 Advanced Baseline Imager (ABI) channels against the target ISRO INSAT-3DS Imager payload using verified official sensor specifications.
6. Formulate a minimal environment specification and an actionable preprocessing roadmap.

---

## 2. Directory Structure and Project Artifacts

### 2.1 Workspace Structure
The project workspace is located at:  
`c:\Users\Krishna\OneDrive\Documents\CS CORE SUBJECTS NOTES\PROJECTS\MAJOR PROJECT`

Prior to this phase, the workspace contained exclusively the downloaded raw satellite data. As part of Phase 1, only the required `docs/` and `scripts/` deliverables were created:

```text
MAJOR PROJECT/
├── data/
│   └── goes19/
│       └── 2026/
│           └── 001/
│               ├── 04/       # 12 GOES-19 NetCDF files (04:01:19 UTC to 04:56:19 UTC)
│               └── 05/       # 12 GOES-19 NetCDF files (05:01:19 UTC to 05:56:19 UTC)
├── docs/
│   └── dataset_inspection.md # [This Document] Technical audit and inspection report
└── scripts/
    └── inspect_goes.py       # Reusable, standalone NetCDF-4 dataset inspection tool
```

### 2.2 Preserved Project Documentation & Research Papers
In an adjacent project directory (`..\MAJOR PROJECT DOCS`), key reference documents and papers were identified and left intact:
- **Base Architecture Paper**:  
  *Self-Supervised Temporal Super-Resolution Framework With METEOR-Net for Enhancing Convective Process Reconstruction in Geostationary Satellite Cloud Data* (Lin, Li, Chen, & Kou, *IEEE Transactions on Geoscience and Remote Sensing*, Vol. 64, published March/April 2026).
- **Other Reference Papers**: 7 additional peer-reviewed papers covering GAN-based frame interpolation, optical-flow baseline interpolation, optical-SAR diffusion models, and cloud cover deep learning.
- **Academic Documents**: Project presentation decks (`CBITIT2026-2027IT1118.ppt`, `our ppt.ppt`), approved project abstract (`major project abstract updated.docx`), and historical research transcripts.

> [!NOTE]
> All existing raw satellite data files (`.nc`) and project documentation files remain completely untouched and unmodified.

---

## 3. Inventory and Cadence Verification

### 3.1 File Counts and Channel Identification
- **Total Downloaded Files**: 24 files
- **Channels Present**: Channel 13 (`M6C13`) exclusively.
- **Channels Absent Locally**: Channels 9 (`M6C09`), 14 (`M6C14`), and 15 (`M6C15`) are currently **not** present in the local folders.
- **Distribution across UTC Hour Folders**:
  - `data/goes19/2026/001/04/`: 12 files (all Channel 13)
  - `data/goes19/2026/001/05/`: 12 files (all Channel 13)

### 3.2 Observation Cadence & Sequence Integrity Analysis
All 24 files were sorted chronologically by their scan start timestamps parsed from both the NetCDF filenames and the internal metadata:

| Scan Index | Filename (`OR_ABI-L2-CMIPC-M6C13_G19_...`) | Scan Start (UTC) | Scan End (UTC) | Cadence Interval | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **00** | `..._s20260010401193_e20260010403578_c20260010404062.nc` | 2026-01-01 04:01:19 | 04:03:57 | — | Initial Scan |
| **01** | `..._s20260010406193_e20260010408578_c20260010409066.nc` | 2026-01-01 04:06:19 | 04:08:57 | 5.0 min | Continuous |
| **02** | `..._s20260010411193_e20260010413580_c20260010414044.nc` | 2026-01-01 04:11:19 | 04:13:58 | 5.0 min | Continuous |
| **03** | `..._s20260010416193_e20260010418578_c20260010419070.nc` | 2026-01-01 04:16:19 | 04:18:57 | 5.0 min | Continuous |
| **04** | `..._s20260010421193_e20260010423579_c20260010424041.nc` | 2026-01-01 04:21:19 | 04:23:57 | 5.0 min | Continuous |
| **05** | `..._s20260010426193_e20260010428578_c20260010429054.nc` | 2026-01-01 04:26:19 | 04:28:57 | 5.0 min | Continuous |
| **06** | `..._s20260010431193_e20260010433578_c20260010434076.nc` | 2026-01-01 04:31:19 | 04:33:57 | 5.0 min | Continuous |
| **07** | `..._s20260010436193_e20260010438578_c20260010439049.nc` | 2026-01-01 04:36:19 | 04:38:57 | 5.0 min | Continuous |
| **08** | `..._s20260010441193_e20260010443578_c20260010444062.nc` | 2026-01-01 04:41:19 | 04:43:57 | 5.0 min | Continuous |
| **09** | `..._s20260010446193_e20260010448578_c20260010449072.nc` | 2026-01-01 04:46:19 | 04:48:57 | 5.0 min | Continuous |
| **10** | `..._s20260010451193_e20260010453578_c20260010454070.nc` | 2026-01-01 04:51:19 | 04:53:57 | 5.0 min | Continuous |
| **11** | `..._s20260010456193_e20260010458578_c20260010459053.nc` | 2026-01-01 04:56:19 | 04:58:57 | 5.0 min | Continuous |
| **12** | `..._s20260010501193_e20260010503578_c20260010504067.nc` | 2026-01-01 05:01:19 | 05:03:57 | 5.0 min | Continuous |
| **13** | `..._s20260010506193_e20260010508578_c20260010509043.nc` | 2026-01-01 05:06:19 | 05:08:57 | 5.0 min | Continuous |
| **14** | `..._s20260010511193_e20260010513578_c20260010514053.nc` | 2026-01-01 05:11:19 | 05:13:57 | 5.0 min | Continuous |
| **15** | `..._s20260010516193_e20260010518578_c20260010519052.nc` | 2026-01-01 05:16:19 | 05:18:57 | 5.0 min | Continuous |
| **16** | `..._s20260010521193_e20260010523578_c20260010524062.nc` | 2026-01-01 05:21:19 | 05:23:57 | 5.0 min | Continuous |
| **17** | `..._s20260010526193_e20260010528578_c20260010529073.nc` | 2026-01-01 05:26:19 | 05:28:57 | 5.0 min | Continuous |
| **18** | `..._s20260010531193_e20260010533578_c20260010534051.nc` | 2026-01-01 05:31:19 | 05:33:57 | 5.0 min | Continuous |
| **19** | `..._s20260010536194_e20260010538580_c20260010539041.nc` | 2026-01-01 05:36:19 | 05:38:58 | 5.0 min | Continuous |
| **20** | `..._s20260010541194_e20260010543578_c20260010544064.nc` | 2026-01-01 05:41:19 | 05:43:57 | 5.0 min | Continuous |
| **21** | `..._s20260010546194_e20260010548578_c20260010549036.nc` | 2026-01-01 05:46:19 | 05:48:57 | 5.0 min | Continuous |
| **22** | `..._s20260010551194_e20260010553578_c20260010554064.nc` | 2026-01-01 05:51:19 | 05:53:57 | 5.0 min | Continuous |
| **23** | `..._s20260010556194_e20260010558578_c20260010559063.nc` | 2026-01-01 05:56:19 | 05:58:57 | 5.0 min | Continuous |

### 3.3 Cadence Findings
1. **Zero Missing Frames**: Over the 2-hour window (120 minutes), there are exactly 24 scans, with **0 dropped frames**.
2. **Perfect 5-Minute Cadence**: The temporal interval between any two consecutive scans is $\Delta t = 5.00\text{ minutes}$ (300 seconds), with scan raster durations of $\approx 2\text{ min } 38\text{ s}$.
3. **Flawless Hour Transition**: The transition from the end of hour 04 (`04:56:19 UTC`) to the start of hour 05 (`05:01:19 UTC`) is exactly 5.0 minutes, confirming seamless continuity across the directory boundary.
4. **Alignment with METEOR-Net Training Formulation**:
   - In the base paper, METEOR-Net takes low-temporal-resolution inputs spaced 15 minutes apart and reconstructs 2 intermediate real frames spaced 5 minutes apart ($t+5\text{ min}$, $t+10\text{ min}$).
   - This exact 5-minute CONUS cadence allows identical multi-step sequence construction: taking frames at $t-30$, $t-15$, $t$, and $t+15$ minutes as inputs, and using the real intermediate scans at $t+5$ and $t+10$ minutes as self-supervised ground truth!

---

## 4. Technical File Inspection: Channel 13 (`M6C13`)

Representative file inspected:  
`OR_ABI-L2-CMIPC-M6C13_G19_s20260010401193_e20260010403578_c20260010404062.nc`

### 4.1 Product Metadata and Identification
- **Satellite ID**: GOES-19 (`G19`), operational GOES-East geostationary satellite (orbital slot $-75.0^\circ\text{ W}$).
- **Product Name**: `ABI-L2-CMIPC` (Advanced Baseline Imager Level 2+ Cloud and Moisture Imagery, CONUS).
- **Scanning Mode**: Mode 6 (`M6`, 10-minute full disk / 5-minute CONUS / 1-minute mesoscale).
- **Channel Identifier**: Channel 13 (`C13` - Clean IR Longwave Window).
- **Central Wavelength**: $10.33\ \mu\text{m}$ (Nominal band range: $10.1 - 10.6\ \mu\text{m}$).
- **Nominal Spatial Resolution at Nadir**: $2\text{ km}$ ($0.000056\text{ radians}$).
- **File Size**: $3.73\text{ MB}$ ($3,911,963\text{ bytes}$). Average file size across all 24 scans: $3.735\text{ MB}$.

### 4.2 Array Dimensions, Coordinates, and Types
- **Array Shape**: $1,500 \times 2,500$ pixels (rows $\times$ columns) = $3,750,000$ total pixels.
- **Coordinate Dimensions**:
  - `y`: $1,500$ points, stored as `int16`, representing the N/S elevation scan angle in the GOES fixed grid projection.
    - Decoded range: $+0.128212\text{ rad}$ (North) down to $+0.044268\text{ rad}$ (South).
    - Angular step: $-0.000056\text{ rad}$ per pixel.
  - `x`: $2,500$ points, stored as `int16`, representing the E/W azimuth scan angle in the GOES fixed grid projection.
    - Decoded range: $-0.101332\text{ rad}$ (West) up to $+0.038612\text{ rad}$ (East).
    - Angular step: $+0.000056\text{ rad}$ per pixel.
  - `t`: scalar `float64`, seconds since J2000 epoch ($820468958.55\text{ s}$).
  - `band`: shape $(1,)$, central wavelength $10.33\ \mu\text{m}$.

### 4.3 Calibration Attributes and Physical Suitability
- **Primary Data Variable**: `CMI`
  - `standard_name`: `toa_brightness_temperature`
  - `long_name`: `ABI L2+ Cloud and Moisture Imagery brightness temperature`
  - `units`: `K` (Kelvin)
  - `raw_dtype`: `int16`
  - `scale_factor`: `0.06145332008600235`
  - `add_offset`: `89.62000274658203`
  - `_FillValue`: `-1`
  - `valid_range`: `[0, 4095]`
- **Calibration Verification**:
  $$\text{Brightness Temperature }(T_b\text{ in Kelvin}) = \text{raw\_int16} \times \text{scale\_factor} + \text{add\_offset}$$
  > [!IMPORTANT]
  > Because these files are **Level 2+ CMIP** products, the conversion from raw instrument detector counts to spectral radiance and then to top-of-atmosphere brightness temperature via the Planck inversion function has **already been performed and calibrated by NOAA’s Ground System**. There is **no need** to apply manual Planck constants (`planck_fk1`, `planck_fk2`, etc.). Simply applying `scale_factor` and `add_offset` yields the true physical brightness temperature in Kelvin.

### 4.4 Pixel Statistics, Fill Masking, and Convective Analysis
Evaluating the decoded $1,500 \times 2,500$ grid for the representative file:
- **Total Pixels**: $3,750,000$
- **Valid Pixels**: $3,702,837$ ($98.74\%$)
- **Space/Fill Pixels**: $47,163$ ($1.26\%$) — these correspond to corners of the CONUS bounding box outside the Earth disc.
- **NaNs in Raw Storage**: None (fill is strictly represented by `_FillValue = -1`).
- **Data Quality Flag (`DQF`)**:
  - `DQF = 0` (`good_pixel_qf`): Exactly $3,702,837$ pixels ($100\%$ of valid pixels).
  - `DQF = -1` (no value / fill): $47,163$ pixels.
  - Flag values `1` (conditionally usable), `2` (out of range), `3` (no value), and `4` (focal plane temp exceeded): $0$ pixels detected.
- **Physical Thermal Distribution**:
  - Minimum Brightness Temperature: $205.83\text{ K}$ (Cloud tops / high-altitude cirrus)
  - Maximum Brightness Temperature: $298.19\text{ K}$ (Warm sea/land surfaces)
  - Mean Brightness Temperature: $269.93\text{ K}$
  - Standard Deviation: $19.61\text{ K}$
- **Convective Activity Breakdown (using METEOR-Net Paper Thresholds)**:
  - **Moderate Convective Clouds ($T_b < 241\text{ K}$)**: $350,934$ pixels ($9.48\%$ of valid earth pixels).
  - **Vigorous Deep Convective Cores ($T_b < 221\text{ K}$)**: $23,754$ pixels ($0.64\%$ of valid earth pixels).

> [!TIP]
> This confirms that the January 1, 2026 dataset captures significant convective cloud systems with strong thermal gradients down to $205\text{ K}$, making it an ideal candidate for testing the convective-weighted loss function ($\alpha=1, \beta=2, \gamma=5$ at $241\text{ K}$ and $221\text{ K}$) defined in Lin et al.

### 4.5 Navigation & Geolocation Feasibility
The file metadata specifies the exact geostationary projection parameters:
- `grid_mapping_name`: `geostationary`
- `perspective_point_height`: $35,786,023.0\text{ m}$
- `semi_major_axis`: $6,378,137.0\text{ m}$ (GRS80/WGS84 ellipsoid)
- `semi_minor_axis`: $6,356,752.31414\text{ m}$
- `inverse_flattening`: $298.2572221$
- `longitude_of_projection_origin`: $-75.0^\circ\text{ W}$
- `latitude_of_projection_origin`: $0.0^\circ$

**Lat/Lon Derivation Verified**:  
We implemented and tested the analytical coordinate transformation equations from the NOAA GOES-R Product Definition and User's Guide (PUG Volume 5). The center pixel of the CONUS array (row 750, column 1250) resolves precisely to:
$$\text{Latitude} = 30.08^\circ\text{ N},\quad \text{Longitude} = 87.10^\circ\text{ W}$$
which corresponds to the Gulf Coast / Florida Panhandle region, perfectly centering the CONUS sector. Latitude and longitude can therefore be computed for every pixel without external third-party geospatial services.

---

## 5. Feasibility of Multi-Channel Expansion (Channels 9, 14, 15)

In METEOR-Net, the **Multichannel Encoding (MCE)** module is explicitly designed to exploit the physical asynchrony and complementarity between infrared channels (e.g., upper/mid-tropospheric water-vapor changes preceding cloud-top cooling). 

We audited the feasibility of adding Channels 9, 14, and 15:

### 5.1 Remote Archive Status (NOAA GOES-19 AWS S3 Bucket)
- **Bucket**: `s3://noaa-goes19/` (Publicly accessible with `--no-sign-request`).
- **Product Path**: `ABI-L2-CMIPC/2026/001/<HH>/`
- **Channel Synchronization**: The GOES-19 ABI sensor scans all 16 spectral channels simultaneously during each 5-minute CONUS sweep. Consequently, whenever Channel 13 is produced, Channels 9 (`M6C09`), 14 (`M6C14`), and 15 (`M6C15`) are guaranteed to exist at the exact identical start and end timestamps.
- **Spatial Alignment**: Channels 9, 13, 14, and 15 all have a nominal nadir spatial resolution of $2\text{ km}$ and share identical $1,500 \times 2,500$ coordinate grids. They can be stacked into an array of shape `(4, 1500, 2500)` without spatial interpolation.

### 5.2 Storage & Computational Impact
- Each compressed NetCDF file is $\approx 3.7\text{ MB}$.
- For 4 channels: $4 \times 3.7\text{ MB} \approx 14.8\text{ MB}$ per 5-minute timestamp.
- For 1 hour (12 timestamps): $\approx 178\text{ MB}$.
- For 1 full day (288 timestamps): $\approx 4.25\text{ GB}$.
- For our intended initial 10-day dataset: $\approx 42.5\text{ GB}$.

> [!NOTE]
> If local disk space is constrained, our preprocessing pipeline can follow a **streaming crop pattern**: download raw NetCDF files for the 4 channels, extract the target convective $128 \times 128$ patches in float32/uint16, save them into structured `.npy` / `.zarr` arrays, and immediately purge the raw files.

---

## 6. Proposed Channel-Selection & Satellite Cross-Mapping Table

To enable cross-satellite domain adaptation from GOES-19 to INSAT-3DS, the neural network's input channels must represent physically equivalent atmospheric absorption and window phenomena across both sensors.

The official sensor specifications for the **INSAT-3DS Imager** payload were retrieved from ISRO MOSDAC and the WMO OSCAR sensor database:

| GOES-19 ABI Channel | ABI Central $\lambda$ ($\mu\text{m}$) | ABI Nadir Res. | Meteorological Phenomenon / Primary Role | INSAT-3DS Imager Counterpart | INSAT-3DS Band Range ($\mu\text{m}$) | INSAT-3DS Central $\lambda$ | INSAT-3DS Nadir Res. | Verification Status | Selection Recommendation |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Band 13** | **10.33** | 2 km | **Clean Longwave IR Window**: Cloud-top temp & convection | **TIR-1** | $10.2 - 11.2$ | $\sim 10.8\ \mu\text{m}$ | 4 km | **Verified** (MOSDAC) | **Primary Anchor (Required)**: Train single-channel baseline first, then anchor channel in 4-band model. |
| **Band 9** | **6.93** | 2 km | **Mid-Level Water Vapor**: Mid-troposphere dynamics & moisture | **WV** | $6.5 - 7.1$ | $\sim 6.8\ \mu\text{m}$ | 8 km | **Verified** (MOSDAC) | **Candidate 2 (High Priority)**: Essential for METEOR-Net MCE module to capture moisture pre-convection. |
| **Band 14** | **11.20** | 2 km | **Traditional Longwave IR**: Standard infrared window | **TIR-1** *(Overlap)* | $10.2 - 11.2$ | $\sim 10.8\ \mu\text{m}$ | 4 km | **Verified** (MOSDAC) | **Candidate 3 (Secondary)**: Band 14 is closer to INSAT TIR-1 center ($10.8\ \mu\text{m}$) than Band 13 ($10.33\ \mu\text{m}$). |
| **Band 15** | **12.30** | 2 km | **Dirty Longwave IR**: Split-window low-level water vapor | **TIR-2** | $11.5 - 12.5$ | $\sim 12.0\ \mu\text{m}$ | 4 km | **Verified** (MOSDAC) | **Candidate 4 (High Priority)**: Direct match to INSAT-3DS TIR-2; enables split-window $(T_{13} - T_{15})$ physical constraints. |
| **Band 7** | **3.90** | 2 km | **Shortwave / Midwave IR**: Nighttime fog & fire detection | **MIR** | $3.8 - 4.0$ | $\sim 3.82\ \mu\text{m}$ | 4 km | **Verified** (MOSDAC) | *Optional Future*: High solar reflectance during daytime complicates 24h continuous models. |

### Verification Distinctions
- **Verified Facts**: 
  1. INSAT-3DS Imager has exactly 6 channels (VIS, SWIR, MIR, WV, TIR-1, TIR-2).
  2. GOES-19 ABI Band 13 ($10.33\ \mu\text{m}$) and Band 14 ($11.20\ \mu\text{m}$) span the exact spectral interval of INSAT-3DS TIR-1 ($10.2 - 11.2\ \mu\text{m}$).
  3. GOES-19 ABI Band 9 ($6.93\ \mu\text{m}$) lies directly within the INSAT-3DS WV filter ($6.5 - 7.1\ \mu\text{m}$).
  4. GOES-19 ABI Band 15 ($12.30\ \mu\text{m}$) matches INSAT-3DS TIR-2 ($11.5 - 12.5\ \mu\text{m}$).
- **Assumptions Requiring Future Empirical Confirmation**:
  1. INSAT-3DS WV resolution is $8\text{ km}$ at nadir, whereas GOES-19 ABI WV is $2\text{ km}$. Preprocessing must resample both to a common grid ($4\text{ km}$) for cross-satellite domain adaptation.
  2. While GOES-19 ABI L2+ CMIP data provides direct brightness temperatures in Kelvin, INSAT-3DS L1B data from MOSDAC contains raw digital counts requiring conversion via lookup calibration tables (or using L1C geometrically corrected products if available).

---

## 7. Python Environment & Dependency Audit

The existing Python environment was inspected:
- **Python Version**: `3.13.3` (Windows AMD64)
- **Currently Available Installed Packages**:
  - `h5py` ($3.14.0$) — *Successfully utilized to inspect NetCDF-4 files*
  - `numpy` ($2.3.5$)
  - `scipy` ($1.17.1$)
  - `pandas` ($3.0.2$)
  - `torch` ($2.11.0$)
  - `pillow` ($12.3.0$)
  - `requests` ($2.33.1$)
- **Missing Packages Required for Standard Satellite Workflow**:
  - `netCDF4` and `xarray` (standard Python satellite data abstraction)
  - `pyproj` (for standard Cartopy/PROJ coordinate reprojection)
  - `boto3` or `s3fs` (optional, for streaming directly from AWS S3)

### Exact Minimal Installation Command
To install only the lightweight inspection and geospatial coordinate tools without disturbing existing ML packages:

```powershell
pip install netCDF4 xarray pyproj
```

*(Note: `h5py` already allows reading and processing all files without installing any additional package, so this installation is optional and recommended only for xarray-based workflows).*

---

## 8. Summary of Confirmed Issues & Next Steps for Preprocessing

### 8.1 Confirmed Issues
1. **Single-Band Limitation in Current Download**: Only Channel 13 is currently downloaded locally. While sufficient for single-channel baseline model verification, training the multichannel MCE module requires fetching Channels 9, 14, and 15.
2. **Space Fill Pixels**: $1.26\%$ of pixels in the CONUS grid are off-earth fill pixels (`_FillValue = -1`). Any patch extraction pipeline must enforce strict bounding boxes over continental land/ocean regions to prevent `-1` values from leaking into convolution kernels.
3. **Spatial Resolution Mismatch for Cross-Satellite Adaptation**: GOES ABI IR channels have $2\text{ km}$ nadir resolution, while INSAT-3DS Imager has $4\text{ km}$ (TIR) and $8\text{ km}$ (WV). Preprocessing must establish a standardized resampling resolution (e.g., $4\text{ km}$).

### 8.2 Recommended Next Steps (Phase 2 Preprocessing Pipeline)
1. **Single-Band Baseline Pipeline First**: Build the sequence dataset generator on the 24 existing Channel 13 files to validate the 4-frame input $\to$ 2-frame ground truth sequence generator ($t-30, t-15, t, t+15\text{ min} \to t+5, t+10\text{ min}$).
2. **Selective Bounding-Box Cropping**: Define a fixed geographic bounding box with rich convective activity (e.g., $25^\circ\text{N} - 35^\circ\text{N}$, $80^\circ\text{W} - 95^\circ\text{W}$) and cut standard $128 \times 128$ patches.
3. **Normalization Strategy**: Normalize brightness temperatures linearly to $[0, 1]$ using a fixed meteorological scale (e.g., $[180\text{ K}, 320\text{ K}]$) rather than min-max per image, preserving physical convective temperature gradients.
4. **AWS S3 Automated Fetch Script**: Create a script using `boto3` or `curl`/`aws s3` to selectively download synchronized Channels 9, 14, and 15 for expanded training windows.

---
*Report prepared and validated via automated inspection scripts on October 13, 2026.*

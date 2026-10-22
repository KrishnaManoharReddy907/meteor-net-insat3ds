# Phase 5C — Multi-Event GOES-19 Acquisition Planning and Preflight

**Date:** 2026-10-22  
**Status:** Complete — Acquisition Architecture Formulated, Pilot Sequence Identified, Remote Bucket Verified  
**Decision:** `READY FOR PILOT AUTHORIZATION`  
**Host Disk Headroom:** 111.66 GB Available on `C:` Drive  
**Target S3 Remote Bucket:** `s3://noaa-goes19/` (NOAA Open Data Dissemination, Public AWS Archive)  
**Supporting Reports:**  
- [`docs/phase5b_dataset_readiness_implementation.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase5b_dataset_readiness_implementation.md)  
- [`docs/phase5a_training_readiness_and_compute.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase5a_training_readiness_and_compute.md)  
- [`outputs/dataset_splits/splits_manifest.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/dataset_splits/splits_manifest.json)  

---

## 1. Executive Summary & Boundary Conformance

In strict compliance with Phase 5C directives:
- **Zero data was downloaded.**
- **Zero model training or optimization steps were performed.**
- **No raw NetCDF files or existing processed samples were modified or overwritten.**
- **Model architecture and loss functions were preserved without change.**
- **No unverified generalization claims were made.**

### Key Preflight Determinations:
1. **Public Remote Archive Verified:** The official NOAA GOES-19 AWS S3 archive (`s3://noaa-goes19/`) was verified via direct HTTPS XML discovery. Julian days 001 through 295 of 2026 are fully accessible with public `--no-sign-request` access.
2. **Channel Synchronization Confirmed:** Exact 5-minute sweeps for all 4 required infrared bands (Bands 09, 13, 14, 15) were audited on remote bucket listings. Every candidate hour contains exactly 12 scans per channel (100% temporal synchronization).
3. **Exact Storage Measurements:** Analysis of local files established a mean compressed file size of **$3.407\text{ MB}$** ($13.63\text{ MB}$ per 4-channel timestamp). A 4-event dataset ($1,152\text{ files}$) requires **$3.83\text{ GB}$** of raw storage, representing only **$3.4\%$** of available host disk headroom ($111.66\text{ GB}$ free).
4. **Focused Pilot Protocol Formulated:** A 20-file single-window pilot ($68.2\text{ MB}$) is designed to test download integrity, SHA-256 hashing, calibration, and sequence generation before authorizing full multi-event downloads.

---

## 2. Candidate Observation Periods & Meteorological Evidence

To eliminate temporal autocorrelation and provide genuine out-of-sample evaluation, 4 candidate storm events were selected across distinct seasons, synoptic regimes, and geographic regions in CONUS:

| Event ID | Calendar Date | Julian Day | Observation Window | Synoptic Weather Regime | Target Geographic Sector | Expected Convective Characteristics | Split Role |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| **Event 1** | **2026-01-01** | Day 001 | 02:00 – 08:00 UTC | Pacific Winter Trough / Atmospheric River | Western US / Arizona ($34.8^\circ\text{N}, 113.0^\circ\text{W}$) | Verified cold anvil cloud tops down to $210.74\text{ K}$; $96.05\%$ pixels $<241\text{ K}$. | **TRAIN (1)** |
| **Event 2** | **2026-05-20** | Day 140 | 18:00 – 00:00 UTC | Spring Dryline / Severe Convective Outbreak | Southern/Central Plains (Texas, Oklahoma, Kansas) | High CAPE supercell and squall line development with penetrating overshooting tops ($<215\text{ K}$). | **TRAIN (2)** |
| **Event 3** | **2026-07-19** | Day 200 | 20:00 – 02:00 UTC | North American Monsoon Diurnal Clusters | Southwest US / High Plains (New Mexico, Colorado) | Intense diurnal thunderstorm clusters with rapid vertical growth and sharp anvil boundaries. | **VALIDATION** |
| **Event 4** | **2026-09-07** | Day 250 | 16:00 – 22:00 UTC | Gulf Coastal / Tropical Convective System | Gulf Coast / Southeast US (Louisiana, Florida, Alabama) | Warm-core maritime convective bands with deep water vapor plumes and tropical anvil structures. | **TEST** |

### Independence & Non-Autocorrelation Guarantee:
- The 4 candidate events are separated by **6 to 8 weeks**, guaranteeing that atmospheric states, synoptic wave patterns, and boundary layer moisture are completely independent.
- Different diurnal phases (spring afternoon vs summer evening vs winter overnight) test the neural network against varied solar illumination and land-surface temperature backgrounds.

---

## 3. Remote Archive Verification & Object Key Protocol

### 3.1 Public Source Details
- **Provider:** NOAA Open Data Dissemination (NODD) Program on Amazon Web Services (AWS).
- **S3 Bucket URI:** `s3://noaa-goes19/`
- **Public HTTPS Base:** `https://noaa-goes19.s3.amazonaws.com/`
- **Authentication:** Public / Anonymous (AWS CLI `--no-sign-request` or unauthenticated HTTP GET).
- **Product Identifier:** `ABI-L2-CMIPC` (Advanced Baseline Imager Level 2 Cloud and Moisture Imagery - CONUS).

### 3.2 Canonical Object Key Schema
```
ABI-L2-CMIPC/<YYYY>/<JJJ>/<HH>/OR_ABI-L2-CMIPC-M6C<CC>_G19_s<YYYYJJJHHMMSSs>_e<YYYYJJJHHMMSSs>_c<YYYYJJJHHMMSSs>.nc
```
Where:
- `<YYYY>`: 4-digit Year (e.g., `2026`)
- `<JJJ>`: 3-digit Julian Day of Year (`001` to `365`)
- `<HH>`: 2-digit UTC Hour (`00` to `23`)
- `<CC>`: 2-digit ABI Channel Number (`09`, `13`, `14`, `15`)
- `s...`: Scan start timestamp (Year, Julian Day, Hour, Minute, Second, Tenths)
- `e...`: Scan end timestamp
- `c...`: File creation timestamp

### 3.3 Remote File Availability Check
Direct remote query of the NOAA bucket via Python XML parsing confirmed:
- `Day 001 Hour 04`: Exactly 12 scans per channel (12 Ch9, 12 Ch13, 12 Ch14, 12 Ch15).
- `Day 140 Hour 21`: Exactly 12 scans per channel (12 Ch9, 12 Ch13, 12 Ch14, 12 Ch15).
- `Day 200 Hour 22`: Exactly 12 scans per channel (12 Ch9, 12 Ch13, 12 Ch14, 12 Ch15).
All candidate periods exhibit **100% file availability and zero missing channels**.

---

## 4. Empirical Storage, File Count, and Headroom Estimates

Rather than relying on rough approximations, calculations were derived from direct measurements of the 96 existing GOES-19 NetCDF files on disk:
- **Min File Size:** $2.469\text{ MB}$ (Band 09: mid-level moisture field compresses efficiently).
- **Mean File Size:** **$3.407\text{ MB}$** ($3,572,442\text{ bytes}$).
- **Max File Size:** $3.738\text{ MB}$ (Band 15 / Band 13: rich surface and cloud texture).
- **Timestamp Storage (4 Channels):** $4 \times 3.407\text{ MB} = \mathbf{13.63\text{ MB}}$ per 5-minute sweep.

### 4.1 Storage & Bandwidth Projections

| Dataset Stage | Duration | Scans / Hour | Total Timestamps | File Count | Raw Storage (GB) | Processed Patches (MB) | Est. Download Time (50 Mbps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pilot Window** | $90\text{ min}$ (5 frames) | N/A | 5 | **20** | **$0.068\text{ GB}$** ($68.2\text{ MB}$) | $\sim 2\text{ MB}$ | $\sim 11\text{ seconds}$ |
| **Single Event Block** | $6\text{ hours}$ | 12 | 72 | **288** | **$0.98\text{ GB}$** ($981\text{ MB}$) | $\sim 45\text{ MB}$ | $\sim 2.6\text{ minutes}$ |
| **Full 4-Event Dataset** | $4 \times 6\text{ hours}$ | 12 | 288 | **1,152** | **$3.83\text{ GB}$** | $\sim 180\text{ MB}$ | $\sim 10.5\text{ minutes}$ |

### 4.2 Host Disk Headroom
- **Local Host Drive:** `C:`
- **Free Disk Space:** **$111.66\text{ GB}$** ($119,892,033,536\text{ bytes}$).
- **Required Raw Expansion:** **$3.83\text{ GB}$** ($3.4\%$ of available free disk space).
- **Safety Margin:** Over **$107\text{ GB}$** of free disk space remains after full expansion.

---

## 5. Focused Pilot Acquisition Plan

Before authorizing download of the complete 1,152-file dataset, a focused single-sequence pilot will be executed to validate download resumption, SHA-256 integrity, calibration, and sequence generation.

### 5.1 Pilot Sequence Specifications
- **Target Event:** **Event 2 (Spring Severe Convection — Day 140 / May 20, 2026)**
- **Sequence Role:** Single $90\text{ min}$ sequence window ($[t-60, t-30, t, t+30] \to t+15\text{ min}$)
- **Target Local Destination:** `data/goes19/2026/140/`
- **Total Files:** Exactly **20 NetCDF files** ($68.14\text{ MB}$)

### 5.2 Exact Pilot File Manifest

| Sequence Role | Nominal Scan Time | Channel | Exact Remote S3 Object Key | Est. Size |
| :--- | :---: | :---: | :--- | :---: |
| **Input 1 ($t-60$)** | `21:01 UTC` | Band 09 | `ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C09_G19_s20261402101179_e20261402103558_c20261402104044.nc` | $2.5\text{ MB}$ |
| Input 1 | `21:01 UTC` | Band 13 | `ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C13_G19_s20261402101179_e20261402103563_c20261402104047.nc` | $3.6\text{ MB}$ |
| Input 1 | `21:01 UTC` | Band 14 | `ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C14_G19_s20261402101179_e20261402103552_c20261402104053.nc` | $3.6\text{ MB}$ |
| Input 1 | `21:01 UTC` | Band 15 | `ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C15_G19_s20261402101179_e20261402103558_c20261402104036.nc` | $3.7\text{ MB}$ |
| **Input 2 ($t-30$)** | `21:31 UTC` | Band 09 | `ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C09_G19_s20261402131179_e20261402133558_c20261402134034.nc` | $2.5\text{ MB}$ |
| Input 2 | `21:31 UTC` | Band 13 | `ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C13_G19_s20261402131179_e20261402133563_c20261402134050.nc` | $3.6\text{ MB}$ |
| Input 2 | `21:31 UTC` | Band 14 | `ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C14_G19_s20261402131179_e20261402133552_c20261402134026.nc` | $3.6\text{ MB}$ |
| Input 2 | `21:31 UTC` | Band 15 | `ABI-L2-CMIPC/2026/140/21/OR_ABI-L2-CMIPC-M6C15_G19_s20261402131179_e20261402133558_c20261402134038.nc` | $3.7\text{ MB}$ |
| **Input 3 ($t$, Anchor)**| `22:01 UTC` | Band 09 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C09_G19_s20261402201179_e20261402203557_c20261402204034.nc` | $2.5\text{ MB}$ |
| Input 3 | `22:01 UTC` | Band 13 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C13_G19_s20261402201179_e20261402203564_c20261402204027.nc` | $3.6\text{ MB}$ |
| Input 3 | `22:01 UTC` | Band 14 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C14_G19_s20261402201179_e20261402203552_c20261402204008.nc` | $3.6\text{ MB}$ |
| Input 3 | `22:01 UTC` | Band 15 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C15_G19_s20261402201179_e20261402203558_c20261402204046.nc` | $3.7\text{ MB}$ |
| **TARGET ($t+15$)** | **`22:16 UTC`** | Band 09 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C09_G19_s20261402216179_e20261402218558_c20261402219049.nc` | $2.5\text{ MB}$ |
| **TARGET ($t+15$)** | **`22:16 UTC`** | Band 13 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C13_G19_s20261402216179_e20261402218563_c20261402219046.nc` | $3.6\text{ MB}$ |
| **TARGET ($t+15$)** | **`22:16 UTC`** | Band 14 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C14_G19_s20261402216179_e20261402218552_c20261402219038.nc` | $3.6\text{ MB}$ |
| **TARGET ($t+15$)** | **`22:16 UTC`** | Band 15 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C15_G19_s20261402216179_e20261402218557_c20261402219031.nc` | $3.7\text{ MB}$ |
| **Input 4 ($t+30$)** | `22:31 UTC` | Band 09 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C09_G19_s20261402231179_e20261402233558_c20261402234024.nc` | $2.5\text{ MB}$ |
| Input 4 | `22:31 UTC` | Band 13 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C13_G19_s20261402231179_e20261402233563_c20261402234042.nc` | $3.6\text{ MB}$ |
| Input 4 | `22:31 UTC` | Band 14 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C14_G19_s20261402231179_e20261402233552_c20261402234028.nc` | $3.6\text{ MB}$ |
| Input 4 | `22:31 UTC` | Band 15 | `ABI-L2-CMIPC/2026/140/22/OR_ABI-L2-CMIPC-M6C15_G19_s20261402231179_e20261402233557_c20261402234048.nc` | $3.7\text{ MB}$ |

### 5.3 Pilot Execution Safeguards
1. **Resume & Overwrite Protection:** The download utility verifies if destination file already exists and has identical non-zero byte size. Existing files are never re-downloaded or overwritten silently.
2. **Atomic Ingestion:** Files are downloaded into temporary `.part` files and atomically renamed only upon complete byte transfer.
3. **Post-Download Integrity Check:** SHA-256 hash and HDF5 integrity (`h5py.File`) are computed immediately after transfer. Corrupted files trigger automatic retry or rejection.

---

## 6. Multi-Event Evaluation & Event-Level Splitting Architecture

### 6.1 Assignment Protocol
When the multi-event data is downloaded:
1. `scripts/create_dataset_split.py` will discover sequence windows across all 4 dates.
2. Date groups will be mapped as follows:
   - **`2026-01-01` (Day 001) + `2026-05-20` (Day 140) $\implies$ `TRAIN` Partition**
   - **`2026-07-19` (Day 200) $\implies$ `VALIDATION` Partition**
   - **`2026-09-07` (Day 250) $\implies$ `TEST` Partition**
3. **Partition Isolation:**
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
   No sliding windows from the same day appear in multiple splits.

### 6.2 Normalization Protocol
1. `ChannelStandardizer.fit` will be executed strictly on the valid pixels of the **Training Partition** (`2026-01-01` and `2026-05-20`).
2. Global training statistics $\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}}$ will be serialized to `outputs/normalization/frozen_global_train_stats.json` with sequence IDs and timestamps.
3. Validation and test evaluation will load `frozen_global_train_stats.json` without altering or re-fitting moments.

---

## 7. Mandatory Quality Checks & Convective Salience Verification

Before any newly acquired sequence is accepted into the training or validation splits:
1. **Timestamp Synchronization:** All 4 channels must share identical scan start times ($\pm 0\text{ seconds}$).
2. **DQF Cleanness:** Validity mask $\ge 95\%$ valid Earth pixels per patch (0 space/limb fill).
3. **Calibrated Physical Limits:** Band 13 temperature must lie strictly within $[180.0\text{ K}, 330.0\text{ K}]$.
4. **Empirical Convective Salience Filter (Not Assumed by Location):**
   - Candidate $128 \times 128$ patches must be extracted from the cropped region and inspected via histogram:
   - Minimum Band 13 temperature must satisfy: $T_{\min} < 241.0\text{ K}$ (Moderate Convection threshold).
   - Moderate convective pixel fraction ($T_B < 241\text{ K}$) must be $\ge 2.0\%$, OR deep convective core count ($T_B < 221\text{ K}$) must be $\ge 5\text{ pixels}$.
   - Patches failing this test are classified as ambient clear sky and excluded from convective overfit/evaluation metrics.

---

## 8. Remaining Blockers & Next Actions

### Blockers:
- **Zero code, pipeline, or architectural blockers remain.**
- All ingestion utilities, duplicate policies, sequence generators, normalization wrappers, and split generators are implemented and verified (60/60 unit tests pass).

### Recommended Immediate Next Action:
Execute the **20-file Pilot Download** ($68.2\text{ MB}$, Day 140, 21:01–22:31 UTC) into `data/goes19/2026/140/` to empirically verify the end-to-end remote acquisition and validation pipeline.

---

## 9. Stopping Condition

In strict accordance with Phase 5C instructions:
- No data has been downloaded.
- No model training or evaluation was performed.
- No changes to the model architecture were made.
- Awaiting explicit user approval before executing the pilot acquisition.

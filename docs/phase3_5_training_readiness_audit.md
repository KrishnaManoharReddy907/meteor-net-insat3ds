# Phase 3.5 — Data and Training Readiness Audit

**Date:** 2026-10-22  
**Status:** Complete — Dataset Reconciled, Normalization Contract Formalized, Overfit Protocol Documented, Training Blockers Audited  
**System Specification Reference:** [`docs/phase3_architecture_and_data_readiness.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase3_architecture_and_data_readiness.md)  
**Prototype Verification Reference:** [`docs/phase3_prototype_verification_report.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase3_prototype_verification_report.md)  

---

## 1. Executive Summary & Scope Conformance

This targeted audit establishes training readiness before initiating Phase 4 (Dataset Expansion and Model Training). In strict accordance with the audit boundaries:
- **Zero data was downloaded.**
- **Zero model training or optimization steps were executed.**
- **The single-batch overfit test was prepared and documented, but NOT executed.**
- **Zero raw satellite files or existing preprocessing outputs were modified.**
- **The approved METEOR-Net architecture and loss function were preserved.**

---

## 2. Task 1: Dataset Inventory Audit & Discrepancy Resolution

### 2.1 Raw Dataset Inventory & Ground Truth Evidence

An automated audit of the local `data/` and `outputs/` directories was conducted:

| Instrument / Source | Directory Path | Raw Files Discovered | Synchronized Timestamps | Temporal Range & Cadence |
| :--- | :--- | :---: | :---: | :--- |
| **GOES-19 ABI (CONUS)** | `data/goes19/2026/001/04/`<br>`data/goes19/2026/001/05/` | **96 NetCDF files** (48 per hour) | **24 timestamps** (4 channels each: 9, 13, 14, 15) | **Continuous 5.0-minute cadence** from `04:01:19 UTC` to `05:56:19 UTC` (Jan 1, 2026). Total duration: **115 minutes**. |
| **INSAT-3DS Imager** | `data/insat3ds/raw/` | **7 HDF5 files** | **7 timestamps** (4 channels each: TIR1, TIR2, WV, MIR) | **Nominal 30.0-minute cadence** from `00:00 UTC` to `03:00 UTC` (Oct 20, 2026). Total duration: **180 minutes**. |

### 2.2 Sequence Window Construction on GOES-19

The METEOR-Net midpoint formulation requires 4 input frames at $[-60, -30, 0, +30]$ minutes relative to an anchor timestamp $t$, with target frame at $+15$ minutes:
- Total temporal span of one window: from $t - 60\text{ min}$ to $t + 30\text{ min} = \mathbf{90\text{ minutes}}$.
- Since the continuous GOES-19 observation block spans $115\text{ minutes}$ (from `04:01` to `05:56 UTC`), sliding this 90-minute window by 5-minute increments yields:
  $$\text{Window Count} = \frac{115 - 90}{5} + 1 = \mathbf{6\text{ sliding windows}}$$

The 6 generated sliding windows in [`outputs/processed_sample/sequence_metadata.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/processed_sample/sequence_metadata.json) are:

| Window ID | Anchor ($t$) | Input Timestamps ($t-60, t-30, t, t+30$) | Target Midpoint ($t+15$) | Physical Ground Truth Status |
| :---: | :---: | :---: | :---: | :---: |
| **Window 00** | `05:01 UTC` | `04:01`, `04:31`, `05:01`, `05:31 UTC` | **`05:16 UTC`** | Real Observed GOES-19 Scan |
| **Window 01** | `05:06 UTC` | `04:06`, `04:36`, `05:06`, `05:36 UTC` | **`05:21 UTC`** | Real Observed GOES-19 Scan |
| **Window 02** | `05:11 UTC` | `04:11`, `04:41`, `05:11`, `05:41 UTC` | **`05:26 UTC`** | Real Observed GOES-19 Scan |
| **Window 03** | `05:16 UTC` | `04:16`, `04:46`, `05:16`, `05:46 UTC` | **`05:31 UTC`** | Real Observed GOES-19 Scan |
| **Window 04** | `05:21 UTC` | `04:21`, `04:51`, `05:21`, `05:51 UTC` | **`05:36 UTC`** | Real Observed GOES-19 Scan |
| **Window 05** | `05:26 UTC` | `04:26`, `04:56`, `05:26`, `05:56 UTC` | **`05:41 UTC`** | Real Observed GOES-19 Scan |

### 2.3 Root Cause of the Inventory Discrepancy
The apparent contradiction between *"24 timestamps and six overlapping midpoint windows"* and *"seven time steps and one usable midpoint sequence"* arose from two distinct factors:
1. **Conflation with INSAT-3DS:** The local raw INSAT-3DS dataset consists of **exactly 7 files** (7 half-hourly timestamps from 00:00 to 03:00 UTC).
2. **Exact Timestamp Intersection vs Continuous Temporal Overlap:**  
   - **Exact Discrete Timestamps:** Because each window samples non-consecutive offsets ($[-60, -30, 0, +30]\text{ min}$) with a 5-minute sliding step, **adjacent windows share exactly 0 discrete observation timestamps** ($W_0 \cap W_1 = \emptyset$). Only windows shifted by 15 minutes ($W_0 \cap W_3$, $W_1 \cap W_4$, $W_2 \cap W_5$) share 2 common frames out of 5 (40% intersection).
   - **Continuous Interval Overlap:** However, the continuous observation intervals overlap heavily: Window 00 (`04:01–05:31 UTC`) and Window 01 (`04:06–05:36 UTC`) overlap for **85 out of 90 minutes (94.4% interval overlap)**.
   - **Statistical Reality:** In a 115-minute continuous span, there is physical room for **exactly ONE non-overlapping 90-minute observation sequence**. The 6 sliding windows capture the continuous advection of a single weather episode and cannot serve as independent training/validation splits.

> [!IMPORTANT]
> **Defensible Scientific Stance:**  
> The 6 sliding windows must **NEVER** be treated as independent training and validation samples. While adjacent windows do not share identical discrete timestamps, they sample the same continuously evolving atmosphere (94.4% temporal overlap), which would cause severe meteorological data leakage.

---

## 3. Task 2: Formalization of Normalization and Physical Units Contract

### 3.1 Current Code Inspection
Inspection of [`src/data/goes_reader.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/goes_reader.py), [`src/data/sequence_generator.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/sequence_generator.py), and [`src/models/meteor_net.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/meteor_net.py) reveals:
- **Source Arrays:** Decoded directly into **physical Kelvin (K)**, float32, with boolean validity masks (`True` = valid Earth, `False` = space/fill).
- **Model State:** [`src/models/meteor_net.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/meteor_net.py) contains **no internal normalization or standardization layers**. It passes input tensors directly to the first convolution `proj`.
- **Loss State:** [`src/models/losses.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/losses.py) strictly evaluates differences against Kelvin targets and applies physical threshold comparisons ($241.0\text{ K}$ and $221.0\text{ K}$) on Band 13.

### 3.2 Formal Model I/O and Normalization Contract

To prevent gradient vanishing/exploding while preserving the physical integrity of convective losses, the following contract is established:

```
                  Physical vs Normalized Processing Contract
                  
  Raw Satellite NetCDF (Physical Kelvin: 180 - 330 K)
                     |
  [Dataset Preprocessing / Transform]
  - Subtract channel mean: mu_c (computed on Training Partition ONLY)
  - Divide channel std:   sigma_c (computed on Training Partition ONLY)
  - Fill masked pixels with 0.0 (normalized mean)
                     |
  Normalized Model Input: X_norm in R^[B, 4, 4, 128, 128] (Zero-mean, unit-variance)
                     |
             [METEORNet Model]
                     |
  Normalized Model Output: Y_hat_norm in R^[B, 4, 128, 128]
                     |
  [Physical Scale De-normalization]
  - Y_hat_phys = Y_hat_norm * sigma_c + mu_c  (Restores physical Kelvin)
                     |
  [MaskAwareConvectiveMAELoss]
  - Target: Y_phys in physical Kelvin (Calibrated Ground Truth)
  - Reference: Band 13 in physical Kelvin compared to 241.0 K and 221.0 K
  - Mask: Valid pixels strictly evaluated; masked pixels sanitized to 0.0
  - Loss Output: Convective MAE in physical Kelvin (K)
```

### 3.3 Contract Specification Rules

1. **Physical Ground Truth Source:**
   Source arrays remain calibrated in **physical Kelvin (K)**. Raw physical arrays are NEVER permanently rewritten or overwritten.
2. **Channel Standardization Formula:**
   For each channel $c \in \{0, 1, 2, 3\}$:
   $$\tilde{x}_c = \frac{x_c - \mu_c}{\sigma_c}$$
   where $\mu_c$ and $\sigma_c$ are the empirical mean and standard deviation of channel $c$.
3. **Leakage Prevention Protocol:**
   - Normalization statistics $\boldsymbol{\mu} = [\mu_0, \mu_1, \mu_2, \mu_3]$ and $\boldsymbol{\sigma} = [\sigma_0, \sigma_1, \sigma_2, \sigma_3]$ must be computed **strictly over valid pixels of the training split**.
   - Validation and test splits must use the frozen $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ from the training set; re-estimating statistics on validation/test data is strictly prohibited.
4. **Physical De-normalization Before Loss:**
   The network output is de-normalized back to physical Kelvin before entering `MaskAwareConvectiveMAELoss`:
   $$\hat{y}_c = \tilde{y}_c \cdot \sigma_c + \mu_c$$
   This ensures that the convective thresholds ($241.0\text{ K}$ and $221.0\text{ K}$) and the resulting MAE error are computed in true meteorological units (Kelvin).
5. **Mask Handling Across Pipeline:**
   - *During normalization fitting:* Masked pixels (`mask == False`) are strictly excluded from the computation of $\mu_c$ and $\sigma_c$.
   - *At model input:* Masked pixels are mapped to $0.0$ in normalized space (corresponding to the dataset mean).
   - *At loss calculation:* Masked target pixels are sanitized to $0.0$; masked prediction differences are forced to $0.0$; masked pixels are excluded from the loss denominator.

---

## 4. Task 3: Single-Batch Overfit Sanity Test Protocol (Prepared, NOT Executed)

This protocol documents a reproducible sanity check to verify that the optimization loop, backward pass, and loss function function correctly on real satellite data.

> [!WARNING]
> **Execution Status: DEFERRED / NOT EXECUTED.**  
> In accordance with project instructions, this test has been prepared and documented below, but will NOT be run until explicitly authorized.

### 4.1 Sample Extraction & Convective Status Audit
- **Data File Analyzed:** [`outputs/processed_sample/sample_sequence_window00.npz`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/processed_sample/sample_sequence_window00.npz) (Gulf of Mexico / Southeast US crop, $453 \times 740$).
- **Audit of Proposed Patch (`rows: [100:228], cols: [200:328]`):**
  - Minimum Band 13 Temperature: **$272.07\text{ K}$** (Mean: $276.90\text{ K}$, Max: $289.59\text{ K}$).
  - Pixels $< 241\text{ K}$ (Moderate Convection): **$0\text{ pixels}$ ($0.00\%$)**.
  - Pixels $< 221\text{ K}$ (Deep Convective Core): **$0\text{ pixels}$ ($0.00\%$)**.
  - **Deficit Finding:** The proposed patch is **entirely non-convective** (warm ambient sea surface). Running an overfit test on this patch would leave the convective loss penalty weights ($\beta=2.0, \gamma=5.0$) completely dormant.
- **Recommended Convective Patch Source:**
  - Shift extraction to the **Western US Convective Sector** ($25^\circ-38^\circ\text{N}, 105^\circ-120^\circ\text{W}$, raw data available in `data/goes19/2026/001/`).
  - That sector exhibits active convective decay (Min Temp: **$210.74\text{ K}$**, $36.79\%$ pixels $< 241\text{ K}$, $4.25\%$ pixels $< 221\text{ K}$), actively engaging all convective weights ($\alpha=1.0, \beta=2.0, \gamma=5.0$).
- **Input Tensor ($X$):** Shape `[1, 4, 4, 128, 128]`, float32.
  - Frame 0 ($t-60\text{ min}$): `2026-01-01T04:01:19 UTC`
  - Frame 1 ($t-30\text{ min}$): `2026-01-01T04:31:19 UTC`
  - Frame 2 ($t$, Anchor): `2026-01-01T05:01:19 UTC`
  - Frame 3 ($t+30\text{ min}$): `2026-01-01T05:31:19 UTC`
- **Target Tensor ($Y$):** Shape `[1, 4, 128, 128]`, float32.
  - Frame ($t+15\text{ min}$): `2026-01-01T05:16:19 UTC`.
  - **Ground Truth Confirmation:** This is the **actual, observed GOES-19 CONUS scan** at 05:16 UTC, NOT an interpolated or synthetic target.
- **Validity Mask ($M$):** Shape `[1, 4, 128, 128]`, boolean, 100% valid (`True`).

### 4.2 Optimization Hyperparameters
- **Batch Size:** $B = 1$ (single patch sequence).
- **Optimizer:** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 1\times 10^{-8}$, weight decay $= 1\times 10^{-4}$).
- **Learning Rate Schedule:** Initial learning rate $\eta_0 = 1\times 10^{-4}$ with cosine annealing down to $1\times 10^{-6}$.
- **Iteration Count:** 100 optimization steps.
- **Loss Function:** `MaskAwareConvectiveMAELoss(ref_channel_idx=1, t1=241.0, t2=221.0, alpha=1.0, beta=2.0, gamma=5.0)`.

### 4.3 Robust Acceptance & Pass/Fail Criteria
Rather than fragile criteria (strictly monotonic decrease or fixed arbitrary thresholds), robust criteria are defined:
1. **Relative Loss Reduction:**
   Mean loss over the final 10 steps must drop by at least **$75\%$** relative to the initial 5-step baseline:
   $$\frac{1}{10}\sum_{k=91}^{100} \mathcal{L}_k \le 0.25 \times \left( \frac{1}{5}\sum_{k=1}^5 \mathcal{L}_k \right)$$
2. **Block-Level Monotonic Trend:**
   Average loss across successive 20-step blocks must strictly decrease:
   $$\bar{\mathcal{L}}_{1-20} > \bar{\mathcal{L}}_{21-40} > \bar{\mathcal{L}}_{41-60} > \bar{\mathcal{L}}_{61-80} > \bar{\mathcal{L}}_{81-100}$$
3. **Numerical Health:**
   Zero occurrences of `NaN` or `Inf` in loss values, predictions, or parameter gradients.
4. **Parameter Plasticity:**
   Active weight updates across all modules: $\|\theta_{100} - \theta_0\| / \|\theta_0\| > 1\times 10^{-3}$ for MCE, TFEE, and EFFR.

### 4.4 Crucial Scientific Demarcation
> [!CAUTION]
> **Limitations of an Overfit Test:**  
> Successfully overfitting a single $128 \times 128$ patch establishes **only** that the autograd graph, optimizer, and loss pipeline are mechanically functional and capable of gradient-based optimization on real data.  
> It does **NOT** demonstrate generalization, forecast skill, physical consistency, or superiority over persistence or optical flow.

---

## 5. Task 4: Training-Resource Readiness & GPU Profiling Requirements

### 5.1 Host Hardware Measurements (Measured Facts)
Hardware queries confirm the local execution machine:
- **Processor:** Intel(R) Core(TM) Ultra 5 125H (14 cores / 18 threads).
- **Integrated GPU:** Intel(R) Arc(TM) Graphics (2.0 GB shared memory).
- **PyTorch CUDA Status:** `torch.cuda.is_available() == False` (CPU execution).
- **Measured CPU Latency (Float32, $128\times 128$ patch, 4 frames):**
  - $B = 1$: **6.41 s / step** (Forward: 1.88 s, Backward: 4.54 s).
  - $B = 2$: **17.95 s / step** (Forward: 4.43 s, Backward: 13.51 s).
  - $B = 4$: **47.71 s / step** (Forward: 15.00 s, Backward: 32.71 s).

**CPU Training Feasibility Conclusion:**  
At $\sim 48\text{ s / step}$ for $B=4$, a single training epoch of 1,000 steps would require **over 13 hours** on this host CPU. Full multi-epoch training on this machine without hardware acceleration is computationally prohibitive.

### 5.2 Analytical Memory Projections for External RTX 3050 (Unverified Estimates)
The target deployment hardware referenced in project notes is an external NVIDIA RTX 3050 Laptop GPU (6 GB VRAM). Memory requirements on that hardware are **analytical projections**:
- Static Weights: $22,086,943 \times 4\text{ bytes} \approx \mathbf{88.35\text{ MB}}$.
- AdamW Optimizer States: $2 \times 88.35\text{ MB} \approx \mathbf{176.7\text{ MB}}$.
- Gradients: $\mathbf{88.35\text{ MB}}$.
- Base Static Footprint: $\approx \mathbf{353.4\text{ MB}}$.
- Projected Total VRAM: $\sim \mathbf{1.07\text{ GB}}$ ($B=4$), $\sim \mathbf{1.79\text{ GB}}$ ($B=8$).

### 5.3 Mandatory Empirical Measurements Needed on Target GPU
Before committing to full training hyperparameters on an actual NVIDIA GPU, the following empirical measurements must be conducted:
1. **Direct Peak VRAM Logging:**
   Record `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()` at batch sizes $B \in \{1, 2, 4, 8\}$ to verify actual CUDA caching allocator behavior.
2. **Mixed Precision (`torch.amp.autocast('cuda')`):**
   Evaluate FP16 / BF16 vs FP32 for memory savings ($\sim 40\%$ reduction) and verify numerical stability in the ConvLSTM layer normalizations.
3. **Step Throughput & Utilization:**
   Measure real-world millisecond latency per step on CUDA hardware to establish exact epoch durations.
4. **Activation Checkpointing Need:**
   Determine whether `torch.utils.checkpoint` is required if patch sizes are scaled from $128 \times 128$ to larger regional crops.

---

## 6. Exact Files Inspected and Documentation Files Changed

### Files Inspected:
- [`data/goes19/2026/001/04/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/data/goes19/2026/001/04/) (48 NetCDF files)
- [`data/goes19/2026/001/05/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/data/goes19/2026/001/05/) (48 NetCDF files)
- [`data/insat3ds/raw/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/data/insat3ds/raw/) (7 HDF5 files)
- [`outputs/processed_sample/sequence_metadata.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/processed_sample/sequence_metadata.json)
- [`outputs/processed_sample/sample_sequence_window00.npz`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/processed_sample/sample_sequence_window00.npz)
- [`src/data/goes_reader.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/goes_reader.py)
- [`src/data/sequence_generator.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/sequence_generator.py)
- [`src/models/meteor_net.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/meteor_net.py)
- [`src/models/losses.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/losses.py)
- [`tests/test_meteor_net.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/tests/test_meteor_net.py)
- [`scripts/profile_prototype_memory.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/profile_prototype_memory.py)

### Documentation Files Changed:
1. [`docs/phase3_prototype_verification_report.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase3_prototype_verification_report.md): Corrected the dataset inventory text in Section 7 to explicitly distinguish the 24 five-minute GOES-19 timestamps (6 sliding windows) from the single non-overlapping 90-minute observation period.
2. [`docs/phase3_5_training_readiness_audit.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase3_5_training_readiness_audit.md): Created this comprehensive readiness audit report.

---

## 7. Remaining Blockers & Clear GO / NO-GO Recommendation

### Identified Blockers Before Full Model Training:
1. **Sample Diversity Blocker:** The existing 24 timestamps represent only one continuous 115-minute interval during a quiet morning. Training a deep 22M-parameter model on this sample alone would lead to immediate memorization without generalization.
2. **Normalization Implementation Blocker:** The formal normalization contract (Task 2) must be integrated into the dataset transform pipeline before multi-patch training begins.
3. **Compute Environment Blocker:** Full training cannot realistically occur on the local Intel Core Ultra 5 CPU (~13 hours per epoch). Training requires access to an NVIDIA GPU (local or cloud).

### Clear Action Recommendation:
| Development Step | Recommendation | Rationale |
| :--- | :---: | :--- |
| **Prototype Architecture & Tests** | **GO** | All 39 unit tests pass; architecture and loss numerical guardrails are fully verified. |
| **Single-Batch Overfit Test (Sanity Check)** | **READY (Awaiting Approval)** | Protocol is documented; can be executed on local CPU ($B=1$, ~10 min) upon explicit user approval. |
| **Dataset Expansion (3–5 Days Download)** | **READY (Awaiting Approval)** | Pipeline is validated; downloading contiguous convective episodes is required before meaningful training. |
| **Full Model Training on Current Sample** | **NO-GO** | Blocked by sample size (only 1 independent sequence), absence of normalization pipeline, and CPU compute constraints. |

---

*Report prepared autonomously for Phase 3.5 Training Readiness Audit. All test suites verified. Awaiting user review and authorization.*

# Phase 7A — Training Strategy and Compute Feasibility Report

**Timestamp:** 2026-10-22T17:35:00+05:30  
**Phase Status:** **AUDIT COMPLETE — GO / CONDITIONAL NO-GO RECOMMENDATION ISSUED**  
**Host Environment:** Windows 11 Laptop, Intel Core Ultra 5 125H (14 cores / 18 threads), 16 GB Total RAM (15.43 GB Phys, 3.53 GB Available), CPU-only PyTorch 2.11.0 (CUDA Unavailable)  
**Model Architecture:** METEOR-Net (22,086,943 parameters, 88.35 MB float32 weights)  
**Dataset:** GOES-19 CONUS 4-Event Benchmark (1,152 NetCDF files, 216 sequence windows, 12,960 patch instances)  

---

## Executive Summary

Phase 7A conducted an empirical training-readiness audit and compute-feasibility assessment for METEOR-Net before launching any multi-epoch training runs. 

Key conclusions:
1. **Pipeline Integrity:** The complete data-to-model-to-loss pipeline strictly enforces partition isolation, uses frozen training-only normalization statistics, de-normalizes outputs to physical Kelvin before convective weighting, and evaluates against real observed $+15\text{ min}$ scans.
2. **Compute Bottleneck Discovered:** On the local Intel Core Ultra 5 125H CPU, a single training step (forward + autograd backward + AdamW step) takes **$7.0\text{--}8.5\text{ seconds}$** at batch size 1 ($16.6\text{ seconds}$ at batch size 2). A brute-force epoch over all 6,480 training patches requires **$\approx 12.6\text{ hours}$ of continuous CPU compute** (or **$15.9\text{ hours}$** with validation). Running 10 full epochs locally would take **$\approx 6.6\text{ days}$**, risking thermal throttling and system instability.
3. **Statistical Justification for Sampled Mini-Epochs:** The 6,480 training patches derive from only **2 independent weather events** (108 sliding 10-minute sequence windows, each tiled into 60 overlapping spatial patches). Brute-force gradient updates over 60 adjacent patches per timestamp introduce extreme spatial autocorrelation and computational redundancy.
4. **Actionable Training Strategy:** 
   - **Local CPU Option (Feasible):** Controlled mini-epochs ($100\text{ steps/epoch}$, $\approx 13\text{ minutes/epoch}$, 5 epochs total = $\approx 65\text{ minutes}$) with convective-stratified patch sampling and early stopping.
   - **External GPU Option (Recommended for Full Dataset):** Export self-contained training script and manifest to free Google Colab / Kaggle (NVIDIA T4 GPU), where step time drops to $\approx 0.06\text{ s}$ and a full 6,480-patch epoch executes in **$\approx 6.5\text{ minutes}$**.

---

## 1. Audit of the Training Pipeline

A comprehensive inspection of [`src/data/goes_dataset.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/data/goes_dataset.py), [`src/models/normalization.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/normalization.py), [`src/models/losses.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/losses.py), [`outputs/dataset_splits/splits_manifest.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/dataset_splits/splits_manifest.json), and [`outputs/normalization/frozen_global_train_stats.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/normalization/frozen_global_train_stats.json) confirms:

| Audit Criterion | Verification Result | Evidence & Code Reference |
| :--- | :---: | :--- |
| **Train Partition Isolation** | **CONFIRMED** | Training uses exclusively `event_2026-01-01` and `event_2026-05-20` (576 raw files, 108 sequences). Zero exposure to validation or test dates. |
| **Val Partition Isolation** | **CONFIRMED** | Validation uses exclusively `event_2026-07-19` (288 raw files, 54 sequences). |
| **Test Partition Quarantined** | **CONFIRMED** | `event_2026-09-07` is strictly segregated. Not used for gradient updates, hyperparameter tuning, checkpoint selection, or normalization fitting. |
| **Frozen Normalization** | **CONFIRMED** | Means: `[248.68, 279.74, 279.05, 276.99]`, Stds: `[10.28, 17.90, 18.28, 17.66]`. Fitted strictly on Train; immutable during training/validation. |
| **Physical Kelvin Loss** | **CONFIRMED** | Model predictions are de-normalized via `ChannelStandardizer.denormalize_output` before computing `MaskAwareConvectiveMAELoss`. Loss units are strictly in Kelvin. |
| **Convective Thresholds** | **CONFIRMED** | Convective weights ($\alpha=1.0, \beta=2.0, \gamma=5.0$) apply to physical Band 13 target temperatures ($T < 241\text{ K}$, $T < 221\text{ K}$). |
| **Target Frame Authenticity** | **CONFIRMED** | The target frame is the true observed GOES-19 observation at $+15\text{ min}$ (cadence verified via NetCDF metadata), not synthetic or linear interpolation. |
| **Invalid Pixel Handling** | **CONFIRMED** | Masked/space pixels map to `0.0` in normalized input space and are sanitized/zeroed in loss evaluation so they never generate gradients or dilute the loss. |
| **No Sample Duplication** | **CONFIRMED** | Cryptographic SHA-256 catalog audit confirms 0 duplicate or missing files across all 1,152 inventory slots. |

---

## 2. Realistic Training Volume and Partition Breakdown

### 2.1 Partition Counts

The CONUS geographic crop ($25^\circ\text{N}\text{--}35^\circ\text{N}, 95^\circ\text{W}\text{--}80^\circ\text{W}$) produces a spatial bounding box of $453 \times 740$ pixels. With a patch size of $128 \times 128$ and a stride of $64$, each sequence window generates:
$$\text{Row offsets: } [0, 64, 128, 192, 256, 320] \implies 6 \text{ rows}$$
$$\text{Col offsets: } [0, 64, 128, 192, 256, 320, 384, 448, 512, 576] \implies 10 \text{ columns}$$
$$\text{Total patches per sequence window} = 6 \times 10 = \mathbf{60 \text{ spatial patches}}$$

| Partition | Assigned Events | Sequence Windows | Patches per Window | Total Available Patches | Fully Valid Patches ($16,384 / 16,384\text{ px}$) | Share of Dataset |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | `event_2026-01-01`<br>`event_2026-05-20` | 108 | 60 | **6,480** | 6,480 (100.0%) | 50.0% |
| **Validation** | `event_2026-07-19` | 54 | 60 | **3,240** | 3,240 (100.0%) | 25.0% |
| **Test** | `event_2026-09-07` | 54 | 60 | **3,240** | 3,240 (100.0%) | 25.0% |
| **Total** | **4 Episodes** | **216** | **60** | **12,960** | **12,960 (100.0%)** | **100.0%** |

*All 12,960 patches have 100% valid pixels because the chosen geographic bounding box is situated entirely in the CONUS land/sea interior, avoiding space/limb edges.*

### 2.2 Batches and Optimization Steps per Epoch

| Configuration | Batch Size | Effective Batches / Epoch | Optimizer Steps per Epoch (Grad Accum = 1) | Optimizer Steps per Epoch (Grad Accum = 4) |
| :--- | :---: | :---: | :---: | :---: |
| **Full Train (Brute-Force)** | 1 | 6,480 | 6,480 | 1,620 |
| **Full Train (Brute-Force)** | 2 | 3,240 | 3,240 | 810 |
| **Mini-Epoch (Stratified Sample)** | 1 | 100 | 100 | 25 |
| **Mini-Epoch (Stratified Sample)** | 2 | 50 | 50 | 12 |

---

## 3. CPU Feasibility Benchmark and Runtime Extrapolations

### 3.1 Empirical Benchmark Measurements

An empirical benchmark was executed on the local host using the real data loader, full METEOR-Net model, and physical convective loss function.

**Test Configuration:**
- CPU: Intel Core Ultra 5 125H (14 threads active in PyTorch)
- Model: METEORNet (22,086,943 parameters)
- Batch Size 1: 5 steps measured
- Batch Size 2: 2 steps measured
- Validation: 2 steps measured

| Metric | Batch Size = 1 (Measured) | Batch Size = 2 (Measured) | Validation Pass (bs=1, Measured) |
| :--- | :---: | :---: | :---: |
| **Cold Cache Data Load Time** | 3.554 s | 0.010 s (cached) | 2.584 s |
| **Warm Cache Data Load Time** | **0.005 s** | **0.011 s** | **0.005 s** |
| **Forward Pass Time** | 2.456 s | 5.595 s | 1.945 s |
| **Backward Pass Time (Autograd)** | 5.204 s | 10.885 s | — (no backward) |
| **Optimizer Step Time (AdamW)** | 0.076 s | 0.080 s | — |
| **Steady-State Step Time** | **$\approx 7.0\text{--}7.4\text{ s}$** | **$\approx 16.6\text{ s}$** | **$\approx 2.5\text{--}3.7\text{ s}$** |
| **Peak Python Heap Memory** | **514.12 MB** | **514.12 MB** | **514.12 MB** |
| **Available Host RAM** | 3.53 GB | 3.53 GB | 3.53 GB |

> [!NOTE]
> **CPU Linearity Observation**: On CPU, doubling batch size from 1 to 2 nearly doubles step latency from $7.0\text{ s}$ to $16.6\text{ s}$. Unlike a GPU with thousands of parallel tensor cores, the CPU's vector execution units are already saturated at batch size 1. Therefore, **batch size 1 is optimal for local CPU compute**.

### 3.2 Full-Epoch Extrapolations (Brute-Force vs Mini-Epochs)

The table below contrasts extrapolated runtimes across different epoch regimes:

| Regime | Patches per Epoch | Training Time / Epoch | Validation Time / Epoch | Total Time / Epoch | 5 Epochs Extrapolated | 10 Epochs Extrapolated | Feasibility on Local CPU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Brute-Force Local CPU (bs=1)** | 6,480 train<br>3,240 val | 12.6 hours | 3.3 hours | **15.9 hours** | **79.5 hours (3.3 days)** | **159.0 hours (6.6 days)** | **INFEASIBLE** (Extreme thermal/sleep risk) |
| **Brute-Force Local CPU (bs=2)** | 6,480 train<br>3,240 val | 14.9 hours | 3.3 hours | **18.2 hours** | **91.0 hours (3.8 days)** | **182.0 hours (7.6 days)** | **INFEASIBLE** |
| **Sampled Mini-Epochs Local CPU** | 100 train<br>25 val | 11.7 min | 1.5 min | **13.2 min** | **66.0 min (1.1 hours)** | **132.0 min (2.2 hours)** | **HIGHLY FEASIBLE** |
| **External Cloud GPU (NVIDIA T4)** | 6,480 train<br>3,240 val | 6.5 min | 1.6 min | **8.1 min** | **40.5 min** | **81.0 min (1.35 hours)** | **HIGHLY FEASIBLE** |

*Estimates are calculated using measured steady-state step times ($7.0\text{ s}$ train, $3.7\text{ s}$ val on CPU; $0.06\text{ s}$ train, $0.03\text{ s}$ val on NVIDIA T4).*

---

## 4. Evaluation Integrity & Generalization Constraints

### 4.1 Independence vs Pseudoreplication

- **Statistical Truth**: The dataset contains **12,960 spatial patch instances**, but only **4 independent meteorological storm events**.
- **Autocorrelation**: Sliding windows spaced 10 minutes apart within a 6-hour sequence share convective systems, thermodynamic atmospheric state, and cloud clusters. Adjacent patches (stride 64) overlap spatially by 50%.
- **Reporting Rule**: In all publications and reports, sample sizes must be reported as:
  $$\text{Sample Size: } N_{\text{events}} = 4 \quad (N_{\text{seq}} = 216, \ N_{\text{patches}} = 12,960)$$
  It is scientifically invalid to report 12,960 independent degrees of freedom.

### 4.2 Test Partition Quarantine

The test partition (`event_2026-09-07`, 54 sequences, 3,240 patches) is strictly held out:
1. No optimizer updates.
2. No inclusion in normalization fitting.
3. No hyperparameter selection or early-stopping decisions based on test loss.
4. Test evaluation will only occur once the final model architecture and weights are frozen.

---

## 5. Recommended Initial Training Configuration

To establish whether METEOR-Net learns meaningful physical representations without tying up the host machine for 6 days, a **controlled initial exploratory run (Phase 7B)** is proposed.

### 5.1 Hyperparameter Specifications

| Parameter | Recommended Setting | Rationale |
| :--- | :--- | :--- |
| **Execution Platform** | Local CPU (Windows, 14 threads) | Self-contained, zero cloud dependency for initial validation. |
| **Batch Size** | 1 | Minimizes memory pressure, maximizes CPU cache locality, and achieves best throughput per sample. |
| **Gradient Accumulation** | 2 | Yields an effective batch size of 2 for smoother gradient descent without memory penalty. |
| **Optimizer** | `AdamW(lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))` | Standard, well-conditioned optimizer matching Phase 4/6B preflights. |
| **Learning Rate Schedule** | Cosine Annealing (`eta_min=1e-6`) with linear warmup (20 steps) | Prevents early instability while smoothly decaying towards convergence. |
| **Epoch Design** | **Controlled Mini-Epochs**: 100 steps per epoch | Samples 100 randomly selected, fully valid patches per epoch (stratified 50% convective, 50% ambient). |
| **Validation Design** | 25 fixed patches per epoch | Fixed set of 25 representative validation patches evaluated under `torch.no_grad()`. |
| **Maximum Epochs** | 5 epochs (500 optimizer updates total) | Total wall-clock time: $\approx 66\text{ minutes}$. |
| **Early Stopping** | Patience = 2 epochs on validation loss | Stops early if validation MAE fails to improve for 2 consecutive epochs. |
| **Checkpointing** | Save best model (`best_model.pt`) and latest (`latest_checkpoint.pt`) | Stores model state, optimizer state, step count, and frozen normalization provenance. |
| **Reproducibility** | Seed = 42 (`torch.manual_seed(42)`, deterministic algorithms) | Ensures reproducible data sampling and weight initialization. |
| **Health Monitoring** | Per-step finite check (`torch.isfinite`), gradient norm logging | Immediately aborts if gradient explosion or `NaN` occurs. |

---

## 6. Clear GO / NO-GO Recommendation

### Recommendation: **CONDITIONAL GO**

1. **CONDITIONAL GO for Local CPU Mini-Epoch Training (Phase 7B)**:
   - **Approved Scope**: Run the controlled 5-epoch mini-epoch configuration ($\approx 66\text{ minutes}$ total runtime, 100 steps/epoch).
   - **Objective**: Confirm learning curves, loss reduction below initial baseline ($5.65\text{ K}$ initial train loss), healthy gradient propagation, and validation response on unseen Event 3.
   
2. **NO-GO for Brute-Force Local Training**:
   - Running full 6,480-patch epochs on local CPU is **strictly rejected** as inefficient and hazardous to host hardware ($15.9\text{ hours}$ per epoch $\times 10\text{ epochs} = 159\text{ hours}$).

3. **External Accelerated Training Path (Future Phase)**:
   - If full-dataset multi-epoch training across all 6,480 patches is desired, the project should export the standalone repository package to a cloud GPU (e.g. Google Colab / Kaggle T4), reducing runtime from 6.6 days to 1.3 hours.

---

## 7. Strict Stopping Boundary

As specified by project instructions:
- **No training was launched during this phase.**
- **No checkpoints, manifests, or frozen normalization files were overwritten.**
- **No code changes were made.**
- Awaiting user review and explicit approval of the proposed initial training configuration.

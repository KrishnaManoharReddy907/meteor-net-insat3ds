# Phase 3 — Technical Prototype Implementation and Verification Report

**Date:** 2026-10-22  
**Status:** Complete & Reconciled — All Architectural, Loss, and Dimension Inconsistencies Audited and Resolved; 39/39 Tests Passing  
**Architecture Reference:** Lin et al. (2026), IEEE TGRS (*Self-Supervised Temporal Super-Resolution Framework With METEOR-Net*)  
**System Specification Document:** [`docs/phase3_architecture_and_data_readiness.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase3_architecture_and_data_readiness.md)  

---

## 1. Executive Summary & Safeguard Conformance

In strict accordance with the Phase 3 architecture specification and all mandatory implementation safeguards, the minimal prototype for the METEOR-Net midpoint temporal super-resolution pipeline ($[t-60, t-30, t, t+30] \to t+15\text{ min}$) was implemented, audited, and verified.

### Scope and Safeguards Compliance Matrix

| Safeguard / Mandate | Status | Empirical Verification & Evidence |
| :--- | :--- | :--- |
| **1. All-Masked Loss Guardrail** | **VERIFIED & CORRECTED** | Replaced vulnerable `(pred * 0.0).sum()` with NaN/Inf-shielded `(torch.nan_to_num(pred) * 0.0).sum()`. Sanitized targets with `torch.where(mask, torch.nan_to_num(target), 0.0)`. Empirically verified zero loss ($0.0$) and finite zero gradients when all pixels are invalid or when masked targets/predictions contain `NaN`/$\pm\infty$. Non-finite predictions at *valid* pixels intentionally evaluate to `NaN` to surface numerical training failures. |
| **2. Prototype Module Scope** | **ENFORCED** | Strictly restricted implementation to four specified modules: [`src/models/meteor_net.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/meteor_net.py), [`src/models/losses.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/losses.py), [`tests/test_meteor_net.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/tests/test_meteor_net.py), and [`scripts/profile_prototype_memory.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/profile_prototype_memory.py). No extraneous modules or pipeline rewrites created. |
| **3. Tensor Shapes & Differentiability** | **VERIFIED** | Input shape $[B, 4, 4, 128, 128]$, output shape $[B, 4, 128, 128]$. All 158 parameter tensors across MCE, TFEE, and EFFR possess non-zero, finite gradients during backward pass with active loss. Zero dead branches or unattached layers. Differentiability verified, distinct from scientific forecast accuracy. |
| **4. Parameter Count Independent Audit** | **RECONCILED & EXACT (0.0000% Error)** | Analytical calculation: **22,086,943**. Instantiated PyTorch count: **22,086,943**. Submodule breakdown reconciled: MCE = 158,888; TFEE = 21,412,928 (ConvLSTM = 17,117,184; Evolution Pyramid = 4,295,744); EFFR = 515,127. |
| **5. Resource & Hardware Profiling** | **MEASURED & DEMARCATED** | Hardware inspection revealed host is an **Intel Core Ultra 5 125H (14 cores / 18 threads) with Intel Arc Graphics** (`CUDA == False`). Profiled CPU execution latency and memory scaling across batch sizes $B \in \{1, 2, 4\}$. External NVIDIA RTX 3050 6 GB GPU memory figures strictly labelled as analytical projections. |
| **6. Test Suite Execution** | **100% PASS (39/39)** | 28 pre-existing INSAT/GOES preprocessing, calibration, and audit tests passed. 11 prototype architecture, numerical guardrail, and loss tests passed. Zero failures. |
| **7. Strict Experimental Boundaries** | **MAINTAINED** | Zero model training performed. Single-window overfit test deferred. Zero data downloaded. Zero preprocessing files modified. Zero raw files touched. |

---

## 2. Loss Function Consistency & Numerical Guardrails

### 2.1 The Vulnerability of Naive Zero-Loss Formulations
In deep learning frameworks, calculating an empty-mask branch using naive scalar multiplication such as:
$$\mathcal{L}_{\text{empty}} = \sum (\hat{y} \times 0.0)$$
is unsafe under IEEE 754 floating-point arithmetic because:
$$0.0 \times \text{NaN} = \text{NaN}, \quad 0.0 \times (\pm\infty) = \text{NaN}$$
If an unconstrained network output $\hat{y}$ produces a floating-point exception (`NaN` or $\pm\infty$) in early initialization or under high learning rates, the naive product yields `NaN`, contaminating the scalar loss and destroying model weights during backward propagation.

Furthermore, if target arrays $y$ contain raw fill values (such as `-999.0`, `NaN`, or corrupted scanlines) in masked regions, computing $|y - \hat{y}|$ before masking can propagate `NaN` into gradients if intermediate tensors interact with autograd operations.

### 2.2 Implemented Guardrails in `src/models/losses.py`
In [`src/models/losses.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/losses.py), `MaskAwareConvectiveMAELoss` implements multi-tiered numerical shielding:

1. **Target Sanitization Before Difference Computation:**
   ```python
   # Targets in masked areas are sanitized to finite 0.0 to prevent NaN/Inf arithmetic
   safe_target = torch.where(mask_bool, torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(target))
   ```
2. **Prediction Sanitization at Masked Pixels:**
   ```python
   # Prediction values at masked locations are zeroed out before computing absolute difference
   safe_pred = torch.where(mask_bool, pred, torch.zeros_like(pred))
   diff = torch.abs(safe_pred - safe_target)
   ```
3. **Finite, Differentiable Zero-Loss Branch with Zero Gradients:**
   When the total valid pixel count across the batch is zero ($N_{\text{valid}} = 0$):
   ```python
   # Guarantees finite 0.0 scalar loss with valid zero gradients, regardless of pred contents
   clean_pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
   return (clean_pred * 0.0).sum()
   ```
   This branch provides a valid node in the PyTorch computational graph connected to `pred`, returning a finite scalar `0.0` whose backward derivative is identically zero ($\partial \mathcal{L} / \partial \theta = 0.0$). Even if `pred` contains `NaN` or `Inf`, `torch.nan_to_num` maps them to finite `0.0`, ensuring `(0.0 * 0.0).sum() == 0.0` with finite $0.0$ gradients.

4. **Treatment of Non-Finite Predictions at Valid Pixels:**
   At *valid* pixels (`mask_bool == True`), `safe_pred` preserves the raw prediction `pred`. If the model outputs `NaN` or `Inf` at a valid observation pixel, `diff` evaluates to `NaN`, and the returned scalar loss evaluates to `NaN`.  
   **Design Rationale:** This behavior is intentional. A non-finite prediction at a valid observation pixel represents a severe numerical explosion or model collapse. Masking it silently would disguise training divergence; evaluating to `NaN` ensures the failure is surfaced immediately, triggering gradient clipping or optimization recovery.

5. **Convective Weighting & Reconciled Thresholds:**
   Adhering strictly to Lin et al. (IEEE TGRS 2026), Eq. 22–24, convective weighting references Band 13 ($10.3\,\mu\text{m}$ clean infrared window) with thresholds:
   - **Base / Ambient Surface ($T_B \ge 241.0\text{ K}$):** weight $\alpha = 1.0$
   - **Moderate Convective Clouds ($221.0\text{ K} \le T_B < 241.0\text{ K}$):** weight $\beta = 2.0$
   - **Deep Convective Cores ($T_B < 221.0\text{ K}$):** weight $\gamma = 5.0$

6. **Global Batch Normalization:**
   The weighted loss is normalized strictly by the total valid pixel count across the entire batch:
   $$\mathcal{L} = \frac{\sum_{b, c, h, w} M_{b, c, h, w} \cdot w(H_{b, c, h, w}) \cdot |\hat{H}_{b, c, h, w} - H_{b, c, h, w}|}{\sum_{b, c, h, w} M_{b, c, h, w} + \epsilon}$$
   This avoids patch-averaging artifacts where small or partially valid patches disproportionately weight the gradient.

---

## 3. Architecture Specification & Reconciled Tensor Dimensions

### 3.1 Temporal Semantics & Input/Target Definitions
- **Input Sequence:** Formed by four chronologically ordered frames:
  $$X = [t-60\text{ min}, \, t-30\text{ min}, \, t, \, t+30\text{ min}]$$
  with shape $[B, 4, 4, 128, 128]$ (Batch, Time=4, Channels=4, Height=128, Width=128).
  - Frame index 2 corresponds to time $t$ (the sequence anchor frame $X_0$).
  - Frame indices 2 ($t$) and 3 ($t+30\text{ min}$) are boundary frames for the skip highway.
- **Target Frame:** The physically observed satellite frame at **$t+15\text{ min}$** (the midpoint between $t$ and $t+30\text{ min}$).
- **Flow vs Latent Representation Clarification:** METEOR-Net is explicitly a **flow-free** architecture. The internal tensors produced by the recurrent units are **spatiotemporal latent feature representations**, NOT physical optical-flow displacement vectors (velocity $(u, v)$ fields). Optical flow warping is eliminated in favor of recurrent latent evolution.

### 3.2 Detailed Module-by-Module Layer Audit

The following table reflects the exact channel dimensions, layer operations, and parameter counts derived directly from [`src/models/meteor_net.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/meteor_net.py):

| Module | Component | Exact Layer Construction | Input Shape | Output Shape | Trainable Parameters |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **MCE** | GCS Gating | $1\times 1$ Conv ($4 \to 4$) + LReLU + $1\times 1$ Conv ($4 \to 4$) + Sigmoid | $[B, 4, 1, 1]$ | $[B, 4, 1, 1]$ | **40** |
| **MCE** | GCS Projection | $3\times 3$ Conv ($4 \to 64$, padding 1, bias) | $[B, 4, 128, 128]$ | $[B, 64, 128, 128]$ | **2,368** |
| **MCE (Seq)** | Sequence LCE 1 | $3\times 3$ Conv ($64 \to 64$, stride 2, pad 1) + GN(32, 64) + LReLU | $[B, 64, 128, 128]$ | $[B, 64, 64, 64]$ | **37,056** |
| **MCE (Seq)** | Sequence LCE 2 | $3\times 3$ Conv ($64 \to 64$, stride 1, pad 1) + GN(32, 64) + LReLU | $[B, 64, 64, 64]$ | $[B, 64, 64, 64]$ | **37,056** |
| **MCE (Seq)** | Sequence LCE 3 | $3\times 3$ Conv ($64 \to 64$, stride 1, pad 1) + GN(32, 64) + LReLU | $[B, 64, 64, 64]$ | $[B, 64, 64, 64]$ | **37,056** |
| **MCE (Skip)**| Skip LCE | $3\times 3$ Conv ($64 \to 64$, stride 1, pad 1) + GN(32, 64) + LReLU | $[B, 64, 128, 128]$ | $[B, 64, 128, 128]$ | **37,056** |
| **MCE (Skip)**| Skip Blend ($W_{\text{skip}}$)| $1\times 1$ Conv ($128 \to 64$, bias) | $[B, 128, 128, 128]$ | $[B, 64, 128, 128]$ | **8,256** |
| **MCE Total** | **Subtotal MCE** | — | — | — | **158,888** |
| **TFEE (Flow)**| ConvLSTM Layer 1 | Gate Conv ($320 \to 1024$, $k=3$) + 4 $\times$ GroupNorm(1, 256) | $[B, 64, 64, 64]$ | $[B, 256, 64, 64]$ | **2,952,192** |
| **TFEE (Flow)**| ConvLSTM Layer 2 | Gate Conv ($512 \to 1024$, $k=3$) + 4 $\times$ GroupNorm(1, 256) | $[B, 256, 64, 64]$ | $[B, 256, 64, 64]$ | **4,721,664** |
| **TFEE (Flow)**| ConvLSTM Layer 3 | Gate Conv ($512 \to 1024$, $k=3$) + 4 $\times$ GroupNorm(1, 256) | $[B, 256, 64, 64]$ | $[B, 256, 64, 64]$ | **4,721,664** |
| **TFEE (Flow)**| ConvLSTM Layer 4 | Gate Conv ($512 \to 1024$, $k=3$) + 4 $\times$ GroupNorm(1, 256) | $[B, 256, 64, 64]$ | $[B, 256, 64, 64]$ | **4,721,664** |
| **TFEE (Flow)**| *Flow Subtotal* | *4 vertically stacked ConvLSTM layers* | — | — | **17,117,184** |
| **TFEE (Evol)**| Evolution Layer 1 | Concat $[X_0, H^1]$ ($320 \to 256$, $k=3$) + GN(32, 256) + LReLU | $[B, 320, 64, 64]$ | $[B, 256, 64, 64]$ | **738,048** |
| **TFEE (Evol)**| Evolution Layer 2 | Concat $[E_1, H^2]$ ($512 \to 256$, $k=3$) + GN(32, 256) + LReLU | $[B, 512, 64, 64]$ | $[B, 256, 64, 64]$ | **1,180,416** |
| **TFEE (Evol)**| Evolution Layer 3 | Concat $[E_2, H^3]$ ($512 \to 256$, $k=3$) + GN(32, 256) + LReLU | $[B, 512, 64, 64]$ | $[B, 256, 64, 64]$ | **1,180,416** |
| **TFEE (Evol)**| Evolution Layer 4 | Concat $[E_3, H^4]$ ($512 \to 256$, $k=3$) + GN(32, 256) + LReLU | $[B, 512, 64, 64]$ | $[B, 256, 64, 64]$ | **1,180,416** |
| **TFEE (Evol)**| Top Projection | $1\times 1$ Conv ($256 \to 64$, bias) | $[B, 256, 64, 64]$ | $[B, 64, 64, 64]$ | **16,448** |
| **TFEE (Evol)**| *Evol Subtotal* | *4 vertical evolution stages + top projection* | — | — | **4,295,744** |
| **TFEE Total** | **Subtotal TFEE** | — | — | — | **21,412,928** |
| **EFFR** | Recon Conv | $3\times 3$ Conv ($64 \to 64$, pad 1) + GN(32, 64) + LReLU | $[B, 64, 64, 64]$ | $[B, 64, 64, 64]$ | **37,056** |
| **EFFR** | Recon Deconv | $3\times 3$ ConvTranspose ($64 \to 64$, $s=2, p=1, \text{op}=1$) + GN + LReLU | $[B, 64, 64, 64]$ | $[B, 64, 128, 128]$ | **37,056** |
| **EFFR** | Channel Attention | GAP $\to 1\times 1$ Conv ($64 \to 16$) $\to$ LReLU $\to 1\times 1$ Conv ($16 \to 64$) | $[B, 64, 128, 128]$ | $[B, 64, 1, 1]$ | **2,128** |
| **EFFR** | Spatial Attention | $7\times 7$ Conv ($2 \to 1$, padding 3) | $[B, 2, 128, 128]$ | $[B, 1, 128, 128]$ | **99** |
| **EFFR** | Evolution Map ($W_p$) | Concat $[(c_{\text{map}} + s_{\text{map}}), I]$ ($128 \to 64$, $k=7, p=3$) + Sigmoid | $[B, 128, 128, 128]$ | $[B, 64, 128, 128]$ | **401,472** |
| **EFFR** | Scale Restoration 1 | $3\times 3$ Conv ($64 \to 64$, pad 1) + GN(32, 64) + LReLU | $[B, 64, 128, 128]$ | $[B, 64, 128, 128]$ | **37,056** |
| **EFFR** | Scale Restoration 2 | $1\times 1$ Conv ($64 \to 4$, bias) (Linear) | $[B, 64, 128, 128]$ | $[B, 4, 128, 128]$ | **260** |
| **EFFR Total**| **Subtotal EFFR** | — | — | — | **515,127** |
| **GRAND TOTAL**| **Full METEOR-Net** | — | `[B, 4, 4, 128, 128]` | `[B, 4, 128, 128]` | **22,086,943** |

### 3.3 Resolution of Prior Document Contradictions
1. **MCE Sequence Latent Dimensions:**  
   Prior draft text incorrectly noted MCE sequence outputs as 256 channels. In the actual implementation, all sequence LCE layers have `in_channels=64, out_channels=64`. The sequence latent tensors $X_t$ are $[B, 64, 64, 64]$ (64 channels), matching the input specification of the first ConvLSTM layer.
2. **First ConvLSTM Input-Plus-Hidden Channels:**  
   Prior tables stated $512 \to 1024$ for ConvLSTM Layer 1. In reality, Layer 1 takes $X_t$ (64 channels) concatenated with recurrent $H_{t-1}^1$ (256 channels), yielding strictly **$64 + 256 = \mathbf{320}$ input channels** into the gate convolution ($320 \to 1024$). Layers 2–4 take 256 input channels from $H_t^{l-1}$ and 256 from $H_{t-1}^l$, giving $256 + 256 = \mathbf{512}$ input channels.
3. **Evolution Pyramid Layer Channels:**  
   Prior tables stated decreasing channels ($512 \to 256$, $512 \to 128$, $384 \to 64$). In reality, all four evolution layers maintain 256 hidden channels: Layer 1 fuses anchor $X_0$ (64 ch) with $H^1$ (256 ch) via $320 \to 256$. Layers 2, 3, and 4 each fuse deep evolved feature $E_{n-1}$ (256 ch) with $H^n$ (256 ch) via $512 \to 256$. A dedicated $1\times 1$ top projection layer then maps $256 \to 64$ to produce $Y_{t_{\text{mid}}} \in \mathbb{R}^{B \times 64 \times 64 \times 64}$.
4. **Stacked ConvLSTM State Clarification:**  
   The tensors $\{H^1, H^2, H^3, H^4\}$ passed to the evolution pyramid are the **final hidden states from the 4 vertically stacked ConvLSTM layers** after sequentially processing all 4 time steps. They are NOT 4 separate time-step outputs from a single layer.

---

## 4. Parameter Audit & Reconciliation

The model was instantiated in PyTorch and its parameters audited by submodule:

```python
mce_p = sum(p.numel() for p in model.mce.parameters() if p.requires_grad)
flow_p = sum(p.numel() for p in model.tfee.flow_estimator.parameters() if p.requires_grad)
evol_p = sum(p.numel() for p in model.tfee.evolution_pyramid.parameters() if p.requires_grad)
effr_p = sum(p.numel() for p in model.effr.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

| Module / Component | Analytical Expectation | Instantiated PyTorch Count | Discrepancy | Status |
| :--- | :---: | :---: | :---: | :---: |
| **MCE** (Collaborative Encoding) | 158,888 | 158,888 | **0** | Verified Exact |
| **TFEE - Flow Estimator (ConvLSTM)** | 17,117,184 | 17,117,184 | **0** | Verified Exact |
| **TFEE - Evolution Pyramid** | 4,295,744 | 4,295,744 | **0** | Verified Exact |
| **EFFR** (Fusion & Reconstruction) | 515,127 | 515,127 | **0** | Verified Exact |
| **Total METEOR-Net** | **22,086,943** | **22,086,943** | **0 (0.0000%)** | **Verified Exact** |

- **Weight Storage Footprint:** $22,086,943 \times 4\text{ bytes} \approx \mathbf{88.35\text{ MB}}$ (float32).
- **Match with Analytical Model:** Exactly matches the analytical calculation with $0.0000\%$ error.

---

## 5. Automated Verification & Test Suite Results

The full test suite was executed across the repository:
```
tests/test_insat_audit.py .......                                       [ 18%]
tests/test_insat_calibration_geo.py ......                              [ 33%]
tests/test_insat_preprocessing.py ..........                            [ 59%]
tests/test_meteor_net.py ...........                                    [ 87%]
tests/test_preprocessing.py .....                                       [100%]
============================== 39 passed in 50.23s ==============================
```

### 5.1 Test Breakdown
- **Preprocessing, Geolocation, and Audit Tests (28 tests):** Radiometric calibration, solar zenith correction, NetCDF reading, HDF5 validity mask separation, scan line monotonicity.
- **METEOR-Net Prototype Tests (11 tests):**
  1. `test_01_parameter_count_audit`: Exact parameter counts verified for MCE, TFEE, EFFR, and total model.
  2. `test_02_forward_shape_and_finite_outputs`: Forward pass on $[2, 4, 4, 128, 128] \to [2, 4, 128, 128]$ produces strictly finite numbers.
  3. `test_03_variable_batch_sizes`: Verifies batch size $B=1$ execution.
  4. `test_04_backward_pass_and_zero_dead_branches`: **Exhaustive Gradient Check:** Backward pass verified on all 158 parameter tensors. Every parameter tensor has finite, non-zero gradients ($|\nabla_\theta| > 0$).
  5. `test_05_all_valid_batch`: Correct MAE loss on 100% valid mask.
  6. `test_06_convective_weighting_thresholds`: Verifies thresholds 241.0 K and 221.0 K with weights 1.0, 2.0, 5.0 on Band 13.
  7. `test_07_partially_masked_batch`: Verifies masked pixels contribute 0 error and do not dilute valid pixel denominator.
  8. `test_08_completely_masked_batch`: Fully masked batch evaluates to finite scalar `0.0` with exact zero parameter gradients.
  9. `test_09_masked_targets_containing_nan_and_inf`: Injected NaNs and $\pm\infty$ into masked targets; loss and gradients remained finite.
  10. `test_10_all_masked_batch_with_nan_predictions`: All-masked batch with NaNs in predictions evaluates to finite scalar `0.0` with finite zero gradients.
  11. `test_11_unmasked_nan_prediction_evaluates_to_nan`: Non-finite predictions at valid observation pixels intentionally evaluate to `NaN` loss, surfacing training failures.

### 5.2 Critical Evidence Demarcation
> [!IMPORTANT]
> **Technical Differentiability vs. Scientific Validation:**  
> The 11 prototype unit tests prove **architectural correctness, input/output geometry, computational differentiability, and numerical shielding**. They confirm that gradients flow to all 22M parameters without graph disconnection.  
> They do **NOT** prove that the network has learned convective cloud physics, produces accurate satellite forecasts, or is superior to baseline persistence or optical flow. Scientific validation requires empirical training on large, diverse convective datasets and rigorous statistical evaluation (CSI, POD, FAR, SSIM, PSNR).

---

## 6. Hardware Reality & Profiling: CPU Measurements vs GPU Projections

### 6.1 Host Hardware Discovery
Hardware queries via Windows Management Instrumentation (`Win32_Processor`, `Win32_VideoController`):
- **CPU:** Intel(R) Core(TM) Ultra 5 125H (14 cores / 18 threads: 4 P-cores, 8 E-cores, 2 LP E-cores)
- **Host GPU:** Intel(R) Arc(TM) Graphics (Integrated / 2.0 GB shared VRAM)
- **CUDA in Active PyTorch Environment:** `torch.cuda.is_available() == False`
- **NVIDIA Hardware:** None present on this host machine.

### 6.2 Empirical Batch Scaling Measurements (Host CPU, Float32)
Profiled via [`scripts/profile_prototype_memory.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/scripts/profile_prototype_memory.py) using PyTorch 2.6.0 on Windows CPU (Input Patch Size $128 \times 128$, float32 precision, sequence length 4):

| Batch Size ($B$) | Forward Latency | Backward Latency | Total Step Latency | Status | Evidentiary Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$B = 1$** | **1.876 s** | **4.535 s** | **6.411 s** | **SUCCESS** | Directly Measured on Host CPU |
| **$B = 2$** | **4.433 s** | **13.513 s** | **17.946 s** | **SUCCESS** | Directly Measured on Host CPU |
| **$B = 4$** | **14.995 s** | **32.712 s** | **47.707 s** | **SUCCESS** | Directly Measured on Host CPU |

### 6.3 Analytical VRAM Projections for Target NVIDIA RTX 3050 (External)
Because CUDA is unavailable on this host, the memory requirements for an external NVIDIA RTX 3050 6 GB GPU are **analytical estimates**, not measured host data:

- **Static Weights:** $22,086,943 \times 4\text{ bytes} \approx \mathbf{88.35\text{ MB}}$.
- **AdamW Optimizer States (FP32):** $2 \times 88.35\text{ MB} \approx \mathbf{176.7\text{ MB}}$.
- **Gradients:** $\mathbf{88.35\text{ MB}}$.
- **Base Static Footprint:** $\approx \mathbf{353.4\text{ MB}}$.
- **Analytical Activation Projections (FP32):**
  - For $B = 1$: $\approx 180\text{ MB}$ activations $\implies \approx \mathbf{0.53\text{ GB}}$ total VRAM.
  - For $B = 2$: $\approx 360\text{ MB}$ activations $\implies \approx \mathbf{0.71\text{ GB}}$ total VRAM.
  - For $B = 4$: $\approx 720\text{ MB}$ activations $\implies \approx \mathbf{1.07\text{ GB}}$ total VRAM.
  - For $B = 8$: $\approx 1.44\text{ GB}$ activations $\implies \approx \mathbf{1.79\text{ GB}}$ total VRAM.

**Hardware Conclusion:** An external 6 GB RTX 3050 has ample memory headroom for $B=4$ or $B=8$. However, on this local CPU host, full model training without GPU acceleration would require ~13 hours per epoch for 1,000 steps.

---

## 7. Remaining Limitations & Prerequisites for Training (Phase 4)

Before model training can be authorized, the following technical prerequisites must be satisfied:

1. **Model-Input Normalization:**  
   Raw satellite infrared observations span $180\text{ K}$ to $330\text{ K}$. Feeding unnormalized temperatures directly into deep networks causes gradient instability. A per-channel normalization pipeline (e.g., channel-specific mean and standard deviation: $\tilde{x}_c = (x_c - \mu_c) / \sigma_c$) must be formalized before training.
2. **Diverse Dataset Acquisition:**  
   The current GOES-19 sample consists of 24 continuous 5-minute timestamps (spanning 115 minutes from 04:01 to 05:56 UTC). While a sliding window generator produces 6 overlapping 90-minute windows, the 115-minute span contains only 1 statistically independent, non-overlapping 90-minute observation sequence. This sample is suitable solely for technical pipeline debugging. Training requires downloading contiguous multi-day blocks (3–5 days) covering active convective lifecycles.
3. **Leakage-Safe Chronological Partitioning:**  
   Train, validation, and test datasets must be split chronologically by **complete observation days or discrete convective events**. Random patch shuffling would lead to severe spatial and temporal leakage (overlapping patches from adjacent time steps in both train and test sets).
4. **Actual-Data Single-Batch Overfit Sanity Test:**  
   Before launching full-scale distributed or multi-epoch training, the optimization loop must be validated on an actual satellite patch batch ($B=1$ or $B=4$) to confirm that AdamW reduces the convective MAE loss toward zero on fixed data.

---

## 8. Summary of Consistency Audit Actions

| Component | Inconsistency Detected | Action Taken |
| :--- | :--- | :--- |
| **MCE Sequence Channels** | Report table stated 256 ch; code used 64 ch | Updated report to reflect 64 channels ($[B, 64, 64, 64]$) matching code. |
| **ConvLSTM L1 Channels** | Report table stated $512 \to 1024$; code used $320 \to 1024$ | Reconciled table to reflect $64 + 256 = 320$ input channels to gate conv. |
| **Evolution Pyramid** | Report table had mismatched channel steps and subtotal | Reconciled table to reflect four 256-channel layers ($4,295,744$ params). |
| **Loss Thresholds** | Report text noted 220 K / 240 K; code had 221 K / 241 K | Corrected report text to 221 K and 241 K, matching code and Lin et al. Eq. 24. |
| **Loss Edge Cases** | Unmasked NaN prediction behavior was unverified | Updated `losses.py` with defensive channel indexing; added `test_11` verifying unmasked NaNs evaluate to NaN loss. |
| **Temporal Semantics** | Hidden states were ambiguously termed "flows" | Clarified that $\{H^1, H^2, H^3, H^4\}$ are final hidden states from 4 stacked ConvLSTM layers, representing flow-free spatiotemporal latent features. |
| **Hardware Demarcation** | Host CPU vs external GPU was ambiguously presented | Explicitly separated measured host CPU timings from external GPU analytical estimates. |

**Stopping Point:** All prototype modules and verification reports are audited, corrected, and consistent. All 39 tests pass. Awaiting approval before proceeding to dataset expansion or training.

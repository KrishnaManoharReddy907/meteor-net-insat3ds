# Phase 3.5.1 — Final Sanity-Test Preflight Audit

**Date:** 2026-10-22  
**Status:** Complete — Overlap Calculations Corrected, Convective Patch Deficit Identified, Preflight Readiness Reconciled  
**Target Specification Documents:**  
- [`docs/phase3_5_training_readiness_audit.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase3_5_training_readiness_audit.md)  
- [`docs/phase3_architecture_and_data_readiness.md`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/docs/phase3_architecture_and_data_readiness.md)  
**Processed Sample Analyzed:** [`outputs/processed_sample/sample_sequence_window00.npz`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/processed_sample/sample_sequence_window00.npz)  

---

## 1. Executive Summary & Scope Conformance

This preflight audit examines the technical and meteorological assumptions underlying the proposed single-batch overfit sanity test. In strict adherence to project safety boundaries:
- **Zero data was downloaded.**
- **Zero model training or optimization loops were executed.**
- **No raw files were modified or overwritten.**
- **No changes to architecture or implementation code were made without authorization.**

### Key Audit Findings
1. **Correction of Temporal Overlap Analysis:**  
   The claim that adjacent windows share *"80% of their frames"* was mathematically inaccurate. Adjacent sliding windows ($5\text{-minute shift}$) share **exactly 0 discrete timestamps**. However, their 90-minute observation intervals overlap by **85 out of 90 minutes (94.4%)**, establishing strong meteorological dependence and precluding their use as independent validation data.
2. **Convective Deficit in Proposed Patch:**  
   The proposed patch in [`outputs/processed_sample/sample_sequence_window00.npz`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/processed_sample/sample_sequence_window00.npz) (rows `100:228`, cols `200:328`) is **entirely non-convective**. It contains **0.00% pixels below 241 K** and **0.00% pixels below 221 K** (minimum temperature: $272.07\text{ K}$). An overfit test on this patch would leave the convective loss penalties ($\beta=2.0, \gamma=5.0$) completely unexercised.
3. **Overfit-Test Readiness & Robust Criteria:**  
   The prototype requires an explicit normalization/de-normalization wrapper before training. Acceptance criteria requiring "strictly monotonic loss decrease" and a "fixed $< 3\text{ K}$ threshold" are rejected as fragile under AdamW momentum; robust relative criteria ($\ge 75\%$ loss reduction, block-level decay, zero NaNs, parameter updates) are established.

---

## 2. Item 1: Exact Temporal Overlap and Frame Intersection Audit

### 2.1 Complete Timestamp Inventory Across the Six Sliding Windows
Each window samples 4 inputs at $[-60, -30, 0, +30]\text{ min}$ relative to an anchor $t$, plus 1 midpoint target at $+15\text{ min}$:

| Window ID | Anchor ($t$) | Input 1 ($t-60$) | Input 2 ($t-30$) | Input 3 ($t$) | Target ($t+15$) | Input 4 ($t+30$) | Total Span |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Window 00** | `05:01 UTC` | `04:01` | `04:31` | `05:01` | **`05:16`** | `05:31` | `04:01 – 05:31` (90 min) |
| **Window 01** | `05:06 UTC` | `04:06` | `04:36` | `05:06` | **`05:21`** | `05:36` | `04:06 – 05:36` (90 min) |
| **Window 02** | `05:11 UTC` | `04:11` | `04:41` | `05:11` | **`05:26`** | `05:41` | `04:11 – 05:41` (90 min) |
| **Window 03** | `05:16 UTC` | `04:16` | `04:46` | `05:16` | **`05:31`** | `05:46` | `04:16 – 05:46` (90 min) |
| **Window 04** | `05:21 UTC` | `04:21` | `04:51` | `05:21` | **`05:36`** | `05:51` | `04:21 – 05:51` (90 min) |
| **Window 05** | `05:26 UTC` | `04:26` | `04:56` | `05:26` | **`05:41`** | `05:56` | `04:26 – 05:56` (90 min) |

### 2.2 Exact Frame Intersections (Pairwise Matrix)
Calculating the exact set intersection of discrete timestamps $\{t_1, t_2, t_3, t_4, t_{\text{target}}\}$ between all pairs of windows:

$$\begin{array}{c|cccccc}
\text{Shared Frames} & \mathbf{W_0} & \mathbf{W_1} & \mathbf{W_2} & \mathbf{W_3} & \mathbf{W_4} & \mathbf{W_5} \\
\hline
\mathbf{W_0} & 5 & 0 & 0 & 2 & 0 & 0 \\
\mathbf{W_1} & 0 & 5 & 0 & 0 & 2 & 0 \\
\mathbf{W_2} & 0 & 0 & 5 & 0 & 0 & 2 \\
\mathbf{W_3} & 2 & 0 & 0 & 5 & 0 & 0 \\
\mathbf{W_4} & 0 & 2 & 0 & 0 & 5 & 0 \\
\mathbf{W_5} & 0 & 0 & 2 & 0 & 0 & 5 \\
\end{array}$$

#### Explicit Timestamp Intersections:
- **$W_0 \cap W_1$ (Adjacent):** **0 shared frames** ($\emptyset$).
- **$W_1 \cap W_2$ (Adjacent):** **0 shared frames** ($\emptyset$).
- **$W_2 \cap W_3$ (Adjacent):** **0 shared frames** ($\emptyset$).
- **$W_3 \cap W_4$ (Adjacent):** **0 shared frames** ($\emptyset$).
- **$W_4 \cap W_5$ (Adjacent):** **0 shared frames** ($\emptyset$).
- **$W_0 \cap W_3$ (Shifted by 15 min):** **2 shared frames** (`05:16` and `05:31`).  
  *Role reversal:* `05:16` is Target in $W_0$, but Input 3 (Anchor) in $W_3$. `05:31` is Input 4 in $W_0$, but Target in $W_3$.
- **$W_1 \cap W_4$ (Shifted by 15 min):** **2 shared frames** (`05:21` and `05:36`).
- **$W_2 \cap W_5$ (Shifted by 15 min):** **2 shared frames** (`05:26` and `05:41`).

### 2.3 Distinction: Discrete Frame Intersections vs. Physical Meteorological Dependence
- **Correction of Previous Claim:**  
  The prior statement that *"adjacent windows share 80% of their frames"* was based on an assumption of consecutive 1-step sampling. Because the sequence uses non-consecutive offsets ($[-60, -30, 0, +30]$) sampled at a 5-minute shift, **adjacent windows share zero identical observation timestamps**.
- **Physical Continuous Overlap:**  
  However, Window 00 spans `04:01 to 05:31 UTC` and Window 01 spans `04:06 to 05:36 UTC`. The physical observation periods overlap continuously for **85 out of 90 minutes (94.4% temporal overlap)**.
- **Meteorological Consequence:**  
  The atmosphere does not reset between 5-minute scans. Cloud formations, convective updrafts, and moisture fields in Window 01 are continuous advections of those in Window 00. Therefore, the 6 sliding windows remain **highly collinear and statistically dependent**, representing a single 115-minute weather event. Independent validation or test partitioning remains impossible on this sample.

---

## 3. Item 2: Convective Audit of Proposed Patch vs. Western US Storm Sector

### 3.1 Quantitative Audit of Proposed Overfit Patch
The overfit test proposed extracting a $128 \times 128$ patch at `rows: [100:228]`, `cols: [200:328]` from [`outputs/processed_sample/sample_sequence_window00.npz`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/processed_sample/sample_sequence_window00.npz).

An empirical analysis of calibrated Band 13 ($10.33\,\mu\text{m}$) brightness temperatures was executed:

| Metric | Proposed Patch (`[100:228, 200:328]`) | Full `sample_sequence_window00.npz` | Western US Storm Sector (Phase 3 Audit) |
| :--- | :---: | :---: | :---: |
| **Spatial Bounds** | Fixed $128 \times 128$ subgrid | $453 \times 740$ (Gulf / SE US) | $585 \times 630$ (Western US / Pacific) |
| **Geographic Region** | Gulf of Mexico / Ambient Sea | Gulf / Florida / Southeast US | Arizona / New Mexico / Pacific Storm |
| **Minimum Brightness Temp ($K$)** | **$272.07\text{ K}$** | **$264.64\text{ K}$** | **$210.74\text{ K}$** (Deep Convective Core) |
| **Mean Brightness Temp ($K$)** | $276.90\text{ K}$ | $282.12\text{ K}$ | $251.90\text{ K}$ |
| **Maximum Brightness Temp ($K$)** | $289.59\text{ K}$ | $296.47\text{ K}$ | $292.15\text{ K}$ |
| **Fraction $< 241\text{ K}$ (Moderate Conv)** | **$0.00\%$ ($0\text{ pixels}$)** | **$0.00\%$ ($0\text{ pixels}$)** | **$36.79\%$ ($135,604\text{ pixels}$)** |
| **Fraction $< 221\text{ K}$ (Deep Core)** | **$0.00\%$ ($0\text{ pixels}$)** | **$0.00\%$ ($0\text{ pixels}$)** | **$4.25\%$ ($15,652\text{ pixels}$)** |
| **Convective Loss Weights Active** | **Only $\alpha = 1.0$** | **Only $\alpha = 1.0$** | **All Active ($\alpha=1.0, \beta=2.0, \gamma=5.0$)** |

### 3.2 Visualization and Contour Audit
A contour visualization was generated and archived at:
`outputs/preprocessing/proposed_overfit_patch_convective_audit.png` (and replicated to the artifact directory at [`proposed_overfit_patch_convective_audit.png`](file:///C:/Users/Krishna/.gemini/antigravity-ide/brain/f208b45a-6641-4e03-a6a2-b07b756a699d/proposed_overfit_patch_convective_audit.png)).

**Contour Evaluation Findings:**
- Neither the $241.0\text{ K}$ nor the $221.0\text{ K}$ contour appears on the proposed patch because every pixel has $T_B \ge 272.07\text{ K}$.
- The patch represents warm sea surface and low-level boundary layer moisture ($272 - 289\text{ K}$).
- **Impact on Loss Verification:** In `MaskAwareConvectiveMAELoss`, because $T_B \ge 241.0\text{ K}$ everywhere, the weight map is identically $w \equiv 1.0$. The convective weighting mechanism ($\beta=2.0, \gamma=5.0$) designed to penalize cold cloud tops would be **completely untested and dormant** during an overfit test on this patch.

### 3.3 Root Cause of the Patch Deficit
In Phase 2B, `scripts/run_preprocessing.py` was executed with regional bounds targeting the Gulf of Mexico / Southeast US ($25^\circ-35^\circ\text{N}, 80^\circ-95^\circ\text{W}$). At 05:00 UTC on Jan 1, 2026, that region was calm and cloud-free.  
The actual synoptic winter storm system on that day was positioned over the **Southwest US / Arizona / New Mexico** (centered near $34.5^\circ\text{N}, 112.8^\circ\text{W}$, with raw minimum temperature of **$206.93\text{ K}$** and $353,423$ pixels below $241\text{ K}$ across full CONUS).

### 3.4 Concrete Preflight Recommendation
> [!IMPORTANT]
> **Actionable Recommendation:**  
> Do **NOT** run the overfit sanity test on the Gulf crop in `sample_sequence_window00.npz`.  
> When the overfit test is authorized, extract the $128 \times 128$ patch from the **Western US Convective Sector** ($25^\circ-38^\circ\text{N}, 105^\circ-120^\circ\text{W}$, raw file data already present locally in `data/goes19/2026/001/`). In that sector, $85.9\%$ of patches contain convective pixels ($< 241\text{ K}$) and $51.6\%$ contain deep cores ($< 221\text{ K}$), ensuring that $\beta=2.0$ and $\gamma=5.0$ are actively verified during optimization.

---

## 4. Item 3: Overfit-Test Readiness & Code Requirements

### 4.1 Required Normalization Code Wrapper
The model implementation [`src/models/meteor_net.py`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/src/models/meteor_net.py) does not contain input normalization. To execute the test cleanly without modifying the core model class, a standardized wrapper must be added to the test script:

```python
class ChannelStandardizer:
    """Channel-wise standardization for model inputs and outputs."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.view(1, 1, -1, 1, 1)  # Broadcastable to [B, T, C, H, W]
        self.std = std.view(1, 1, -1, 1, 1)
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)
        
    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, C, H, W]
        mean_4d = self.mean.squeeze(1) # [1, C, 1, 1]
        std_4d = self.std.squeeze(1)
        return y * (std_4d + 1e-6) + mean_4d
```

### 4.2 Step-by-Step Training-Step Flow
During each optimization step:
1. Input $X_{\text{phys}} \in \mathbb{R}^{1 \times 4 \times 4 \times 128 \times 128}$ (in Kelvin) is normalized: $\tilde{X} = \text{standardizer.normalize}(X_{\text{phys}})$.
2. Forward pass: $\tilde{Y} = \text{model}(\tilde{X})$ produces normalized prediction in $\mathbb{R}^{1 \times 4 \times 128 \times 128}$.
3. De-normalization: $\hat{Y}_{\text{phys}} = \text{standardizer.denormalize}(\tilde{Y})$ restores physical Kelvin.
4. Loss computation: $\mathcal{L} = \text{loss\_fn}(\hat{Y}_{\text{phys}}, Y_{\text{phys}}, M)$.
   - `loss_fn` inspects Band 13 of $Y_{\text{phys}}$ at $241.0\text{ K}$ and $221.0\text{ K}$ to compute convective weights.
   - Gradients backpropagate through de-normalization and the entire network.

### 4.3 Critique & Redefinition of Acceptance Criteria

#### Why Prior Criteria Were Fragile:
1. **"Strictly Monotonic Loss Reduction":**  
   AdamW maintains first and second momentum buffers ($\beta_1=0.9, \beta_2=0.999$). Momentum inertia, group normalization statistics, and curvature adjustments routinely produce minor step-to-step variations (e.g., loss may rise from $4.10\text{ K}$ to $4.14\text{ K}$ before dropping to $3.80\text{ K}$). Demanding strictly monotonic decrease on every single step is an unrobust criterion that would falsely fail valid optimization runs.
2. **"Fixed Final Loss Below 3 K":**  
   If an overfit patch is deeply convective, Band 13 pixels below $221\text{ K}$ carry a $5\times$ penalty weight ($\gamma=5.0$). An initial loss starting at $35\text{ K}$ may drop by $80\%$ down to $7\text{ K}$ in 100 steps. A fixed threshold of $3\text{ K}$ is arbitrary and does not scale with convective weighting.

#### Recommended Reproducible Acceptance Criteria:
1. **Relative Loss Reduction:**  
   The mean loss over the final 10 steps must decrease by at least **$75\%$** relative to the initial 5-step baseline:
   $$\frac{1}{10}\sum_{k=91}^{100} \mathcal{L}_k \le 0.25 \times \left( \frac{1}{5}\sum_{k=1}^5 \mathcal{L}_k \right)$$
2. **Block-Level Monotonic Trend:**  
   The average loss across successive 20-step blocks must strictly decrease:
   $$\bar{\mathcal{L}}_{1-20} > \bar{\mathcal{L}}_{21-40} > \bar{\mathcal{L}}_{41-60} > \bar{\mathcal{L}}_{61-80} > \bar{\mathcal{L}}_{81-100}$$
3. **Numerical Health:**  
   Zero occurrences of `NaN`, $\pm\infty$, or unhandled zero-gradient exceptions across all 100 steps.
4. **Parameter Plasticity:**  
   All parameter groups must show measurable weight updates:
   $$\frac{\|\theta_{100} - \theta_0\|}{\|\theta_0\|} > 1\times 10^{-3} \quad \text{for MCE, TFEE, and EFFR}$$

### 4.4 Target Frame Confirmation
- In Window 00, the target frame timestamp is `2026-01-01T05:16:19 UTC`.
- Inspection of the raw file `OR_ABI-L2-CMIPC-M6C13_G19_s20260010516193...nc` confirms this is the **real physical GOES-19 CONUS scan**, not a synthetic or temporally interpolated target.

---

## 5. Summary of Audit Updates & Deliverables

| Item | Status in Previous Draft | Audited & Corrected Status |
| :--- | :--- | :--- |
| **Window Frame Overlap** | Stated "adjacent windows share 80% of frames" | **Corrected:** Adjacent windows share **0 discrete frames**. Overlap is continuous in time (85/90 min, 94.4%), confirming meteorological dependence. |
| **Proposed Patch Convection** | Assumed convective (cold cloud) | **Identified Deficit:** Proposed Gulf patch has **0% pixels < 241 K** (Min: 272.1 K). Convective weights inactive. |
| **Overfit Patch Recommendation** | Proposing Gulf `[100:228, 200:328]` | **Recommended:** Shift extraction to **Western US storm sector** ($25^\circ-38^\circ\text{N}, 105^\circ-120^\circ\text{W}$), which has 36.8% convective pixels. |
| **Acceptance Criteria** | Strictly monotonic; $< 3\text{ K}$ fixed | **Replaced:** $\ge 75\%$ relative drop from initial 5-step baseline, 20-step block decay, zero NaNs, active parameter plasticity. |
| **Normalization Wrapper** | Unspecified in code | **Specified:** Detailed `ChannelStandardizer` wrapper for input normalization and physical de-normalization. |

**Audit Conclusion:** Preflight verification is complete. The system is ready to construct the overfit script upon authorization. No training or data acquisition was initiated.

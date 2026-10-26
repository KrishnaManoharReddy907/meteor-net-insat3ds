# METEOR-Net Google Colab T4 Migration & Execution Guide

---

## 1. Overview & Architectural Principles

This guide establishes the reproducible workflow for running METEOR-Net on **Google Colab (NVIDIA T4 GPU)** while maintaining **Antigravity (local Windows workstation)** as your primary development environment.

### Core Safeguards:
1. **Primary Workspace Invariance:** Local source code, tests, and configuration remain authoritative in Antigravity.
2. **Local Experiment Immutability:** All Phase 7B CPU baseline artifacts in [`outputs/phase7b/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/phase7b/) remain **read-only and untouched**.
3. **Dedicated GPU Experiment Partition:** Colab executions output strictly to [`outputs/colab_t4_smoke_test/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/colab_t4_smoke_test/) and [`outputs/colab_t4_phase8/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/colab_t4_phase8/).
4. **Strict Meteorological Event Isolation:**
   * **Train:** Event 1 (Jan 1, 2026) and Event 2 (May 20, 2026) — 6,480 patches.
   * **Validation:** Event 3 (July 19, 2026) — 3,240 patches.
   * **Quarantined Test:** Event 4 (Sept 7, 2026) — 3,240 patches (strictly excluded from gradient updates and model selection).
5. **Frozen Normalization Invariance:** Normalization parameters (Means: `[248.6812, 279.7365, 279.0453, 276.9893]` K, Stds: `[10.2850, 17.9001, 18.2796, 17.6614]` K) are loaded exclusively from [`outputs/normalization/frozen_global_train_stats.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/normalization/frozen_global_train_stats.json) (SHA-256 `71ef64...`) and are never refit.
6. **Numerical Precision:** Full FP32 is maintained across all GPU tensor calculations to ensure exact mathematical agreement with the physical Kelvin baseline.
7. **Periodic Live Drive Checkpoint Mirroring:** `latest_checkpoint.pt`, `best_model.pt`, and `training_history.json` are copied to Google Drive at every epoch boundary during training using safe `.tmp` write and atomic replacement.
8. **Dual-Key Training Execution Guard:** Stage 7 (Full Training) cannot be launched accidentally by "Run All". It requires both setting `CONFIRM_LAUNCH_TRAINING = True` and verified completion of Stage 6 (GPU Smoke Test) in the active session.

---

## 2. Verified Storage & Dataset Metrics

| Component | Files | Physical Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Uncompressed NetCDF Dataset** | 1,152 files | **3.974 GB** (4,267,041,370 bytes) | Indexed in `data_archive_manifest.json` |
| **Compressed Archive (`data_goes19_4events.tar.gz`)** | 1 archive | **3.905 GB** (4,192,771,771 bytes) | `d640cfdbaf09e9add5e8bfb0d82ddb63a97c30f4c09807e4ded222ccf0975aa5` |
| **Source Bundle Zip (`meteor_net_source_bundle.zip`)** | 59 files | **0.51 MB** (538,133 bytes) | `b5f5a0510a116a24341825a58681146109d8bc27631491418f65723d43deb303` |
| **Active Colab Run Storage (Phase 8)** | Checkpoints + History | **~0.76 GB** (253 MB per checkpoint) | Generated during training |
| **Total Google Drive Storage Required** | — | **~4.67 GB** | Fits within Google Drive free tier (15 GB) |

---

## 3. Step-by-Step Migration Instructions

### Step 1: Package the Dataset & Source Locally
In Antigravity PowerShell, run:
```powershell
python scripts/archive_dataset.py
python scripts/package_source_bundle.py
```
* **Dataset archive created:** `data_goes19_4events.tar.gz` (3.905 GB, SHA-256: `d640cfdbaf09e9add5e8bfb0d82ddb63a97c30f4c09807e4ded222ccf0975aa5`).
* **Dataset manifest generated:** [`outputs/full_acquisition/data_archive_manifest.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/full_acquisition/data_archive_manifest.json) containing individual sizes and SHA-256 checksums for all 1,152 files.
* **Source bundle created:** `meteor_net_source_bundle.zip` (0.51 MB, 538,133 bytes, SHA-256: `b5f5a0510a116a24341825a58681146109d8bc27631491418f65723d43deb303`), along with authoritative sidecar checksum `meteor_net_source_bundle.zip.sha256`.

### Step 2: Upload Files to Google Drive
1. In Google Drive (`My Drive`), create a folder named `meteor_net_data`.
2. Upload `data_goes19_4events.tar.gz` into `meteor_net_data/`.
3. *(Recommended)* Upload `meteor_net_source_bundle.zip` and `meteor_net_source_bundle.zip.sha256` into `meteor_net_data/` for instantaneous, offline cryptographic source ingestion.

### Step 3: Open Notebook in Google Colab
1. Upload and open [`notebooks/meteor_net_colab_t4_training.ipynb`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/notebooks/meteor_net_colab_t4_training.ipynb) in Colab.
2. Ensure GPU acceleration is enabled:
   * **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Hardware accelerator: **T4 GPU** $\rightarrow$ Click **Save**.

### Step 4: Execute Verification & GPU Smoke Test
Execute notebook cells in sequence:
* **Stage 1 (Drive Mount):** Authorizes access to Google Drive (`/content/drive`).
* **Stage 2 (Source Ingestion - Two Distinct Verification Methods):**
  * **Route A (Offline Source Bundle on Drive - Recommended):**
    * Verification Method: Cryptographic SHA-256 checksum (`EXPECTED_BUNDLE_SHA256 = "b5f5a0510a116a24341825a58681146109d8bc27631491418f65723d43deb303"`) and authoritative sidecar `meteor_net_source_bundle.zip.sha256`.
    * Completely offline: requires no Git, no network clone, and no GitHub tokens. Extracts verified files directly into `/content/meteor_net`.
  * **Route B (Private GitHub Clone):**
    * Verification Method: Pinned, audited Git commit (`PINNED_GIT_COMMIT = "c273fc0603030d884e5fbb181733ab77080d5339"`).
    * Prompts interactively for a GitHub Personal Access Token (PAT) via `getpass.getpass()`, uses temporary restricted-permission `GIT_ASKPASS` (0600 mode) without exposing token in command line or remote URL, clones the repository, asserts `current_commit == PINNED_GIT_COMMIT`, and scrubs the token and remote URL.
* **Stage 3 (Dataset Ingestion & Deep Verification - 100% Coverage):**
  * Verifies `data_goes19_4events.tar.gz` SHA-256 against `d640cfdb...` before unpacking.
  * Extracts archive to the local fast VM disk (`/content/meteor_net/data/`).
  * Audits **all 1,152 extracted NetCDF files** against `data_archive_manifest.json` (verifying file presence, exact byte sizes, and cryptographic SHA-256 hashes for 100% of files; asserts no unexpected extra NetCDF files exist).
* **Stage 4 (Dependencies & Hardware):** Installs pinned utilities (`requirements.txt`) without touching Colab's pre-installed CUDA PyTorch, then inspects GPU VRAM and runs `nvidia-smi`.
* **Stage 5 (Provenance & Quarantine):** Verifies split manifest (`6a173d...`), frozen normalization (`71ef64...`), acquisition inventory (`26cf30...`), and confirms Event 4 (Sept 7, 2026) is quarantined as test.
* **Stage 6 (GPU Smoke Test):**
  * Runs `python scripts/run_gpu_smoke_test.py --device cuda --output_dir outputs/colab_t4_smoke_test`.
  * Tests: CUDA memory allocation, 22,086,943 model parameters, forward pass, autograd backward pass, AdamW step, and atomic checkpoint serialization under 15 seconds.
  * Sets in-kernel session authorization flags (`GLOBAL_SMOKE_TEST_PASSED = True`).

### Step 5: Full Controlled Training (Next Experiment: Natural-Ratio Single-Pass Full Training)

> [!IMPORTANT]
> **NEXT EXPERIMENT: FULL-DATA TRAINING:**
> The primary, authorized next experiment trains across all 6,480 training patches using `--sampler_mode natural_single_pass` and full-event validation on all 3,240 validation patches (`--val_mode full_event`).

1. Open Stage 7 in the notebook.
2. Ensure Stage 6 (GPU Smoke Test) passed in the current session.
3. Change the execution guard:
   ```python
   CONFIRM_LAUNCH_TRAINING = True
   ```
4. Run the cell:
   ```bash
   python scripts/train_meteor_net.py \
       --device cuda \
       --output_dir outputs/colab_t4_full_training \
       --drive_sync_dir /content/drive/MyDrive/meteor_net_runs/colab_t4_full_training \
       --sampler_mode natural_single_pass \
       --val_mode full_event \
       --epochs 5 \
       --grad_accum_steps 16 \
       --lr 1e-4 \
       --weight_decay 1e-4 \
       --seed 42 \
       --early_stopping_patience 2
   ```

---

> [!CAUTION]
> **COMPLETED PILOT — DO NOT RERUN:**
> Phase 8 pilot training (100 microbatches/epoch, balanced_pilot) successfully completed 5 epochs on Tesla T4 with best validation loss 9.9753 K. Its command is archived below strictly for reference and reproducibility. **Do not re-execute this command for active training.**
>
> ```bash
> # [ARCHIVED PILOT COMMAND - DO NOT RERUN]
> python scripts/train_meteor_net.py \
>     --device cuda \
>     --output_dir outputs/colab_t4_phase8 \
>     --drive_sync_dir /content/drive/MyDrive/meteor_net_runs/colab_t4_phase8 \
>     --epochs 5 \
>     --microbatches_per_epoch 100 \
>     --grad_accum_steps 2 \
>     --lr 1e-4 \
>     --weight_decay 1e-4 \
>     --seed 42 \
>     --val_patches 25 \
>     --early_stopping_patience 2
> ```

### Step 6: Checkpoint Recovery Guarantees, Resume Preflight & Limitations
* **Recovery Guarantee:** At the completion of each epoch, `latest_checkpoint.pt`, `best_model.pt` (if validation improved), and `training_history.json` are written atomically locally and immediately live-mirrored to Google Drive (`/content/drive/MyDrive/meteor_net_runs/colab_t4_full_training/`).
* **Drive Sync Observability & Risk Logging:** After mirroring, `sync_file_to_drive` explicitly verifies that the destination exists on Drive and that its byte size matches the local file exactly. If Google Drive encounters transient network dropouts or FUSE latency, `sync_file_to_drive` logs a clear `[Drive Sync WARNING]` and explicitly warns that the latest local checkpoint is not yet backed up on Drive and would be lost if the Colab VM terminates before the next sync. Training continues safely on the local NVMe SSD.
* **Resume Preflight:**
  Before launching a resumed run, Stage 8 executes a preflight check:
  1. Verifies CUDA availability on the active runtime.
  2. Deserializes the Drive checkpoint (`torch.load(..., map_location="cpu")`) and verifies that `epoch`, `optimizer_update_count`, `microbatch_count`, and `best_val_loss` are valid.
  3. Displays a preview of the saved state and the exact epoch/microbatch training will resume at.
* **Artifact Preservation on Resume:** When `--resume` is supplied, the training runner checks for existing untrained baseline metrics (`baseline_metrics.json`) and preflight serialization artifacts (`preflight_checkpoint_test.pt`). If found, they are **preserved without re-evaluating or overwriting**, keeping original preflight timestamps and baseline metrics intact.

> [!WARNING]
> **CRITICAL RESTRICTION — PILOT CHECKPOINT CANNOT RESUME FULL-DATA TRAINING:**
> The Phase 8 pilot checkpoint (`outputs/colab_t4_phase8/latest_checkpoint.pt`) must **NEVER** be used to resume full-data training. Resumption across different sampler modes (`balanced_pilot` vs `natural_single_pass`), mismatched microbatch budgets, mismatched gradient accumulation steps, or conflicting manifest hashes is fundamentally invalid and is strictly rejected fail-closed with an explicit error by `verify_checkpoint_compatibility()`. Full-data training must be resumed exclusively using `latest_checkpoint.pt` from `/content/drive/MyDrive/meteor_net_runs/colab_t4_full_training/`.

* **Full-Training Resume Procedure:**
  In Stage 8 of the notebook, set `CONFIRM_RESUME_TRAINING = True` and run the cell:
  ```bash
  python scripts/train_meteor_net.py \
      --device cuda \
      --output_dir outputs/colab_t4_full_training \
      --drive_sync_dir /content/drive/MyDrive/meteor_net_runs/colab_t4_full_training \
      --sampler_mode natural_single_pass \
      --val_mode full_event \
      --epochs 5 \
      --grad_accum_steps 16 \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --seed 42 \
      --early_stopping_patience 2 \
      --resume /content/drive/MyDrive/meteor_net_runs/colab_t4_full_training/latest_checkpoint.pt
  ```
  The runner restores: model weights, optimizer state, scheduler state, global update count, microbatch count, CPU RNG state, CUDA RNG state, Python random state, and training history, safely beginning from microbatch 0 of epoch $E+1$.
* **Limitation:** Incomplete microbatches within an epoch that abort abnormally (e.g. abrupt Colab VM termination before epoch completion) are not checkpointed. The training loop safely re-executes the aborted epoch from microbatch 0.

### Step 7: Returning Artifacts to Antigravity
1. In Stage 9, verify that `latest_checkpoint.pt`, `best_model.pt`, and `training_history.json` exist in `/content/drive/MyDrive/meteor_net_runs/colab_t4_full_training/`.
2. Download or copy the `colab_t4_full_training/` folder to your local Antigravity workspace at:
   [`outputs/colab_t4_full_training/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/colab_t4_full_training/)
3. Inspect `training_history.json` and compare convergence against the completed Phase 8 pilot baseline.

### Step 8: Held-Out Test Evaluation (Stage 10)
1. **Scientific Integrity & Quarantined Split:**
   - The test partition comprises the September 7, 2026 convective event (`event_2026-09-07`), consisting of 54 sequence windows (3,240 patches with stride 64 and patch size 128x128). This event was **strictly quarantined** and excluded from training, validation, and normalization calculation.
2. **Benchmark Baselines:**
   - Evaluates METEOR-Net against **Linear Temporal Interpolation** ($0.5 \times X[t=0] + 0.5 \times X[t=+30]$) and **Persistence** ($t=0$ min and $t=+30$ min) on the exact same valid pixels and physical Kelvin scale.
3. **Overlap-Aware Unique-Pixel Scene Evaluation:**
   - Because patches are extracted with stride 64 and size 128, adjacent patches overlap by 50%. In addition to patch-aggregated metrics, Stage 10 reconstructs the full cropped observation grid ($453 \times 740$) for each sequence, averages overlapping patch predictions, and computes scene MAE/RMSE over unique valid pixels without double-counting.
4. **SSIM & PSNR Guardrails:**
   - SSIM and PSNR are computed exclusively on patches that are 100% valid (`is_fully_valid == True`). Dynamic ranges are derived from frozen training statistics ($6\sigma$: Band 9 = 61.7 K, Band 13 = 107.4 K, Band 14 = 109.7 K, Band 15 = 106.0 K).
5. **Execution Procedure (Separate Configurations for Pilot vs. Full-Training):**
   - **Configuration A: Completed Phase 8 Pilot Evaluation (Immediate / Active Target):**
     Evaluates the completed 5-epoch pilot checkpoint (`best_model.pt`) verified on Drive.
     In Stage 10 of the notebook, set `EVAL_TARGET = "pilot"`, verify preflight output, set `CONFIRM_EVALUATE_TEST = True`, and execute:
     ```bash
     python scripts/evaluate_meteor_net_test.py \
         --checkpoint /content/drive/MyDrive/meteor_net_runs/colab_t4_phase8/best_model.pt \
         --manifest outputs/dataset_splits/splits_manifest.json \
         --stats outputs/normalization/frozen_global_train_stats.json \
         --output_dir outputs/colab_t4_phase8_test_evaluation \
         --drive_sync_dir /content/drive/MyDrive/meteor_net_runs/colab_t4_phase8_test_evaluation \
         --device cuda
     ```
     - Artifacts are saved locally at [`outputs/colab_t4_phase8_test_evaluation/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/colab_t4_phase8_test_evaluation/) and mirrored to Google Drive at `/content/drive/MyDrive/meteor_net_runs/colab_t4_phase8_test_evaluation/`.

   - **Configuration B: Full-Training Held-Out Evaluation (Future Run after full training completes):**
     Once full-dataset training has finished all 5 epochs and saved its final `best_model.pt`:
     Set `EVAL_TARGET = "full_training"`, verify preflight, and execute:
     ```bash
     python scripts/evaluate_meteor_net_test.py \
         --checkpoint /content/drive/MyDrive/meteor_net_runs/colab_t4_full_training/best_model.pt \
         --manifest outputs/dataset_splits/splits_manifest.json \
         --stats outputs/normalization/frozen_global_train_stats.json \
         --output_dir outputs/colab_t4_full_training_test_evaluation \
         --drive_sync_dir /content/drive/MyDrive/meteor_net_runs/colab_t4_full_training_test_evaluation \
         --device cuda
     ```
     - Artifacts are saved locally at [`outputs/colab_t4_full_training_test_evaluation/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/colab_t4_full_training_test_evaluation/) and mirrored to Google Drive at `/content/drive/MyDrive/meteor_net_runs/colab_t4_full_training_test_evaluation/`.

---

## 4. Scientific Training Scope & Configuration: Pilot vs. Full-Dataset Run

### A. Completed Pilot Configuration (Phase 8 Colab T4 Initial Run — COMPLETED PILOT — DO NOT RERUN)
* **Configuration:**
  - Microbatches per Epoch: `100` (`--microbatches_per_epoch 100`)
  - Gradient Accumulation: `2` (`--grad_accum_steps 2`) -> `50` optimizer updates per epoch
  - Total Updates: `250` across 5 mini-epochs
  - Sampled Patches: `50` convective + `50` ambient per epoch = `500` unique patches total (zero replacement across 5 epochs)
  - Validation: `25` fixed patches from Event 3 (July 19, 2026)
  - Result: Completed 5 epochs successfully (Best Val Loss: 9.9753 K).
* **Status:** Archived baseline reference. **Do not re-execute.**

### B. Next Experiment: Full-Dataset Training Scope & Natural-Ratio Single-Pass Sampler (ACTIVE/READY)
* **Dataset Partition Audit:**
  - Total Training Partition: `6,480` patches (Event 1: 3,240, Event 2: 3,240).
  - Genuine Convective Pool: `2,267` patches (34.98%).
  - Ambient Pool: `4,213` patches (65.02%).
  - Validation Partition: `3,240` patches (Event 3: July 19, 2026).
  - Test Partition: `3,240` patches (Event 4: September 7, 2026 — strictly quarantined).
* **Sampler Implementation (`natural_single_pass`):**
  - **Single-Pass Permutation:** Presents all 6,480 eligible training patches exactly once per epoch (zero duplicates, zero omissions within each epoch).
  - **Deterministic Global Shuffle (Decision 1: Option A):** Seeded deterministic global shuffle using `random.Random(seed + epoch)`, yielding identical ordering on rerun and distinct permutations across epochs.
  - **Natural Ratio Preservation:** Preserves the natural ~35:65 convective/ambient distribution without artificial oversampling. Convective weighting is maintained physically by `MaskAwareConvectiveMAELoss` ($\beta=2.0, \gamma=5.0$).
  - **Full-Event Validation (Decision 2: Option B):** Evaluates all `3,240` validation patches from Event 3 at every epoch boundary with progress reporting every 500 patches.
  - **Dynamic Manifest Scaling:** Sampler dynamically queries `len(train_ds)` and manifest metadata, avoiding any hardcoded 6,480 patch counts or 35:65 ratios.
* **Gradient Accumulation & Learning Rate Optimization:**
  - Effective Batch Size: `16` (`--grad_accum_steps 16`)
  - Updates per Epoch: $\lceil 6,480 / 16 \rceil = 405$ optimizer updates per epoch
  - Total Scheduled Updates: $5 \times 405 = 2,025$ updates across 5 epochs
  - Dynamic Normalization: Unbiased mathematical gradient scaling using actual group size `loss / current_group_size` (including final incomplete group).
  - Scheduler Synchronization: Scheduler steps strictly once per optimizer update (never per microbatch).
  - Linear Warmup: `162` updates (8% of 2,025 total scheduled updates)
  - Learning Rate: `1e-4` with cosine decay to `1e-6` (`eta_min_ratio=0.01`)
  - Estimated Colab T4 Runtime: `~18–22 hours` total (~3.5–4 hours per epoch)
* **Dedicated Output Directories:**
  - Local VM: `outputs/colab_t4_full_training/`
  - Persistent Drive Mirror: `/content/drive/MyDrive/meteor_net_runs/colab_t4_full_training/`
* **Execution Command (Fresh Run):**
  ```bash
  python scripts/train_meteor_net.py \
      --device cuda \
      --output_dir outputs/colab_t4_full_training \
      --drive_sync_dir /content/drive/MyDrive/meteor_net_runs/colab_t4_full_training \
      --sampler_mode natural_single_pass \
      --val_mode full_event \
      --epochs 5 \
      --grad_accum_steps 16 \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --seed 42 \
      --early_stopping_patience 2
  ```
* **Matching Resume Command:**
  ```bash
  python scripts/train_meteor_net.py \
      --device cuda \
      --output_dir outputs/colab_t4_full_training \
      --drive_sync_dir /content/drive/MyDrive/meteor_net_runs/colab_t4_full_training \
      --sampler_mode natural_single_pass \
      --val_mode full_event \
      --epochs 5 \
      --grad_accum_steps 16 \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --seed 42 \
      --early_stopping_patience 2 \
      --resume /content/drive/MyDrive/meteor_net_runs/colab_t4_full_training/latest_checkpoint.pt
  ```
* **Backward Compatibility:**
  The balanced-pilot mode remains 100% available via `--sampler_mode balanced_pilot` for reproducibility of Phase 7B and Phase 8 runs. Cross-sampler resume attempts are rejected fail-closed with explicit error diagnostics. Pilot checkpoints must never be used to resume full-data training runs.


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
| **Source Bundle Zip (`meteor_net_source_bundle.zip`)** | 55 files | **0.48 MB** (501,854 bytes) | `a03cbce9b76acfe015b0346e5cee06ca89025494e7e4a730cec8785a0dc0b49b` |
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
* **Source bundle created:** `meteor_net_source_bundle.zip` (0.48 MB, SHA-256: `a03cbce9b76acfe015b0346e5cee06ca89025494e7e4a730cec8785a0dc0b49b`).

### Step 2: Upload Files to Google Drive
1. In Google Drive (`My Drive`), create a folder named `meteor_net_data`.
2. Upload `data_goes19_4events.tar.gz` into `meteor_net_data/`.
3. *(Recommended)* Upload `meteor_net_source_bundle.zip` into `meteor_net_data/` for instantaneous, offline cryptographic source ingestion.

### Step 3: Open Notebook in Google Colab
1. Upload and open [`notebooks/meteor_net_colab_t4_training.ipynb`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/notebooks/meteor_net_colab_t4_training.ipynb) in Colab.
2. Ensure GPU acceleration is enabled:
   * **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Hardware accelerator: **T4 GPU** $\rightarrow$ Click **Save**.

### Step 4: Execute Verification & GPU Smoke Test
Execute notebook cells in sequence:
* **Stage 1 (Drive Mount):** Authorizes access to Google Drive (`/content/drive`).
* **Stage 2 (Source Ingestion):**
  * If `meteor_net_source_bundle.zip` is on Drive, verifies SHA-256 (`a03cbce...`) and unpacks into `/content/meteor_net`.
  * Otherwise, prompts interactively for a GitHub Personal Access Token (PAT) via `getpass.getpass()`, uses temporary restricted-permission `GIT_ASKPASS` (0600 mode) without exposing token in command line or remote URL, clones the repository, and verifies the authorized migration commit (`c273fc0...` or authorized list).
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

### Step 5: Full Controlled Training (Phase 8 GPU Experiment)
1. Open Stage 7 in the notebook.
2. Change the execution guard:
   ```python
   CONFIRM_LAUNCH_TRAINING = True
   ```
3. Run the cell:
   ```bash
   python scripts/train_meteor_net.py \
       --device cuda \
       --output_dir outputs/colab_t4_phase8 \
       --drive_sync_dir /content/drive/MyDrive/meteor_net_runs/colab_t4_phase8 \
       --epochs 5 \
       --microbatches_per_epoch 100 \
       --grad_accum_steps 2 \
       --lr 1e-4 \
       --weight_decay 1e-4 \
       --seed 42 \
       --val_patches 25 \
       --early_stopping_patience 2
   ```

### Step 6: Checkpoint Recovery Guarantees & Limitations
* **Recovery Guarantee:** At the completion of each epoch, `latest_checkpoint.pt`, `best_model.pt` (if validation improved), and `training_history.json` are written atomically locally and immediately live-mirrored to Google Drive (`/content/drive/MyDrive/meteor_net_runs/colab_t4_phase8/`).
* **Drive Sync Resilience:** If Google Drive encounters transient network dropouts or FUSE latency, `sync_file_to_drive` catches the exception, logs a warning, and allows training to continue. Local NVMe SSD checkpoints remain intact and uncorrupted.
* **Resume Procedure:**
  In Stage 8 of the notebook (or via terminal), set `CONFIRM_RESUME_TRAINING = True` and pass `--resume`:
  ```bash
  python scripts/train_meteor_net.py \
      --device cuda \
      --output_dir outputs/colab_t4_phase8 \
      --drive_sync_dir /content/drive/MyDrive/meteor_net_runs/colab_t4_phase8 \
      --epochs 5 \
      --microbatches_per_epoch 100 \
      --grad_accum_steps 2 \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --seed 42 \
      --val_patches 25 \
      --early_stopping_patience 2 \
      --resume /content/drive/MyDrive/meteor_net_runs/colab_t4_phase8/latest_checkpoint.pt
  ```
  The runner restores: model weights, optimizer state, scheduler state, global update count, microbatch count, CPU RNG state, CUDA RNG state, Python random state, and training history, safely beginning from microbatch 0 of epoch $E$.
* **Limitation:** Incomplete microbatches within an epoch that abort abnormally (e.g. abrupt Colab VM termination before epoch completion) are not checkpointed. The training loop safely re-executes the aborted epoch from microbatch 0.

### Step 7: Returning Artifacts to Antigravity
1. In Stage 9, verify that `latest_checkpoint.pt`, `best_model.pt`, and `training_history.json` exist in `/content/drive/MyDrive/meteor_net_runs/colab_t4_phase8/`.
2. Download or copy the `colab_t4_phase8/` folder to your local Antigravity workspace at:
   [`outputs/colab_t4_phase8/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/colab_t4_phase8/)
3. Inspect `training_history.json` and compare convergence against the CPU Phase 7B baseline.

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

---

## 2. Directory Layout & Storage Architecture

```
GOOGLE DRIVE (Persistent Storage)
  MyDrive/
    ├── meteor_net_data/
    │     ├── data_goes19_4events.tar.gz  (~7.93 GB compressed dataset archive)
    │     └── data_archive_manifest.json  (Cryptographic manifest with SHA-256)
    └── meteor_net_runs/
          └── colab_t4_phase8/            (Live synced GPU checkpoints and history)
                ├── best_model.pt
                ├── latest_checkpoint.pt
                ├── training_history.json
                └── baseline_metrics.json

COLAB VM LOCAL STORAGE (High-Speed NVMe Storage)
  /content/meteor_net/
    ├── data/                            (Extracted NetCDF files for high-IOPS lazy loading)
    ├── src/                             (Synchronized source code)
    ├── scripts/                         (train_meteor_net.py & run_gpu_smoke_test.py)
    ├── outputs/
    │     ├── dataset_splits/            (splits_manifest.json)
    │     ├── normalization/             (frozen_global_train_stats.json)
    │     ├── phase7b/                   (Read-only CPU baseline)
    │     ├── colab_t4_smoke_test/       (GPU smoke test outputs)
    │     └── colab_t4_phase8/           (Active Colab run outputs)
    └── notebooks/
          └── meteor_net_colab_t4_training.ipynb
```

---

## 3. Step-by-Step Migration Instructions

### Step 1: Package the Dataset Locally
In Antigravity PowerShell, run:
```powershell
python scripts/archive_dataset.py
```
* **Archive created:** `data_goes19_4events.tar.gz` (contains all 1,152 NetCDF files).
* **Manifest generated:** [`outputs/full_acquisition/data_archive_manifest.json`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/full_acquisition/data_archive_manifest.json) recording the exact file count, sizes, and SHA-256 hash.

### Step 2: Upload Archive to Google Drive
1. Open Google Drive in your browser.
2. In `My Drive`, create a folder named `meteor_net_data`.
3. Upload `data_goes19_4events.tar.gz` into `meteor_net_data/`.

### Step 3: Open Notebook in Google Colab
1. Upload [`notebooks/meteor_net_colab_t4_training.ipynb`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/notebooks/meteor_net_colab_t4_training.ipynb) to Google Colab.
2. Ensure GPU acceleration is enabled:
   * **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Hardware accelerator: **T4 GPU** $\rightarrow$ Click **Save**.

### Step 4: Execute Verification & GPU Smoke Test
Run through Stages 1 through 6 of the notebook:
* **Stage 1:** Authorize Google Drive mount.
* **Stage 2:** Ingest repository to `/content/meteor_net`.
* **Stage 3:** Unpack `data_goes19_4events.tar.gz` to local VM disk `/content/meteor_net/data` (verifies exactly 1,152 `.nc` files).
* **Stage 4:** Pinned dependency installation (`requirements.txt`) and `nvidia-smi` GPU inspection.
* **Stage 5:** Cryptographic verification of split manifest (`6a173d...`), frozen normalization (`71ef64...`), and acquisition inventory (`26cf30...`).
* **Stage 6 (GPU Smoke Test):**
  ```bash
  python scripts/run_gpu_smoke_test.py --device cuda --output_dir outputs/colab_t4_smoke_test
  ```
  *Verifies: CUDA memory allocation, 22,086,943 model parameters, forward pass, autograd backward pass, AdamW step, and atomic checkpoint serialization under 15 seconds.*

### Step 5: Execute Full Controlled Training (Phase 8 GPU Experiment)
Run Stage 7 of the notebook:
```bash
python scripts/train_meteor_net.py \
    --device cuda \
    --output_dir outputs/colab_t4_phase8 \
    --epochs 5 \
    --microbatches_per_epoch 100 \
    --grad_accum_steps 2 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --seed 42 \
    --val_patches 25 \
    --early_stopping_patience 2
```

### Step 6: Session Interruption & Checkpoint Resume
If your Colab session disconnects or reaches runtime limits:
1. Re-mount Drive and re-extract data (Stages 1–4).
2. Resume training directly from the latest saved checkpoint:
   ```bash
   python scripts/train_meteor_net.py \
       --device cuda \
       --output_dir outputs/colab_t4_phase8 \
       --epochs 5 \
       --microbatches_per_epoch 100 \
       --grad_accum_steps 2 \
       --resume outputs/colab_t4_phase8/latest_checkpoint.pt
   ```
   The resume engine restores: model weights, optimizer state, scheduler state, current epoch, global update counter, microbatch counter, RNG states, and frozen normalization provenance.

### Step 7: Returning Artifacts to Antigravity
1. Run Stage 8 of the notebook to sync completed artifacts to `/content/drive/MyDrive/meteor_net_runs/colab_t4_phase8/`.
2. From Google Drive on your local PC, copy `colab_t4_phase8/` into your local workspace at:
   [`outputs/colab_t4_phase8/`](file:///c:/Users/Krishna/OneDrive/Documents/CS%20CORE%20SUBJECTS%20NOTES/PROJECTS/MAJOR%20PROJECT/outputs/colab_t4_phase8/)
3. Inspect `training_history.json` and compare convergence against the CPU Phase 7B baseline.

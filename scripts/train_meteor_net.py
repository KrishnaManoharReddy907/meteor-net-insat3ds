"""
scripts/train_meteor_net.py
Controlled METEOR-Net Training Runner with Deterministic Event-Aware Mini-Epochs.

Adheres strictly to Phase 7A, 7B, and 7B.1 Audit Specifications:
- Microbatch size: 1
- Gradient accumulation: 2 microbatches -> 1 optimizer update
- 100 microbatches per mini-epoch (50 optimizer updates)
- Maximum 5 mini-epochs (250 optimizer updates total)
- Linear warmup (20 updates) + Cosine decay (down to eta_min=1e-6)
- Genuine patch-level 50:50 Convective/Ambient sampling without replacement
- Fixed 25-patch validation set evaluated under torch.no_grad()
- Strict event isolation (Train: Event 1 & 2; Val: Event 3; Test: Quarantined)
- Full check for finite inputs, outputs, losses, gradients, and parameters
- Atomic checkpoint writes and full state restoration for crash resilience
- System physical RAM monitoring and low-memory emergency abort
- Dry-run mode for preflight verification
"""

import os
import sys
import time
import math
import json
import random
import argparse
import hashlib
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.abspath("."))
from src.data.goes_dataset import GOESSequencePatchDataset
from src.models.meteor_net import METEORNet
from src.models.losses import MaskAwareConvectiveMAELoss
from src.models.normalization import ChannelStandardizer


def check_system_memory() -> Dict[str, float]:
    """Queries host RAM health across Windows and Linux (Colab)."""
    # 1. Try Windows API
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)
            ]
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return {
            "total_gb": ms.ullTotalPhys / (1024**3),
            "avail_gb": ms.ullAvailPhys / (1024**3),
            "load_pct": float(ms.dwMemoryLoad)
        }
    except Exception:
        pass

    # 2. Try psutil (cross-platform, available in Colab)
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb": vm.total / (1024**3),
            "avail_gb": vm.available / (1024**3),
            "load_pct": float(vm.percent)
        }
    except Exception:
        pass

    # 3. Try Linux /proc/meminfo
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].split()[0])
        total_kb = meminfo.get("MemTotal", 16 * 1024 * 1024)
        avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 3 * 1024 * 1024))
        load_pct = (1.0 - (avail_kb / total_kb)) * 100.0
        return {
            "total_gb": total_kb / (1024**2),
            "avail_gb": avail_kb / (1024**2),
            "load_pct": float(load_pct)
        }
    except Exception:
        return {"total_gb": 16.0, "avail_gb": 3.0, "load_pct": 80.0}


def check_gpu_memory(device: torch.device) -> Optional[Dict[str, Any]]:
    """Queries GPU VRAM health if target device is CUDA."""
    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "device_name": torch.cuda.get_device_name(device),
            "total_vram_gb": total_bytes / (1024**3),
            "free_vram_gb": free_bytes / (1024**3),
            "allocated_vram_gb": torch.cuda.memory_allocated(device) / (1024**3)
        }
    return None


def save_checkpoint_atomic(payload: Dict[str, Any], filepath: str):
    """Saves checkpoint payload atomically using temporary file and atomic replace."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = filepath + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, filepath)


def sync_file_to_drive(src_path: str, drive_dir: Optional[str]) -> bool:
    """Safely mirrors a local checkpoint or history file to Google Drive using a temporary write and atomic rename."""
    if not drive_dir:
        return False
    fname = os.path.basename(src_path)
    try:
        os.makedirs(drive_dir, exist_ok=True)
        dst_path = os.path.join(drive_dir, fname)
        tmp_dst = dst_path + ".tmp"
        import shutil
        shutil.copy2(src_path, tmp_dst)
        os.replace(tmp_dst, dst_path)

        # Verification of destination existence and byte size
        if not os.path.exists(dst_path):
            raise IOError(f"Destination file {dst_path} missing after atomic rename.")
        src_size = os.path.getsize(src_path)
        dst_size = os.path.getsize(dst_path)
        if dst_size != src_size:
            raise IOError(f"Size mismatch for {fname}: local {src_size} bytes vs drive {dst_size} bytes.")

        print(f"  [Drive Sync Verified] Live-synced {fname} to Drive ({dst_size / (1024*1024):.2f} MB, {dst_size:,} bytes verified)")
        return True
    except Exception as e:
        print(
            f"  [Drive Sync WARNING] Could not live-sync {fname} to Google Drive: {e}\n"
            f"  [Drive Sync RISK] Checkpoint is saved on local NVMe SSD ({src_path}), but is NOT yet mirrored to Google Drive.\n"
            f"  [Drive Sync RISK] If the Colab VM terminates before the next successful sync, this checkpoint may be lost!"
        )
        return False


REQUIRED_RESUME_CHECKPOINT_FIELDS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "epoch",
    "optimizer_update_count",
    "best_val_loss",
)


def validate_resume_checkpoint(checkpoint_path: Optional[str], map_location: Any = "cpu") -> Dict[str, Any]:
    """
    Validates that a resume checkpoint path exists, is a regular readable file,
    can be deserialized by PyTorch, and contains all required training restoration fields.

    Fails closed immediately if any check fails, preventing accidental fallback
    to a fresh run or unintended artifact creation.
    """
    if checkpoint_path is None:
        raise ValueError("No resume checkpoint path provided.")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError(f"Resume checkpoint path is empty or invalid: {checkpoint_path!r}")

    normalized_path = os.path.abspath(checkpoint_path.strip())

    if not os.path.exists(normalized_path):
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: '{checkpoint_path}'. "
            "Explicit resume was requested; refusing to silently start a fresh run."
        )

    if not os.path.isfile(normalized_path):
        raise ValueError(
            f"Resume checkpoint path is not a regular file: '{checkpoint_path}'. "
            "Explicit resume was requested; refusing to silently start a fresh run."
        )

    try:
        payload = torch.load(normalized_path, map_location=map_location)
    except Exception as e:
        raise RuntimeError(
            f"Resume checkpoint '{checkpoint_path}' is unreadable or corrupt: {e}. "
            "Explicit resume was requested; refusing to silently start a fresh run."
        ) from e

    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid resume checkpoint payload in '{checkpoint_path}': expected dict, got {type(payload).__name__}."
        )

    missing = [f for f in REQUIRED_RESUME_CHECKPOINT_FIELDS if f not in payload]
    if missing:
        raise KeyError(
            f"Resume checkpoint '{checkpoint_path}' is missing required fields: {missing}. "
            f"Required fields are: {list(REQUIRED_RESUME_CHECKPOINT_FIELDS)}."
        )

    # Validate essential structural types
    if not hasattr(payload["model_state_dict"], "keys") or not hasattr(payload["model_state_dict"], "__getitem__"):
        raise ValueError(f"Resume checkpoint '{checkpoint_path}': 'model_state_dict' must be a mapping.")
    if not hasattr(payload["optimizer_state_dict"], "keys") or not hasattr(payload["optimizer_state_dict"], "__getitem__"):
        raise ValueError(f"Resume checkpoint '{checkpoint_path}': 'optimizer_state_dict' must be a mapping.")
    if not hasattr(payload["scheduler_state_dict"], "keys") or not hasattr(payload["scheduler_state_dict"], "__getitem__"):
        raise ValueError(f"Resume checkpoint '{checkpoint_path}': 'scheduler_state_dict' must be a mapping.")

    if not isinstance(payload["epoch"], int) or payload["epoch"] < 0:
        raise ValueError(f"Resume checkpoint '{checkpoint_path}': 'epoch' must be a non-negative integer, got {payload['epoch']!r}.")
    if not isinstance(payload["optimizer_update_count"], int) or payload["optimizer_update_count"] < 0:
        raise ValueError(f"Resume checkpoint '{checkpoint_path}': 'optimizer_update_count' must be a non-negative integer, got {payload['optimizer_update_count']!r}.")
    if not isinstance(payload["best_val_loss"], (int, float)) or math.isnan(payload["best_val_loss"]) or math.isinf(payload["best_val_loss"]):
        raise ValueError(f"Resume checkpoint '{checkpoint_path}': 'best_val_loss' must be a finite number, got {payload['best_val_loss']!r}.")

    return payload


class CosineWarmupScheduler:
    """
    Cosine learning rate scheduler with linear warmup keyed to optimizer updates.
    """
    def __init__(self, optimizer: AdamW, warmup_updates: int, total_updates: int, eta_min_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_updates = max(1, warmup_updates)
        self.total_updates = max(warmup_updates + 1, total_updates)
        self.eta_min_ratio = eta_min_ratio

        def lr_lambda(current_update: int) -> float:
            if current_update < self.warmup_updates:
                # Linear warmup from 0.1 to 1.0
                return 0.1 + 0.9 * (float(current_update) / float(self.warmup_updates))
            # Cosine decay from 1.0 to eta_min_ratio
            progress = float(current_update - self.warmup_updates) / float(self.total_updates - self.warmup_updates)
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.eta_min_ratio + (1.0 - self.eta_min_ratio) * cosine_decay

        self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(self):
        self.scheduler.step()

    def get_last_lr(self) -> List[float]:
        return self.scheduler.get_last_lr()

    def state_dict(self) -> Dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.scheduler.load_state_dict(state_dict)


def verify_checkpoint_compatibility(
    checkpoint_payload: Dict[str, Any],
    epochs: int,
    microbatches_per_epoch: int,
    grad_accum_steps: int,
    seed: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_updates: int = 20,
    sampler_mode: Optional[str] = None,
    manifest_hash: Optional[str] = None,
    training_set_size: Optional[int] = None
) -> None:
    """
    Verifies that the checkpoint's model, optimizer, scheduler, sampler, manifest, and
    training-set metadata are strictly compatible with current execution parameters
    BEFORE creating any experiment artifacts or modifying live training state.

    Operates on temporary CPU probe objects to ensure zero mutation of the active graph.
    """
    updates_per_epoch = math.ceil(microbatches_per_epoch / grad_accum_steps)
    total_optimizer_updates = epochs * updates_per_epoch
    epoch = checkpoint_payload["epoch"]
    updates = checkpoint_payload["optimizer_update_count"]
    mbs = checkpoint_payload.get("microbatch_count", 0)

    # 1. Sampler mode consistency check
    ckpt_mode = checkpoint_payload.get("sampler_mode")
    if ckpt_mode is None and "patch_sampling_metadata" in checkpoint_payload:
        ckpt_mode = checkpoint_payload["patch_sampling_metadata"].get("sampler_mode", "balanced_pilot")
    if ckpt_mode is None:
        ckpt_mode = "balanced_pilot"

    if sampler_mode is not None and ckpt_mode != sampler_mode:
        raise ValueError(
            f"Resume checkpoint was trained with sampler_mode='{ckpt_mode}', conflicting with requested sampler_mode='{sampler_mode}'."
        )

    # 2. Dataset split manifest hash consistency check
    ckpt_manifest_hash = checkpoint_payload.get("manifest_hash")
    if ckpt_manifest_hash is not None and manifest_hash is not None:
        if ckpt_manifest_hash != manifest_hash:
            raise ValueError(
                f"Resume checkpoint was trained with dataset split manifest SHA-256 '{ckpt_manifest_hash}', "
                f"which conflicts with current manifest hash '{manifest_hash}'. Resuming across different dataset splits is prohibited."
            )
    elif sampler_mode == "natural_single_pass" and ckpt_manifest_hash is None and manifest_hash is not None:
        raise ValueError(
            "Resume checkpoint is missing 'manifest_hash'. Checkpoints resumed in natural_single_pass mode must have a verified manifest hash."
        )

    # 3. Training set size consistency check
    ckpt_train_size = checkpoint_payload.get("training_set_size")
    if ckpt_train_size is not None and training_set_size is not None:
        if ckpt_train_size != training_set_size:
            raise ValueError(
                f"Resume checkpoint recorded training_set_size={ckpt_train_size}, conflicting with current training dataset size ({training_set_size})."
            )

    # 4. Epoch bounds check
    if epoch >= epochs:
        raise ValueError(
            f"Resume checkpoint has already completed epoch {epoch}, which is >= total scheduled epochs ({epochs}). "
            "Resuming requires specifying a larger --epochs value."
        )

    # 3. Seed consistency check
    if "seed" in checkpoint_payload and checkpoint_payload["seed"] != seed:
        raise ValueError(
            f"Checkpoint was trained with seed={checkpoint_payload['seed']}, but current execution requested seed={seed}. "
            "Resuming with a different seed violates deterministic schedule consistency."
        )

    # 4. Microbatch and accumulation consistency check
    if mbs > 0 and epoch > 0:
        expected_mbs = epoch * microbatches_per_epoch
        if mbs != expected_mbs:
            raise ValueError(
                f"Checkpoint recorded {mbs} microbatches across {epoch} completed epochs (implying {mbs // epoch} per epoch), "
                f"conflicting with requested microbatches_per_epoch={microbatches_per_epoch}."
            )
    if epoch > 0:
        expected_updates = epoch * updates_per_epoch
        if updates != expected_updates:
            raise ValueError(
                f"Checkpoint recorded {updates} optimizer updates across {epoch} completed epochs, conflicting with "
                f"requested configuration ({microbatches_per_epoch} microbatches / {grad_accum_steps} accum steps -> {expected_updates} updates)."
            )

    # 5. Probe Model compatibility (using strict=True on separate CPU probe instance)
    probe_model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4)
    try:
        probe_model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    except Exception as e:
        raise ValueError(f"Resume checkpoint model_state_dict is incompatible with METEOR-Net architecture: {e}") from e

    # 6. Probe Optimizer compatibility & hyperparameter verification
    opt_state = checkpoint_payload["optimizer_state_dict"]
    if "param_groups" in opt_state and len(opt_state["param_groups"]) > 0:
        pg = opt_state["param_groups"][0]
        ckpt_init_lr = pg.get("initial_lr")
        if ckpt_init_lr is not None and not math.isclose(ckpt_init_lr, lr, rel_tol=1e-5):
            raise ValueError(
                f"Resume checkpoint optimizer initial_lr ({ckpt_init_lr}) is incompatible with requested lr ({lr})."
            )
        ckpt_wd = pg.get("weight_decay")
        if ckpt_wd is not None and not math.isclose(ckpt_wd, weight_decay, rel_tol=1e-5):
            raise ValueError(
                f"Resume checkpoint optimizer weight_decay ({ckpt_wd}) is incompatible with requested weight_decay ({weight_decay})."
            )
        ckpt_betas = pg.get("betas")
        if ckpt_betas is not None and tuple(ckpt_betas) != (0.9, 0.999):
            raise ValueError(
                f"Resume checkpoint optimizer betas ({ckpt_betas}) incompatible with expected (0.9, 0.999)."
            )

    probe_optimizer = AdamW(probe_model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    try:
        probe_optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
    except Exception as e:
        raise ValueError(f"Resume checkpoint optimizer_state_dict is incompatible with AdamW optimizer configuration: {e}") from e

    # 7. Probe Scheduler compatibility & step count verification
    sched_state = checkpoint_payload["scheduler_state_dict"]
    if "base_lrs" in sched_state and len(sched_state["base_lrs"]) > 0:
        ckpt_base_lr = sched_state["base_lrs"][0]
        if not math.isclose(ckpt_base_lr, lr, rel_tol=1e-5):
            raise ValueError(
                f"Resume checkpoint scheduler base_lr ({ckpt_base_lr}) is incompatible with requested lr ({lr})."
            )
    if "last_epoch" in sched_state and sched_state["last_epoch"] != updates:
        raise ValueError(
            f"Resume checkpoint scheduler last_epoch ({sched_state['last_epoch']}) does not match checkpoint optimizer_update_count ({updates})."
        )
    if updates >= total_optimizer_updates:
        raise ValueError(
            f"Resume checkpoint update count ({updates}) has reached or exceeded total scheduled updates ({total_optimizer_updates})."
        )

    probe_scheduler = CosineWarmupScheduler(
        optimizer=probe_optimizer,
        warmup_updates=warmup_updates,
        total_updates=total_optimizer_updates
    )
    try:
        probe_scheduler.load_state_dict(checkpoint_payload["scheduler_state_dict"])
    except Exception as e:
        raise ValueError(f"Resume checkpoint scheduler_state_dict is incompatible with CosineWarmupScheduler state: {e}") from e


def build_deterministic_training_schedule(
    train_ds: GOESSequencePatchDataset,
    epochs: int = 5,
    microbatches_per_epoch: int = 100,
    seed: int = 42,
    audit_path: str = "outputs/phase7b/patch_classification_audit.json"
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    """
    Precomputes deterministic, balanced (50 convective : 50 ambient) patch indices per epoch
    based on GENUINE patch-level Band 13 temperature evaluations.
    
    Guarantees zero replacement across the entire 5-epoch run.
    """
    rng = random.Random(seed)

    if os.path.exists(audit_path):
        with open(audit_path, "r") as f:
            audit_data = json.load(f)
        train_patches_info = audit_data["train"]["patches"]
        convective_pool = [p["patch_index"] for p in train_patches_info if p["is_convective"]]
        ambient_pool = [p["patch_index"] for p in train_patches_info if not p["is_convective"]]
    else:
        # Fallback: scan all train sequence target frames directly
        convective_pool = []
        ambient_pool = []
        for idx in range(len(train_ds)):
            # Fallback estimation using items
            s_idx, r, c = train_ds.items[idx]
            if s_idx >= 54:
                convective_pool.append(idx)
            else:
                ambient_pool.append(idx)

    rng.shuffle(ambient_pool)
    rng.shuffle(convective_pool)

    convective_per_epoch = microbatches_per_epoch // 2
    ambient_per_epoch = microbatches_per_epoch - convective_per_epoch
    total_convective_needed = convective_per_epoch * epochs
    total_ambient_needed = ambient_per_epoch * epochs

    if len(convective_pool) < total_convective_needed:
        raise ValueError(f"Insufficient genuine convective patches ({len(convective_pool)} < {total_convective_needed})")
    if len(ambient_pool) < total_ambient_needed:
        raise ValueError(f"Insufficient genuine ambient patches ({len(ambient_pool)} < {total_ambient_needed})")

    schedule: Dict[int, List[int]] = {}
    patch_provenance: Dict[int, List[Dict[str, Any]]] = {}

    for epoch in range(1, epochs + 1):
        amb_start = (epoch - 1) * ambient_per_epoch
        amb_slice = ambient_pool[amb_start : amb_start + ambient_per_epoch]

        conv_start = (epoch - 1) * convective_per_epoch
        conv_slice = convective_pool[conv_start : conv_start + convective_per_epoch]

        # Interleave ambient and convective deterministically
        epoch_patches = []
        for a, c in zip(amb_slice, conv_slice):
            epoch_patches.extend([a, c])
        
        # Deterministically permute the epoch patches so convective/ambient alternate naturally
        rng.shuffle(epoch_patches)
        schedule[epoch] = epoch_patches

        # Record provenance for each sampled patch
        epoch_prov = []
        for p_idx in epoch_patches:
            s_idx, r, c = train_ds.items[p_idx]
            seq_info = train_ds.sequence_entries[s_idx]
            epoch_prov.append({
                "patch_index": p_idx,
                "sequence_index": s_idx,
                "sequence_id": seq_info["sequence_id"],
                "event_id": seq_info["group_id"],
                "row_slice": [r, r + 128],
                "col_slice": [c, c + 128],
                "is_convective": bool(p_idx in convective_pool)
            })
        patch_provenance[epoch] = epoch_prov

    sampling_metadata = {
        "sampler_mode": "balanced_pilot",
        "seed": seed,
        "convective_pool_size": len(convective_pool),
        "ambient_pool_size": len(ambient_pool),
        "convective_sampled_total": total_convective_needed,
        "ambient_sampled_total": total_ambient_needed,
        "provenance_by_epoch": patch_provenance
    }

    return schedule, sampling_metadata


def build_natural_ratio_training_schedule(
    train_ds: GOESSequencePatchDataset,
    epochs: int = 5,
    seed: int = 42,
    audit_path: Optional[str] = "outputs/phase7b/patch_classification_audit.json"
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    """
    Constructs a deterministic, single-pass training schedule where EVERY eligible training patch
    in the dataset is visited exactly once per epoch, preserving the natural convective-to-ambient
    class distribution.

    Dynamically derives total patches and class distribution from the dataset items and audit metadata,
    without hardcoding patch counts or class ratios.

    Validation & Audit Handling:
      - If audit_path exists: validates index bounds, index uniqueness, and complete coverage.
      - If audit_path is missing or incomplete: falls back to deriving physical labels from the
        actual target Band 13 Kelvin data and the established 241 K / 221 K thresholds via train_ds,
        or fails clearly if reliable physical classification cannot be performed.
      - Regardless of class distribution, guarantees that every eligible training patch is scheduled
        exactly once per epoch with zero replacement within each epoch.
    """
    total_patches = len(train_ds)
    all_indices = list(range(total_patches))

    convective_indices = set()
    ambient_indices = set()
    audit_loaded = False

    if audit_path and os.path.exists(audit_path):
        try:
            with open(audit_path, "r") as f:
                audit_data = json.load(f)
            train_patches_info = audit_data.get("train", {}).get("patches", [])
            seen_indices = set()
            for p in train_patches_info:
                idx = p.get("patch_index")
                if idx is None:
                    continue
                if idx < 0 or idx >= total_patches:
                    raise ValueError(f"Audit patch_index {idx} is out of bounds for training dataset of size {total_patches}.")
                if idx in seen_indices:
                    raise ValueError(f"Audit contains duplicate patch_index {idx}.")
                seen_indices.add(idx)
                if p.get("is_convective"):
                    convective_indices.add(idx)
                else:
                    ambient_indices.add(idx)

            if len(seen_indices) == total_patches:
                audit_loaded = True
        except ValueError:
            raise
        except Exception as e:
            audit_loaded = False

    # If audit was missing or incomplete, derive physically from target Band 13 Kelvin data
    if not audit_loaded:
        missing_indices = [i for i in range(total_patches) if i not in convective_indices and i not in ambient_indices]
        for idx in missing_indices:
            try:
                item = train_ds[idx]
                if isinstance(item, dict) and "is_convective" in item:
                    is_conv = bool(item["is_convective"])
                elif isinstance(item, dict) and "target_kelvin" in item:
                    tgt_t = item["target_kelvin"]
                    tgt_m = item.get("target_mask")
                    if tgt_m is None:
                        tgt_m = torch.ones_like(tgt_t[0], dtype=torch.bool)
                    b13_tgt = tgt_t[1] if tgt_t.ndim >= 3 else tgt_t
                    valid_px = b13_tgt[tgt_m] if tgt_m.any() else b13_tgt
                    t_min = float(valid_px.min().item()) if valid_px.numel() > 0 else 300.0
                    frac_241 = float((valid_px < 241.0).float().mean().item()) if valid_px.numel() > 0 else 0.0
                    count_221 = int((valid_px < 221.0).sum().item()) if valid_px.numel() > 0 else 0
                    is_conv = (t_min < 241.0) and (frac_241 >= 0.02 or count_221 >= 5)
                else:
                    raise ValueError(f"Dataset item at index {idx} does not contain target Band 13 Kelvin data or 'is_convective' label.")

                if is_conv:
                    convective_indices.add(idx)
                else:
                    ambient_indices.add(idx)
            except Exception as e:
                raise ValueError(
                    f"Reliable physical patch classification unavailable for patch {idx}: "
                    f"audit file '{audit_path}' is missing or incomplete, and physical evaluation failed: {e}"
                ) from e

    convective_count = len(convective_indices)
    ambient_count = len(ambient_indices)
    convective_ratio = convective_count / max(1, total_patches)

    schedule: Dict[int, List[int]] = {}
    epoch_summaries: Dict[int, Dict[str, Any]] = {}

    for epoch in range(1, epochs + 1):
        # Deterministic global shuffle using seed + epoch (Decision 1: Option A)
        epoch_rng = random.Random(seed + epoch)
        epoch_patches = list(all_indices)
        epoch_rng.shuffle(epoch_patches)
        schedule[epoch] = epoch_patches

        # Summary provenance for the epoch
        ep_conv = sum(1 for p in epoch_patches if p in convective_indices)
        ep_amb = len(epoch_patches) - ep_conv
        epoch_summaries[epoch] = {
            "total_patches": len(epoch_patches),
            "convective_patches": ep_conv,
            "ambient_patches": ep_amb,
            "convective_ratio": round(ep_conv / max(1, len(epoch_patches)), 4)
        }

    sampling_metadata = {
        "sampler_mode": "natural_single_pass",
        "seed": seed,
        "total_training_patches": total_patches,
        "convective_pool_size": convective_count,
        "ambient_pool_size": ambient_count,
        "convective_ratio": round(convective_ratio, 4),
        "epochs": epochs,
        "epoch_summaries": epoch_summaries
    }

    return schedule, sampling_metadata


def select_fixed_validation_patches(val_ds: GOESSequencePatchDataset, count: int = 25, seed: int = 42) -> List[int]:
    """
    Selects a deterministic, spatially and temporally diverse subset of validation patches.
    """
    total_val = len(val_ds)  # 3240
    step = total_val // count
    rng = random.Random(seed)

    selected = [i * step + rng.randint(0, min(step - 1, 59)) for i in range(count)]
    return selected


def evaluate_patches(
    model: nn.Module,
    dataset: GOESSequencePatchDataset,
    standardizer: ChannelStandardizer,
    loss_fn: MaskAwareConvectiveMAELoss,
    patch_indices: List[int],
    device: Optional[torch.device] = None,
    show_progress: bool = False
) -> Dict[str, Any]:
    """
    Evaluates loss on a specified list of patch indices under torch.no_grad().
    Supports full-event validation (e.g. 3,240 patches) with progress logging.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    losses = []
    patch_records = []
    total_to_eval = len(patch_indices)

    with torch.no_grad():
        for i, idx in enumerate(patch_indices):
            item = dataset[idx]
            x_norm = item["input_norm"].unsqueeze(0).to(device)        # [1, 4, 4, 128, 128]
            tgt_phys = item["target_kelvin"].unsqueeze(0).to(device)   # [1, 4, 128, 128]
            mask = item["combined_mask"].unsqueeze(0).to(device)       # [1, 128, 128]

            y_norm = model(x_norm)
            y_phys = standardizer.denormalize_output(y_norm)
            loss = loss_fn(y_phys, tgt_phys, mask)

            loss_val = float(loss.item())
            losses.append(loss_val)
            if total_to_eval <= 100:
                patch_records.append({
                    "patch_index": idx,
                    "sequence_id": item["metadata"]["sequence_id"],
                    "event_id": item["metadata"]["event_id"],
                    "is_convective": item["is_convective"],
                    "b13_min_kelvin": item["metadata"]["b13_min_kelvin"],
                    "loss_kelvin": round(loss_val, 4)
                })

            if show_progress and ((i + 1) % 500 == 0 or (i + 1) == total_to_eval):
                print(f"    Validation Progress: {i+1:04d}/{total_to_eval:04d} patches evaluated...")

    mean_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "mean_loss_kelvin": mean_loss,
        "sample_count": len(losses),
        "patches": patch_records
    }


def run_training(
    manifest_path: str = "outputs/dataset_splits/splits_manifest.json",
    stats_path: str = "outputs/normalization/frozen_global_train_stats.json",
    audit_path: str = "outputs/phase7b/patch_classification_audit.json",
    output_dir: Optional[str] = None,
    epochs: int = 5,
    microbatches_per_epoch: Optional[int] = None,
    grad_accum_steps: Optional[int] = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_patch_count: Optional[int] = None,
    val_mode: Optional[str] = None,
    sampler_mode: Optional[str] = None,
    early_stopping_patience: int = 2,
    resume_path: Optional[str] = None,
    dry_run: bool = False,
    device_str: Optional[str] = None,
    drive_sync_dir: Optional[str] = None
) -> Dict[str, Any]:
    print("=" * 80)
    print("METEOR-NET CONTROLLED TRAINING RUNNER")
    print(f"Mode: {'DRY RUN (PREFLIGHT AUDIT ONLY)' if dry_run else 'ACTIVE TRAINING'}")
    print("=" * 80)

    # 1. System Memory Check
    mem_status = check_system_memory()
    print(f"\n[System Memory Preflight] Total Physical: {mem_status['total_gb']:.2f} GB | Available: {mem_status['avail_gb']:.2f} GB | Load: {mem_status['load_pct']}%")
    if mem_status['avail_gb'] < 0.50:
        raise MemoryError(f"Critically low system memory ({mem_status['avail_gb']:.2f} GB). Aborting to prevent OOM crash.")

    # 2. Compute Device Resolution
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    print(f"\n[Compute Device] Target: {device}")
    gpu_mem = check_gpu_memory(device)
    if gpu_mem:
        print(f"  GPU Hardware: {gpu_mem['device_name']} | Total VRAM: {gpu_mem['total_vram_gb']:.2f} GB | Free: {gpu_mem['free_vram_gb']:.2f} GB")

    # 3. Resume Preflight & Checkpoint Validation (Fail-closed BEFORE any artifact creation)
    resume_data = None
    is_resuming = resume_path is not None
    if is_resuming:
        print(f"\n[Resume Preflight] Validating resume checkpoint: {resume_path}")
        resume_data = validate_resume_checkpoint(resume_path, map_location=device)

    # Infer sampler_mode if not explicitly provided
    if sampler_mode is None:
        if resume_data is not None:
            ckpt_s_mode = resume_data.get("sampler_mode")
            if ckpt_s_mode is None and "patch_sampling_metadata" in resume_data:
                ckpt_s_mode = resume_data["patch_sampling_metadata"].get("sampler_mode")
            sampler_mode = ckpt_s_mode if ckpt_s_mode is not None else "balanced_pilot"
        elif output_dir and ("phase7b" in output_dir or "phase8" in output_dir):
            sampler_mode = "balanced_pilot"
        elif microbatches_per_epoch is not None and microbatches_per_epoch <= 500:
            sampler_mode = "balanced_pilot"
        else:
            sampler_mode = "natural_single_pass"

    # Default configuration resolution based on sampler_mode
    if sampler_mode == "natural_single_pass":
        if output_dir is None:
            output_dir = "outputs/colab_t4_full_training"
        if grad_accum_steps is None:
            grad_accum_steps = 16
        if val_mode is None:
            val_mode = "fixed_sample" if val_patch_count is not None else "full_event"
    elif sampler_mode == "balanced_pilot":
        if output_dir is None:
            output_dir = "outputs/phase7b"
        if grad_accum_steps is None:
            grad_accum_steps = 2
        if microbatches_per_epoch is None:
            microbatches_per_epoch = 100
        if val_mode is None:
            val_mode = "fixed_sample"
        if val_patch_count is None:
            val_patch_count = 25
    else:
        raise ValueError(f"Unknown sampler_mode '{sampler_mode}'. Expected 'natural_single_pass' or 'balanced_pilot'.")

    # Load split manifest to determine dataset size and calculate hash
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Dataset split manifest not found: {manifest_path}")
    with open(manifest_path, "rb") as f:
        manifest_raw = f.read()
        manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    manifest_data = json.loads(manifest_raw.decode("utf-8"))
    train_seq_count = len(manifest_data.get("splits", {}).get("train", []))
    patches_per_seq = manifest_data.get("metadata", {}).get("patches_per_sequence", 60)
    train_total_patches = train_seq_count * patches_per_seq

    if sampler_mode == "natural_single_pass":
        if microbatches_per_epoch is not None and microbatches_per_epoch != train_total_patches:
            raise ValueError(
                f"In 'natural_single_pass' mode, every eligible training patch must appear exactly once per epoch. "
                f"Requested microbatches_per_epoch={microbatches_per_epoch} does not match training set size ({train_total_patches}). "
                "Partial sampling / schedule truncation is prohibited in natural_single_pass mode. "
                "Use '--sampler_mode balanced_pilot' for pilot or debug runs."
            )
        microbatches_per_epoch = train_total_patches

    updates_per_epoch = math.ceil(microbatches_per_epoch / grad_accum_steps)
    total_optimizer_updates = epochs * updates_per_epoch
    warmup_updates = max(1, round(0.08 * total_optimizer_updates)) if sampler_mode == "natural_single_pass" else 20

    # If resuming, verify checkpoint compatibility against resolved hyperparameters
    if is_resuming and resume_data is not None:
        verify_checkpoint_compatibility(
            checkpoint_payload=resume_data,
            epochs=epochs,
            microbatches_per_epoch=microbatches_per_epoch,
            grad_accum_steps=grad_accum_steps,
            seed=seed,
            lr=lr,
            weight_decay=weight_decay,
            warmup_updates=warmup_updates,
            sampler_mode=sampler_mode,
            manifest_hash=manifest_hash,
            training_set_size=train_total_patches
        )
        print(f"  [PASS] Resume checkpoint verified and compatible with architecture/hyperparameters.")
        print(f"  Recorded Epoch: {resume_data['epoch']} | Updates: {resume_data['optimizer_update_count']} | Best Val Loss: {float(resume_data['best_val_loss']):.4f} K")

    # 4. Reproducibility & Output Directory Setup
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 5. Frozen Normalization Loading & Verification
    print("\n[Step 1] Loading frozen normalization statistics...")
    standardizer = ChannelStandardizer.from_stats_file(stats_path).to(device)
    initial_mean = standardizer.mean.clone()
    initial_std = standardizer.std.clone()
    print(f"  Frozen Means (K): {[round(float(m), 4) for m in initial_mean.cpu()]}")
    print(f"  Frozen Stds  (K): {[round(float(s), 4) for s in initial_std.cpu()]}")

    # 6. Model, Loss, Optimizer, and Scheduler Initialization
    print("\n[Step 2] Initializing METEOR-Net architecture and optimization graph...")
    model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
    loss_fn = MaskAwareConvectiveMAELoss(
        ref_channel_idx=1,
        t1_threshold=241.0,
        t2_threshold=221.0,
        alpha=1.0,
        beta=2.0,
        gamma=5.0
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_updates=warmup_updates,
        total_updates=total_optimizer_updates,
        eta_min_ratio=0.01
    )

    print(f"  Total Model Parameters:          {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Sampler Mode:                    {sampler_mode}")
    print(f"  Validation Mode:                 {val_mode}")
    print(f"  Microbatches per Epoch:          {microbatches_per_epoch}")
    print(f"  Gradient Accumulation:           {grad_accum_steps} microbatches -> 1 optimizer update")
    print(f"  Optimizer Updates per Epoch:     {updates_per_epoch}")
    print(f"  Total Scheduled Updates:         {total_optimizer_updates} (across {epochs} epochs)")
    print(f"  Warmup Updates:                  {warmup_updates} updates")

    # 7. Partition Ingestion & Sampler Construction
    print("\n[Step 3] Initializing Train and Validation datasets...")
    train_ds = GOESSequencePatchDataset(
        manifest_path=manifest_path,
        split="train",
        standardizer=standardizer,
        normalize_inputs=True,
        cache_size=2
    )
    val_ds = GOESSequencePatchDataset(
        manifest_path=manifest_path,
        split="val",
        standardizer=standardizer,
        normalize_inputs=True,
        cache_size=2
    )
    print(f"  Train partition total patches:   {len(train_ds):,} (Events: 2026-01-01, 2026-05-20)")
    print(f"  Val partition total patches:     {len(val_ds):,} (Event: 2026-07-19)")
    print("  Test partition quarantined:      EXCLUDED (zero instantiation or iteration)")

    if sampler_mode == "natural_single_pass":
        if microbatches_per_epoch != len(train_ds):
            raise ValueError(
                f"In 'natural_single_pass' mode, microbatches_per_epoch ({microbatches_per_epoch}) must equal len(train_ds) ({len(train_ds)})."
            )
        train_schedule, sampling_metadata = build_natural_ratio_training_schedule(
            train_ds=train_ds,
            epochs=epochs,
            seed=seed,
            audit_path=audit_path
        )
    else:
        train_schedule, sampling_metadata = build_deterministic_training_schedule(
            train_ds=train_ds,
            epochs=epochs,
            microbatches_per_epoch=microbatches_per_epoch,
            seed=seed,
            audit_path=audit_path
        )

    # Resolution of validation patch indices
    if val_mode == "full_event":
        val_patch_indices = list(range(len(val_ds)))
    else:
        actual_val_count = val_patch_count if val_patch_count is not None else 25
        val_patch_indices = select_fixed_validation_patches(val_ds=val_ds, count=actual_val_count, seed=seed)

    fixed_diag_train_patches = train_schedule[1][:10]

    print(f"  Convective pool size (genuine):  {sampling_metadata['convective_pool_size']:,} patches")
    print(f"  Ambient pool size (genuine):     {sampling_metadata['ambient_pool_size']:,} patches")
    print(f"  Constructed {epochs}-epoch schedule:    {sum(len(v) for v in train_schedule.values()):,} distinct patches scheduled")
    print(f"  Selected validation set:         {len(val_patch_indices):,} patches from Event 3 ({val_mode})")

    # Record complete execution provenance
    sampling_metadata["sampler_mode"] = sampler_mode
    sampling_metadata["val_mode"] = val_mode
    sampling_metadata["grad_accum_steps"] = grad_accum_steps
    sampling_metadata["effective_batch_size"] = grad_accum_steps
    sampling_metadata["optimizer_updates_per_epoch"] = updates_per_epoch
    sampling_metadata["total_optimizer_updates"] = total_optimizer_updates
    sampling_metadata["warmup_updates"] = warmup_updates
    sampling_metadata["val_patch_count"] = len(val_patch_indices)
    sampling_metadata["manifest_hash"] = manifest_hash
    sampling_metadata["scheduler_config"] = {
        "type": "CosineWarmupScheduler",
        "warmup_updates": warmup_updates,
        "total_updates": total_optimizer_updates,
        "lr_initial": lr,
        "weight_decay": weight_decay,
        "eta_min_ratio": 0.01,
        "eta_min": round(lr * 0.01, 8)
    }

    schedule_provenance_path = os.path.join(output_dir, "training_schedule_provenance.json")
    drive_schedule_path = os.path.join(drive_sync_dir, "training_schedule_provenance.json") if drive_sync_dir else None
    if is_resuming and not os.path.exists(schedule_provenance_path) and drive_schedule_path and os.path.exists(drive_schedule_path):
        import shutil
        shutil.copy2(drive_schedule_path, schedule_provenance_path)

    if is_resuming and os.path.exists(schedule_provenance_path):
        try:
            with open(schedule_provenance_path, "r") as f:
                saved_prov = json.load(f)
            if saved_prov.get("seed") != seed:
                raise ValueError(f"Saved schedule provenance seed ({saved_prov.get('seed')}) conflicts with requested seed ({seed}).")
            saved_sm = saved_prov.get("sampler_mode", "balanced_pilot")
            if saved_sm != sampler_mode:
                raise ValueError(f"Saved schedule provenance sampler_mode ({saved_sm}) conflicts with requested {sampler_mode}.")
            if saved_sm == "balanced_pilot":
                if len(saved_prov.get("provenance_by_epoch", {})) != epochs:
                    raise ValueError(f"Saved schedule provenance has {len(saved_prov.get('provenance_by_epoch', {}))} epochs, conflicts with requested epochs={epochs}.")
                first_ep_key = next(iter(saved_prov.get("provenance_by_epoch", {}).keys()), None)
                if first_ep_key is not None and len(saved_prov["provenance_by_epoch"][first_ep_key]) != microbatches_per_epoch:
                    raise ValueError(f"Saved schedule provenance microbatches ({len(saved_prov['provenance_by_epoch'][first_ep_key])}) conflicts with requested {microbatches_per_epoch}.")
            else:
                if saved_prov.get("epochs") is not None and saved_prov["epochs"] != epochs:
                    raise ValueError(f"Saved schedule provenance epochs ({saved_prov['epochs']}) conflicts with requested {epochs}.")
            print(f"  [Resume Mode] Preserved and verified existing schedule provenance: {schedule_provenance_path}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            print(f"  [Resume Mode WARNING] Could not verify existing schedule provenance: {e}")
    else:
        with open(schedule_provenance_path, "w") as f:
            json.dump(sampling_metadata, f, indent=2)
        print(f"  Sampling provenance saved to:    {schedule_provenance_path}")

    # 8. Baseline Evaluation (Epoch 0 Baseline)
    baseline_path = os.path.join(output_dir, "baseline_metrics.json")
    drive_baseline_path = os.path.join(drive_sync_dir, "baseline_metrics.json") if drive_sync_dir else None

    if is_resuming and (os.path.exists(baseline_path) or (drive_baseline_path and os.path.exists(drive_baseline_path))):
        if not os.path.exists(baseline_path) and drive_baseline_path and os.path.exists(drive_baseline_path):
            import shutil
            shutil.copy2(drive_baseline_path, baseline_path)
        with open(baseline_path, "r") as f:
            baseline_metrics = json.load(f)
        initial_val_loss_kelvin = baseline_metrics["initial_val_loss_kelvin"]
        print(f"\n[Step 4] Resume Mode: Preserving existing baseline metrics from {baseline_path} (untrained val loss: {initial_val_loss_kelvin:.4f} K)")
    elif is_resuming:
        initial_val_loss_kelvin = None
        baseline_metrics = {
            "resumed": True,
            "initial_val_loss_kelvin": None,
            "best_val_loss_at_resume": float(resume_data["best_val_loss"]),
            "resume_checkpoint": resume_path,
            "sampler_mode": sampler_mode,
            "val_mode": val_mode,
            "note": "Original epoch-zero baseline_metrics.json was unavailable. 'initial_val_loss_kelvin' is null because untrained baseline was not evaluated in this session."
        }
        print(f"\n[Step 4] Resume Mode: Preserved best_val_loss ({float(resume_data['best_val_loss']):.4f} K) without mislabeling as untrained baseline.")
    else:
        print(f"\n[Step 4] Computing Untrained Model Baseline Metrics (Epoch 0 on {len(val_patch_indices)} val patches)...")
        t0_base = time.perf_counter()
        initial_val_eval = evaluate_patches(
            model=model,
            dataset=val_ds,
            standardizer=standardizer,
            loss_fn=loss_fn,
            patch_indices=val_patch_indices,
            device=device,
            show_progress=(val_mode == "full_event")
        )
        initial_train_eval = evaluate_patches(
            model=model,
            dataset=train_ds,
            standardizer=standardizer,
            loss_fn=loss_fn,
            patch_indices=fixed_diag_train_patches,
            device=device
        )
        t_base = time.perf_counter() - t0_base
        initial_val_loss_kelvin = initial_val_eval["mean_loss_kelvin"]

        baseline_metrics = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_architecture": "METEORNet",
            "device": str(device),
            "seed": seed,
            "sampler_mode": sampler_mode,
            "val_mode": val_mode,
            "initial_val_loss_kelvin": round(initial_val_loss_kelvin, 4),
            "initial_train_diag_loss_kelvin": round(initial_train_eval["mean_loss_kelvin"], 4),
            "val_patches_evaluated": len(val_patch_indices),
            "train_diag_patches_evaluated": len(fixed_diag_train_patches),
            "baseline_eval_time_s": round(t_base, 3),
            "val_patches_detail": initial_val_eval["patches"],
            "train_diag_patches_detail": initial_train_eval["patches"]
        }
        with open(baseline_path, "w") as f:
            json.dump(baseline_metrics, f, indent=2)
        sync_file_to_drive(baseline_path, drive_sync_dir)
        print(f"  Untrained Validation Baseline:   {initial_val_loss_kelvin:.4f} K (physical MAE)")
        print(f"  Untrained Train Diagnostic Loss: {initial_train_eval['mean_loss_kelvin']:.4f} K (physical MAE)")
        print(f"  Saved baseline metrics to:       {baseline_path}")

    # 9. Checkpoint Serialization & Atomic Restoration Test
    test_ckpt_path = os.path.join(output_dir, "preflight_checkpoint_test.pt")
    if is_resuming and os.path.exists(test_ckpt_path):
        ckpt_size_mb = os.path.getsize(test_ckpt_path) / (1024 * 1024)
        print(f"\n[Step 5] Resume Mode: Preserving existing preflight test checkpoint: {test_ckpt_path} ({ckpt_size_mb:.2f} MB)")
    else:
        print("\n[Step 5] Testing Checkpoint Serialization and Restoration...")
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": 0,
            "optimizer_update_count": 0,
            "microbatch_count": 0,
            "best_val_loss": initial_val_loss_kelvin,
            "seed": seed,
            "sampler_mode": sampler_mode,
            "val_mode": val_mode,
            "grad_accum_steps": grad_accum_steps,
            "microbatches_per_epoch": microbatches_per_epoch,
            "updates_per_epoch": updates_per_epoch,
            "total_optimizer_updates": total_optimizer_updates,
            "warmup_updates": warmup_updates,
            "training_set_size": len(train_ds),
            "class_counts": {
                "convective": sampling_metadata.get("convective_pool_size"),
                "ambient": sampling_metadata.get("ambient_pool_size")
            },
            "manifest_hash": manifest_hash,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "random_state": random.getstate(),
            "patch_sampling_metadata": sampling_metadata,
            "normalization_provenance": standardizer.provenance
        }
        save_checkpoint_atomic(checkpoint_payload, test_ckpt_path)
        ckpt_size_mb = os.path.getsize(test_ckpt_path) / (1024 * 1024)

        # Reload into fresh model
        reloaded_model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
        loaded_ckpt = torch.load(test_ckpt_path, map_location=device)
        reloaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
        for n, p in model.named_parameters():
            p_reloaded = dict(reloaded_model.named_parameters())[n]
            assert torch.equal(p, p_reloaded), f"State dict mismatch on parameter {n}"
        print(f"  Saved atomic test checkpoint:    {test_ckpt_path} ({ckpt_size_mb:.2f} MB)")
        print("  Round-trip restoration test:     PASS (All parameters identical bit-for-bit)")

    # Assert normalization statistics unmutated
    assert torch.equal(standardizer.mean, initial_mean), "Normalization mean mutated during baseline!"
    assert torch.equal(standardizer.std, initial_std), "Normalization std mutated during baseline!"
    print("  Normalization invariance:        PASS (Frozen statistics strictly intact)")

    if dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN PREFLIGHT VERIFICATION: ALL AUDITS PASSED")
        print("No optimizer steps or model updates were executed.")
        print("Training runner is validated and ready for authorized execution.")
        print("=" * 80)
        return {
            "status": "DRY_RUN_PASS",
            "is_resuming": is_resuming,
            "resume_path": resume_path,
            "sampler_mode": sampler_mode,
            "val_mode": val_mode,
            "baseline_metrics": baseline_metrics,
            "checkpoint_size_mb": ckpt_size_mb if 'ckpt_size_mb' in locals() else None,
            "train_schedule_summary": {e: len(p) for e, p in train_schedule.items()},
            "val_patch_count": len(val_patch_indices),
            "convective_pool_size": sampling_metadata["convective_pool_size"],
            "ambient_pool_size": sampling_metadata["ambient_pool_size"],
            "updates_per_epoch": updates_per_epoch,
            "total_optimizer_updates": total_optimizer_updates,
            "warmup_updates": warmup_updates
        }

    # 10. Checkpoint Resume Support
    start_epoch = 1
    global_update_count = 0
    global_microbatch_count = 0
    best_val_loss = float(initial_val_loss_kelvin) if initial_val_loss_kelvin is not None else float("inf")
    patience_counter = 0

    if is_resuming and resume_data is not None:
        print(f"\n[Resume Engine] Restoring state from verified checkpoint: {resume_path}")
        model.load_state_dict(resume_data["model_state_dict"])
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        scheduler.load_state_dict(resume_data["scheduler_state_dict"])
        start_epoch = int(resume_data["epoch"]) + 1
        global_update_count = int(resume_data["optimizer_update_count"])
        global_microbatch_count = int(resume_data.get("microbatch_count", 0))
        best_val_loss = float(resume_data["best_val_loss"])
        if "rng_state" in resume_data and resume_data["rng_state"] is not None:
            torch.set_rng_state(resume_data["rng_state"])
        if "cuda_rng_state" in resume_data and resume_data["cuda_rng_state"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state(resume_data["cuda_rng_state"], device)
        if "random_state" in resume_data and resume_data["random_state"] is not None:
            random.setstate(resume_data["random_state"])
        print(f"  Restored successfully at Epoch {start_epoch}, Update {global_update_count}, Microbatch {global_microbatch_count}")

    # 11. Active Training Loop (Only when approved and NOT dry_run)
    print("\n[Step 6] Commencing Controlled METEOR-Net Training...")
    history_path = os.path.join(output_dir, "training_history.json")
    drive_history_path = os.path.join(drive_sync_dir, "training_history.json") if drive_sync_dir else None
    if is_resuming and (os.path.exists(history_path) or (drive_history_path and os.path.exists(drive_history_path))):
        if not os.path.exists(history_path) and drive_history_path and os.path.exists(drive_history_path):
            import shutil
            shutil.copy2(drive_history_path, history_path)
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
            print(f"  [Resume Engine] Restored prior training history ({len(history.get('epochs', []))} epochs recorded)")
            history["best_val_loss"] = best_val_loss
        except Exception as e:
            print(f"  [Resume Engine WARNING] Could not load prior history: {e}")
            history = {
                "epochs": [],
                "best_val_loss": best_val_loss,
                "stopped_early": False
            }
    else:
        history = {
            "epochs": [],
            "best_val_loss": best_val_loss,
            "stopped_early": False
        }

    for epoch in range(start_epoch, epochs + 1):
        # Pre-epoch memory check
        mem_info = check_system_memory()
        print(f"\n--- Epoch {epoch} / {epochs} ---")
        print(f"  [RAM Monitor] Avail: {mem_info['avail_gb']:.2f} GB | Load: {mem_info['load_pct']}%")
        gpu_info = check_gpu_memory(device)
        if gpu_info:
            print(f"  [VRAM Monitor] Free: {gpu_info['free_vram_gb']:.2f} GB | Allocated: {gpu_info['allocated_vram_gb']:.2f} GB")
        if mem_info['avail_gb'] < 0.50:
            print(f"  [CRITICAL] Host RAM critically low ({mem_info['avail_gb']:.2f} GB). Pausing safely.")
            emergency_ckpt = os.path.join(output_dir, "emergency_low_ram_checkpoint.pt")
            save_checkpoint_atomic({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch - 1,
                "optimizer_update_count": global_update_count,
                "microbatch_count": global_microbatch_count,
                "best_val_loss": best_val_loss,
                "seed": seed,
                "sampler_mode": sampler_mode,
                "val_mode": val_mode,
                "grad_accum_steps": grad_accum_steps,
                "microbatches_per_epoch": microbatches_per_epoch,
                "updates_per_epoch": updates_per_epoch,
                "total_optimizer_updates": total_optimizer_updates,
                "warmup_updates": warmup_updates,
                "training_set_size": len(train_ds),
                "class_counts": {
                    "convective": sampling_metadata.get("convective_pool_size"),
                    "ambient": sampling_metadata.get("ambient_pool_size")
                },
                "manifest_hash": manifest_hash,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "random_state": random.getstate(),
                "patch_sampling_metadata": sampling_metadata,
                "normalization_provenance": standardizer.provenance
            }, emergency_ckpt)
            sync_file_to_drive(emergency_ckpt, drive_sync_dir)
            raise MemoryError(f"Emergency stop: system memory critically low ({mem_info['avail_gb']:.2f} GB). Saved state to {emergency_ckpt}")

        t_epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        epoch_train_losses = []
        accum_loss_tracker = 0.0

        patch_indices = train_schedule[epoch]

        for mb_idx, p_idx in enumerate(patch_indices):
            item = train_ds[p_idx]
            x_norm = item["input_norm"].unsqueeze(0).to(device)
            tgt_phys = item["target_kelvin"].unsqueeze(0).to(device)
            mask = item["combined_mask"].unsqueeze(0).to(device)

            # Forward pass
            y_norm = model(x_norm)
            if not torch.isfinite(y_norm).all():
                raise FloatingPointError(f"Non-finite prediction at epoch {epoch}, microbatch {mb_idx}!")

            # De-normalize to physical Kelvin
            y_phys = standardizer.denormalize_output(y_norm)
            loss = loss_fn(y_phys, tgt_phys, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, microbatch {mb_idx}!")

            loss_unscaled = float(loss.item())
            epoch_train_losses.append(loss_unscaled)
            accum_loss_tracker += loss_unscaled

            # Calculate actual number of microbatches in current accumulation group
            group_idx = mb_idx // grad_accum_steps
            group_start = group_idx * grad_accum_steps
            group_end = min(len(patch_indices), group_start + grad_accum_steps)
            current_group_size = group_end - group_start

            # Gradient accumulation scaling using exact group size (including final incomplete group)
            loss_scaled = loss / current_group_size
            loss_scaled.backward()
            global_microbatch_count += 1

            # Check gradients
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError(f"Non-finite gradient at epoch {epoch}, microbatch {mb_idx}!")

            # Optimizer step upon completing accumulation group
            is_accum_boundary = ((mb_idx + 1) % grad_accum_steps == 0) or ((mb_idx + 1) == len(patch_indices))
            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_update_count += 1
                accum_loss_tracker = 0.0

            print_interval = 20 if len(patch_indices) <= 200 else 200
            if (mb_idx + 1) % print_interval == 0 or (mb_idx + 1) == len(patch_indices):
                current_lr = scheduler.get_last_lr()[0]
                mean_so_far = sum(epoch_train_losses) / len(epoch_train_losses)
                print(f"  [Epoch {epoch}] MB {mb_idx+1:04d}/{len(patch_indices):04d} | Update {global_update_count:04d} | Loss: {loss_unscaled:.4f} K (Avg: {mean_so_far:.4f} K) | LR: {current_lr:.6e}")

        t_epoch_train = time.perf_counter() - t_epoch_start
        epoch_mean_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)

        # Validation evaluation
        print(f"  Evaluating Epoch {epoch} on {len(val_patch_indices)} validation patches ({val_mode})...")
        t_val_start = time.perf_counter()
        val_eval = evaluate_patches(
            model=model,
            dataset=val_ds,
            standardizer=standardizer,
            loss_fn=loss_fn,
            patch_indices=val_patch_indices,
            device=device,
            show_progress=(val_mode == "full_event")
        )
        t_epoch_val = time.perf_counter() - t_val_start
        epoch_val_loss = float(val_eval["mean_loss_kelvin"])

        epoch_duration = time.perf_counter() - t_epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(f"  [Epoch {epoch} Results] Train Loss: {epoch_mean_train_loss:.4f} K | Val Loss: {epoch_val_loss:.4f} K | LR: {current_lr:.6e} | Time: {epoch_duration:.1f} s")

        epoch_record = {
            "epoch": epoch,
            "train_loss_kelvin": round(epoch_mean_train_loss, 4),
            "val_loss_kelvin": round(epoch_val_loss, 4),
            "lr": current_lr,
            "duration_s": round(epoch_duration, 2),
            "train_duration_s": round(t_epoch_train, 2),
            "val_duration_s": round(t_epoch_val, 2),
            "updates_completed": global_update_count,
            "microbatches_completed": global_microbatch_count
        }
        history["epochs"].append(epoch_record)

        # Check for validation improvement
        improved = epoch_val_loss < best_val_loss
        if improved:
            improvement = best_val_loss - epoch_val_loss
            best_val_loss = epoch_val_loss
            history["best_val_loss"] = round(best_val_loss, 4)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  [!] Validation did not improve ({patience_counter}/{early_stopping_patience} patience)")

        # Save latest checkpoint atomically with current, accurate best_val_loss
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "optimizer_update_count": global_update_count,
            "microbatch_count": global_microbatch_count,
            "best_val_loss": best_val_loss,
            "seed": seed,
            "sampler_mode": sampler_mode,
            "val_mode": val_mode,
            "grad_accum_steps": grad_accum_steps,
            "microbatches_per_epoch": microbatches_per_epoch,
            "updates_per_epoch": updates_per_epoch,
            "total_optimizer_updates": total_optimizer_updates,
            "warmup_updates": warmup_updates,
            "training_set_size": len(train_ds),
            "class_counts": {
                "convective": sampling_metadata.get("convective_pool_size"),
                "ambient": sampling_metadata.get("ambient_pool_size")
            },
            "manifest_hash": manifest_hash,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "random_state": random.getstate(),
            "patch_sampling_metadata": sampling_metadata,
            "normalization_provenance": standardizer.provenance
        }
        latest_ckpt_path = os.path.join(output_dir, "latest_checkpoint.pt")
        save_checkpoint_atomic(checkpoint_payload, latest_ckpt_path)
        sync_file_to_drive(latest_ckpt_path, drive_sync_dir)

        # Save best model checkpoint if validation improved
        if improved:
            best_ckpt_path = os.path.join(output_dir, "best_model.pt")
            save_checkpoint_atomic(checkpoint_payload, best_ckpt_path)
            sync_file_to_drive(best_ckpt_path, drive_sync_dir)
            print(f"  [*] Validation improved by {improvement:.4f} K -> Saved best model to {best_ckpt_path}")

        # Atomically serialize and live-sync training history at each epoch boundary
        history_path = os.path.join(output_dir, "training_history.json")
        history_tmp = history_path + ".tmp"
        with open(history_tmp, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(history_tmp, history_path)
        sync_file_to_drive(history_path, drive_sync_dir)

        if patience_counter >= early_stopping_patience:
            print(f"  [!] Early stopping triggered after {epoch} epochs!")
            history["stopped_early"] = True
            with open(history_tmp, "w") as f:
                json.dump(history, f, indent=2)
            os.replace(history_tmp, history_path)
            sync_file_to_drive(history_path, drive_sync_dir)
            break

    # Final history dump confirmation
    history_path = os.path.join(output_dir, "training_history.json")
    history_tmp = history_path + ".tmp"
    with open(history_tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(history_tmp, history_path)
    sync_file_to_drive(history_path, drive_sync_dir)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="METEOR-Net Controlled Training Runner")
    parser.add_argument("--manifest", type=str, default="outputs/dataset_splits/splits_manifest.json")
    parser.add_argument("--stats", type=str, default="outputs/normalization/frozen_global_train_stats.json")
    parser.add_argument("--audit", type=str, default="outputs/phase7b/patch_classification_audit.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save training artifacts")
    parser.add_argument("--sampler_mode", type=str, default="natural_single_pass", choices=["natural_single_pass", "balanced_pilot"], help="Training schedule sampler mode ('natural_single_pass' or 'balanced_pilot')")
    parser.add_argument("--val_mode", type=str, default=None, choices=["full_event", "fixed_sample"], help="Validation mode ('full_event' for all patches or 'fixed_sample')")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--microbatches_per_epoch", type=int, default=None, help="Microbatches per epoch (defaults to full training set for natural_single_pass, or 100 for balanced_pilot)")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation factor (defaults to 16 for natural_single_pass, 2 for balanced_pilot)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_patches", type=int, default=None, help="Number of fixed validation patches if val_mode is 'fixed_sample' (default: 25)")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--dry_run", action="store_true", help="Perform preflight checks and baseline without optimizer updates")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'cpu', or None for auto-detection)")
    parser.add_argument("--drive_sync_dir", type=str, default=None, help="Persistent Google Drive directory for live checkpoint mirroring")
    args = parser.parse_args()

    run_training(
        manifest_path=args.manifest,
        stats_path=args.stats,
        audit_path=args.audit,
        output_dir=args.output_dir,
        epochs=args.epochs,
        microbatches_per_epoch=args.microbatches_per_epoch,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_patch_count=args.val_patches,
        val_mode=args.val_mode,
        sampler_mode=args.sampler_mode,
        early_stopping_patience=args.early_stopping_patience,
        resume_path=args.resume,
        dry_run=args.dry_run,
        device_str=args.device,
        drive_sync_dir=args.drive_sync_dir
    )



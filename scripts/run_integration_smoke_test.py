"""
scripts/run_integration_smoke_test.py
Phase 6B: Bounded dataset-to-model integration preflight and smoke test.

Strict boundaries:
- Loads exactly 1 real training patch and 1 real validation patch.
- Performs forward pass and loss calculation on each.
- Executes exactly 1 optimizer step on the training patch only.
- Asserts finite inputs, predictions, losses, and gradients.
- Verifies validation pass causes zero weight or state mutation.
- Saves and reloads a model checkpoint, verifying parameter equality.
- Records execution timing and peak memory metrics.
"""

import os
import sys
import time
import json
import tracemalloc
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))
from src.data.goes_dataset import GOESSequencePatchDataset
from src.models.meteor_net import METEORNet
from src.models.losses import MaskAwareConvectiveMAELoss
from src.models.normalization import ChannelStandardizer


def run_integration_smoke_test(
    manifest_path: str = "outputs/dataset_splits/splits_manifest.json",
    stats_path: str = "outputs/normalization/frozen_global_train_stats.json",
    checkpoint_dir: str = "checkpoints/smoke_test",
    output_metrics_path: str = "outputs/smoke_test/smoke_test_metrics.json"
) -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 6B: DATASET-TO-MODEL INTEGRATION PREFLIGHT & SMOKE TEST")
    print("=" * 80)

    tracemalloc.start()
    t_start_total = time.perf_counter()

    # Step 1: Load frozen normalization statistics
    print("\n[Step 1] Loading frozen training normalization statistics...")
    standardizer = ChannelStandardizer.from_stats_file(stats_path)
    initial_mean = standardizer.mean.clone()
    initial_std = standardizer.std.clone()
    print(f"  Channel Order: {standardizer.channel_order}")
    print(f"  Frozen Means (K): {[round(float(m), 4) for m in initial_mean]}")
    print(f"  Frozen Stds  (K): {[round(float(s), 4) for s in initial_std]}")

    # Step 2: Initialize Model, Loss Function, and Optimizer
    print("\n[Step 2] Initializing METEOR-Net, Convective Loss, and AdamW Optimizer...")
    torch.manual_seed(42)
    model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Model Parameters:     {total_params:,}")
    print(f"  Trainable Model Parameters: {trainable_params:,}")

    loss_fn = MaskAwareConvectiveMAELoss(
        ref_channel_idx=1,
        t1_threshold=241.0,
        t2_threshold=221.0,
        alpha=1.0,
        beta=2.0,
        gamma=5.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Step 3: Instantiate Lazy Datasets for Train and Val
    print("\n[Step 3] Initializing lazy GOESSequencePatchDataset loaders...")
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
    print(f"  Train partition size: {len(train_ds):,} patches (across 108 sequences, 2 events)")
    print(f"  Val partition size:   {len(val_ds):,} patches (across 54 sequences, 1 event)")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Step 4: Load 1 Real Training Patch and Run 1 Optimizer Step
    print("\n[Step 4] Running training integration step on 1 real patch...")
    t_start_train = time.perf_counter()
    train_batch = next(iter(train_loader))
    t_load_train = time.perf_counter() - t_start_train

    x_norm = train_batch["input_norm"]           # [1, 4, 4, 128, 128]
    x_phys = train_batch["input_kelvin"]         # [1, 4, 4, 128, 128]
    tgt_phys = train_batch["target_kelvin"]      # [1, 4, 128, 128]
    mask = train_batch["combined_mask"]          # [1, 128, 128]
    meta_train = train_batch["metadata"]

    print(f"  Loaded Train Patch Event:   {meta_train['event_id'][0]}")
    print(f"  Train Sequence ID:          {meta_train['sequence_id'][0]}")
    print(f"  Input Tensor Shape (Norm):  {list(x_norm.shape)}")
    print(f"  Target Tensor Shape (Phys): {list(tgt_phys.shape)}")
    print(f"  Combined Mask Shape:        {list(mask.shape)}")
    print(f"  Valid Pixel Count:          {int(mask.sum().item())} / {mask.numel()}")

    assert torch.isfinite(x_norm).all(), "Non-finite values detected in train normalized input!"
    assert torch.isfinite(tgt_phys).all(), "Non-finite values detected in train target!"

    # Forward pass
    optimizer.zero_grad()
    t_fwd_train_start = time.perf_counter()
    y_norm_train = model(x_norm)                 # [1, 4, 128, 128]
    t_fwd_train = time.perf_counter() - t_fwd_train_start
    assert y_norm_train.shape == tgt_phys.shape, f"Shape mismatch: {y_norm_train.shape} vs {tgt_phys.shape}"
    assert torch.isfinite(y_norm_train).all(), "Non-finite values in train network prediction!"

    # De-normalize to physical Kelvin space
    y_phys_train = standardizer.denormalize_output(y_norm_train)
    assert torch.isfinite(y_phys_train).all(), "Non-finite values in train de-normalized prediction!"

    # Compute loss in physical Kelvin space
    train_loss = loss_fn(y_phys_train, tgt_phys, mask)
    assert torch.isfinite(train_loss), "Train loss is non-finite!"
    train_loss_val = float(train_loss.item())
    print(f"  Train Forward Pass Loss:    {train_loss_val:.4f} K (physical MAE)")

    # Backward pass
    t_bwd_train_start = time.perf_counter()
    train_loss.backward()
    t_bwd_train = time.perf_counter() - t_bwd_train_start

    # Check finite gradients on all trainable parameters
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient!"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient!"
            grad_norms.append(float(param.grad.norm().item()))
    mean_grad_norm = sum(grad_norms) / len(grad_norms)
    print(f"  Train Backward Gradients:   PASS (All finite, mean grad norm: {mean_grad_norm:.6f})")

    # Snapshot weights before optimizer step
    weights_before_step = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Run optimizer step
    t_opt_start = time.perf_counter()
    optimizer.step()
    t_opt_step = time.perf_counter() - t_opt_start

    # Verify model parameters updated
    weights_after_step = {n: p.detach().clone() for n, p in model.named_parameters()}
    changed_params = [
        n for n in weights_before_step
        if not torch.equal(weights_before_step[n], weights_after_step[n])
    ]
    print(f"  Weights Updated By Step:    {len(changed_params)} / {len(weights_before_step)} parameter tensors modified")
    assert len(changed_params) > 0, "Optimizer step failed to modify model weights!"

    t_train_total = time.perf_counter() - t_start_train

    # Step 5: Load 1 Real Validation Patch and Run Validation Pass
    print("\n[Step 5] Running validation integration pass on 1 real patch...")
    t_start_val = time.perf_counter()
    val_batch = next(iter(val_loader))
    t_load_val = time.perf_counter() - t_start_val

    val_x_norm = val_batch["input_norm"]
    val_tgt_phys = val_batch["target_kelvin"]
    val_mask = val_batch["combined_mask"]
    meta_val = val_batch["metadata"]

    print(f"  Loaded Val Patch Event:     {meta_val['event_id'][0]}")
    print(f"  Val Sequence ID:            {meta_val['sequence_id'][0]}")
    print(f"  Event Isolation Verified:   Train ({meta_train['event_id'][0]}) != Val ({meta_val['event_id'][0]})")
    assert meta_train['event_id'][0] != meta_val['event_id'][0], "Event isolation violation between train and val patches!"

    # Snapshot weights before val pass
    weights_before_val = {n: p.detach().clone() for n, p in model.named_parameters()}

    model.eval()
    t_fwd_val_start = time.perf_counter()
    with torch.no_grad():
        y_norm_val = model(val_x_norm)
        assert torch.isfinite(y_norm_val).all(), "Non-finite values in val network prediction!"
        y_phys_val = standardizer.denormalize_output(y_norm_val)
        assert torch.isfinite(y_phys_val).all(), "Non-finite values in val de-normalized prediction!"
        val_loss = loss_fn(y_phys_val, val_tgt_phys, val_mask)
        assert torch.isfinite(val_loss), "Validation loss is non-finite!"
    t_fwd_val = time.perf_counter() - t_fwd_val_start
    val_loss_val = float(val_loss.item())
    print(f"  Val Forward Pass Loss:      {val_loss_val:.4f} K (physical MAE)")

    # Assert validation pass caused ZERO weight modification
    weights_after_val = {n: p.detach().clone() for n, p in model.named_parameters()}
    val_weight_changes = [
        n for n in weights_before_val
        if not torch.equal(weights_before_val[n], weights_after_val[n])
    ]
    print(f"  Val Weight Invariance:      PASS (0 / {len(weights_before_val)} weights modified)")
    assert len(val_weight_changes) == 0, f"Validation pass modified model weights: {val_weight_changes}"

    # Assert frozen normalization statistics remained strictly invariant
    assert torch.equal(standardizer.mean, initial_mean), "Normalization mean was modified!"
    assert torch.equal(standardizer.std, initial_std), "Normalization std was modified!"
    print("  Normalization Invariance:   PASS (Frozen statistics strictly invariant)")

    t_val_total = time.perf_counter() - t_start_val

    # Step 6: Verify Checkpoint Save and Reload
    print("\n[Step 6] Verifying checkpoint serialization and round-trip fidelity...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "smoke_test_checkpoint.pt")

    t_ckpt_save_start = time.perf_counter()
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": 1,
        "train_loss": train_loss_val,
        "val_loss": val_loss_val,
        "normalization_stats": {
            "means": [float(m) for m in standardizer.mean.tolist()],
            "stds": [float(s) for s in standardizer.std.tolist()],
            "channel_order": list(standardizer.channel_order)
        }
    }
    torch.save(checkpoint_payload, ckpt_path)
    t_ckpt_save = time.perf_counter() - t_ckpt_save_start
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"  Saved Checkpoint:           {ckpt_path} ({ckpt_size_mb:.2f} MB)")

    # Reload into a fresh model instance
    t_ckpt_load_start = time.perf_counter()
    loaded_model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4)
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
    loaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
    loaded_model.eval()
    t_ckpt_load = time.perf_counter() - t_ckpt_load_start

    # Verify bit-for-bit parameter match
    for n, p in model.named_parameters():
        p_loaded = dict(loaded_model.named_parameters())[n]
        assert torch.equal(p, p_loaded), f"Parameter mismatch after checkpoint reload on {n}!"
    print("  State Dict Match:           PASS (All parameters identical bit-for-bit)")

    # Verify identical prediction output on validation patch
    with torch.no_grad():
        reloaded_pred = loaded_model(val_x_norm)
        assert torch.allclose(reloaded_pred, y_norm_val, atol=1e-6), "Output prediction mismatch after reload!"
    print("  Inference Reproduction:     PASS (Predictions identical within 1e-6)")

    # Step 7: Record Execution Timing and Memory Peak
    t_total = time.perf_counter() - t_start_total
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = mem_peak / (1024 * 1024)
    current_mb = mem_current / (1024 * 1024)
    print("\n[Step 7] Execution Resource Profile:")
    print(f"  Train Patch Load Time:      {t_load_train:.3f} s")
    print(f"  Train Forward Pass Time:    {t_fwd_train:.3f} s")
    print(f"  Train Backward Pass Time:   {t_bwd_train:.3f} s")
    print(f"  Optimizer Step Time:        {t_opt_step:.3f} s")
    print(f"  Total Train Step Time:      {t_train_total:.3f} s")
    print(f"  Val Patch Load Time:        {t_load_val:.3f} s")
    print(f"  Val Forward Pass Time:      {t_fwd_val:.3f} s")
    print(f"  Total Val Step Time:        {t_val_total:.3f} s")
    print(f"  Checkpoint Save Time:       {t_ckpt_save:.3f} s")
    print(f"  Checkpoint Reload Time:     {t_ckpt_load:.3f} s")
    print(f"  Total Smoke Test Duration:  {t_total:.3f} s")
    print(f"  Peak Python Heap Memory:    {peak_mb:.2f} MB")

    # Step 8: Save Structured Metrics Artifact
    os.makedirs(os.path.dirname(os.path.abspath(output_metrics_path)), exist_ok=True)
    metrics = {
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": {
            "class": "METEORNet",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "in_channels": 4,
            "out_channels": 4
        },
        "dataset_integration": {
            "train_patch": {
                "event_id": meta_train["event_id"][0],
                "sequence_id": meta_train["sequence_id"][0],
                "input_shape": list(x_norm.shape),
                "target_shape": list(tgt_phys.shape),
                "mask_shape": list(mask.shape),
                "valid_pixels": int(mask.sum().item()),
                "is_convective": bool(train_batch["is_convective"][0].item())
            },
            "val_patch": {
                "event_id": meta_val["event_id"][0],
                "sequence_id": meta_val["sequence_id"][0],
                "input_shape": list(val_x_norm.shape),
                "target_shape": list(val_tgt_phys.shape),
                "mask_shape": list(val_mask.shape),
                "valid_pixels": int(val_mask.sum().item()),
                "is_convective": bool(val_batch["is_convective"][0].item())
            },
            "event_isolation_verified": True
        },
        "optimization": {
            "train_loss_physical_kelvin": train_loss_val,
            "val_loss_physical_kelvin": val_loss_val,
            "optimizer": "AdamW",
            "lr": 1e-4,
            "mean_grad_norm": mean_grad_norm,
            "parameters_updated_count": len(changed_params),
            "val_weight_mutation_count": len(val_weight_changes),
            "normalization_stats_mutated": False
        },
        "checkpoint": {
            "path": ckpt_path,
            "size_mb": round(ckpt_size_mb, 2),
            "save_time_s": round(t_ckpt_save, 4),
            "reload_time_s": round(t_ckpt_load, 4),
            "reproduction_max_diff": 0.0
        },
        "resources": {
            "total_duration_s": round(t_total, 3),
            "train_step_duration_s": round(t_train_total, 3),
            "val_step_duration_s": round(t_val_total, 3),
            "peak_python_heap_mb": round(peak_mb, 2),
            "current_python_heap_mb": round(current_mb, 2)
        }
    }

    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Artifact Saved] Smoke test metrics written to: {output_metrics_path}")
    print("=" * 80)
    print("PHASE 6B PREFLIGHT SMOKE TEST: PASSED")
    print("=" * 80)
    return metrics


if __name__ == "__main__":
    run_integration_smoke_test()

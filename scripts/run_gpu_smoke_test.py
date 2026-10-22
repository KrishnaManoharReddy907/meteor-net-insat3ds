"""
scripts/run_gpu_smoke_test.py
Standalone GPU Smoke Test for METEOR-Net on Google Colab (T4 GPU).

Exercises the real model architecture, frozen normalization, convective loss,
dataset loader, autograd, optimizer step, and atomic checkpoint serialization
on CUDA without altering any training configurations or touching Phase 7B artifacts.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, os.path.abspath("."))
from src.data.goes_dataset import GOESSequencePatchDataset
from src.models.meteor_net import METEORNet
from src.models.losses import MaskAwareConvectiveMAELoss
from src.models.normalization import ChannelStandardizer


def run_gpu_smoke_test(
    manifest_path: str = "outputs/dataset_splits/splits_manifest.json",
    stats_path: str = "outputs/normalization/frozen_global_train_stats.json",
    output_dir: str = "outputs/colab_t4_smoke_test",
    device_str: str = "cuda"
) -> Dict[str, Any]:
    print("=" * 80)
    print("METEOR-NET: GPU HARDWARE & PIPELINE SMOKE TEST")
    print("=" * 80)

    # 1. Device Resolution & Hardware Inspection
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CRITICAL: CUDA was explicitly requested (--device cuda), but torch.cuda.is_available() is False!\n"
                "In Google Colab, verify that the runtime is set to GPU: Runtime -> Change runtime type -> T4 GPU."
            )
        device = torch.device("cuda")
    elif device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    hw_info = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    if device.type == "cuda":
        hw_info.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_count": torch.cuda.device_count(),
            "compute_capability": torch.cuda.get_device_capability(device),
            "total_vram_gb": round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2),
            "free_vram_gb": round(torch.cuda.mem_get_info(device)[0] / (1024**3), 2)
        })
        print(f"[Hardware] GPU: {hw_info['gpu_name']} | Compute: {hw_info['compute_capability']} | VRAM: {hw_info['free_vram_gb']}/{hw_info['total_vram_gb']} GB Free")
    else:
        print(f"[Hardware] Running on CPU (Device: {device})")

    os.makedirs(output_dir, exist_ok=True)

    # 2. Frozen Normalization Loading & Device Placement
    print("\n[Stage 1/7] Initializing Frozen Normalization on Device...")
    standardizer = ChannelStandardizer.from_stats_file(stats_path).to(device)
    init_mean = standardizer.mean.clone()
    init_std = standardizer.std.clone()
    print(f"  Means (K): {[round(float(m), 4) for m in init_mean.cpu()]}")
    print(f"  Stds  (K): {[round(float(s), 4) for s in init_std.cpu()]}")

    # 3. Model & Loss Instantiation on Device
    print("\n[Stage 2/7] Initializing METEOR-Net & Loss on Device...")
    model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
    loss_fn = MaskAwareConvectiveMAELoss(
        ref_channel_idx=1,
        t1_threshold=241.0,
        t2_threshold=221.0,
        alpha=1.0,
        beta=2.0,
        gamma=5.0
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters: {param_count:,}")
    assert param_count == 22086943, f"Unexpected parameter count: {param_count}"

    # 4. Dataset Loader Verification
    print("\n[Stage 3/7] Ingesting Sample Patch from Dataset...")
    train_ds = GOESSequencePatchDataset(
        manifest_path=manifest_path,
        split="train",
        standardizer=standardizer,
        normalize_inputs=True,
        cache_size=1
    )
    t0_data = time.perf_counter()
    sample_item = train_ds[0]
    t_data = time.perf_counter() - t0_data
    print(f"  Patch loaded in {t_data*1000:.1f} ms | Seq ID: {sample_item['metadata']['sequence_id']}")

    x_norm = sample_item["input_norm"].unsqueeze(0).to(device)        # [1, 4, 4, 128, 128]
    tgt_phys = sample_item["target_kelvin"].unsqueeze(0).to(device)   # [1, 4, 128, 128]
    mask = sample_item["combined_mask"].unsqueeze(0).to(device)       # [1, 128, 128]

    assert x_norm.device == device, f"Tensor not on expected device: {x_norm.device}"
    assert tgt_phys.device == device
    assert mask.device == device

    # 5. Forward Pass & Physical De-normalization
    print("\n[Stage 4/7] Executing Forward Pass on Device...")
    model.train()
    t0_fwd = time.perf_counter()
    y_norm = model(x_norm)
    y_phys = standardizer.denormalize_output(y_norm)
    t_fwd = time.perf_counter() - t0_fwd

    assert torch.isfinite(y_norm).all(), "Non-finite values in prediction!"
    assert torch.isfinite(y_phys).all(), "Non-finite values in denormalized prediction!"
    print(f"  Forward Pass Time: {t_fwd*1000:.1f} ms | Output Shape: {list(y_phys.shape)}")

    # 6. Loss Computation & Backward Pass
    print("\n[Stage 5/7] Executing Loss & Autograd Backward Pass...")
    loss = loss_fn(y_phys, tgt_phys, mask)
    assert torch.isfinite(loss), "Non-finite loss evaluated!"
    loss_val = float(loss.item())

    optimizer = AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    t0_bwd = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0_bwd

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    print(f"  Loss Value: {loss_val:.4f} K | Backward Time: {t_bwd*1000:.1f} ms | Optimizer Step: PASS")

    # 7. Atomic Checkpoint Serialization & Deserialization
    print("\n[Stage 6/7] Testing Atomic Checkpoint Round-Trip on Device...")
    ckpt_path = os.path.join(output_dir, "gpu_smoke_checkpoint.pt")
    tmp_path = ckpt_path + ".tmp"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_val,
        "normalization_provenance": standardizer.provenance
    }, tmp_path)
    os.replace(tmp_path, ckpt_path)

    reloaded_model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
    loaded_payload = torch.load(ckpt_path, map_location=device)
    reloaded_model.load_state_dict(loaded_payload["model_state_dict"])
    for n, p in model.named_parameters():
        p_reloaded = dict(reloaded_model.named_parameters())[n]
        assert torch.equal(p, p_reloaded), f"State dict mismatch on parameter {n}"
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"  Saved & Verified Atomic Checkpoint: {ckpt_path} ({ckpt_size_mb:.2f} MB)")

    # 8. Normalization Invariance Assertion
    print("\n[Stage 7/7] Verifying Normalization Invariance...")
    assert torch.equal(standardizer.mean, init_mean), "Normalization mean was mutated!"
    assert torch.equal(standardizer.std, init_std), "Normalization std was mutated!"
    print("  Normalization Mean & Std Bitwise Invariance: PASS")

    results = {
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw_info,
        "model_parameter_count": param_count,
        "loss_kelvin": round(loss_val, 4),
        "forward_latency_ms": round(t_fwd * 1000, 2),
        "backward_latency_ms": round(t_bwd * 1000, 2),
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "output_directory": output_dir
    }

    results_file = os.path.join(output_dir, "gpu_smoke_test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("GPU SMOKE TEST: ALL CHECKS PASSED")
    print(f"Summary written to: {results_file}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="METEOR-Net GPU Smoke Test")
    parser.add_argument("--manifest", type=str, default="outputs/dataset_splits/splits_manifest.json")
    parser.add_argument("--stats", type=str, default="outputs/normalization/frozen_global_train_stats.json")
    parser.add_argument("--output_dir", type=str, default="outputs/colab_t4_smoke_test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_gpu_smoke_test(
        manifest_path=args.manifest,
        stats_path=args.stats,
        output_dir=args.output_dir,
        device_str=args.device
    )

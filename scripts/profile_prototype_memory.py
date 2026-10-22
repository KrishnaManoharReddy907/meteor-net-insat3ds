"""
scripts/profile_prototype_memory.py
Profiles model memory usage, activation graph memory, and parameter footprint
for the METEOR-Net prototype across batch sizes B in {1, 2, 4}.
Detects whether CUDA GPU is available or CPU execution is active.
"""

import os
import sys
import time
import tracemalloc
import torch

sys.path.insert(0, os.path.abspath("."))
from src.models.meteor_net import METEORNet
from src.models.losses import MaskAwareConvectiveMAELoss


def profile_prototype(batch_sizes=(1, 2, 4)):
    print("=" * 80)
    print("METEOR-NET PROTOTYPE HARDWARE AND MEMORY PROFILING REPORT")
    print("=" * 80)
    
    # 1. Hardware Inspection
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Detected CUDA Device: {device_name}")
        print(f"Total Dedicated VRAM: {total_vram_gb:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Running on: CPU / System Host Memory (No CUDA device detected in active PyTorch build)")
        
    # 2. Instantiated Model Parameter Audit
    model = METEORNet().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameter Audit:")
    print(f"  Trainable Parameters: {total_params:,}")
    print(f"  Model Weight Storage: {weight_bytes / (1024 * 1024):.2f} MB ({weight_bytes / 1e6:.2f} MB decimal)")
    print(f"  MCE Subtotal:         {sum(p.numel() for p in model.mce.parameters() if p.requires_grad):,}")
    print(f"  TFEE Subtotal:        {sum(p.numel() for p in model.tfee.parameters() if p.requires_grad):,}")
    print(f"  EFFR Subtotal:        {sum(p.numel() for p in model.effr.parameters() if p.requires_grad):,}")
    
    loss_fn = MaskAwareConvectiveMAELoss().to(device)
    
    print("\n" + "-" * 80)
    print("BATCH SIZE SCALING EXPERIMENT (Patch Size 128x128, Precision float32)")
    print("-" * 80)
    print(f"{'Batch':<8}{'Forward (s)':<15}{'Backward (s)':<15}{'Memory Metric':<35}{'Status'}")
    print("-" * 80)
    
    for B in batch_sizes:
        try:
            # Create synthetic input, target, and mask
            inputs = torch.randn(B, 4, 4, 128, 128, device=device)
            targets = torch.full((B, 4, 128, 128), 280.0, device=device)
            mask = torch.ones((B, 4, 128, 128), dtype=torch.bool, device=device)
            
            if cuda_avail:
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.synchronize()
                
            t0 = time.perf_counter()
            outputs = model(inputs)
            if cuda_avail:
                torch.cuda.synchronize()
            t_fwd = time.perf_counter() - t0
            
            loss = loss_fn(outputs, targets, mask)
            
            t0 = time.perf_counter()
            model.zero_grad()
            loss.backward()
            if cuda_avail:
                torch.cuda.synchronize()
            t_bwd = time.perf_counter() - t0
            
            if cuda_avail:
                alloc_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
                reserv_mb = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
                mem_str = f"Alloc: {alloc_mb:.1f} MB | Res: {reserv_mb:.1f} MB"
            else:
                # CPU tensor size estimation for activations
                mem_str = f"Loss={loss.item():.2f} (CPU execution)"
                
            print(f"B={B:<6}{t_fwd:<15.3f}{t_bwd:<15.3f}{mem_str:<35}SUCCESS")
            
        except Exception as e:
            print(f"B={B:<6}{'FAILED':<15}{'FAILED':<15}{str(e)[:35]:<35}OOM/ERROR")
            
    print("=" * 80)


if __name__ == "__main__":
    profile_prototype(batch_sizes=(1, 2, 4))

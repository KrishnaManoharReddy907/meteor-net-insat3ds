"""
scripts/profile_fast_pipeline.py
Measures (not guesses) where time goes, and projects GPU-hours for a training budget.

Reports
  1. loader throughput  : numpy gather of mixed-gap batches (+ prefetcher)
  2. model step time    : fwd + bwd + AdamW for several microbatch sizes, fp32 vs fp16-AMP,
                          with peak GPU memory and a finite-loss check
  3. projections        : hours for N optimizer updates at an effective batch size
  4. (optional) legacy  : time per item of the OLD NetCDF dataset, to confirm the I/O diagnosis

It trains nothing persistent: no checkpoints, no writes outside --output_dir.
It refuses to load any event that the manifest lists in the 'test' split.

Example (Colab, after building the cache)
  python scripts/profile_fast_pipeline.py --cache_dir /content/cache/goes19_v1 \
      --events event_2026-01-01,event_2026-05-20 \
      --stats outputs/normalization/frozen_global_train_stats.json \
      --manifest outputs/dataset_splits/splits_manifest.json --device cuda
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.fast_dataset import FastSequenceStore, BatchPrefetcher, to_device  # noqa: E402


def test_event_ids(manifest_path):
    if not manifest_path or not os.path.exists(manifest_path):
        return set()
    m = json.load(open(manifest_path))
    return {w["group_id"] for w in m["splits"].get("test", [])}


def bench_loader(store, batch_size, n_batches, seed):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        store.gather(store.sample_train_ids(rng, batch_size))
    serial = (time.perf_counter() - t0) / n_batches
    rng = np.random.default_rng(seed)
    pf = BatchPrefetcher(lambda i: store.gather(store.sample_train_ids(rng, batch_size)), n_batches)
    t0 = time.perf_counter()
    for _ in pf:
        pass
    pre = (time.perf_counter() - t0) / n_batches
    return {"batch_size": batch_size, "ms_per_batch_serial": 1e3 * serial, "ms_per_batch_prefetch": 1e3 * pre,
            "patches_per_s_serial": batch_size / serial}


def bench_model(store, standardizer, device, batch_size, amp, steps, warmup, seed):
    import torch
    from src.models.meteor_net import METEORNet
    from src.models.losses import MaskAwareConvectiveMAELoss

    torch.manual_seed(seed)
    model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
    loss_fn = MaskAwareConvectiveMAELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = np.random.default_rng(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batches = [to_device(store.gather(store.sample_train_ids(rng, batch_size)), device, standardizer)
               for _ in range(warmup + steps)]
    finite, times = True, []
    model.train()
    for i, b in enumerate(batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            y_norm = model(b["input_norm"])
        y_phys = standardizer.denormalize_output(y_norm.float())
        loss = loss_fn(y_phys, b["target_kelvin"], b["combined_mask"])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        finite &= bool(torch.isfinite(loss).item())
        if i >= warmup:
            times.append(dt)
    sec = float(np.mean(times))
    return {
        "batch_size": batch_size, "amp_fp16": use_amp, "sec_per_step": sec,
        "patches_per_s": batch_size / sec, "finite_loss": finite,
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
    }


def bench_legacy(manifest, n_items, seed):
    from src.data.goes_dataset import GOESSequencePatchDataset  # legacy NetCDF-backed dataset
    ds = GOESSequencePatchDataset(manifest_path=manifest, split="train", cache_size=2)
    rnd = random.Random(seed)
    idx = [rnd.randrange(len(ds)) for _ in range(n_items)]
    t0 = time.perf_counter()
    for i in idx:
        _ = ds[i]
    return {"items": n_items, "sec_per_item": (time.perf_counter() - t0) / n_items}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--events", required=True, help="Comma-separated TRAIN/VAL event ids (never test).")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gaps", default="10,20,30")
    ap.add_argument("--batch_sizes", default="1,4,8,16")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--in_ram", action="store_true")
    ap.add_argument("--legacy_manifest", default=None, help="Optional: time the old dataset too (needs raw .nc files).")
    ap.add_argument("--output_dir", default="outputs/step_a_profile")
    args = ap.parse_args()

    import torch
    from src.models.normalization import ChannelStandardizer

    events = [e for e in args.events.split(",") if e]
    forbidden = set(events) & test_event_ids(args.manifest)
    if forbidden:
        sys.exit(f"SAFETY: refusing to load held-out test event(s): {sorted(forbidden)}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    standardizer = ChannelStandardizer.from_stats_file(args.stats).to(device)
    gaps = [int(g) for g in args.gaps.split(",")]

    t0 = time.perf_counter()
    store = FastSequenceStore(args.cache_dir, events, gaps_min=gaps, in_ram=args.in_ram)
    index_s = time.perf_counter() - t0
    desc = store.describe()
    print(f"Indexed {desc['n_samples']:,} samples from {desc['n_events']} events in {index_s:.1f}s")
    print("Pools:", desc["pool_sizes"])

    results = {"device": str(device), "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "index_seconds": index_s, "dataset": desc, "loader": [], "model": []}

    for bs in (int(b) for b in args.batch_sizes.split(",")):
        r = bench_loader(store, bs, 20, args.seed)
        results["loader"].append(r)
        print(f"[loader] bs={bs:>2}: {r['ms_per_batch_serial']:.1f} ms/batch serial, "
              f"{r['ms_per_batch_prefetch']:.1f} ms/batch prefetched")

    for bs in (int(b) for b in args.batch_sizes.split(",")):
        for amp in ([False, True] if device.type == "cuda" else [False]):
            try:
                r = bench_model(store, standardizer, device, bs, amp, args.steps, args.warmup, args.seed)
            except RuntimeError as e:  # typically CUDA OOM
                r = {"batch_size": bs, "amp_fp16": amp, "error": str(e)[:160]}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            results["model"].append(r)
            if "error" in r:
                print(f"[model]  bs={bs:>2} amp={amp}: FAILED {r['error']}")
            else:
                print(f"[model]  bs={bs:>2} amp={amp!s:5}: {r['sec_per_step']*1e3:7.0f} ms/step, "
                      f"{r['patches_per_s']:6.1f} patches/s, finite={r['finite_loss']}, "
                      f"peak={r['peak_gpu_mem_mb'] and round(r['peak_gpu_mem_mb'])} MB")

    ok = [r for r in results["model"] if "error" not in r and r["finite_loss"]]
    if ok:
        best = max(ok, key=lambda r: r["patches_per_s"])
        results["best_config"] = best
        print(f"\nBest stable config: bs={best['batch_size']} amp={best['amp_fp16']} -> {best['patches_per_s']:.1f} patches/s")
        print("Projected GPU-hours for N training patch-samples (1 'update' = 16 samples):")
        proj = {}
        for updates in (2000, 4000, 6000, 10000):
            hrs = updates * 16 / best["patches_per_s"] / 3600
            proj[str(updates)] = round(hrs, 2)
            print(f"  {updates:>6} updates ({updates*16:>7,} samples): {hrs:5.2f} h")
        results["projected_hours_by_updates"] = proj

    if args.legacy_manifest:
        try:
            results["legacy"] = bench_legacy(args.legacy_manifest, 20, args.seed)
            print(f"[legacy] {results['legacy']['sec_per_item']:.2f} s/item (old NetCDF dataset, cache_size=2)")
        except Exception as e:  # raw data may not be present on this machine
            results["legacy"] = {"error": str(e)[:200]}
            print("[legacy] skipped:", results["legacy"]["error"])

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "profile_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", out)


if __name__ == "__main__":
    main()

"""
scripts/eval_image_metrics.py -- SSIM / PSNR / inter-band (BTD) error / cold-core error, on the SAME fixed
validation patches as eval_baselines.py. Validation events only; refuses a test cache.

Per gap (10/20/30): B13 SSIM and PSNR (fixed 100 K range; SSIM mean over channels also reported),
MAE of the inter-band differences B9-B13, B14-B13, B15-B13, and for patches whose true B13 minimum is below
221 K: error of the patch's predicted coldest-pixel temperature (cold-core MAE and bias).

  python scripts/eval_image_metrics.py --cache_dir /content/cache/dev --events_json configs/events_v2.json \
     --stats outputs/normalization/v2_train_stats.json --out outputs/image_metrics_val.json \
     --ckpt residual_12k=<runs>/residual_12k/best.pt:residual --ckpt spec05=<runs>/residual_spec05/best.pt:residual
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="outputs/image_metrics_val.json")
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=PATH[:residual|:farneback]")
    ap.add_argument("--per_cell", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from src.data.fast_dataset import FastSequenceStore, to_device
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from src.models.flow_base import farneback_batch
    from src.eval.image_metrics import psnr, ssim
    from train_v2 import build_val_ids, load_split_ids, predict

    idx = os.path.join(args.cache_dir, "cache_index.json")
    if os.path.exists(idx) and json.load(open(idx)).get("cache_role") == "test":
        sys.exit("REFUSED: cache_dir is a TEST cache.")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = device.type == "cuda"
    st = ChannelStandardizer.from_stats_file(args.stats).to(device)
    store = FastSequenceStore(args.cache_dir, load_split_ids(args.events_json, "val"))
    ids = build_val_ids(store, args.per_cell, seed=1234)

    models = {}
    for spec in args.ckpt:
        name, rest = spec.split("=", 1)
        mode, path = None, rest
        for tag in ("residual", "farneback"):
            if rest.endswith(":" + tag):
                mode, path = (True if tag == "residual" else "farneback"), rest[:-(len(tag) + 1)]
        net = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"]); net.eval()
        models[name] = (net, mode)
    methods = ["linear", "farneback"] + list(models)
    rows = {}    # (method, gap) -> list of dicts

    n_batches = (len(ids) + args.bs - 1) // args.bs
    print(f"Evaluating {len(ids)} patches across {n_batches} batches...")
    with torch.no_grad():
        for b_idx, batch in enumerate(store.iter_eval_batches(ids, args.bs)):
            if b_idx % 20 == 0 or b_idx == n_batches - 1:
                print(f"  [Eval] Batch {b_idx + 1}/{n_batches} ({min((b_idx + 1) * args.bs, len(ids))}/{len(ids)} patches)...")
            b = to_device(batch, device, st)
            x, tgt = b["input_kelvin"], torch.nan_to_num(b["target_kelvin"], nan=250.0)
            preds = {"linear": torch.nan_to_num(0.5 * (x[:, 2] + x[:, 3]), nan=250.0),
                     "farneback": torch.from_numpy(farneback_batch(x[:, 2].cpu().numpy(), x[:, 3].cpu().numpy())).to(device)}
            for n, (net, mode) in models.items():
                preds[n] = predict(net, st, b, mode, amp, device)
            T = tgt.cpu().numpy()
            gaps = b["gap_min"].cpu().numpy()
            for meth in methods:
                P = torch.nan_to_num(preds[meth], nan=250.0).cpu().numpy()
                for i in range(P.shape[0]):
                    r = {"psnr13": psnr(P[i, 1], T[i, 1]), "ssim13": ssim(P[i, 1], T[i, 1]),
                         "ssim_mean4": float(np.mean([ssim(P[i, c], T[i, c]) for c in range(4)]))}
                    for nm, c in (("btd_9_13", 0), ("btd_14_13", 2), ("btd_15_13", 3)):
                        r[nm] = float(np.abs((P[i, c] - P[i, 1]) - (T[i, c] - T[i, 1])).mean())
                    tmin = float(T[i, 1].min())
                    if tmin < 221.0:
                        r["core_err"] = float(P[i, 1].min() - tmin)
                    rows.setdefault((meth, int(gaps[i])), []).append(r)

    out, keys = {}, ["psnr13", "ssim13", "ssim_mean4", "btd_9_13", "btd_14_13", "btd_15_13"]
    for (meth, g), lst in rows.items():
        r = {k: float(np.mean([d[k] for d in lst if np.isfinite(d[k])])) for k in keys}
        ce = [d["core_err"] for d in lst if "core_err" in d]
        r["cold_core_n"] = len(ce)
        r["cold_core_mae"] = float(np.mean(np.abs(ce))) if ce else float("nan")
        r["cold_core_bias"] = float(np.mean(ce)) if ce else float("nan")
        out[f"{meth}|gap{g}"] = r
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    for g in (10, 20, 30):
        print(f"\n=== gap {g} min (validation) ===")
        print(f"{'method':<16}{'PSNR13':>8}{'SSIM13':>8}{'SSIM4':>8}{'BTD9-13':>9}{'BTD14-13':>9}{'BTD15-13':>9}{'coreMAE':>9}{'coreBias':>9}{'n':>5}")
        for meth in methods:
            r = out[f"{meth}|gap{g}"]
            print(f"{meth:<16}{r['psnr13']:8.2f}{r['ssim13']:8.4f}{r['ssim_mean4']:8.4f}{r['btd_9_13']:9.3f}"
                  f"{r['btd_14_13']:9.3f}{r['btd_15_13']:9.3f}{r['cold_core_mae']:9.3f}{r['cold_core_bias']:9.3f}{r['cold_core_n']:5d}")
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()

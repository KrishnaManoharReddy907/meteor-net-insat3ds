"""
scripts/uq_baselines.py -- is the learned confidence head better than simpler uncertainty?

Same calibration/validation pixels as calibrate_conformal.py (same patches, seeds). Three interval methods, each
split-conformal calibrated on the CAL events with Mondrian groups (gap x predicted intensity class), 90% and 95%:
  constant : scale = 1                     (width depends only on gap and predicted class)
  ensemble : scale = spread of B13 across the given checkpoints (+0.05 K floor)
  head     : scale = learned Laplace scale
Compared on VALIDATION by coverage, mean width (lower = tighter at the same coverage), width on strong pixels,
Spearman(scale, |err|). Validation + cal events only; refuses a test cache.

  python scripts/uq_baselines.py ... --ckpt <runs>/residual_12k/best.pt --base linear --head <runs>/conf_head_12k/head.pt \
     --ens <runs>/residual_s0/best.pt:residual --ens <runs>/residual_s1/best.pt:residual --out outputs/uq_baselines_val.json
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

from src.eval import conformal as C  # noqa: E402
from src.data.fast_dataset import FastSequenceStore, to_device  # noqa: E402

ALPHAS = (0.1, 0.05)


def collect(store, ids, forward, st, device, pix, seed, bs=16):
    import torch
    rng = np.random.default_rng(seed)
    n_batches = (len(ids) + bs - 1) // bs
    print(f"Collecting pixels across {len(ids)} patches ({n_batches} batches)...")
    cols = {}
    with torch.no_grad():
        for b_idx, batch in enumerate(store.iter_eval_batches(ids, bs)):
            if b_idx % 25 == 0 or b_idx == n_batches - 1:
                print(f"  [Progress] Batch {b_idx + 1}/{n_batches} ({min((b_idx + 1) * bs, len(ids))}/{len(ids)} patches)...")
            b = to_device(batch, device, st)
            yhat, scales = forward(b)                                   # scales: dict name -> [B,H,W]
            m = b["combined_mask"].cpu().numpy()
            e = (b["target_kelvin"][:, 1] - yhat[:, 1]).abs().cpu().numpy()
            p = yhat[:, 1].cpu().numpy()
            t = torch.nan_to_num(b["target_kelvin"][:, 1], nan=300.0).cpu().numpy()
            sc = {k: v.cpu().numpy() for k, v in scales.items()}
            for i in range(m.shape[0]):
                flat = np.flatnonzero(m[i].ravel())
                if len(flat) == 0:
                    continue
                pick = rng.choice(flat, size=min(pix, len(flat)), replace=False)
                parts = {"err": e[i].ravel()[pick], "pred": p[i].ravel()[pick], "tgt": t[i].ravel()[pick],
                         "gap": np.full(len(pick), batch["gap_min"][i])}
                parts.update({f"s_{k}": v[i].ravel()[pick] for k, v in sc.items()})
                for k, v in parts.items():
                    cols.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", choices=["none", "linear", "farneback"], default="linear")
    ap.add_argument("--head", required=True)
    ap.add_argument("--ens", action="append", default=[], help="extra checkpoint PATH[:residual] for the spread")
    ap.add_argument("--out", default="outputs/uq_baselines_val.json")
    ap.add_argument("--cal_per_cell", type=int, default=300)
    ap.add_argument("--val_per_cell", type=int, default=300)
    ap.add_argument("--pix_per_patch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from src.models.conf_head import ConfHead, build_features
    from train_v2 import build_val_ids, load_split_ids, predict

    idx = os.path.join(args.cache_dir, "cache_index.json")
    if os.path.exists(idx) and json.load(open(idx)).get("cache_role") == "test":
        sys.exit("REFUSED: cache_dir is a TEST cache.")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = device.type == "cuda"
    st = ChannelStandardizer.from_stats_file(args.stats).to(device)

    def load(path):
        net = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"]); net.eval()
        return net

    main_net, main_mode = load(args.ckpt), (None if args.base == "none" else args.base)
    ens = []
    for spec in args.ens:
        mode = True if spec.endswith(":residual") else None
        ens.append((load(spec[:-9] if spec.endswith(":residual") else spec), mode))
    head = ConfHead().to(device)
    head.load_state_dict(torch.load(args.head, map_location=device, weights_only=False)["head"]); head.eval()

    def forward(b):
        yhat = predict(main_net, st, b, main_mode, amp, device).float()
        scales = {"head": head(build_features(st, b, yhat))[:, 1]}
        if ens:
            stack = [yhat[:, 1]] + [predict(n, st, b, md, amp, device).float()[:, 1] for n, md in ens]
            scales["ens"] = torch.stack(stack).std(dim=0) + 0.05
        return yhat, scales

    cal_events, val_events = load_split_ids(args.events_json, "cal"), load_split_ids(args.events_json, "val")
    cs, vs = FastSequenceStore(args.cache_dir, cal_events), FastSequenceStore(args.cache_dir, val_events)
    d_cal = collect(cs, build_val_ids(cs, args.cal_per_cell, 777), forward, st, device, args.pix_per_patch, 1)
    d_val = collect(vs, build_val_ids(vs, args.val_per_cell, 1234), forward, st, device, args.pix_per_patch, 2)

    kc = C.group_keys(d_cal["gap"], C.intensity_class(d_cal["pred"]))
    kv = C.group_keys(d_val["gap"], C.intensity_class(d_val["pred"]))
    cls_true = C.intensity_class(d_val["tgt"])
    methods = {"constant": None, "head": "s_head"}
    if ens:
        methods["ensemble"] = "s_ens"
    res = {}
    for a in ALPHAS:
        for name, col in methods.items():
            sc_c = np.ones_like(d_cal["err"]) if col is None else d_cal[col]
            sc_v = np.ones_like(d_val["err"]) if col is None else d_val[col]
            cal = C.fit_mondrian(d_cal["err"] / sc_c, kc, a, 500)
            q = C.q_for(cal, kv)
            width = 2 * q * sc_v
            res[f"{name}|{a}"] = {
                "coverage": C.coverage(d_val["err"], sc_v, q), "mean_width_K": float(width.mean()),
                "median_width_K": float(np.median(width)),
                "width_strong_true_K": float(width[cls_true == 0].mean()), "cov_strong_true": C.coverage(d_val["err"][cls_true == 0], sc_v[cls_true == 0], q[cls_true == 0]),
                "spearman": C.spearman(sc_v, d_val["err"]) if col is not None else float("nan")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    for a in ALPHAS:
        print(f"\n=== {100*(1-a):.0f}% intervals, VALIDATION, B13 ===")
        print(f"{'scale':<10}{'coverage':>9}{'mean W':>8}{'median W':>9}{'W strong':>9}{'cov strong':>11}{'spearman':>9}")
        for name in methods:
            r = res[f"{name}|{a}"]
            print(f"{name:<10}{r['coverage']:9.3f}{r['mean_width_K']:8.2f}{r['median_width_K']:9.2f}"
                  f"{r['width_strong_true_K']:9.2f}{r['cov_strong_true']:11.3f}{r['spearman']:9.3f}")
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()

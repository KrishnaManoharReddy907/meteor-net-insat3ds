"""
scripts/calibrate_conformal.py -- Extension 1, stage 2: split-conformal calibration + validation report.

Calibrates on the CAL events (never used for training or model selection), evaluates on VAL events.
Band 13, pixel-level. Sampling: --pix_per_patch random valid pixels from each sampled patch.
Mondrian groups = (gap, predicted-intensity class: strong <221 K / weak 221-241 K / ambient >=241 K).
Writes <out> (JSON) with the calibration table so the final TEST evaluation reuses it unchanged.

  python scripts/calibrate_conformal.py --cache_dir /content/cache/dev --events_json configs/events_v2.json \
     --stats outputs/normalization/v2_train_stats.json --ckpt <runs>/residual_12k/best.pt --base linear \
     --head <runs>/conf_head_12k/head.pt --out outputs/conformal_val.json
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


def collect(store, ids, forward, st, device, pix_per_patch, seed, bs=16):
    import torch
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("err", "scale", "pred", "tgt", "gap", "event")}
    with torch.no_grad():
        for batch in store.iter_eval_batches(ids, bs):
            b = to_device(batch, device, st)
            yhat, sc = forward(b)
            m = b["combined_mask"].cpu().numpy()
            e = (b["target_kelvin"][:, 1] - yhat[:, 1]).abs().cpu().numpy()
            s, p = sc[:, 1].cpu().numpy(), yhat[:, 1].cpu().numpy()
            t = torch.nan_to_num(b["target_kelvin"][:, 1], nan=300.0).cpu().numpy()
            for i in range(m.shape[0]):
                flat = np.flatnonzero(m[i].ravel())
                if len(flat) == 0:
                    continue
                pick = rng.choice(flat, size=min(pix_per_patch, len(flat)), replace=False)
                cols["err"].append(e[i].ravel()[pick]); cols["scale"].append(s[i].ravel()[pick])
                cols["pred"].append(p[i].ravel()[pick]); cols["tgt"].append(t[i].ravel()[pick])
                cols["gap"].append(np.full(len(pick), batch["gap_min"][i])); cols["event"].append(np.full(len(pick), batch["event_idx"][i]))
    return {k: np.concatenate(v) for k, v in cols.items()}


def report(d, cal_by_alpha, event_names):
    out = {}
    cls_pred = C.intensity_class(d["pred"])
    cls_true = C.intensity_class(d["tgt"])
    keys = C.group_keys(d["gap"], cls_pred)
    for a in ALPHAS:
        cal = cal_by_alpha[a]
        q_m = C.q_for(cal, keys)
        q_g = cal["global"]["q"]
        raw_q = float(np.log(1.0 / a))                       # Laplace-implied half-width in units of scale
        r = {"nominal": 1 - a,
             "raw_laplace_cov": C.coverage(d["err"], d["scale"], raw_q),
             "global_cov": C.coverage(d["err"], d["scale"], q_g),
             "mondrian_cov": C.coverage(d["err"], d["scale"], q_m),
             "mondrian_mean_width_K": float(np.mean(2 * q_m * d["scale"])), "by_gap": {}, "by_pred_class": {},
             "by_true_class": {}, "by_event": {}}
        for g in sorted(set(d["gap"].astype(int))):
            m = d["gap"] == g
            r["by_gap"][str(g)] = {"mondrian_cov": C.coverage(d["err"][m], d["scale"][m], q_m[m]),
                                   "width_K": float(np.mean(2 * q_m[m] * d["scale"][m]))}
        for ci, cn in enumerate(C.CLASSES):
            mp, mt = cls_pred == ci, cls_true == ci
            r["by_pred_class"][cn] = {"n": int(mp.sum()), "mondrian_cov": C.coverage(d["err"][mp], d["scale"][mp], q_m[mp]),
                                      "global_cov": C.coverage(d["err"][mp], d["scale"][mp], q_g)}
            r["by_true_class"][cn] = {"n": int(mt.sum()), "mondrian_cov": C.coverage(d["err"][mt], d["scale"][mt], q_m[mt]),
                                      "global_cov": C.coverage(d["err"][mt], d["scale"][mt], q_g)}
        for ev in sorted(set(d["event"].astype(int))):
            m = d["event"] == ev
            r["by_event"][event_names[ev]] = {"mondrian_cov": C.coverage(d["err"][m], d["scale"][m], q_m[m]), "n": int(m.sum())}
        out[str(a)] = r
    out["uncertainty_quality"] = {
        "spearman_scale_vs_abs_err": C.spearman(d["scale"], d["err"]),
        "spearman_by_gap": {str(g): C.spearman(d["scale"][d["gap"] == g], d["err"][d["gap"] == g]) for g in sorted(set(d["gap"].astype(int)))},
        "mean_abs_err_by_scale_decile_K": C.decile_table(d["scale"], d["err"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", choices=["none", "linear", "farneback"], default="linear")
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", default="outputs/conformal_val.json")
    ap.add_argument("--cal_per_cell", type=int, default=300)
    ap.add_argument("--val_per_cell", type=int, default=300)
    ap.add_argument("--pix_per_patch", type=int, default=1024)
    ap.add_argument("--min_group_n", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from src.models.conf_head import ConfHead, build_features
    from train_v2 import build_val_ids, load_split_ids, predict

    idx = os.path.join(args.cache_dir, "cache_index.json")
    if os.path.exists(idx) and json.load(open(idx)).get("cache_role") == "test":
        sys.exit("REFUSED: cache_dir is a TEST cache. The final test run uses a separate script.")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = device.type == "cuda"
    base_mode = None if args.base == "none" else args.base
    st = ChannelStandardizer.from_stats_file(args.stats).to(device)
    net = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"]); net.eval()
    head = ConfHead().to(device)
    head.load_state_dict(torch.load(args.head, map_location=device, weights_only=False)["head"]); head.eval()

    def forward(b):
        yhat = predict(net, st, b, base_mode, amp, device).float()
        return yhat, head(build_features(st, b, yhat))

    cal_events, val_events = load_split_ids(args.events_json, "cal"), load_split_ids(args.events_json, "val")
    cal_store = FastSequenceStore(args.cache_dir, cal_events)
    val_store = FastSequenceStore(args.cache_dir, val_events)
    d_cal = collect(cal_store, build_val_ids(cal_store, args.cal_per_cell, 777), forward, st, device, args.pix_per_patch, 1)
    d_val = collect(val_store, build_val_ids(val_store, args.val_per_cell, 1234), forward, st, device, args.pix_per_patch, 2)
    print(f"calibration pixels {len(d_cal['err']):,} ({len(cal_events)} events); validation pixels {len(d_val['err']):,}")

    keys = C.group_keys(d_cal["gap"], C.intensity_class(d_cal["pred"]))
    s_cal = d_cal["err"] / d_cal["scale"]
    cal_by_alpha = {a: C.fit_mondrian(s_cal, keys, a, args.min_group_n) for a in ALPHAS}
    res = {"calibration": {str(a): cal_by_alpha[a] for a in ALPHAS}, "cal_events": cal_events, "val_events": val_events,
           "val_report": report(d_val, cal_by_alpha, val_events), "cal_self_report": report(d_cal, cal_by_alpha, cal_events),
           "settings": vars(args)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)

    for a in ALPHAS:
        r = res["val_report"][str(a)]
        print(f"\n=== nominal {100*(1-a):.0f}% coverage (VALIDATION, B13) ===")
        print(f"raw Laplace {r['raw_laplace_cov']:.3f} | global conformal {r['global_cov']:.3f} | Mondrian {r['mondrian_cov']:.3f} "
              f"| mean width {r['mondrian_mean_width_K']:.2f} K")
        print("by gap        :", {k: round(v["mondrian_cov"], 3) for k, v in r["by_gap"].items()})
        print("by pred class :", {k: (round(v["mondrian_cov"], 3), v["n"]) for k, v in r["by_pred_class"].items()})
        print("by TRUE class :", {k: (round(v["mondrian_cov"], 3), v["n"]) for k, v in r["by_true_class"].items()})
        print("by event      :", {k: round(v["mondrian_cov"], 3) for k, v in r["by_event"].items()})
    uq = res["val_report"]["uncertainty_quality"]
    print("\nSpearman(scale, |err|):", round(uq["spearman_scale_vs_abs_err"], 3), uq["spearman_by_gap"])
    print("mean |err| by scale decile (low->high):", [round(x, 2) for x in uq["mean_abs_err_by_scale_decile_K"]])
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()

"""
scripts/eval_utility.py -- does the 15-min product help? Cold-core detection timing and cooling-rate error.

For patches containing convective cold cores, build a 15-min series from 30-min observations plus 15-min midpoints
(gap-30 predictions), and compare with the 5-min ground truth:
  * detection timing: first time the patch's cold-core temperature (5th-coldest B13 pixel) falls below 241 K / 221 K,
    for onsets (patch above threshold at the start). Delay = detection time - true crossing time (minutes).
      obs30      : only the 30-min observations (what INSAT-3DS gives today)
      linear15 / farneback15 / <model>15 : observations + midpoints from that method
  * cooling rate over 15-min intervals (K per 15 min): MAE vs truth, and rapid-cooling (<= -rapid_k) POD/FAR/CSI.
Validation by default; --final_test --test_cache_dir for the single final run.
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

from src.data.fast_cache import OFFSET_K, SCALE_K, SENTINEL  # noqa: E402
from src.data.fast_dataset import FastSequenceStore  # noqa: E402
from src.eval import utility as U  # noqa: E402
from src.eval.final_guard import resolve_eval_cache  # noqa: E402

T0, T_LAST = 12, 66          # step indices (5 min) of the evaluated window inside each 72-step event


def decode(raw):
    k = raw.astype(np.float32) * SCALE_K + OFFSET_K
    k[raw == SENTINEL] = np.nan
    return k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--test_cache_dir", default=None)
    ap.add_argument("--final_test", action="store_true")
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH[:residual|:farneback] (repeatable)")
    ap.add_argument("--out", default="outputs/utility_val.json")
    ap.add_argument("--max_patches", type=int, default=200, help="per event")
    ap.add_argument("--rapid_k", type=float, default=4.0, help="rapid cooling = drop >= k K per 15 min")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from src.models.flow_base import farneback_batch
    from train_v2 import load_split_ids, predict

    cache_dir, split = resolve_eval_cache(args, "eval_utility")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = device.type == "cuda"
    st = ChannelStandardizer.from_stats_file(args.stats).to(device)
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
    methods = ["linear15", "farneback15"] + [f"{n}15" for n in models]

    events = load_split_ids(args.events_json, split)
    store = FastSequenceStore(cache_dir, events)
    rng = np.random.default_rng(11)
    obs_steps = list(range(T0, T_LAST + 1, 6))                  # 30-min observations
    mid_in = list(range(T0, T_LAST - 5, 6))                     # t of each midpoint (t+3 steps)
    times15 = np.arange(0, (T_LAST - T0) * 5.0 + 1, 15.0)       # 0,15,...,270 min
    ps = store.ps
    delays = {thr: {m: [] for m in ["obs30"] + methods} for thr in (241.0, 221.0)}
    rate_err = {m: [] for m in methods}
    rate_true, rate_pred = [], {m: [] for m in methods}
    used = 0

    for ev_i, ev in enumerate(events):
        arr = store.arrays[ev_i]
        if arr.shape[0] < T_LAST + 1:
            continue
        rows, cols = store.grids[ev_i]
        pos = [(r, c) for r in rows for c in cols]
        rng.shuffle(pos)
        n_ev = 0
        for (r, c) in pos:
            if n_ev >= args.max_patches:
                break
            blk = decode(np.asarray(arr[T0 - 12:T_LAST + 1, :, r:r + ps, c:c + ps]))     # steps T0-12 .. T_LAST
            if not np.isfinite(blk).all():
                continue
            off = 12                                              # blk index of step T0
            cc = np.array([U.coldcore(blk[off + k, 1]) for k in range(T_LAST - T0 + 1)])     # 5-min truth series
            if cc.min() >= 241.0:
                continue
            n_ev += 1; used += 1
            x = torch.from_numpy(blk).to(device)                  # [T,C,H,W]
            idx = lambda step: step - (T0 - 12)
            mids = {}
            ins = torch.stack([torch.stack([x[idx(t - 12)], x[idx(t - 6)], x[idx(t)], x[idx(t + 6)]]) for t in mid_in])
            m_ok = torch.isfinite(ins).flatten(1).all(dim=1)
            ins = torch.nan_to_num(ins, nan=250.0)
            b = {"input_kelvin": ins, "input_norm": st.normalize_input(ins, torch.isfinite(ins).all(dim=2)),
                 "gap_min": torch.full((ins.shape[0],), 30, device=device)}
            with torch.no_grad():
                mids["linear15"] = 0.5 * (ins[:, 2] + ins[:, 3])
                mids["farneback15"] = torch.from_numpy(farneback_batch(ins[:, 2].cpu().numpy(), ins[:, 3].cpu().numpy())).to(device)
                for n, (net, md) in models.items():
                    mids[f"{n}15"] = predict(net, st, b, md, amp, device).float()
            true15 = cc[::3]                                      # truth on the 15-min grid
            obs_c = np.array([cc[s - T0] for s in obs_steps])
            obs_t = np.array([(s - T0) * 5.0 for s in obs_steps])
            true_t5 = np.arange(len(cc)) * 5.0
            for m in methods:
                mid_c = np.array([U.coldcore(mids[m][i, 1].cpu().numpy()) for i in range(len(mid_in))])
                series = np.empty(len(times15)); series[0::2] = obs_c; series[1::2] = mid_c
                for thr in (241.0, 221.0):
                    if cc[0] < thr:
                        continue
                    tt = U.first_cross(true_t5, cc, thr)
                    if tt is None:
                        continue
                    d = U.first_cross(times15, series, thr)
                    delays[thr][m].append(None if d is None else d - tt)
                    if m == methods[0]:                           # obs-only entry added once per onset event
                        d0 = U.first_cross(obs_t, obs_c, thr)
                        delays[thr]["obs30"].append(None if d0 is None else d0 - tt)
                rt, rp = np.diff(true15), np.diff(series)
                rate_err[m] += list(np.abs(rp - rt))
                rate_pred[m].append(rp)
            rate_true.append(np.diff(true15))
    print(f"{used} patches from {len(events)} {split} events")
    rt_all = np.concatenate(rate_true)
    res = {"split": split, "n_patches": used, "rapid_k": args.rapid_k, "detection": {}, "cooling": {}}
    for thr in (241.0, 221.0):
        res["detection"][str(int(thr))] = {m: U.delay_summary(v) for m, v in delays[thr].items()}
    for m in methods:
        rp = np.concatenate(rate_pred[m])
        res["cooling"][m] = {"MAE_rate_K_per_15min": float(np.mean(rate_err[m])),
                             "rapid": U.rapid_scores(rp, rt_all, args.rapid_k)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    for thr in ("241", "221"):
        print(f"\n=== cold-core detection delay, onset below {thr} K ({split}) ===")
        print(f"{'series':<18}{'events':>7}{'detected':>9}{'mean min':>9}{'median':>8}{'<=10min':>8}{'early>5':>8}")
        for m, s in res["detection"][thr].items():
            print(f"{m:<18}{s['n_events']:7d}{s['detected']:9d}{s.get('mean_delay_min', float('nan')):9.1f}"
                  f"{s.get('median_delay_min', float('nan')):8.1f}{s.get('within_10min', float('nan')):8.2f}{s.get('early_gt5min', float('nan')):8.2f}")
    print(f"\n=== 15-min cooling rate (K per 15 min), rapid = drop >= {args.rapid_k} K ===")
    print(f"{'series':<18}{'rate MAE':>9}{'n_rapid':>8}{'POD':>7}{'FAR':>7}{'CSI':>7}")
    for m, s in res["cooling"].items():
        r = s["rapid"]
        print(f"{m:<18}{s['MAE_rate_K_per_15min']:9.3f}{r['n_true']:8d}{r['POD']:7.2f}{r['FAR']:7.2f}{r['CSI']:7.2f}")
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()

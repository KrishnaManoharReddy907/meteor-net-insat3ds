"""
scripts/loeo_conformal.py -- leave-one-event-out conformal check.

Pool = all cal + val events. Pixel samples are collected ONCE per event (same sampler as calibrate_conformal.py).
For each event E: calibrate (Mondrian by gap x predicted class) on the OTHER events, test coverage on E.
Reports the per-event coverage and its spread across events: the honest "does it hold from event to event" answer.
Refuses a test cache; the test events are never touched.

  python scripts/loeo_conformal.py --cache_dir /content/cache/dev --events_json configs/events_v3.json \
     --stats outputs/normalization/v2_train_stats.json --ckpt <runs>/residual_12k/best.pt --base linear \
     --head <runs>/conf_head_12k/head.pt --out outputs/loeo_conformal.json
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
from src.data.fast_dataset import FastSequenceStore  # noqa: E402
from calibrate_conformal import collect  # noqa: E402

ALPHAS = (0.1, 0.05)


def ids_for_event(store, ev_name, per_cell, seed):
    rng = np.random.default_rng(seed)
    ids = []
    for g in store.gaps_min:
        for c in (False, True):
            pool = store.select(gap_min=g, event_ids=[ev_name], convective=c)
            if len(pool):
                ids.append(rng.choice(pool, size=min(per_cell, len(pool)), replace=False))
    return np.sort(np.concatenate(ids)) if ids else np.array([], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", choices=["none", "linear", "farneback"], default="linear")
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", default="outputs/loeo_conformal.json")
    ap.add_argument("--per_cell", type=int, default=100, help="patches per (gap, conv/amb) cell PER EVENT")
    ap.add_argument("--pix_per_patch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from src.models.conf_head import ConfHead, build_features
    from train_v2 import load_split_ids, predict

    idx = os.path.join(args.cache_dir, "cache_index.json")
    if os.path.exists(idx) and json.load(open(idx)).get("cache_role") == "test":
        sys.exit("REFUSED: cache_dir is a TEST cache.")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = device.type == "cuda"
    st = ChannelStandardizer.from_stats_file(args.stats).to(device)
    net = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"]); net.eval()
    head = ConfHead().to(device)
    head.load_state_dict(torch.load(args.head, map_location=device, weights_only=False)["head"]); head.eval()
    mode = None if args.base == "none" else args.base

    def forward(b):
        yhat = predict(net, st, b, mode, amp, device).float()
        return yhat, head(build_features(st, b, yhat))

    events = load_split_ids(args.events_json, "cal") + load_split_ids(args.events_json, "val")
    store = FastSequenceStore(args.cache_dir, events)
    data = {}
    for k, ev in enumerate(events):
        ids = ids_for_event(store, ev, args.per_cell, 100 + k)
        if len(ids):
            data[ev] = collect(store, ids, forward, st, device, args.pix_per_patch, 200 + k)
            data[ev]["key"] = C.group_keys(data[ev]["gap"], C.intensity_class(data[ev]["pred"]))
            data[ev]["cls_true"] = C.intensity_class(data[ev]["tgt"])
    print(f"collected {len(data)} events")

    res = {}
    for a in ALPHAS:
        rows = {}
        for ev in data:
            others = [e for e in data if e != ev]
            sc = np.concatenate([data[e]["err"] / data[e]["scale"] for e in others])
            ky = np.concatenate([data[e]["key"] for e in others])
            cal = C.fit_mondrian(sc, ky, a, 500)
            d = data[ev]
            q = C.q_for(cal, d["key"])
            strong = d["cls_true"] == 0
            rows[ev] = {"coverage": C.coverage(d["err"], d["scale"], q), "n": int(len(d["err"])),
                        "cov_strong_true": C.coverage(d["err"][strong], d["scale"][strong], q[strong]) if strong.sum() >= 200 else None,
                        "n_strong": int(strong.sum()), "mean_width_K": float(np.mean(2 * q * d["scale"]))}
        covs = np.array([r["coverage"] for r in rows.values()])
        res[str(a)] = {"per_event": rows, "mean": float(covs.mean()), "std": float(covs.std(ddof=1)) if len(covs) > 1 else None,
                       "min": float(covs.min()), "max": float(covs.max())}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    for a in ALPHAS:
        r = res[str(a)]
        print(f"\n=== leave-one-event-out, nominal {100*(1-a):.0f}% ===")
        print(f"{'event':<18}{'coverage':>9}{'strong cov':>11}{'n strong':>9}{'width K':>9}")
        for ev, x in r["per_event"].items():
            sc = "   n/a" if x["cov_strong_true"] is None else f"{x['cov_strong_true']:11.3f}"
            print(f"{ev:<18}{x['coverage']:9.3f}{sc:>11}{x['n_strong']:9d}{x['mean_width_K']:9.2f}")
        print(f"mean {r['mean']:.3f}  std {r['std'] if r['std'] is None else round(r['std'], 3)}  range [{r['min']:.3f}, {r['max']:.3f}]")
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()

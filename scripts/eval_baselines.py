"""
scripts/eval_baselines.py -- baselines and trained models on the SAME fixed validation patches.

Methods
  linear        : 0.5 * (frame t + frame t+G)               (the bar every model must beat)
  persist_t     : frame t        persist_tg : frame t+G
  farneback     : dense optical flow (cv2 Farneback) on Band 13, half-flow warp of both frames, averaged
  --ckpt NAME=PATH[:residual] (repeatable): trained METEOR-Net checkpoints (add ':residual' for the ablation)

Metrics per gap (10/20/30 min): MAE per channel, B13 RMSE and bias, B13 MAE split strong (<221 K) /
weak (221-241 K) / ambient (>=241 K) by the TARGET value, and B13 CSI/POD/FAR for cold pixels at 241 K and 221 K.
Evaluates VALIDATION events only (split == 'val' in events_json). Refuses a test cache.

Example
  python scripts/eval_baselines.py --cache_dir /content/cache/dev --events_json configs/events_v2.json \
    --stats outputs/normalization/v2_train_stats.json --out outputs/baselines_val.json \
    --ckpt main_s0=/content/drive/MyDrive/runs/main_s0/best.pt \
    --ckpt residual_s0=/content/drive/MyDrive/runs/residual_s0/best.pt:residual
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

FILL_K = 250.0
BLOCK_STEPS = 6          # bootstrap block = 30 min of consecutive start times within an event


class Acc:
    """Accumulates sums per (method, gap)."""
    def __init__(self):
        self.d = {}

    def add(self, key, **kw):
        s = self.d.setdefault(key, {})
        for k, v in kw.items():
            s[k] = s.get(k, 0.0) + float(v)

    def summary(self):
        out = {}
        for key, s in self.d.items():
            n = max(s.get("n", 0.0), 1.0)
            r = {f"mae_b{b}": s[f"ae{i}"] / n for i, b in enumerate((9, 13, 14, 15))}
            r["mae_mean4"] = float(np.mean([r[k] for k in r]))
            r["rmse_b13"] = (s["se1"] / n) ** 0.5
            r["bias_b13"] = s["bias1"] / n
            for nm in ("strong", "weak", "ambient"):
                m = s.get(f"n_{nm}", 0.0)
                r[f"mae_b13_{nm}"] = s.get(f"ae_{nm}", 0.0) / m if m else float("nan")
                r[f"n_{nm}"] = int(m)
            for thr in (241, 221):
                h, mi, fa = s.get(f"hit{thr}", 0.0), s.get(f"miss{thr}", 0.0), s.get(f"fa{thr}", 0.0)
                r[f"POD{thr}"] = h / (h + mi) if h + mi else float("nan")
                r[f"FAR{thr}"] = fa / (h + fa) if h + fa else float("nan")
                r[f"CSI{thr}"] = h / (h + mi + fa) if h + mi + fa else float("nan")
            out[key] = r
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="outputs/baselines_val.json")
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=PATH or NAME=PATH:residual")
    ap.add_argument("--per_cell", type=int, default=400, help="patches per (gap, convective/ambient) cell")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--final_test", action="store_true", help="ONE-TIME final run on the test events")
    ap.add_argument("--test_cache_dir", default=None, help="test cache (cache_role == 'test'), with --final_test")
    ap.add_argument("--no_farneback", action="store_true")
    ap.add_argument("--rife_dir", default=None, help="Practical-RIFE checkout with train_log/ (optional extra baseline)")
    args = ap.parse_args()

    import torch
    from src.data.fast_dataset import FastSequenceStore, to_device
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from src.eval.final_guard import resolve_eval_cache
    from train_v2 import build_val_ids, load_split_ids, predict
    from src.models.flow_base import farneback_batch

    eval_cache, eval_split = resolve_eval_cache(args, "eval_baselines")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = device.type == "cuda"
    st = ChannelStandardizer.from_stats_file(args.stats).to(device)
    store = FastSequenceStore(eval_cache, load_split_ids(args.events_json, eval_split), gaps_min=(10, 20, 30))
    ids = build_val_ids(store, args.per_cell, seed=1234)
    print(f"{len(ids)} validation patches")

    models = {}
    for spec in args.ckpt:
        name, rest = spec.split("=", 1)
        residual, path = False, rest
        for tag in ("residual", "farneback"):
            if rest.endswith(":" + tag):
                residual, path = (True if tag == "residual" else "farneback"), rest[:-(len(tag) + 1)]
        net = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
        net.eval()
        models[name] = (net, residual)

    rife = None
    if args.rife_dir:
        from src.models.rife_base import RifeMidpoint
        rife = RifeMidpoint(args.rife_dir, device)
    methods = ["linear", "persist_t", "persist_tg"] + ([] if args.no_farneback else ["farneback"]) + (["rife"] if rife else []) + list(models)
    acc = Acc()
    rec = {}      # (method, gap) -> list of per-patch rows: block, ae13_sum, n, ae13_strong_sum, n_strong

    with torch.no_grad():
        for batch in store.iter_eval_batches(ids, args.bs):
            b = to_device(batch, device, st)
            x, tgt, m = b["input_kelvin"], b["target_kelvin"], b["combined_mask"]
            preds = {
                "linear": 0.5 * (x[:, 2] + x[:, 3]),
                "persist_t": x[:, 2], "persist_tg": x[:, 3],
            }
            if "farneback" in methods:
                preds["farneback"] = torch.from_numpy(
                    farneback_batch(x[:, 2].cpu().numpy(), x[:, 3].cpu().numpy())).to(device)
            if rife is not None:
                preds["rife"] = rife(x[:, 2], x[:, 3])
            for name, (net, residual) in models.items():
                preds[name] = predict(net, st, b, residual, amp, device)
            tg13 = torch.nan_to_num(tgt[:, 1], nan=300.0)
            gaps = b["gap_min"]
            for g in torch.unique(gaps).tolist():
                sel = gaps == g
                mm, t13 = m[sel], tg13[sel]
                for meth in methods:
                    p = torch.nan_to_num(preds[meth][sel], nan=FILL_K)
                    e = p - torch.nan_to_num(tgt[sel], nan=FILL_K)
                    kw = {"n": mm.sum()}
                    for c in range(4):
                        kw[f"ae{c}"] = e[:, c].abs()[mm].sum()
                    kw["se1"] = (e[:, 1] ** 2)[mm].sum()
                    kw["bias1"] = e[:, 1][mm].sum()
                    for nm, mk in (("strong", mm & (t13 < 221)), ("weak", mm & (t13 >= 221) & (t13 < 241)),
                                   ("ambient", mm & (t13 >= 241))):
                        kw[f"ae_{nm}"] = e[:, 1].abs()[mk].sum()
                        kw[f"n_{nm}"] = mk.sum()
                    p13 = p[:, 1]
                    for thr in (241, 221):
                        ev, pr = (t13 < thr) & mm, (p13 < thr) & mm
                        kw[f"hit{thr}"] = (ev & pr).sum()
                        kw[f"miss{thr}"] = (ev & ~pr).sum()
                        kw[f"fa{thr}"] = (~ev & pr & mm).sum()
                    a13 = e[:, 1].abs()
                    stg = mm & (t13 < 221)
                    blk = (batch["event_idx"][sel.cpu().numpy()].astype(np.int64) * 100000
                           + batch["t_index"][sel.cpu().numpy()].astype(np.int64) // BLOCK_STEPS)
                    rows = np.stack([blk.astype(np.float64),
                                     (a13 * mm).sum((1, 2)).double().cpu().numpy(),
                                     mm.sum((1, 2)).double().cpu().numpy(),
                                     (a13 * stg).sum((1, 2)).double().cpu().numpy(),
                                     stg.sum((1, 2)).double().cpu().numpy()], axis=1)
                    rec.setdefault((meth, int(g)), []).append(rows)
                    acc.add((meth, int(g)), **{k: v.item() if hasattr(v, "item") else v for k, v in kw.items()})

    res = acc.summary()
    out = {f"{m}|gap{g}": r for (m, g), r in res.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"n_patches": int(len(ids)), "results": out}, open(args.out, "w"), indent=2)
    for g in (10, 20, 30):
        print(f"\n=== gap {g} min (validation) ===")
        print(f"{'method':<14}{'MAE4':>7}{'B13':>7}{'strong':>8}{'weak':>7}{'amb':>7}{'RMSE':>7}{'bias':>7}{'CSI241':>8}{'CSI221':>8}")
        for meth in methods:
            r = res[(meth, g)]
            print(f"{meth:<14}{r['mae_mean4']:7.3f}{r['mae_b13']:7.3f}{r['mae_b13_strong']:8.3f}"
                  f"{r['mae_b13_weak']:7.3f}{r['mae_b13_ambient']:7.3f}{r['rmse_b13']:7.3f}"
                  f"{r['bias_b13']:7.3f}{r['CSI241']:8.3f}{r['CSI221']:8.3f}")
    # paired block bootstrap: difference in B13 MAE (model minus reference); negative = model better
    rng = np.random.default_rng(0)
    boot = {}
    for g in (10, 20, 30):
        for m in models:
            for ref in ("linear", "farneback"):
                if ref not in methods:
                    continue
                A, R = np.concatenate(rec[(m, g)]), np.concatenate(rec[(ref, g)])
                blocks = np.unique(A[:, 0])
                agg = lambda M: np.stack([M[M[:, 0] == bk][:, 1:].sum(0) for bk in blocks])   # [nb,4]
                SA, SR = agg(A), agg(R)
                ii = rng.integers(0, len(blocks), size=(2000, len(blocks)))
                for name, (c, n) in (("b13_all", (0, 1)), ("b13_strong", (2, 3))):
                    d = (SA[ii][:, :, c].sum(1) / SA[ii][:, :, n].sum(1).clip(1)
                         - SR[ii][:, :, c].sum(1) / SR[ii][:, :, n].sum(1).clip(1))
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    pt = SA[:, c].sum() / max(SA[:, n].sum(), 1) - SR[:, c].sum() / max(SR[:, n].sum(), 1)
                    boot[f"{m} vs {ref}|gap{g}|{name}"] = {"diff_K": float(pt), "ci95": [float(lo), float(hi)],
                                                         "n_blocks": int(len(blocks))}
    # per-event B13 MAE (all pixels / strong): answers "does it hold event by event?"
    per_event = {}
    for (meth, g), parts in rec.items():
        M = np.concatenate(parts)
        evs = (M[:, 0] // 100000).astype(int)
        for e in np.unique(evs):
            r = M[evs == e]
            per_event.setdefault(store.event_ids[e], {}).setdefault(f"gap{g}", {})[meth] = {
                "mae_all": float(r[:, 1].sum() / max(r[:, 2].sum(), 1)),
                "mae_strong": float(r[:, 3].sum() / r[:, 4].sum()) if r[:, 4].sum() else None}
    wins = {}
    for g in (10, 20, 30):
        for m in models:
            for ref in ("linear", "farneback"):
                if ref not in methods:
                    continue
                n_ev = [e for e in per_event if f"gap{g}" in per_event[e]]
                w = sum(per_event[e][f"gap{g}"][m]["mae_all"] < per_event[e][f"gap{g}"][ref]["mae_all"] for e in n_ev)
                wins[f"{m} beats {ref}|gap{g}"] = f"{w}/{len(n_ev)} events"
    print("\nPer-event wins on B13 MAE (all pixels):")
    for g in (30,):
        for k, v in wins.items():
            if k.endswith(f"gap{g}"):
                print(f"  {k:<44} {v}")
    print("\nPer-event B13 MAE at gap 30 (all pixels):")
    ref_methods = [mm_ for mm_ in ("linear", "farneback") if mm_ in methods] + list(models)[:6]
    print(f"{'event':<18}" + "".join(f"{x[:12]:>14}" for x in ref_methods))
    for e in sorted(per_event):
        print(f"{e:<18}" + "".join(f"{per_event[e]['gap30'][x]['mae_all']:14.3f}" for x in ref_methods))
    out_obj = json.load(open(args.out)); out_obj["bootstrap"] = boot
    out_obj["per_event"] = per_event; out_obj["event_wins"] = wins
    json.dump(out_obj, open(args.out, "w"), indent=2)
    print("\nPaired block-bootstrap, B13 MAE difference in K (model - reference; negative = model better)")
    for k, v in boot.items():
        print(f"  {k:<52} {v['diff_K']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  blocks={v['n_blocks']}")
    print("  (blocks are 30-min chunks within events; with 2 val events this does not capture between-event spread)")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

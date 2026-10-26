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


def farneback_midpoint(a: np.ndarray, b: np.ndarray, ref: int = 1) -> np.ndarray:
    """a, b: [C,H,W] Kelvin at t and t+G. Returns midpoint estimate [C,H,W] (flow from channel `ref`)."""
    import cv2
    a = np.nan_to_num(a, nan=FILL_K).astype(np.float32)
    b = np.nan_to_num(b, nan=FILL_K).astype(np.float32)
    lo, hi = min(a[ref].min(), b[ref].min()), max(a[ref].max(), b[ref].max())
    sc = 255.0 / max(hi - lo, 1e-3)
    ua = ((a[ref] - lo) * sc).astype(np.uint8)
    ub = ((b[ref] - lo) * sc).astype(np.uint8)
    f = cv2.calcOpticalFlowFarneback(ua, ub, None, 0.5, 3, 15, 3, 5, 1.2, 0)   # ua(x) ~ ub(x + f)
    h, w = ua.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx_a, my_a = gx - 0.5 * f[..., 0], gy - 0.5 * f[..., 1]     # where midpoint pixel came from in a
    mx_b, my_b = gx + 0.5 * f[..., 0], gy + 0.5 * f[..., 1]     # where it goes to in b
    out = np.empty_like(a)
    for c in range(a.shape[0]):
        wa = cv2.remap(a[c], mx_a, my_a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(b[c], mx_b, my_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[c] = 0.5 * (wa + wb)
    return out


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
    ap.add_argument("--no_farneback", action="store_true")
    args = ap.parse_args()

    import torch
    from src.data.fast_dataset import FastSequenceStore, to_device
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from train_v2 import build_val_ids, load_split_ids, predict

    idx = os.path.join(args.cache_dir, "cache_index.json")
    if os.path.exists(idx) and json.load(open(idx)).get("cache_role") == "test":
        sys.exit("REFUSED: cache_dir is a TEST cache.")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = device.type == "cuda"
    st = ChannelStandardizer.from_stats_file(args.stats).to(device)
    store = FastSequenceStore(args.cache_dir, load_split_ids(args.events_json, "val"), gaps_min=(10, 20, 30))
    ids = build_val_ids(store, args.per_cell, seed=1234)
    print(f"{len(ids)} validation patches")

    models = {}
    for spec in args.ckpt:
        name, rest = spec.split("=", 1)
        residual = rest.endswith(":residual")
        path = rest[:-9] if residual else rest
        net = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
        net.eval()
        models[name] = (net, residual)

    methods = ["linear", "persist_t", "persist_tg"] + ([] if args.no_farneback else ["farneback"]) + list(models)
    acc = Acc()

    with torch.no_grad():
        for batch in store.iter_eval_batches(ids, args.bs):
            b = to_device(batch, device, st)
            x, tgt, m = b["input_kelvin"], b["target_kelvin"], b["combined_mask"]
            preds = {
                "linear": 0.5 * (x[:, 2] + x[:, 3]),
                "persist_t": x[:, 2], "persist_tg": x[:, 3],
            }
            if "farneback" in methods:
                xa, xb = x[:, 2].cpu().numpy(), x[:, 3].cpu().numpy()
                preds["farneback"] = torch.from_numpy(
                    np.stack([farneback_midpoint(xa[i], xb[i]) for i in range(len(xa))])).to(device)
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
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

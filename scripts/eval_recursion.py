"""
scripts/eval_recursion.py -- recursive halving 20 -> 10 -> 5 min with REAL ground truth (validation or final test).

From observations every 20 min, predict the 10-min midpoints with the gap-20 model, feed them back as inputs to
the gap-10 model and predict the 5-min frame (ground truth exists on the 5-min grid).
  level 1 : frame at t+10  from obs at t-40..t+20            (gap-20 model, true inputs)
  level 2 : frame at t+5   from [obs t-20, pred t-10, obs t, pred t+10]  (gap-10 model on PREDICTED inputs)
  oracle  : same gap-10 model but with the TRUE frames at t-10 and t+10 (isolates error propagation)
Baselines are applied recursively the same way (linear, Farneback). Optionally checks whether the confidence head and
the frozen conformal table still cover at level 2 (coverage under recursion).

Validation:  python scripts/eval_recursion.py --cache_dir /content/cache/dev --events_json configs/events_v3.json ...
Final test:  add --final_test --test_cache_dir /content/cache/test   (see STEP_I_INSTRUCTIONS.md)
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
from src.eval import conformal as C  # noqa: E402
from src.eval.final_guard import resolve_eval_cache  # noqa: E402


def decode(raw):
    k = raw.astype(np.float32) * SCALE_K + OFFSET_K
    k[raw == SENTINEL] = np.nan
    return k


class Acc:
    def __init__(self):
        self.s = {}

    def add(self, key, p, g):
        """p, g: [B,C,H,W] float32 torch tensors (no NaN)."""
        e = (p - g)
        d = self.s.setdefault(key, {})
        def inc(k, v): d[k] = d.get(k, 0.0) + float(v)
        n = p.shape[0] * p.shape[2] * p.shape[3]
        inc("n", n)
        for c in range(4):
            inc(f"ae{c}", e[:, c].abs().sum())
        inc("se1", (e[:, 1] ** 2).sum())
        g13, e13 = g[:, 1], e[:, 1].abs()
        for nm, m in (("strong", g13 < 221), ("weak", (g13 >= 221) & (g13 < 241)), ("amb", g13 >= 241)):
            inc(f"ae_{nm}", (e13 * m).sum()); inc(f"n_{nm}", m.sum())
        for thr in (241, 221):
            ev, pr = g13 < thr, p[:, 1] < thr
            inc(f"h{thr}", (ev & pr).sum()); inc(f"m{thr}", (ev & ~pr).sum()); inc(f"f{thr}", (~ev & pr).sum())

    def table(self):
        out = {}
        for k, d in self.s.items():
            n = d["n"]
            r = {"MAE4": sum(d[f"ae{c}"] for c in range(4)) / (4 * n), "MAE_B13": d["ae1"] / n, "RMSE_B13": (d["se1"] / n) ** 0.5}
            for nm in ("strong", "weak", "amb"):
                r[f"MAE_B13_{nm}"] = d[f"ae_{nm}"] / d[f"n_{nm}"] if d[f"n_{nm}"] else float("nan")
            for thr in (241, 221):
                h, m, f = d[f"h{thr}"], d[f"m{thr}"], d[f"f{thr}"]
                r[f"CSI{thr}"] = h / (h + m + f) if h + m + f else float("nan")
            out[k] = r
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--test_cache_dir", default=None)
    ap.add_argument("--final_test", action="store_true")
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ckpt", required=True, help="main model checkpoint (residual over linear)")
    ap.add_argument("--base", choices=["none", "linear", "farneback"], default="linear")
    ap.add_argument("--head", default=None)
    ap.add_argument("--cal_json", default=None, help="conformal_val.json (calibration table) for coverage under recursion")
    ap.add_argument("--out", default="outputs/recursion_val.json")
    ap.add_argument("--per_cell", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from src.models.flow_base import farneback_batch
    from src.models.conf_head import ConfHead, build_features
    from train_v2 import load_split_ids, predict

    cache_dir, split = resolve_eval_cache(args, "eval_recursion")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = device.type == "cuda"
    st = ChannelStandardizer.from_stats_file(args.stats).to(device)
    net = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"]); net.eval()
    mode = None if args.base == "none" else args.base
    head = None
    if args.head:
        head = ConfHead().to(device)
        head.load_state_dict(torch.load(args.head, map_location=device, weights_only=False)["head"]); head.eval()
    cal = json.load(open(args.cal_json))["calibration"]["0.1"] if args.cal_json else None

    events = load_split_ids(args.events_json, split)
    store = FastSequenceStore(cache_dir, events, gaps_min=(20,))
    valid = set(zip(store.it_ev.tolist(), store.it_t.tolist(), store.it_p.tolist()))
    rng = np.random.default_rng(5)
    cand = []
    for conv in (False, True):
        pool = store.select(gap_min=20, convective=conv)
        pool = [i for i in pool if (int(store.it_ev[i]), int(store.it_t[i]) - 4, int(store.it_p[i])) in valid
                and int(store.it_t[i]) - 12 >= 0]
        if pool:
            cand += list(rng.choice(pool, size=min(args.per_cell, len(pool)), replace=False))
    print(f"{len(cand)} patches from {len(events)} {split} events")

    def mk(x):         # x: [B,4,C,H,W] kelvin tensor -> batch dict for predict()/head
        m = torch.isfinite(x).all(dim=2)
        xk = torch.nan_to_num(x, nan=250.0)
        return {"input_kelvin": xk, "input_norm": st.normalize_input(xk, m)}

    acc, pix = Acc(), {"l2": [], "oracle": []}
    ps = store.ps
    with torch.no_grad():
        for s in range(0, len(cand), args.bs):
            ids = cand[s:s + args.bs]
            blocks = []
            for i in ids:
                ev, t, p = int(store.it_ev[i]), int(store.it_t[i]), int(store.it_p[i])
                rows, cols = store.grids[ev]
                r, c = rows[p // len(cols)], cols[p % len(cols)]
                blk = decode(np.asarray(store.arrays[ev][t - 12:t + 5, :, r:r + ps, c:c + ps]))   # idx j <-> step t-12+j
                blocks.append(blk)
            B = torch.from_numpy(np.stack(blocks)).to(device)               # [B,17,C,H,W]
            ok = torch.isfinite(B).flatten(1).all(dim=1)
            if not ok.any():
                continue
            B = B[ok]
            f = lambda k: B[:, 12 + k]                                    # frame at step t+k
            in_a = torch.stack([f(-8), f(-4), f(0), f(4)], dim=1)          # -> t+2 steps (10 min)
            in_b = torch.stack([f(-12), f(-8), f(-4), f(0)], dim=1)        # -> t-2 steps
            gt10, gt5 = f(2), f(1)
            g20 = torch.full((B.shape[0],), 20, device=device)
            ba, bb = mk(in_a), mk(in_b)
            ba["gap_min"] = bb["gap_min"] = g20
            p_pos = predict(net, st, ba, mode, amp, device).float()       # pred at t+10
            p_neg = predict(net, st, bb, mode, amp, device).float()       # pred at t-10
            g10 = torch.full((B.shape[0],), 10, device=device)
            rec = mk(torch.stack([f(-4), p_neg, f(0), p_pos], dim=1)); rec["gap_min"] = g10
            orc = mk(torch.stack([f(-4), f(-2), f(0), f(2)], dim=1)); orc["gap_min"] = g10
            y_rec = predict(net, st, rec, mode, amp, device).float()
            y_orc = predict(net, st, orc, mode, amp, device).float()
            # baselines
            lin = lambda a, b_: 0.5 * (a + b_)
            l_pos, l_neg = lin(f(0), f(4)), lin(f(-4), f(0))
            l5 = lin(f(0), l_pos)
            fb = lambda a, b_: torch.from_numpy(farneback_batch(a.cpu().numpy(), b_.cpu().numpy())).to(device)
            fb_pos = fb(f(0), f(4)); fb5 = fb(f(0), fb_pos)
            clean = lambda x: torch.nan_to_num(x, nan=250.0)
            acc.add("L1 (10 min) linear", clean(l_pos), clean(gt10)); acc.add("L1 (10 min) farneback", clean(fb_pos), clean(gt10))
            acc.add("L1 (10 min) model", clean(p_pos), clean(gt10))
            acc.add("L2 (5 min) linear-recursive", clean(l5), clean(gt5)); acc.add("L2 (5 min) farneback-recursive", clean(fb5), clean(gt5))
            acc.add("L2 (5 min) model-recursive", clean(y_rec), clean(gt5)); acc.add("L2 (5 min) model-oracle-inputs", clean(y_orc), clean(gt5))
            if head is not None:
                for nm, bt, y in (("l2", rec, y_rec), ("oracle", orc, y_orc)):
                    sc = head(build_features(st, bt, y))[:, 1]
                    e13 = (clean(gt5)[:, 1] - y[:, 1]).abs()
                    for j in range(e13.shape[0]):
                        k = rng.choice(e13[j].numel(), size=min(1024, e13[j].numel()), replace=False)
                        pix[nm].append(np.stack([e13[j].flatten()[k].cpu().numpy(), sc[j].flatten()[k].cpu().numpy(),
                                                 y[j, 1].flatten()[k].cpu().numpy()]))
    tab = acc.table()
    res = {"split": split, "table": tab}
    if head is not None:
        res["uncertainty"] = {}
        for nm, lst in pix.items():
            if not lst:
                continue
            d = np.concatenate(lst, axis=1)
            err, sc, pr = d
            r = {"spearman": C.spearman(sc, err), "mean_scale": float(sc.mean()), "mean_abs_err": float(err.mean())}
            if cal is not None:
                keys = C.group_keys(np.full(len(err), 10), C.intensity_class(pr))
                q = C.q_for(cal, keys)
                r["coverage_90_frozen_gap10_table"] = C.coverage(err, sc, q)
            res["uncertainty"][nm] = r
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n=== recursive halving, B13 etc., {split} ===")
    print(f"{'method':<34}{'MAE4':>7}{'B13':>7}{'strong':>8}{'weak':>7}{'amb':>7}{'RMSE':>7}{'CSI241':>8}{'CSI221':>8}")
    for k, r in tab.items():
        print(f"{k:<34}{r['MAE4']:7.3f}{r['MAE_B13']:7.3f}{r['MAE_B13_strong']:8.3f}{r['MAE_B13_weak']:7.3f}"
              f"{r['MAE_B13_amb']:7.3f}{r['RMSE_B13']:7.3f}{r['CSI241']:8.3f}{r['CSI221']:8.3f}")
    if "uncertainty" in res:
        print("\nconfidence under recursion:", json.dumps(res["uncertainty"], indent=1))
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()

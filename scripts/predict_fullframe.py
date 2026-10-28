"""
scripts/predict_fullframe.py -- stitched full-frame (453x740) prediction of one midpoint + figure + saved arrays.

Overlapping 128x128 tiles (stride 64) are blended with a Hann window. Compares linear / Farneback / the model on the
whole frame, and (with --head and --cal_json) draws the 90% conformal half-width and marks pixels whose true error
exceeds it. Choose the time automatically: --pick strong (most B13<221 K pixels in the target) or ambient.
Validation events by default; --final_test --test_cache_dir for test events (use after everything is frozen).

  python scripts/predict_fullframe.py --cache_dir /content/cache/dev --events_json configs/events_v3.json \
     --stats outputs/normalization/v2_train_stats.json --ckpt <runs>/residual_12k/best.pt --base linear \
     --head <runs>/conf_head_12k/head.pt --cal_json outputs/conformal_val.json \
     --event event_2026-07-19 --gap 30 --pick strong --out_dir outputs/figs
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

from src.data.fast_cache import OFFSET_K, SCALE_K, SENTINEL, load_event_cache  # noqa: E402
from src.eval import conformal as C  # noqa: E402
from src.eval.final_guard import resolve_eval_cache  # noqa: E402


def decode(raw):
    k = raw.astype(np.float32) * SCALE_K + OFFSET_K
    k[raw == SENTINEL] = np.nan
    return k


def tile_starts(n, ps, stride):
    return sorted(set(list(range(0, n - ps + 1, stride)) + [n - ps]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--test_cache_dir", default=None)
    ap.add_argument("--final_test", action="store_true")
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", choices=["none", "linear", "farneback"], default="linear")
    ap.add_argument("--head", default=None)
    ap.add_argument("--cal_json", default=None)
    ap.add_argument("--event", required=True)
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--pick", choices=["strong", "ambient"], default="strong")
    ap.add_argument("--t_index", type=int, default=None)
    ap.add_argument("--out_dir", default="outputs/figs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from src.models.flow_base import farneback_midpoint
    from src.models.conf_head import ConfHead, build_features
    from train_v2 import load_split_ids, predict

    cache_dir, split = resolve_eval_cache(args, "predict_fullframe")
    if args.event not in load_split_ids(args.events_json, split):
        sys.exit(f"{args.event} is not a '{split}' event in {args.events_json}")
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

    arr, meta = load_event_cache(cache_dir, args.event)
    T, Cn, H, W = arr.shape
    G, s = args.gap, args.gap // 5
    valid_t = list(range(2 * s, T - s))
    if args.t_index is not None:
        t = args.t_index
    else:
        cnt = [(decode(np.asarray(arr[u + s // 2, 1])) < 221.0).sum() for u in valid_t]
        t = valid_t[int(np.argmax(cnt) if args.pick == "strong" else np.argmin(cnt))]
    frames = [decode(np.asarray(arr[t + o])) for o in (-2 * s, -s, 0, s)]
    x = np.stack(frames)                                             # [4,C,H,W]
    tgt = decode(np.asarray(arr[t + s // 2]))                        # [C,H,W]
    print(f"{args.event}: t_index={t} ({meta['timestamps'][t] if 'timestamps' in meta else ''}), gap {G} min")

    ps, stride = 128, 64
    w1 = np.hanning(ps + 2)[1:-1].astype(np.float32); win = np.outer(w1, w1)
    ys, xs = tile_starts(H, ps, stride), tile_starts(W, ps, stride)
    acc = np.zeros((Cn, H, W), np.float32); sacc = np.zeros((Cn, H, W), np.float32); wacc = np.zeros((H, W), np.float32)
    coords = [(r, c) for r in ys for c in xs]
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        for i in range(0, len(coords), 8):
            chunk = coords[i:i + 8]
            xb = torch.stack([xt[:, :, r:r + ps, c:c + ps] for r, c in chunk])           # [B,4,C,ps,ps]
            m = torch.isfinite(xb).all(dim=2)
            xk = torch.nan_to_num(xb, nan=250.0)
            b = {"input_kelvin": xk, "input_norm": st.normalize_input(xk, m),
                 "gap_min": torch.full((len(chunk),), G, device=device)}
            y = predict(net, st, b, mode, amp, device).float()
            sc = head(build_features(st, b, y)) if head is not None else None
            for j, (r, c) in enumerate(chunk):
                acc[:, r:r + ps, c:c + ps] += y[j].cpu().numpy() * win
                if sc is not None:
                    sacc[:, r:r + ps, c:c + ps] += sc[j].cpu().numpy() * win
                wacc[r:r + ps, c:c + ps] += win
    pred = acc / wacc
    scale = sacc / wacc if head is not None else None
    lin = 0.5 * (x[2] + x[3])
    fb = farneback_midpoint(x[2], x[3])
    valid = np.isfinite(tgt).all(axis=0) & np.isfinite(x).all(axis=(0, 1))
    strong = valid & (tgt[1] < 221.0)

    def mae(p, m=valid):
        return float(np.abs(p[1] - tgt[1])[m].mean()) if m.any() else float("nan")
    res = {"event": args.event, "t_index": int(t), "gap": G, "split": split, "valid_frac": float(valid.mean()),
           "n_strong_pix": int(strong.sum()),
           "B13_MAE": {"linear": mae(lin), "farneback": mae(fb), "model": mae(pred)},
           "B13_MAE_strong": {"linear": mae(lin, strong), "farneback": mae(fb, strong), "model": mae(pred, strong)}}
    hw = None
    if scale is not None and cal is not None:
        keys = C.group_keys(np.full(valid.sum(), G), C.intensity_class(pred[1][valid]))
        q = np.zeros(valid.sum()); q[:] = C.q_for(cal, keys)
        hw = np.full((H, W), np.nan, np.float32); hw[valid] = q * scale[1][valid]
        res["conformal90_coverage_frame"] = float((np.abs(pred[1] - tgt[1])[valid] <= hw[valid]).mean())
        res["conformal90_mean_width_K"] = float(2 * np.nanmean(hw))
    print(json.dumps(res, indent=1))
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.event}_t{t}_g{G}_{args.pick}"
    json.dump(res, open(os.path.join(args.out_dir, tag + ".json"), "w"), indent=2)
    np.savez_compressed(os.path.join(args.out_dir, tag + ".npz"), pred=pred.astype(np.float16), target=tgt.astype(np.float16),
                        linear=lin.astype(np.float16), farneback=fb.astype(np.float16),
                        scale=(scale.astype(np.float16) if scale is not None else np.zeros(1, np.float16)))

    # ---- figure (B13)
    ir = dict(cmap="turbo_r", vmin=190, vmax=300)
    fig, ax = plt.subplots(2, 4, figsize=(17, 6.6), constrained_layout=True)
    panels = [("Observed t", x[2][1], ir), (f"Observed t+{G} min", x[3][1], ir),
              (f"Truth (t+{G//2} min)", tgt[1], ir), (f"Model  (MAE {res['B13_MAE']['model']:.2f} K)", pred[1], ir),
              (f"|Error| linear ({res['B13_MAE']['linear']:.2f} K)", np.abs(lin[1] - tgt[1]), dict(cmap="magma", vmin=0, vmax=8)),
              (f"|Error| Farneback ({res['B13_MAE']['farneback']:.2f} K)", np.abs(fb[1] - tgt[1]), dict(cmap="magma", vmin=0, vmax=8)),
              (f"|Error| model", np.abs(pred[1] - tgt[1]), dict(cmap="magma", vmin=0, vmax=8))]
    for a, (title, img, kw) in zip(ax.ravel(), panels):
        im = a.imshow(img, **kw); a.set_title(title, fontsize=10); a.axis("off")
        a.contour(tgt[1], levels=[221.0], colors="white", linewidths=0.6)
        if "vmax" in kw and kw["vmax"] == 300:
            fig.colorbar(im, ax=a, fraction=0.035, label="B13 (K)")
    last = ax.ravel()[-1]
    if hw is not None:
        im = last.imshow(hw, cmap="viridis", vmin=0, vmax=8); last.set_title("90% conformal half-width (K)", fontsize=10)
        miss = valid & (np.abs(pred[1] - tgt[1]) > np.nan_to_num(hw, nan=0))
        ov = np.zeros((H, W, 4), np.float32); ov[miss] = (1, 0.1, 0.1, 1)
        last.imshow(ov); last.axis("off"); fig.colorbar(im, ax=last, fraction=0.035)
    else:
        last.axis("off")
    fig.suptitle(f"{args.event}  t={t}  gap {G} min  [{split}]   white contour = 221 K (truth)", fontsize=11)
    out = os.path.join(args.out_dir, tag + ".png")
    fig.savefig(out, dpi=170); print("Saved", out)


if __name__ == "__main__":
    main()

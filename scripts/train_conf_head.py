"""
scripts/train_conf_head.py -- Extension 1, stage 1: train the uncertainty head on a FROZEN METEOR-Net.

Laplace negative log-likelihood, TRAIN events only (val used for monitoring). The reconstruction network is
loaded from --ckpt, put in eval mode and never updated. Resumable. Refuses a test cache.

  python scripts/train_conf_head.py --cache_dir /content/cache/dev --events_json configs/events_v2.json \
     --stats outputs/normalization/v2_train_stats.json --ckpt <runs>/residual_12k/best.pt --base linear \
     --out_dir <runs>/conf_head_12k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

from src.data.fast_dataset import FastSequenceStore, BatchPrefetcher, to_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ckpt", required=True, help="trained METEOR-Net checkpoint (best.pt)")
    ap.add_argument("--base", choices=["none", "linear", "farneback"], default="linear")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_per_cell", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from src.models.meteor_net import METEORNet
    from src.models.normalization import ChannelStandardizer
    from src.models.conf_head import ConfHead, build_features, laplace_nll
    from train_v2 import build_val_ids, load_split_ids, predict
    from src.eval.conformal import spearman

    idx = os.path.join(args.cache_dir, "cache_index.json")
    if os.path.exists(idx) and json.load(open(idx)).get("cache_role") == "test":
        sys.exit("REFUSED: cache_dir is a TEST cache.")
    if args.smoke:
        args.updates, args.val_every, args.val_per_cell = 20, 10, 8
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = device.type == "cuda"
    base_mode = None if args.base == "none" else args.base
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)

    st = ChannelStandardizer.from_stats_file(args.stats).to(device)
    net = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    train_store = FastSequenceStore(args.cache_dir, load_split_ids(args.events_json, "train"))
    val_store = FastSequenceStore(args.cache_dir, load_split_ids(args.events_json, "val"))
    val_ids = build_val_ids(val_store, args.val_per_cell, seed=1234)

    torch.manual_seed(args.seed)
    head = ConfHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    path = os.path.join(args.out_dir, "head.pt")
    step = 0
    if os.path.exists(path):
        ck = torch.load(path, map_location=device, weights_only=False)
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print(f"Resumed at update {step}")

    def forward(b):
        with torch.no_grad():
            yhat = predict(net, st, b, base_mode, amp, device).float()
            feats = build_features(st, b, yhat)
        return yhat, head(feats)

    def validate():
        head.eval()
        nll, n, errs, scales = 0.0, 0, [], []
        with torch.no_grad():
            for batch in val_store.iter_eval_batches(val_ids, 16):
                b = to_device(batch, device, st)
                yhat, sc = forward(b)
                loss = laplace_nll(b["target_kelvin"], yhat, sc, b["combined_mask"])
                nll += float(loss); n += 1
                m = b["combined_mask"]
                e = (b["target_kelvin"][:, 1] - yhat[:, 1]).abs()[m]
                s = sc[:, 1][m]
                k = min(4000, e.numel())
                sel = torch.randperm(e.numel(), device=e.device)[:k]
                errs.append(e[sel].cpu().numpy()); scales.append(s[sel].cpu().numpy())
        head.train()
        e, s = np.concatenate(errs), np.concatenate(scales)
        return {"val_nll": nll / max(n, 1), "spearman_b13": spearman(s, e),
                "mean_abs_err_b13": float(e.mean()), "mean_scale_b13": float(s.mean())}

    rng = np.random.default_rng([args.seed, step])
    remaining = args.updates - step
    pf = BatchPrefetcher(lambda i: train_store.gather(train_store.sample_train_ids(rng, args.bs)), max(remaining, 0), depth=6)
    it = iter(pf)
    log = open(os.path.join(args.out_dir, "head_log.jsonl"), "a")
    head.train()
    t0, run = time.time(), []
    while step < args.updates:
        b = to_device(next(it), device, st)
        yhat, sc = forward(b)
        loss = laplace_nll(b["target_kelvin"], yhat, sc, b["combined_mask"])
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        step += 1
        run.append(float(loss))
        if step % 50 == 0:
            rec = {"step": step, "nll": float(np.mean(run)), "min": round((time.time() - t0) / 60, 1)}
            print(rec); log.write(json.dumps(rec) + "\n"); log.flush(); run = []
        if step % args.val_every == 0 or step == args.updates:
            v = validate(); v["step"] = step
            print("VAL", {k: round(x, 4) for k, x in v.items()}); log.write(json.dumps({"val": v}) + "\n"); log.flush()
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, path + ".tmp")
            os.replace(path + ".tmp", path)
    print("Done:", path)


if __name__ == "__main__":
    main()

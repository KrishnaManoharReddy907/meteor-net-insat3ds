"""
scripts/train_v2.py  --  METEOR-Net mixed-gap training on the v2 fast cache (Objective 1).

* Mixed gaps {10,20,30} min (gap 30 = 50% of samples), 50/50 convective/ambient patches.
* AMP fp16, micro-batch x grad-accumulation, linear warm-up + cosine LR, grad clipping.
* Resumable: checkpoint every --ckpt_every updates (model, optimizer, scaler, step, RNG, history).
  Re-run the SAME command after a Colab disconnect and it continues.
* Validation on a FIXED subset of val events (same patches every time), reporting MAE per gap and
  convective / strong (<221 K) / ambient on Band 13. Best checkpoint by gap-30 B13 MAE.
* Optional ablation --residual: network predicts a correction on top of the linear midpoint.
* Safety: only events whose split is train (training) or val (validation) in --events_json are
  loaded; refuses a cache whose cache_index.json says cache_role == 'test'. Cal/test never touched.

Example (Colab T4)
  python scripts/train_v2.py --cache_dir /content/cache/dev --events_json configs/events_v2.json \
      --stats outputs/normalization/v2_train_stats.json --out_dir /content/drive/MyDrive/runs/main_s0 \
      --updates 6000 --micro_bs 8 --accum 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.fast_dataset import FastSequenceStore, BatchPrefetcher, to_device  # noqa: E402


def load_split_ids(events_json: str, split: str):
    raw = json.load(open(events_json))
    ev = raw["events"] if isinstance(raw, dict) else raw
    return [e["event_id"] for e in ev if e["split"] == split]


def build_val_ids(store: FastSequenceStore, per_cell: int, seed: int):
    """Fixed validation set: per_cell patches for every (gap, convective/ambient) cell."""
    rng = np.random.default_rng(seed)
    ids = []
    for g in store.gaps_min:
        for c in (False, True):
            pool = store.select(gap_min=g, convective=c)
            if len(pool):
                ids.append(rng.choice(pool, size=min(per_cell, len(pool)), replace=False))
    return np.sort(np.concatenate(ids))


def predict(model, standardizer, b, residual, amp: bool, device):
    """residual: False/None = direct; True or 'linear' = correction over linear midpoint;
    'farneback' = correction over the optical-flow midpoint (hybrid, not flow-free)."""
    import torch
    base = "linear" if residual is True else (residual or None)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        y_norm = model(b["input_norm"])
    y_norm = y_norm.float()
    if base:
        xa, xb = b["input_kelvin"][:, 2], b["input_kelvin"][:, 3]            # frames t and t+G
        if base == "linear":
            bs = 0.5 * (xa + xb)
        elif base == "farneback":
            from src.models.flow_base import farneback_batch
            bs = torch.from_numpy(farneback_batch(xa.cpu().numpy(), xb.cpu().numpy())).to(xa.device)
        else:
            raise ValueError(base)
        bs = torch.nan_to_num(bs, nan=250.0)
        return bs + y_norm * standardizer.std.reshape(1, -1, 1, 1)
    return standardizer.denormalize_output(y_norm)


def evaluate(model, standardizer, store, ids, device, residual, amp, bs=16):
    import torch
    model.eval()
    acc = {}

    def add(key, err, n):
        s = acc.setdefault(key, [0.0, 0])
        s[0] += float(err); s[1] += int(n)

    with torch.no_grad():
        for batch in store.iter_eval_batches(ids, bs):
            b = to_device(batch, device, standardizer)
            y = predict(model, standardizer, b, residual, amp, device)
            m = b["combined_mask"]                                    # [B,H,W]
            err = (y - b["target_kelvin"]).abs()                       # [B,C,H,W]
            err = torch.where(m.unsqueeze(1), err, torch.zeros_like(err))
            tg13 = torch.nan_to_num(b["target_kelvin"][:, 1], nan=300.0)
            gaps = b["gap_min"]
            for g in torch.unique(gaps).tolist():
                sel = gaps == g
                mm = m[sel]
                add(f"mae_all_gap{g}", err[sel].sum() / 4.0, mm.sum())          # mean over 4 channels
                add(f"mae_b13_gap{g}", err[sel][:, 1].sum(), mm.sum())
                strong = mm & (tg13[sel] < 221.0)
                weak = mm & (tg13[sel] >= 221.0) & (tg13[sel] < 241.0)
                amb = mm & (tg13[sel] >= 241.0)
                for name, mk in (("strong", strong), ("weak", weak), ("ambient", amb)):
                    add(f"mae_b13_{name}_gap{g}", err[sel][:, 1][mk].sum(), mk.sum())
    model.train()
    out = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}
    out["n_pix_strong_gap30"] = acc.get("mae_b13_strong_gap30", [0, 0])[1]
    return out


def lr_at(step, total, base, warm, min_frac=0.05):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--events_json", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--micro_bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--conv_frac", type=float, default=0.5)
    ap.add_argument("--gaps", default="10,20,30")
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_per_cell", type=int, default=96)
    ap.add_argument("--residual", action="store_true", help="same as --base linear")
    ap.add_argument("--base", choices=["none", "linear", "farneback"], default="none",
                    help="predict a correction over this base (farneback = hybrid, uses optical flow)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="20 updates, tiny val; for a quick test")
    args = ap.parse_args()

    import torch
    from src.models.meteor_net import METEORNet
    from src.models.losses import MaskAwareConvectiveMAELoss
    from src.models.normalization import ChannelStandardizer

    idx = os.path.join(args.cache_dir, "cache_index.json")
    if os.path.exists(idx) and json.load(open(idx)).get("cache_role") == "test":
        sys.exit("REFUSED: cache_dir is a TEST cache.")
    train_ids = load_split_ids(args.events_json, "train")
    val_ids_ev = load_split_ids(args.events_json, "val")
    if not train_ids or not val_ids_ev:
        sys.exit("events_json needs train and val events.")
    base_mode = "linear" if (args.residual and args.base == "none") else (None if args.base == "none" else args.base)
    if args.smoke:
        args.updates, args.val_every, args.ckpt_every, args.val_per_cell, args.warmup = 20, 10, 10, 8, 5

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = (device.type == "cuda") and not args.no_amp
    gaps = [int(g) for g in args.gaps.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)

    standardizer = ChannelStandardizer.from_stats_file(args.stats).to(device)
    train_store = FastSequenceStore(args.cache_dir, train_ids, gaps_min=gaps)
    val_store = FastSequenceStore(args.cache_dir, val_ids_ev, gaps_min=gaps)
    val_ids = build_val_ids(val_store, args.val_per_cell, seed=1234)
    print(f"train samples {len(train_store):,} ({len(train_ids)} events); "
          f"val fixed subset {len(val_ids)} patches ({len(val_ids_ev)} events); amp={amp}; base={base_mode}")

    torch.manual_seed(args.seed)
    model = METEORNet(in_channels=4, feat_channels=64, tfee_hidden_channels=256, out_channels=4).to(device)
    loss_fn = MaskAwareConvectiveMAELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_path = os.path.join(args.out_dir, "last.pt")
    best_path = os.path.join(args.out_dir, "best.pt")
    step, best, history = 0, float("inf"), []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        step, best, history = ck["step"], ck["best"], ck["history"]
        print(f"Resumed from update {step}")

    def save(path):
        tmp = path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best": best, "history": history, "args": vars(args),
                    "stats_path": args.stats}, tmp)
        os.replace(tmp, path)

    # sampling RNG depends on (seed, step) so a resumed run does not repeat earlier batches
    rng = np.random.default_rng([args.seed, step])
    remaining = (args.updates - step) * args.accum + 25 * args.accum   # margin for skipped non-finite updates
    pf = BatchPrefetcher(lambda i: train_store.gather(train_store.sample_train_ids(
        rng, args.micro_bs, convective_frac=args.conv_frac)), remaining, depth=6)
    it = iter(pf)

    model.train()
    log = open(os.path.join(args.out_dir, "train_log.jsonl"), "a")
    t0, run_loss, run_n, skipped = time.time(), 0.0, 0, 0
    while step < args.updates:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.updates, args.lr, args.warmup)
        opt.zero_grad(set_to_none=True)
        ok = True
        for _ in range(args.accum):
            b = to_device(next(it), device, standardizer)
            y = predict(model, standardizer, b, base_mode, amp, device)
            loss = loss_fn(y, b["target_kelvin"], b["combined_mask"])
            if not torch.isfinite(loss):
                ok = False
                break
            scaler.scale(loss / args.accum).backward()
            run_loss += float(loss.item()); run_n += 1
        if not ok:
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if skipped > 20:
                sys.exit("Too many non-finite losses; stopping. Last checkpoint is intact.")
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt); scaler.update()
        step += 1

        if step % 50 == 0 or (args.smoke and step % 5 == 0):
            rec = {"step": step, "loss": run_loss / max(1, run_n), "lr": opt.param_groups[0]["lr"],
                   "scale": float(scaler.get_scale()) if amp else None, "skipped": skipped,
                   "min": round((time.time() - t0) / 60, 1)}
            print(rec); log.write(json.dumps(rec) + "\n"); log.flush()
            run_loss, run_n = 0.0, 0
        if step % args.val_every == 0 or step == args.updates:
            v = evaluate(model, standardizer, val_store, val_ids, device, base_mode, amp)
            v["step"] = step
            history.append(v)
            log.write(json.dumps({"val": v}) + "\n"); log.flush()
            print("VAL", {k: round(x, 3) for k, x in v.items() if k.startswith("mae_b13_gap") or k == "step"})
            score = v.get("mae_b13_gap30", float("inf"))
            if score < best:
                best = score
                save(best_path)
        if step % args.ckpt_every == 0 or step == args.updates:
            save(ckpt_path)

    print(f"Done. best val B13 gap-30 MAE = {best:.3f} K. Checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()

"""
scripts/make_figures.py -- paper figures from the JSON results (matplotlib, no GPU). Any input may be omitted.

  python scripts/make_figures.py --baselines outputs/baselines_val.json --conformal outputs/conformal_val.json \
     --loeo outputs/loeo_conformal.json --recursion outputs/recursion_val.json --utility outputs/utility_val.json \
     --out_dir outputs/figs
Colors are tied to the METHOD (fixed order), never to rank. Single-axis charts, light grid, legend always shown.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COLORS = {"linear": "#8a8985", "persist_t": "#c9c8c3", "persist_tg": "#b3b2ad", "farneback": "#eb6834",
          "rife": "#eda100", "main_s0": "#e87ba4", "hybrid_s0": "#1baf7a", "residual_s0": "#6250d6",
          "residual_s1": "#9085e9", "residual_12k": "#2a78d6"}
FALLBACK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948", "#008300"]
LABELS = {"residual_12k": "METEOR-Net (residual, 12k)", "farneback": "Farneback flow", "linear": "linear",
          "main_s0": "direct", "hybrid_s0": "hybrid (flow base)"}


def col(name, i=0):
    return COLORS.get(name, FALLBACK[i % len(FALLBACK)])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED)


def grouped(ax, groups, series, values, ylabel, title):
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        ax.bar(np.arange(len(groups)) + i * w - 0.4 + w / 2, [values[s][g] for g in range(len(groups))], w * 0.92,
               color=col(s, i), label=LABELS.get(s, s))
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups); ax.set_ylabel(ylabel); ax.set_title(title, color=INK, fontsize=11)
    style(ax)


def main():
    ap = argparse.ArgumentParser()
    for k in ("baselines", "conformal", "loeo", "recursion", "utility"):
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--out_dir", default="outputs/figs")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.baselines:
        R = json.load(open(a.baselines))["results"]
        methods = [m for m in dict.fromkeys(k.split("|")[0] for k in R) if m not in ("persist_t", "persist_tg")]
        gaps = [10, 20, 30]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, key, title in ((axs[0], "mae_b13", "Band 13 MAE, all pixels"), (axs[1], "mae_b13_strong", "Band 13 MAE, strong convection (<221 K)")):
            grouped(ax, [f"{g} min" for g in gaps], methods, {m: [R[f"{m}|gap{g}"][key] for g in gaps] for m in methods}, "MAE (K)", title)
        axs[0].legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(a.out_dir, "fig_mae_by_gap.png"), dpi=200)

    if a.conformal:
        V = json.load(open(a.conformal))["val_report"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        r = V["0.1"]
        names = [f"gap {g}" for g in r["by_gap"]] + list(r["by_true_class"]) 
        vals = [v["mondrian_cov"] for v in r["by_gap"].values()] + [v["mondrian_cov"] for v in r["by_true_class"].values()]
        axs[0].bar(range(len(vals)), vals, color="#2a78d6", width=0.6); axs[0].axhline(0.9, color=INK, lw=1, ls="--")
        axs[0].set_ylim(0.8, 1.0); axs[0].set_xticks(range(len(vals))); axs[0].set_xticklabels(names, rotation=20)
        axs[0].set_ylabel("coverage"); axs[0].set_title("90% intervals: validation coverage (dashed = nominal)", fontsize=11); style(axs[0])
        d = V["uncertainty_quality"]["mean_abs_err_by_scale_decile_K"]
        axs[1].plot(range(1, len(d) + 1), d, marker="o", color="#2a78d6", lw=2); axs[1].set_yscale("log")
        axs[1].set_xlabel("predicted-scale decile (low to high)"); axs[1].set_ylabel("mean |error| (K)")
        axs[1].set_title(f"Uncertainty tracks error (Spearman {V['uncertainty_quality']['spearman_scale_vs_abs_err']:.2f})", fontsize=11); style(axs[1])
        fig.savefig(os.path.join(a.out_dir, "fig_conformal.png"), dpi=200)

    if a.loeo:
        L = json.load(open(a.loeo))
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for i, (al, c) in enumerate((("0.1", "#2a78d6"), ("0.05", "#eb6834"))):
            ev = dict(sorted(L[al]["per_event"].items()))
            ax.scatter(range(len(ev)), [v["coverage"] for v in ev.values()], color=c, s=40, label=f"{100*(1-float(al)):.0f}% nominal", zorder=3)
            ax.axhline(1 - float(al), color=c, lw=1, ls="--")
        ax.set_xticks(range(len(ev))); ax.set_xticklabels([e.replace("event_", "") for e in ev], rotation=45, ha="right")
        ax.set_ylabel("coverage on held-out event"); ax.set_title("Leave-one-event-out conformal coverage", fontsize=11); ax.legend(frameon=False); style(ax)
        fig.savefig(os.path.join(a.out_dir, "fig_loeo.png"), dpi=200)

    if a.recursion:
        T = json.load(open(a.recursion))["table"]
        names = list(T)[::-1]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        h = 0.38
        y = np.arange(len(names))
        ax.barh(y + h / 2, [T[n]["MAE_B13"] for n in names], h, color="#2a78d6", label="all pixels")
        ax.barh(y - h / 2, [T[n]["MAE_B13_strong"] for n in names], h, color="#eb6834", label="strong (<221 K)")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("B13 MAE (K)"); ax.set_title("Recursive halving 20 to 10 to 5 min (ground truth at every level)", fontsize=11)
        ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.savefig(os.path.join(a.out_dir, "fig_recursion.png"), dpi=200)

    if a.utility:
        U = json.load(open(a.utility))["detection"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, thr in zip(axs, ("241", "221")):
            D = U[thr]; names = list(D)
            ax.bar(range(len(names)), [D[n].get("mean_delay_min", np.nan) for n in names],
                   color=[col(n.replace("15", ""), i) if n != "obs30" else "#8a8985" for i, n in enumerate(names)], width=0.6)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, ha="right")
            ax.set_ylabel("mean detection delay (min)"); ax.set_title(f"Cold-core onset below {thr} K", fontsize=11); style(ax)
        fig.savefig(os.path.join(a.out_dir, "fig_detection_delay.png"), dpi=200)
    print("figures in", a.out_dir)


if __name__ == "__main__":
    main()

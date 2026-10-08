#!/usr/bin/env python3
"""Figures of the AAMAS paper (writes paper/figs/*.pdf and .png).

    frontier.pdf          utility vs tokens per dataset (learned methods: mean +- sd over seeds;
                          static top-k selectors: one point per k)
    training_curves.pdf   training-episode return vs steps (PPO, lambda=1; mean +- sd over seeds x folds)
    algorithms.pdf        same, per RL algorithm (PPO, A2C, DQN, Recurrent PPO) for MA+msg

Colors follow a fixed categorical order (dataviz default palette, light mode) and a method keeps
its color in every figure.

Usage (from the repo root):  python scripts/paper/figures.py
"""

import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
VAL = RESULTS / "validation"
OUT = REPO / "paper" / "figs"
DATASETS = {"crag": "CRAG", "hotpotqa": "HotpotQA", "musique": "MuSiQue"}
BASELINES = {"crag": RESULTS / "ma_chunk/baselines.parquet",
             "hotpotqa": RESULTS / "ma_chunk/hotpotqa_n1000/baselines.parquet",
             "musique": RESULTS / "ma_chunk/musique_n1000/baselines.parquet"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e5e0"
METHODS = {  # key -> (label, color); fixed across figures
    "ma_msg": ("MA-Chunk (MA+msg)", "#2a78d6"),
    "single": ("Single (centralized RL)", "#eb6834"),
    "ma_msg_drop0.2": ("MA-Chunk, failure-aware", "#1baf7a"),
    "ma_msg_rich": ("MA-Chunk, rich obs.", "#eda100"),
    "sup_gbm": ("Supervised, rich feat.", "#e87ba4"),
    "sup_gbm_obsfeat": ("Supervised, parity feat.", "#008300"),
    "static_ce": ("Static: cross-encoder top-k", "#4a3aa7"),
    "static_rrf": ("Static: RRF top-k", "#e34948"),
}
ALGOS = {"ppo": ("PPO", "#2a78d6"), "a2c": ("A2C", "#eb6834"), "dqn": ("DQN", "#1baf7a"),
         "recurrent_ppo": ("R-PPO", "#eda100")}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# --------------------------------------------------------------------------- frontier
def frontier():
    runs = pd.read_csv(VAL / "analysis" / "all_runs.csv")
    runs = runs.sort_values("source", ascending=False).drop_duplicates(["dataset", "variant", "lam", "seed"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
    for ax, (ds, label) in zip(axes, DATASETS.items()):
        b = pd.read_parquet(BASELINES[ds]).groupby("condition")[["utility", "tokens"]].mean()
        for key, prefix in [("static_ce", "topk_ce"), ("static_rrf", "rrf")]:
            pts = b[b.index.str.match(rf"{prefix}_\d+$")].sort_values("tokens")
            ax.plot(pts["tokens"], pts["utility"], color=METHODS[key][1], linewidth=1.2, linestyle="--",
                    marker="o", markersize=3, label=METHODS[key][0])
        for key in ["sup_gbm", "sup_gbm_obsfeat", "single", "ma_msg"]:
            x = runs[(runs.dataset == ds) & (runs.variant == key) & (runs.lam > 0)]
            if x.empty:
                continue
            g = x.groupby("lam").agg(u=("utility", "mean"), usd=("utility", "std"), t=("tokens", "mean")).sort_values("t")
            lw = 2.2 if key == "ma_msg" else 1.4
            ax.errorbar(g["t"], g["u"], yerr=g["usd"].fillna(0), color=METHODS[key][1], linewidth=lw,
                        marker="o", markersize=4 if key == "ma_msg" else 3, capsize=2, label=METHODS[key][0],
                        zorder=5 if key == "ma_msg" else 3)
        style(ax)
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel("input tokens sent to the reader (log scale)", fontsize=7, color=MUTED)
        ax.set_xscale("log")
        ax.set_xlim(70, 1300)
        ax.set_xticks([100, 200, 500, 1000])
        ax.set_xticklabels(["100", "200", "500", "1000"])
        ax.minorticks_off()
    axes[0].set_ylabel("utility (evidence in context)", fontsize=7, color=MUTED)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, "frontier")


# --------------------------------------------------------------------------- learning curves
def curve_stats(paths):
    curves = []
    for p in paths:
        for _, pts in json.loads(Path(p).read_text()).items():
            if len(pts) > 1:
                curves.append(np.array(pts, dtype=float))
    if not curves:
        return None
    t_max = min(c[-1, 0] for c in curves)
    grid = np.linspace(min(c[0, 0] for c in curves), t_max, 80)
    ys = np.vstack([np.interp(grid, c[:, 0], c[:, 1]) for c in curves])
    return grid, ys.mean(0), ys.std(0), len(curves)


def training_curves(keys, colors, name, lam=1.0, tag_fmt="{key}_lam{lam}_s*"):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    drawn = False
    for ax, (ds, label) in zip(axes, DATASETS.items()):
        for key, (lab, col) in zip(keys, colors):
            paths = [p for p in glob.glob(str(VAL / ds / "runs" / "curves" / (tag_fmt.format(key=key, lam=lam) + ".json")))]
            st = curve_stats(paths)
            if st is None:
                continue
            grid, m, sd, n = st
            ax.fill_between(grid / 1000, m - sd, m + sd, color=col, alpha=0.15, linewidth=0)
            ax.plot(grid / 1000, m, color=col, linewidth=1.8, label=lab)
            drawn = True
        style(ax)
        ax.set_title(f"{label} ($\\lambda$={lam:g})", fontsize=9, color=INK)
        ax.set_xlabel("training steps (thousands, per fold)", fontsize=7, color=MUTED)
    axes[0].set_ylabel("mean training-episode return", fontsize=7, color=MUTED)
    if not drawn:
        plt.close(fig)
        return
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, name)


def main():
    frontier()
    keys = ["ma_msg", "single", "ma_msg_drop0.2", "ma_msg_rich"]
    training_curves(keys, [METHODS[k] for k in keys], "training_curves")
    # per algorithm (MA+msg): PPO tags have no algo suffix; the others end with _algo-<name>
    fig_keys, fig_cols = [], []
    for algo, (lab, col) in ALGOS.items():
        fig_keys.append("ma_msg" if algo == "ppo" else f"ma_msg_algo-{algo}")
        fig_cols.append((lab, col))
    training_curves(fig_keys, fig_cols, "algorithms")


if __name__ == "__main__":
    main()

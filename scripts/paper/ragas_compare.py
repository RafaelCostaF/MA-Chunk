#!/usr/bin/env python3
"""RAGAS comparison of methods (docs/PROTOCOL.md, sections 3-4).

Reads results/v4/ragas/<dataset>/<condition>.parquet. Conditions that differ only by the
seed suffix (_s0, _s1, _s2) form one method; per question, metrics are averaged over its seeds.
RAGAS-5 = per-question mean of the five metrics that are defined for that question (faithfulness
is NaN when the answer has no statements, e.g. an empty answer or a refusal; its NaN rate is
reported).

Outputs (results/v4/ragas/analysis/):
    summary_<dataset>.csv  per method: n, chunks, each metric (mean, 95% bootstrap CI), RAGAS-5, faithfulness NaN rate
    paired_<dataset>.csv   every method vs --ref: paired difference per metric (questions present in
                           both), Wilcoxon signed-rank, Holm within each metric across comparisons

Usage (from the repo root):
    python scripts/paper/ragas_compare.py --dataset crag --ref ma_msg_lam0.3
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
R = RESULTS / "v4" / "ragas"
METRICS = ["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"]


def method_of(cond):
    return re.sub(r"_s\d+(?=($|_reord$))", "", cond)


def load(ds):
    frames = [pd.read_parquet(p) for p in sorted((R / ds).glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    df["method"] = df["condition"].map(method_of)
    df["ragas5"] = df[METRICS].mean(axis=1, skipna=True)
    df["faith_nan"] = df["faithfulness"].isna().astype(float)
    per_q = df.groupby(["method", "interaction_id"]).agg(
        **{m: (m, "mean") for m in METRICS + ["ragas5", "faith_nan", "n_contexts"]},
        n_seeds=("condition", "nunique")).reset_index()
    return per_q


def boot_ci(x, rng, n=2000):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ref", required=True, help="reference method (seed suffix removed), e.g. ma_msg_lam0.3")
    args = ap.parse_args()
    out = R / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pq = load(args.dataset)
    rng = np.random.default_rng(0)

    rows = []
    for m, g in pq.groupby("method"):
        row = {"method": m, "n": len(g), "seeds": int(g["n_seeds"].max()), "chunks": g["n_contexts"].mean(),
               "faith_nan_rate": g["faith_nan"].mean()}
        for k in METRICS + ["ragas5"]:
            x = g[k].to_numpy(float)
            lo, hi = boot_ci(x, rng)
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = np.nanmean(x), lo, hi
        rows.append(row)
    summ = pd.DataFrame(rows).sort_values("ragas5", ascending=False)
    summ.to_csv(out / f"summary_{args.dataset}.csv", index=False, float_format="%.4f")

    ref = pq[pq["method"] == args.ref].set_index("interaction_id")
    if ref.empty:
        raise SystemExit(f"reference {args.ref} not found")
    prow = []
    for m, g in pq[pq["method"] != args.ref].groupby("method"):
        g = g.set_index("interaction_id")
        common = g.index.intersection(ref.index)
        for k in METRICS + ["ragas5"]:
            d = (g.loc[common, k] - ref.loc[common, k]).dropna().to_numpy()
            nz = d[d != 0]
            p = stats.wilcoxon(nz).pvalue if len(nz) > 1 else 1.0
            prow.append({"method": m, "metric": k, "n": len(d), "diff_vs_ref": d.mean() if len(d) else np.nan,
                         "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": p})
    paired = pd.DataFrame(prow)
    paired["p_holm"] = np.nan
    for k, g in paired.groupby("metric"):
        paired.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    paired.to_csv(out / f"paired_{args.dataset}.csv", index=False, float_format="%.4f")

    show = ["method", "n", "seeds", "chunks"] + METRICS + ["ragas5", "faith_nan_rate"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ[show].round(3).to_string(index=False))
        print(f"\npaired vs {args.ref} (RAGAS-5 and context precision):")
        print(paired[paired.metric.isin(["ragas5", "context_precision"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

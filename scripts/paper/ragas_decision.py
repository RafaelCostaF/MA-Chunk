#!/usr/bin/env python3
"""Precision-aware reward, decision (docs/PROTOCOL.md, section 4): each v4 configuration vs v3 (R0) on RAGAS-5 and on
each RAGAS metric. Unit = question; per question, the score is averaged over seeds and over
lambda in {0.3, 1} (primary) or kept per lambda (secondary). Wilcoxon signed-rank, Holm across the
three datasets for each (configuration, metric). Chunk criterion: mean chunks vs v3 at the same lambda.

Output: results/v4/ragas/analysis/v4_decision.csv
Usage: python scripts/paper/ragas_decision.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
R = RESULTS / "v4/ragas"
METRICS = ["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"]
CONFIGS = {"R0 base": "ma_msg", "R1 reorder": "ma_msg@reord", "R2 confidence": "ma_msg_to-confidence",
           "R3 AP": "ma_msg_rw-marginal_ap", "R4 AP+conf": "ma_msg_rw-marginal_ap_to-confidence",
           "R5 AP+conf+sup2": "ma_msg_rw-marginal_ap_to-confidence_sup2"}


def load(ds):
    df = pd.concat([pd.read_parquet(p) for p in (R / ds).glob("*.parquet")], ignore_index=True)
    m = df["condition"].str.extract(r"^(?P<prefix>.+)_lam(?P<lam>[\d.]+)_s(?P<seed>\d)(?P<reord>_reord)?$")
    df = pd.concat([df, m], axis=1).dropna(subset=["prefix"])
    df["prefix"] = np.where(df["reord"].notna(), df["prefix"] + "@reord", df["prefix"])
    df["ragas5"] = df[METRICS].mean(axis=1, skipna=True)
    return df


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


rows = []
for ds in ["crag", "hotpotqa", "musique"]:
    df = load(ds)
    for scope in ["both", "0.3", "1.0"]:
        d = df if scope == "both" else df[df["lam"] == scope]
        per_q = d.groupby(["prefix", "interaction_id"])[METRICS + ["ragas5", "n_contexts"]].mean()
        chunks = d.groupby("prefix")["n_contexts"].mean()
        if "ma_msg" not in per_q.index.get_level_values(0):
            continue
        ref = per_q.loc["ma_msg"]
        for name, pref in CONFIGS.items():
            if pref == "ma_msg" or pref not in per_q.index.get_level_values(0):
                continue
            x = per_q.loc[pref]
            common = x.index.intersection(ref.index)
            for k in METRICS + ["ragas5"]:
                diff = (x.loc[common, k] - ref.loc[common, k]).dropna().to_numpy()
                nz = diff[diff != 0]
                p = stats.wilcoxon(nz).pvalue if len(nz) > 1 else 1.0
                rows.append({"dataset": ds, "lam": scope, "config": name, "metric": k, "n": len(diff),
                             "v3": ref.loc[common, k].mean(), "v4": x.loc[common, k].mean(), "diff": diff.mean(),
                             "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": p,
                             "chunks_v3": chunks["ma_msg"], "chunks": chunks[pref],
                             "chunk_change": chunks[pref] / chunks["ma_msg"] - 1})
out = pd.DataFrame(rows)
out["p_holm"] = np.nan
for _, g in out.groupby(["lam", "config", "metric"]):
    out.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
out.to_csv(R / "analysis" / "v4_decision.csv", index=False, float_format="%.5f")
show = ["dataset", "lam", "config", "n", "v3", "v4", "diff", "wins", "losses", "p_holm", "chunks_v3", "chunks", "chunk_change"]
with pd.option_context("display.width", 220):
    print("=== PRIMARY: RAGAS-5, lambda-averaged ===")
    print(out[(out.metric == "ragas5") & (out.lam == "both")][show].round(4).to_string(index=False))
    print("\n=== RAGAS-5 per lambda ===")
    print(out[(out.metric == "ragas5") & (out.lam != "both")][show].round(4).to_string(index=False))

#!/usr/bin/env python3
"""Builds every number used in the AAMAS paper draft from the saved results.

Outputs (results/paper_tables/):
  end_task.csv       - CRAG score / correct / incorrect / missing / tokens per condition, with a
                       95% bootstrap CI of the CRAG score, without and with the abstention gate;
  frontier.csv       - offline utility-vs-tokens frontier (mean +- std over seeds) per variant/lambda;
  specialization.csv - share of SEND messages per agent (MA variants);
  signals.csv        - AUC of each agent's score for predicting a useful chunk.

The abstention gate here is the PRELIMINARY one: abstain when the query's best cross-encoder
score is below a threshold chosen on the training folds (5-fold CV, 5 seeds) to maximise the
CRAG score. It never sees the test fold. It is applied identically to every condition.

Usage (from the repo root):
    PYTHONDONTWRITEBYTECODE=1 python scripts/paper/tables_endtask.py
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
from ma_chunk.env_ma import AGENTS  # noqa: E402
from ma_chunk.train_ma import load_pool  # noqa: E402

RES = RESULTS
OUT = RES / "paper_tables"


def bootstrap_ci(scores, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores)
    means = [s[rng.integers(0, len(s), len(s))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def gated_scores(x: pd.DataFrame, seeds=range(5)) -> np.ndarray:
    """Per-query CRAG score after the CV-selected abstention threshold, averaged over seeds."""
    out = np.zeros((len(seeds), len(x)))
    for k, seed in enumerate(seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(x, x["domain"]):
            grid = np.quantile(x["qmax"].iloc[tr], np.linspace(0, 0.8, 41))
            best = max(grid, key=lambda t: np.where(x["qmax"].iloc[tr] < t, 0, x["score"].iloc[tr]).mean())
            out[k, te] = np.where(x["qmax"].iloc[te] < best, 0, x["score"].iloc[te])
    return out.mean(axis=0)


def end_task(pool: pd.DataFrame) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in [
        RES / "ma_chunk" / "eval" / "ma_round1_per_query.parquet",
        RES / "ma_chunk" / "eval" / "paper_baselines_per_query.parquet",
    ]]
    prior = pd.read_parquet(RES / "prior_selector" / "per_query.parquet")
    prior = prior[prior["condition"].str.startswith("rl_")].assign(condition=lambda d: "prior_" + d["condition"])
    res = pd.concat(frames + [prior], ignore_index=True)
    res["score"] = res["verdict"].map({"correct": 1, "incorrect": -1}).fillna(0)
    res["qmax"] = res["interaction_id"].map(pool.groupby("interaction_id")["s_ce"].max())

    rows = []
    for cond, x in res.groupby("condition"):
        x = x.reset_index(drop=True)
        g = gated_scores(x)
        lo, hi = bootstrap_ci(x["score"])
        glo, ghi = bootstrap_ci(g)
        rows.append({
            "condition": cond, "n": len(x),
            "crag_score": x["score"].mean(), "ci_lo": lo, "ci_hi": hi,
            "correct": (x["verdict"] == "correct").mean(), "incorrect": (x["verdict"] == "incorrect").mean(),
            "missing": (x["verdict"] == "missing").mean(), "input_tokens": x["input_tokens"].mean(),
            "n_chunks": x["n_chunks"].mean(),
            "gated_score": g.mean(), "gated_ci_lo": glo, "gated_ci_hi": ghi,
        })
    return pd.DataFrame(rows).sort_values("gated_score", ascending=False)


def frontier() -> pd.DataFrame:
    d = pd.read_json(RES / "ma_chunk" / "sweep_summary.jsonl", lines=True)
    f = d.groupby(["variant", "lam"]).agg(utility=("utility", "mean"), utility_std=("utility", "std"),
                                          tokens=("tokens", "mean"), n_chunks=("n_chunks", "mean"))
    b = pd.read_parquet(RES / "ma_chunk" / "baselines.parquet").groupby("condition")[["utility", "tokens"]].mean()
    ce = b[b.index.str.match(r"topk_ce_\d+$")].sort_values("tokens")
    f["ce_topk_same_tokens"] = [np.interp(t, ce["tokens"], ce["utility"]) for t in f["tokens"]]
    return f.reset_index()


def specialization() -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(RES / "ma_chunk" / "runs" / "ma*_s*.parquet")):
        d = pd.read_parquet(path)
        tag = d["condition"].iloc[0]
        sends = d[[f"sends_{a}" for a in AGENTS]].sum()
        rows.append({"variant": tag.rsplit("_lam", 1)[0], "lam": float(tag.split("lam")[1].split("_")[0]),
                     **{a: sends[f"sends_{a}"] / sends.sum() for a in AGENTS}})
    return pd.DataFrame(rows).groupby(["variant", "lam"]).mean().reset_index()


def signals(pool: pd.DataFrame) -> pd.DataFrame:
    y = pool["useful"] == 1
    return pd.DataFrame([{"agent": a, "auc": roc_auc_score(y, pool[f"s_{a}"])} for a in AGENTS])


def multihop(ds: str):
    """Frontier, end-task (mean of the 3 seeds per query), paired tests and gates for one dataset."""
    from scipy.stats import wilcoxon

    R = RES / "ma_chunk" / f"{ds}_n1000"
    d = pd.read_json(R / "sweep_summary.jsonl", lines=True)
    f = d.groupby(["variant", "lam"]).agg(utility=("utility", "mean"), utility_std=("utility", "std"),
                                          tokens=("tokens", "mean")).reset_index()
    b = pd.read_parquet(R / "baselines.parquet").groupby("condition")[["utility", "tokens"]].mean()
    ce = b[b.index.str.match(r"topk_ce_\d+$")].sort_values("tokens")
    f["ce_topk_same_tokens"] = [np.interp(t, ce["tokens"], ce["utility"]) for t in f["tokens"]]

    r = pd.read_parquet(R / "eval" / "endtask_per_query.parquet")
    r["score"] = r["verdict"].map({"correct": 1, "incorrect": -1}).fillna(0)
    r["group"] = r["condition"].str.replace(r"_s\d$", "", regex=True)
    per_q = r.groupby(["group", "interaction_id"])[["score", "f1", "em", "input_tokens"]].mean()
    rows = []
    for g, x in per_q.groupby(level=0):
        lo, hi = bootstrap_ci(x["score"])
        v = r[r["group"] == g]["verdict"]
        rows.append({"group": g, "score": x["score"].mean(), "ci_lo": lo, "ci_hi": hi,
                     "correct": (v == "correct").mean(), "incorrect": (v == "incorrect").mean(),
                     "f1": x["f1"].mean(), "em": x["em"].mean(), "input_tokens": x["input_tokens"].mean()})
    end = pd.DataFrame(rows).sort_values("score", ascending=False)

    m = per_q["score"].unstack(0)
    rng = np.random.default_rng(0)
    tests = []
    for a in ["ma_lam0.3", "ma_msg_lam0.3", "ma_lam1.0", "ma_msg_lam1.0"]:
        for c in ["topk_ce_1", "topk_ce_2", "rrf_3", "single_lam0.3", "single_lam1.0"]:
            x = (m[a] - m[c]).to_numpy()
            boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
            nz = x[x != 0]
            tests.append({"method": a, "vs": c, "diff": x.mean(), "ci_lo": np.percentile(boots, 2.5),
                          "ci_hi": np.percentile(boots, 97.5), "p_wilcoxon": wilcoxon(nz).pvalue if len(nz) > 10 else np.nan})
    gate = pd.read_csv(R / "eval" / "gate.csv") if (R / "eval" / "gate.csv").exists() else pd.DataFrame()
    return {f"frontier_{ds}": f, f"end_task_{ds}": end, f"tests_{ds}": pd.DataFrame(tests), f"gate_{ds}": gate}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in ("hotpotqa", "musique"):
        for name, df in multihop(ds).items():
            df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
            with pd.option_context("display.width", 220, "display.max_columns", 30):
                print(f"\n== {name}\n{df.round(3).to_string(index=False)}")
    pool = load_pool(8)
    for name, df in [("end_task", end_task(pool)), ("frontier", frontier()),
                     ("specialization", specialization()), ("signals", signals(pool))]:
        df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(f"\n== {name}\n{df.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()

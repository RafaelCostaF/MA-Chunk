#!/usr/bin/env python3
"""RAGAS-aligned labels, decision (docs/PROTOCOL.md, section 9): R6 (primary) and R7 (secondary) at the
budget-matched lambdas (a3/lambda_choice.json) vs v3 (lambda 0.3 and 1.0), per dataset. Unit =
question, score averaged over seeds and over the two budget pairs (primary) or per pair. Wilcoxon
signed-rank; Holm across the three datasets for each (variant, metric, scope).

Output: results/v4/a3/decision.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
R = RESULTS / "v4/ragas"
M = ["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"]
PREFIX = {"R6": "ma_msg_rw-marginal_ap_to-confidence_u-mix", "R7": "ma_msg_rw-marginal_ap_to-confidence_u-attr"}


def load(ds, prefix, lams):
    frames = []
    for pair, l in enumerate(lams):
        for s in (0, 1, 2):
            f = R / ds / f"{prefix}_lam{l}_s{s}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f).assign(pair=pair))
    d = pd.concat(frames, ignore_index=True)
    d["ragas5"] = d[M].mean(axis=1, skipna=True)
    return d


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def main():
    choice = json.loads((RESULTS / "v4/a3/lambda_choice.json").read_text())
    rows = []
    for ds in ["crag", "hotpotqa", "musique"]:
        v3 = load(ds, "ma_msg", ["0.3", "1.0"])
        for vname, prefix in PREFIX.items():
            lams = [choice[f"{ds}|{vname}|0.3"]["choice"], choice[f"{ds}|{vname}|1.0"]["choice"]]
            x = load(ds, prefix, lams)
            for scope, sel in [("both", [0, 1]), ("pair0.3", [0]), ("pair1.0", [1])]:
                a, b = v3[v3.pair.isin(sel)], x[x.pair.isin(sel)]
                qa = a.groupby("interaction_id")[M + ["ragas5"]].mean()
                qb = b.groupby("interaction_id")[M + ["ragas5"]].mean()
                common = qa.index.intersection(qb.index)
                for k in M + ["ragas5"]:
                    d = (qb.loc[common, k] - qa.loc[common, k]).dropna().to_numpy()
                    nz = d[d != 0]
                    rows.append({"dataset": ds, "variant": vname, "lams": "/".join(lams), "scope": scope, "metric": k,
                                 "n": len(d), "v3": qa.loc[common, k].mean(), "v4": qb.loc[common, k].mean(),
                                 "diff": d.mean(), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                                 "p": stats.wilcoxon(nz).pvalue if len(nz) > 1 else 1.0,
                                 "chunks_v3": a["n_contexts"].mean(), "chunks_v4": b["n_contexts"].mean()})
    out = pd.DataFrame(rows)
    out["chunk_change"] = out["chunks_v4"] / out["chunks_v3"] - 1
    out["p_holm"] = np.nan
    for _, g in out.groupby(["variant", "scope", "metric"]):
        out.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    out.to_csv(RESULTS / "v4/a3/decision.csv", index=False, float_format="%.5f")
    show = ["dataset", "variant", "lams", "scope", "n", "v3", "v4", "diff", "wins", "losses", "p_holm", "chunks_v3", "chunks_v4", "chunk_change"]
    with pd.option_context("display.width", 230):
        print(out[out.metric == "ragas5"][show].round(4).to_string(index=False))
        print()
        print(out[(out.scope == "both")].pivot_table(index=["dataset", "variant"], columns="metric", values="diff").round(3).to_string())


if __name__ == "__main__":
    main()

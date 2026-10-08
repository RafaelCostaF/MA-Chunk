#!/usr/bin/env python3
"""Conservative re-analysis of the pooled comparisons (guards against pseudo-replication).

The pooled tests of validation_analysis.py treat every (dataset, lambda, seed) difference as
independent. The two lambdas of the same (dataset, seed) share folds and data, so here the unit
is the cluster (dataset, seed): paired differences are averaged over lambda inside each cluster,
and only clusters enter the test (n = datasets x seeds). Reported per comparison:
    mean cluster difference, 95% cluster-bootstrap CI, wins/losses over clusters,
    exact Wilcoxon signed-rank p, exact sign-test p, Holm-corrected within each family.

Usage (from the repo root):  python scripts/paper/conservative_stats.py
Output: results/validation/analysis/conservative_tests.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
A = RESULTS / "validation" / "analysis"

FAMILIES = {
    "decomposition": [("ma_msg", "single"), ("ma_msg_rich", "single_rich"), ("ma_msg", "ma")],
    "reward": [("ma_msg", "ma_msg_rw-terminal"), ("ma_msg", "ma_msg_rw-individual")],
    "orchestration": [("ma_msg", "ma_msg_to-random"), ("ma_msg", "ma_msg_to-confidence")],
    "robustness": [("ma_msg", "ma_msg_rand"), ("ma_msg", "ma_msg_m4"), ("ma_msg", "ma_msg_m12"),
                   ("ma_msg", "ma_msg_drop0.2")],
    "why_rl": [("ma_msg", "sup_gbm_obsfeat"), ("ma_msg", "sup_gbm"), ("ma_msg", "sup_logreg"),
               ("ma_msg_rich", "sup_gbm"), ("ma_msg_rich", "sup_logreg")],
}
LAMS = (0.3, 1.0)
# Comparisons restricted to the datasets where they are informative: by Proposition 2 the selfish and
# marginal rewards coincide when utility is modular (multi-hop), so those clusters are exact ties and
# are reported separately (n_ties_excluded) instead of diluting the CRAG effect.
RESTRICT = {("ma_msg", "ma_msg_rw-individual"): ["crag"]}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    runs = pd.read_csv(A / "all_runs.csv")
    runs = runs.sort_values("source", ascending=False).drop_duplicates(["dataset", "variant", "lam", "seed"])
    runs = runs[runs["lam"].isin(LAMS)]
    rng = np.random.default_rng(0)
    rows = []
    for fam, pairs in FAMILIES.items():
        fam_rows = []
        for a, b in pairs:
            ra = runs[runs.variant == a].set_index(["dataset", "lam", "seed"])["team_reward"]
            rb = runs[runs.variant == b].set_index(["dataset", "lam", "seed"])["team_reward"]
            common = ra.index.intersection(rb.index)
            if len(common) == 0:
                continue
            d = (ra.loc[common] - rb.loc[common]).groupby(level=["dataset", "seed"]).mean()  # cluster = (dataset, seed)
            n_excl = 0
            if (a, b) in RESTRICT:
                keep = d.index.get_level_values("dataset").isin(RESTRICT[(a, b)])
                n_excl = int((~keep & (d.abs() < 1e-12)).sum())
                d = d[keep]
            x = d.to_numpy()
            nz = x[x != 0]
            p_w = stats.wilcoxon(nz, method="exact").pvalue if len(nz) >= 2 else 1.0
            p_s = stats.binomtest(int((nz > 0).sum()), len(nz)).pvalue if len(nz) else 1.0
            # Percentile bootstrap over clusters; with fewer than 5 clusters it is degenerate, so the
            # min-max range over clusters is reported instead (ci_method).
            if len(x) >= 5:
                boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
                lo, hi, ci_method = np.percentile(boots, 2.5), np.percentile(boots, 97.5), "bootstrap"
            else:
                lo, hi, ci_method = x.min(), x.max(), "range"
            fam_rows.append({"family": fam, "a": a, "b": b, "n_clusters": len(x), "n_ties_excluded": n_excl,
                             "mean_diff": x.mean(), "ci_lo": lo, "ci_hi": hi, "ci_method": ci_method,
                             "wins": int((x > 0).sum()), "losses": int((x < 0).sum()),
                             "p_wilcoxon_exact": p_w, "p_sign": p_s})
        if fam_rows:
            adj = holm([r["p_wilcoxon_exact"] for r in fam_rows])
            for r, pa in zip(fam_rows, adj):
                r["p_holm"] = pa
            rows += fam_rows
    out = pd.DataFrame(rows)
    out.to_csv(A / "conservative_tests.csv", index=False, float_format="%.5f")
    with pd.option_context("display.width", 220):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

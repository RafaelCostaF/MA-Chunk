#!/usr/bin/env python3
"""Tests every hypothesis of docs/PROTOCOL.md from the saved runs.

Writes results/validation/analysis/*.csv and prints a compact report. Missing runs are
skipped (the script can be re-run as batches finish).

Statistics: the replication unit is the seed (same CV folds per seed across variants), so
variants are compared with paired t-tests over seeds (and over seed x dataset x lambda when
pooled); Holm correction within each hypothesis family. Effect sizes are mean paired differences.

Usage (from the repo root):
    PYTHONDONTWRITEBYTECODE=1 python scripts/paper/validation_analysis.py
"""

import glob
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
from ma_chunk.env_ma import AGENTS  # noqa: E402
from ma_chunk.train_ma import load_pool  # noqa: E402

VAL = RESULTS / "validation"
OUT = VAL / "analysis"
OLD = {"crag": RESULTS / "ma_chunk", "hotpotqa": RESULTS / "ma_chunk/hotpotqa_n1000",
       "musique": RESULTS / "ma_chunk/musique_n1000"}
POOLS = {"crag": None, "hotpotqa": OLD["hotpotqa"] / "candidates.parquet", "musique": OLD["musique"] / "candidates.parquet"}
DATASETS = list(OLD)
LAMS_GRID = [0.1, 0.3, 0.6, 1.0, 2.0]
TOKENS_REF = 1200.0


# --------------------------------------------------------------------------- loading
def parse_tag(tag):
    m = re.match(r"(?P<variant>.+)_lam(?P<lam>[\d.]+)_s(?P<seed>\d+)$", tag)
    return m.group("variant"), float(m.group("lam")), int(m.group("seed"))


def load_runs(ds):
    """One row per (variant, lam, seed) with test-fold means; validation runs take precedence."""
    rows = {}
    paths = sorted(glob.glob(str(OLD[ds] / "runs" / "*.parquet"))) + sorted(glob.glob(str(VAL / ds / "runs" / "*.parquet")))
    for p in paths:
        d = pd.read_parquet(p)
        if d.empty or "condition" not in d:
            continue
        variant, lam, seed = parse_tag(d["condition"].iloc[0])
        tr = d["team_reward"] if "team_reward" in d else d["utility"] - lam * d["tokens"] / 1000.0
        row = {"dataset": ds, "variant": variant, "lam": lam, "seed": seed, "team_reward": tr.mean(),
               "utility": d["utility"].mean(), "tokens": d["tokens"].mean(), "n_chunks": d["n_chunks"].mean(),
               "source": "validation" if str(VAL) in p else "original", "path": p}
        rows[(variant, lam, seed, row["source"])] = row
    return pd.DataFrame(rows.values())


def pick(runs, variant, lam):
    """Seed -> team reward, preferring validation re-runs (same code version) over originals."""
    x = runs[(runs["variant"] == variant) & (np.isclose(runs["lam"], lam))]
    x = x.sort_values("source", ascending=False).drop_duplicates("seed")  # 'validation' > 'original'
    return x.set_index("seed")


def paired(a, b, metric="team_reward"):
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 2:
        return None
    d = a.loc[common, metric].to_numpy() - b.loc[common, metric].to_numpy()
    t = stats.ttest_rel(a.loc[common, metric], b.loc[common, metric])
    return {"n_seeds": len(common), "diff": d.mean(), "diff_sd": d.std(ddof=1), "p": t.pvalue}


def holm(df, col="p"):
    df = df.copy()
    valid = df[col].notna()
    p = df.loc[valid, col].to_numpy()
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    df.loc[valid, "p_holm"] = adj
    return df


def compare_family(all_runs, pairs, lams=(0.3, 1.0)):
    """Per (dataset, lambda) paired t-tests over seeds, plus one POOLED test per pair over every
    (dataset, lambda, seed) difference (Wilcoxon signed-rank; sign count reported). Holm over the
    family (per-condition rows and pooled rows corrected separately)."""
    rows, pooled = [], []
    for a, b in pairs:
        diffs = []
        for ds in DATASETS:
            runs = all_runs[ds]
            for lam in lams:
                A, B = pick(runs, a, lam), pick(runs, b, lam)
                r = paired(A, B)
                if r:
                    rows.append({"dataset": ds, "lam": lam, "a": a, "b": b, **r})
                    common = sorted(set(A.index) & set(B.index))
                    diffs += list(A.loc[common, "team_reward"].to_numpy() - B.loc[common, "team_reward"].to_numpy())
        d = np.asarray(diffs)
        if len(d) >= 5 and np.any(d != 0):
            pooled.append({"dataset": "POOLED", "lam": np.nan, "a": a, "b": b, "n_seeds": len(d), "diff": d.mean(),
                           "diff_sd": d.std(ddof=1), "p": stats.wilcoxon(d[d != 0]).pvalue,
                           "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())})
        elif len(d):
            pooled.append({"dataset": "POOLED", "lam": np.nan, "a": a, "b": b, "n_seeds": len(d), "diff": d.mean(),
                           "diff_sd": d.std(ddof=1) if len(d) > 1 else np.nan, "p": np.nan,
                           "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())})
    if not rows:
        return pd.DataFrame()
    return pd.concat([holm(pd.DataFrame(rows)), holm(pd.DataFrame(pooled))], ignore_index=True)


# --------------------------------------------------------------------------- frontier
def hypervolume(points):
    """Area dominated by (tokens, utility) points w.r.t. reference (TOKENS_REF, 0), tokens normalized."""
    pts = sorted((min(t, TOKENS_REF) / TOKENS_REF, u) for t, u in points)
    area, best = 0.0, 0.0
    # sweep from the cheapest point: each point dominates [t, 1] x [0, u]
    nd = []
    for t, u in pts:
        if u > best:
            nd.append((t, u))
            best = u
    for i, (t, u) in enumerate(nd):
        t_next = nd[i + 1][0] if i + 1 < len(nd) else 1.0
        u_level = u
        area += (t_next - t) * u_level
    return area


def window_auc(points, lo, hi):
    """Mean of the staircase frontier f(t) = max{u_i : t_i <= t} over the token window [lo, hi].
    Comparing methods on the SAME token window avoids rewarding a method only for covering a
    wider token range (a flaw of the plain hypervolume when the lambda grids differ)."""
    pts = sorted(points)
    grid = np.linspace(lo, hi, 401)
    f = np.array([max([u for t, u in pts if t <= g], default=0.0) for g in grid])
    return float(np.trapz(f, grid) / (hi - lo))


def frontier_table(all_runs):
    curves = {}
    for ds in DATASETS:
        runs = all_runs[ds]
        for variant in sorted(runs["variant"].unique()):
            x = runs[(runs["variant"] == variant) & runs["lam"].isin(LAMS_GRID)]
            x = x.sort_values("source", ascending=False).drop_duplicates(["lam", "seed"])
            for seed, g in x.groupby("seed"):
                if g["lam"].nunique() == len(LAMS_GRID):
                    curves[(ds, variant, seed)] = list(zip(g["tokens"], g["utility"]))
        b = pd.read_parquet(OLD[ds] / "baselines.parquet").groupby("condition")[["utility", "tokens"]].mean()
        for prefix in ["topk_ce", "topk_e5", "topk_minilm", "rrf"]:
            pts = b[b.index.str.match(rf"{prefix}_\d+$")]
            curves[(ds, f"static_{prefix}", -1)] = list(zip(pts["tokens"], pts["utility"]))
    rows = []
    for ds in DATASETS:
        keys = [k for k in curves if k[0] == ds]
        lo = max(min(t for t, _ in curves[k]) for k in keys)   # every method has reached this cost
        hi = min(max(t for t, _ in curves[k]) for k in keys)   # no method needs extrapolation
        for k in keys:
            rows.append({"dataset": ds, "variant": k[1], "seed": k[2], "window_lo": lo, "window_hi": hi,
                         "frontier_auc": window_auc(curves[k], lo, hi),
                         "hypervolume_full": hypervolume(curves[k])})
    hv = pd.DataFrame(rows)
    return hv.groupby(["dataset", "variant", "window_lo", "window_hi"]).agg(
        frontier_auc=("frontier_auc", "mean"), frontier_auc_sd=("frontier_auc", "std"),
        hypervolume_full=("hypervolume_full", "mean"), n_seeds=("seed", "count")).reset_index()


# --------------------------------------------------------------------------- multi-agent metrics
def shapley(sent_sets, useful, need):
    """Exact Shapley values of the agents for v(T) = min(1, |useful in U_{i in T} S_i| / need)."""
    agents = list(sent_sets)
    n = len(agents)

    def v(T):
        chunks = set().union(*(sent_sets[a] for a in T)) if T else set()
        return min(1.0, sum(useful.get(c, 0) for c in chunks) / max(1, need))

    phi = {}
    for a in agents:
        others = [b for b in agents if b != a]
        total = 0.0
        for k in range(n):
            for T in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                total += w * (v(set(T) | {a}) - v(set(T)))
        phi[a] = total
    return phi


def ma_metrics(ds, path, useful_map):
    d = pd.read_parquet(path)
    agents = [a for a in AGENTS + ["random"] if f"sends_{a}" in d]
    sends = d[[f"sends_{a}" for a in agents]].sum()
    share = sends / max(1, sends.sum())
    nz = share[share > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(len(agents)))
    out = {"messages_per_query": d["n_chunks"].mean(), "utility_per_100_tokens": 100 * d["utility"].mean() / max(1e-9, d["tokens"].mean()),
           "specialization_entropy": entropy, "redundancy": d["redundancy"].mean(skipna=True) if "redundancy" in d else np.nan,
           "first_useful_step": d["first_useful_step"].mean(skipna=True) if "first_useful_step" in d else np.nan,
           "stops_per_query": d["stops"].mean(), **{f"share_{a}": share[f"sends_{a}"] for a in agents}}
    # H3: credit faithfulness (only runs that stored per-agent sends / credits)
    if all(f"sent_{a}" in d for a in agents):
        cm, ci, sh = [], [], []
        for r in d.itertuples():
            umap = useful_map.get(r.interaction_id, {})
            sets = {a: set(getattr(r, f"sent_{a}")) for a in agents}
            phi = shapley(sets, umap, r.n_support)
            for a in agents:
                cm.append(getattr(r, f"credit_{a}"))
                ci.append(min(1.0, sum(umap.get(c, 0) for c in sets[a]) / max(1, r.n_support)))  # selfish credit
                sh.append(phi[a])
        cm, ci, sh = map(np.asarray, (cm, ci, sh))
        out.update({"credit_pearson_marginal": stats.pearsonr(cm, sh)[0], "credit_spearman_marginal": stats.spearmanr(cm, sh)[0],
                    "credit_pearson_selfish": stats.pearsonr(ci, sh)[0], "credit_spearman_selfish": stats.spearmanr(ci, sh)[0],
                    "credit_mae_marginal": np.abs(cm - sh).mean(), "credit_mae_selfish": np.abs(ci - sh).mean()})
    return out


# --------------------------------------------------------------------------- learning curves
def curve_stats(ds):
    rows = []
    for p in glob.glob(str(VAL / ds / "runs" / "curves" / "*.json")):
        tag = Path(p).stem
        curves = json.loads(Path(p).read_text())
        for fold, c in curves.items():
            if len(c) < 10:
                continue
            y = np.array([v for _, v in c])
            last, prev = y[-max(1, len(y) // 10):].mean(), y[-2 * max(1, len(y) // 10):-max(1, len(y) // 10)].mean()
            rows.append({"dataset": ds, "tag": tag, "fold": fold, "final_return": last,
                         "late_gain": last - prev, "rel_late_gain": (last - prev) / (abs(last) + 1e-9)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- H10 alignment
def alignment():
    rows = []
    srcs = {"crag": (RESULTS / "ma_chunk/eval/ma_round1_per_query.parquet",
                     [RESULTS / "ma_chunk/baselines.parquet"] + glob.glob(str(OLD["crag"] / "runs" / "*.parquet"))),
            "hotpotqa": (OLD["hotpotqa"] / "eval/endtask_per_query.parquet",
                         [OLD["hotpotqa"] / "baselines.parquet"] + glob.glob(str(OLD["hotpotqa"] / "runs" / "*.parquet"))),
            "musique": (OLD["musique"] / "eval/endtask_per_query.parquet",
                        [OLD["musique"] / "baselines.parquet"] + glob.glob(str(OLD["musique"] / "runs" / "*.parquet")))}
    for ds, (ev, sel) in srcs.items():
        r = pd.read_parquet(ev)
        r = r.drop(columns=[c for c in ("utility",) if c in r])
        s = pd.concat([pd.read_parquet(p, columns=["condition", "interaction_id", "utility"]) for p in sel], ignore_index=True)
        m = r.merge(s.drop_duplicates(["condition", "interaction_id"]), on=["condition", "interaction_id"], how="inner")
        y = (m["verdict"] == "correct").astype(float)
        hi, lo = y[m["utility"] >= 0.999], y[m["utility"] <= 0.001]
        rows.append({"dataset": ds, "n": len(m), "pointbiserial_r": stats.pearsonr(m["utility"], y)[0],
                     "p_correct_U1": hi.mean(), "p_correct_U0": lo.mean(), "gap": hi.mean() - lo.mean(),
                     "p_incorrect_U1": (m.loc[m["utility"] >= 0.999, "verdict"] == "incorrect").mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- H5 / H6a / H8 robustness
def robustness_tables(all_runs):
    frames = [pd.read_csv(p) for p in glob.glob(str(VAL / "robustness_*.csv")) if "robustness_drop_" not in p]
    if not frames:
        return {}
    r = pd.concat(frames, ignore_index=True)
    # out-of-fold mean per (dataset, lam, seed, test, level): average over folds weighted by #queries
    r["w"] = r["team_reward"] * r["n"]
    g = r.groupby(["dataset", "lam", "seed", "test", "level"]).agg(w=("w", "sum"), n=("n", "sum"),
                                                                    tokens=("tokens", "mean")).reset_index()
    g["team_reward"] = g["w"] / g["n"]
    base = g[(g["test"] == "noise") & (g["level"].astype(str).isin(["0.0", "0"]))][["dataset", "lam", "seed", "team_reward"]]
    base = base.rename(columns={"team_reward": "R_clean"})
    g = g.merge(base, on=["dataset", "lam", "seed"], how="left")
    g["delta"] = g["team_reward"] - g["R_clean"]

    # sanity: clean re-evaluation must equal the training-run test reward
    chk = []
    for ds in DATASETS:
        for lam in (0.3, 1.0):
            runs = pick(all_runs[ds], "ma_msg", lam)
            for _, row in base[(base.dataset == ds) & (np.isclose(base.lam, lam))].iterrows():
                if row.seed in runs.index:
                    chk.append(abs(row.R_clean - runs.loc[row.seed, "team_reward"]))
    out = {"robustness_sanity_max_abs_diff": pd.DataFrame({"max_abs_diff": [max(chk) if chk else np.nan]})}

    noise = g[g["test"] == "noise"].copy()
    noise["sigma"] = noise["level"].astype(float)
    nt = noise.groupby(["dataset", "lam", "sigma"]).agg(R=("team_reward", "mean"), R_sd=("team_reward", "std"),
                                                         delta=("delta", "mean")).reset_index()
    # reference: the no-message team (ma) at the same lambda, seeds 0-2
    nt["R_ma_no_messages"] = [pick(all_runs[d], "ma", l)["team_reward"].mean() for d, l in zip(nt["dataset"], nt["lam"])]
    out["H5_noise"] = nt
    fail = g[g["test"] == "failure"].groupby(["dataset", "lam", "level"]).agg(
        R=("team_reward", "mean"), delta=("delta", "mean"), delta_sd=("delta", "std")).reset_index()
    out["H6a_failure"] = fail.rename(columns={"level": "failed_agent"})
    tr = g[g["test"] == "transfer"].copy()
    tr["in_domain_R"] = [pick(all_runs[t], "ma_msg", l).loc[s, "team_reward"] if s in pick(all_runs[t], "ma_msg", l).index else np.nan
                         for t, l, s in zip(tr["level"], tr["lam"], tr["seed"])]
    tt = tr.groupby(["dataset", "level", "lam"]).agg(R_transfer=("team_reward", "mean"),
                                                     R_in_domain=("in_domain_R", "mean")).reset_index()
    tt["relative"] = tt["R_transfer"] / tt["R_in_domain"]
    out["H8_transfer"] = tt.rename(columns={"dataset": "source", "level": "target"})

    # Failure-aware training (agent dropout + presence mask) vs standard training, seeds 0-2.
    dframes = [pd.read_csv(p) for p in glob.glob(str(VAL / "robustness_drop_*.csv"))]
    if dframes:
        def per_condition(df):
            df = df.copy()
            df["w"] = df["team_reward"] * df["n"]
            x = df.groupby(["dataset", "lam", "seed", "test", "level"]).agg(w=("w", "sum"), n=("n", "sum")).reset_index()
            x["R"] = x["w"] / x["n"]
            clean = x[(x["test"] == "noise") & (x["level"].astype(str).isin(["0.0", "0"]))][["dataset", "lam", "seed", "R"]]
            fail = x[x["test"] == "failure"][["dataset", "lam", "seed", "level", "R"]].rename(columns={"level": "failed_agent"})
            return clean, fail
        c_std, f_std = per_condition(r[r["seed"].isin([0, 1, 2])])
        c_drop, f_drop = per_condition(pd.concat(dframes, ignore_index=True))
        f = f_std.merge(c_std.rename(columns={"R": "R_clean"}), on=["dataset", "lam", "seed"]) \
                 .assign(loss=lambda d: d["R_clean"] - d["R"])
        fd = f_drop.merge(c_drop.rename(columns={"R": "R_clean"}), on=["dataset", "lam", "seed"]) \
                   .assign(loss=lambda d: d["R_clean"] - d["R"])
        comp = f.groupby(["dataset", "lam", "failed_agent"]).agg(R_clean_std=("R_clean", "mean"), loss_std=("loss", "mean")).reset_index() \
            .merge(fd.groupby(["dataset", "lam", "failed_agent"]).agg(R_clean_drop=("R_clean", "mean"), loss_drop=("loss", "mean")).reset_index(),
                   on=["dataset", "lam", "failed_agent"])
        out["B18_failure_dropout"] = comp
        worst = comp.groupby(["dataset", "lam"]).agg(worst_loss_std=("loss_std", "max"), worst_loss_drop=("loss_drop", "max"),
                                                     R_clean_std=("R_clean_std", "first"), R_clean_drop=("R_clean_drop", "first")).reset_index()
        worst["worst_loss_reduction"] = 1 - worst["worst_loss_drop"] / worst["worst_loss_std"]
        worst["clean_cost"] = worst["R_clean_std"] - worst["R_clean_drop"]
        worst["criterion_a_(>=50%)"] = worst["worst_loss_reduction"] >= 0.5
        worst["criterion_b_(<=0.01)"] = worst["clean_cost"] <= 0.01
        out["B18_criteria"] = worst
    return out


# --------------------------------------------------------------------------- end task (H7 at the reader)
def endtask_comparisons():
    files = {"crag": [RESULTS / "ma_chunk/eval/ma_round1_per_query.parquet"],
             "hotpotqa": [OLD["hotpotqa"] / "eval/endtask_per_query.parquet"],
             "musique": [OLD["musique"] / "eval/endtask_per_query.parquet"]}
    rng = np.random.default_rng(0)
    summary, tests = [], []
    for ds in DATASETS:
        paths = files[ds] + [VAL / ds / "eval" / "endtask_validation_per_query.parquet"]
        r = pd.concat([pd.read_parquet(p, columns=["condition", "interaction_id", "verdict", "f1", "input_tokens"])
                       for p in paths if Path(p).exists()], ignore_index=True)
        r["score"] = r["verdict"].map({"correct": 1, "incorrect": -1}).fillna(0)
        r["group"] = r["condition"].str.replace(r"_s\d$", "", regex=True)
        per_q = r.groupby(["group", "interaction_id"])[["score", "f1", "input_tokens"]].mean()
        for g, x in per_q.groupby(level=0):
            seeds = r.loc[r["group"] == g, "condition"].nunique()
            summary.append({"dataset": ds, "group": g, "n_seeds": seeds, "score": x["score"].mean(),
                            "f1": x["f1"].mean(), "tokens": x["input_tokens"].mean()})
        m = per_q["score"].unstack(0)
        rows = []
        for lam in (0.3, 1.0):
            for a, b in [("ma_msg", "sup_gbm"), ("ma_msg", "sup_logreg"), ("ma_msg_rich", "sup_gbm"),
                         ("ma_msg_rich", "sup_logreg"), ("ma_msg_rich", "ma_msg")]:
                A, B = f"{a}_lam{lam}", f"{b}_lam{lam}"
                if A not in m or B not in m:
                    continue
                d = (m[A] - m[B]).dropna().to_numpy()
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
                nz = d[d != 0]
                rows.append({"dataset": ds, "lam": lam, "a": a, "b": b, "n": len(d), "diff": d.mean(),
                             "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                             "p": stats.wilcoxon(nz).pvalue if len(nz) > 10 else np.nan})
        if rows:
            tests.append(holm(pd.DataFrame(rows)))
    return pd.DataFrame(summary), (pd.concat(tests, ignore_index=True) if tests else pd.DataFrame())


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs = {ds: load_runs(ds) for ds in DATASETS}
    pd.concat(all_runs.values()).drop(columns="path").to_csv(OUT / "all_runs.csv", index=False, float_format="%.5f")

    # Reproducibility: validation re-runs (new code) vs original runs, same tag.
    rep = []
    for ds, runs in all_runs.items():
        for (v, l, s), g in runs.groupby(["variant", "lam", "seed"]):
            if set(g["source"]) == {"original", "validation"}:
                o, n = g[g.source == "original"].iloc[0], g[g.source == "validation"].iloc[0]
                rep.append({"dataset": ds, "variant": v, "lam": l, "seed": s, "utility_orig": o.utility,
                            "utility_new": n.utility, "abs_diff": abs(o.utility - n.utility), "tokens_diff": abs(o.tokens - n.tokens)})
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT / "reproducibility.csv", index=False, float_format="%.6f")

    fam = {
        "H1_reward": [("ma_msg", "ma_msg_rw-terminal"), ("ma_msg", "ma_msg_rw-individual")],
        "H4_orchestration": [("ma_msg", "ma_msg_to-random"), ("ma_msg", "ma_msg_to-confidence")],
        "H6b_random_agent": [("ma_msg", "ma_msg_rand")],
        "H7_why_rl": [("ma_msg", "sup_gbm_obsfeat"), ("ma_msg", "sup_gbm"), ("ma_msg", "sup_logreg"),
                      ("ma_msg_rich", "sup_gbm"), ("ma_msg_rich", "sup_logreg"), ("ma_msg_rich", "ma_msg")],
        "decomposition": [("ma_msg", "single"), ("ma_msg_rich", "single_rich"), ("ma_msg", "ma")],
        "H9_top_m": [("ma_msg", "ma_msg_m4"), ("ma_msg", "ma_msg_m12")],
    }
    for name, pairs in fam.items():
        lams = (1.0,) if name == "H9_top_m" else (0.3, 1.0)
        t = compare_family(all_runs, pairs, lams)
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.5f")

    # H2: price of anarchy and over-sending under selfish rewards.
    poa = []
    for ds in DATASETS:
        for lam in (0.3, 1.0):
            c, i = pick(all_runs[ds], "ma_msg", lam), pick(all_runs[ds], "ma_msg_rw-individual", lam)
            common = sorted(set(c.index) & set(i.index))
            if common:
                rc, rs = c.loc[common, "team_reward"].mean(), i.loc[common, "team_reward"].mean()
                poa.append({"dataset": ds, "lam": lam, "R_coop": rc, "R_selfish": rs,
                            "efficiency_loss": rc - rs,
                            # ratio only meaningful when the selfish team still has positive welfare
                            "price_of_anarchy": rc / rs if rs > 0 else np.inf,
                            "tokens_coop": c.loc[common, "tokens"].mean(), "tokens_selfish": i.loc[common, "tokens"].mean(),
                            "identical_policies": bool(np.allclose(c.loc[common, "team_reward"], i.loc[common, "team_reward"]))})
    pd.DataFrame(poa).to_csv(OUT / "H2_price_of_anarchy.csv", index=False, float_format="%.5f")

    frontier_table(all_runs).to_csv(OUT / "hypervolume.csv", index=False, float_format="%.5f")

    # Multi-agent metrics (+ H3 credit faithfulness) on the validation runs that store them.
    mam = []
    for ds in DATASETS:
        pool = load_pool(8, pool_path=POOLS[ds], with_random=True)
        useful_map = {q: dict(zip(g["chunk_id"], (g["useful"] == 1).astype(int))) for q, g in pool.groupby("interaction_id")}
        for p in sorted(glob.glob(str(VAL / ds / "runs" / "ma*.parquet"))):
            tag = Path(p).stem
            variant, lam, seed = parse_tag(tag)
            d = pd.read_parquet(p, columns=None)
            if "sent_ce" not in d:
                continue
            mam.append({"dataset": ds, "variant": variant, "lam": lam, "seed": seed, **ma_metrics(ds, p, useful_map)})
    mam = pd.DataFrame(mam)
    mam.to_csv(OUT / "ma_metrics.csv", index=False, float_format="%.5f")

    curves = pd.concat([curve_stats(ds) for ds in DATASETS], ignore_index=True)
    curves.to_csv(OUT / "learning_curves.csv", index=False, float_format="%.5f")
    alignment().to_csv(OUT / "H10_alignment.csv", index=False, float_format="%.5f")
    et_sum, et_tests = endtask_comparisons()
    et_sum.to_csv(OUT / "endtask_summary.csv", index=False, float_format="%.5f")
    et_tests.to_csv(OUT / "endtask_tests.csv", index=False, float_format="%.5f")
    rob = robustness_tables(all_runs)
    for name, df in rob.items():
        df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.5f")

    # ---- compact report
    pd.set_option("display.width", 220)
    print("== reproducibility (new code vs original runs):",
          f"{len(rep)} pairs, max |utility diff| = {rep['abs_diff'].max() if len(rep) else float('nan'):.2e}")
    for name in fam:
        t = pd.read_csv(OUT / f"{name}.csv") if (OUT / f"{name}.csv").stat().st_size > 1 else pd.DataFrame()
        if len(t):
            cols = [c for c in ["dataset", "lam", "a", "b", "n_seeds", "diff", "p", "p_holm", "n_positive", "n_negative"] if c in t]
            print(f"\n== {name}\n" + t[cols].round(4).to_string(index=False))
    print("\n== H2 price of anarchy\n" + pd.read_csv(OUT / "H2_price_of_anarchy.csv").round(3).to_string(index=False)
          if (OUT / "H2_price_of_anarchy.csv").stat().st_size > 1 else "")
    print("\n== hypervolume\n" + pd.read_csv(OUT / "hypervolume.csv").round(4).to_string(index=False))
    if len(mam):
        cols = [c for c in ["dataset", "variant", "lam", "specialization_entropy", "redundancy", "utility_per_100_tokens",
                            "first_useful_step", "stops_per_query", "credit_spearman_marginal", "credit_spearman_selfish",
                            "credit_mae_marginal", "credit_mae_selfish", "share_random"] if c in mam]
        print("\n== multi-agent metrics (mean over seeds)\n" +
              mam.groupby(["dataset", "variant", "lam"])[cols[3:]].mean().round(3).to_string())
    if len(curves):
        print("\n== learning curves: median relative gain in the last 10% of training",
              curves.groupby("dataset")["rel_late_gain"].median().round(4).to_dict())
    print("\n== H10 alignment\n" + pd.read_csv(OUT / "H10_alignment.csv").round(3).to_string(index=False))
    for name, df in rob.items():
        print(f"\n== {name}\n" + df.round(4).to_string(index=False))
    print("\n== end task (reader + judge): groups\n" + et_sum.round(3).to_string(index=False))
    print("\n== end task: paired tests\n" + et_tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

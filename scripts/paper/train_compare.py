#!/usr/bin/env python3
"""Training-proxy comparison (no LLM): utility U (original n_G), label AP (= RAGAS context-precision
formula with the utility labels), hit rate (some useful chunk sent), tokens and chunks, for v3 (R0),
R2 (confidence order), R3 (AP reward), R4 (AP + confidence), the beta ablation and R5.
All quantities are recomputed from the sent chunk ids and the labels, so v3 and v4 runs are
measured identically. Mean over seeds per (dataset, lambda); test folds only.

Output: results/v4/train/analysis/train_compare.csv
Usage (from the repo root):  python scripts/paper/train_compare.py
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
V3 = {"crag": RESULTS, "hotpotqa": RESULTS, "musique": RESULTS}
V4 = RESULTS / "v4/train"
POOLS = {"crag": None, "hotpotqa": DATA / "hotpotqa/candidates.parquet",
         "musique": DATA / "musique/candidates.parquet"}
CONFIGS = {  # name -> (directory kind, tag prefix)
    "R0 v3": ("v3", "ma_msg"),
    "R2 confidence": ("val", "ma_msg_to-confidence"),
    "R3 AP": ("v4", "ma_msg_rw-marginal_ap"),
    "R4 AP+conf": ("v4", "ma_msg_rw-marginal_ap_to-confidence"),
    "R4 beta=0.5": ("v4", "ma_msg_rw-marginal_ap_to-confidence_ap0.5"),
    "R4 beta=2": ("v4", "ma_msg_rw-marginal_ap_to-confidence_ap2"),
    "R5 AP+conf+sup2": ("v4", "ma_msg_rw-marginal_ap_to-confidence_sup2"),
}


def labels(ds):
    if ds == "crag":
        lab = pd.read_parquet(DATA / "crag/labels.parquet")[["interaction_id", "chunk_id", "useful"]]
        nsup = None
    else:
        lab = pd.read_parquet(POOLS[ds], columns=["interaction_id", "chunk_id", "useful", "n_support"])
        nsup = lab.drop_duplicates("interaction_id").set_index("interaction_id")["n_support"].to_dict()
    L = {(q, int(c)): int(u == 1) for q, c, u in zip(lab["interaction_id"], lab["chunk_id"], lab["useful"])}
    return L, nsup


def ap(v):
    s = sum(v)
    return 0.0 if s == 0 else sum(sum(v[:i + 1]) / (i + 1) * v[i] for i in range(len(v))) / s


def run_files(ds, kind, prefix, lam):
    if kind == "v3":
        base = RESULTS / "ma_chunk" / ("" if ds == "crag" else f"{ds}_n1000") / "runs"
        files = [base / f"{prefix}_lam{lam}_s0.parquet"]
        files += [Path(p) for p in glob.glob(str(RESULTS / f"validation/{ds}/runs/{prefix}_lam{lam}_s[12].parquet"))]
        if not files[0].exists():
            files = [Path(p) for p in glob.glob(str(base / f"{prefix}_lam{lam}_s[0-2].parquet"))]
        return [f for f in files if f.exists()]
    root = RESULTS / "validation" / ds / "runs" if kind == "val" else V4 / ds / "runs"
    return [Path(p) for p in glob.glob(str(root / f"{prefix}_lam{lam}_s[0-2].parquet"))]


def main():
    rows = []
    for ds in ["crag", "hotpotqa", "musique"]:
        L, nsup = labels(ds)
        for name, (kind, prefix) in CONFIGS.items():
            for lam in ("0.3", "1.0"):
                per_seed = []
                for f in run_files(ds, kind, prefix, lam):
                    if not re.search(rf"/{re.escape(prefix)}_lam{lam}_s\d\.parquet$", str(f)):
                        continue
                    d = pd.read_parquet(f, columns=["interaction_id", "chunk_ids", "tokens"])
                    v = [[L.get((q, int(c)), 0) for c in ids] for q, ids in zip(d["interaction_id"], d["chunk_ids"])]
                    ng = [1 if nsup is None else max(1, nsup.get(q, 1)) for q in d["interaction_id"]]
                    per_seed.append({
                        "U": np.mean([min(1.0, sum(x) / n) for x, n in zip(v, ng)]),
                        "AP": np.mean([ap(x) for x in v]),
                        "hit": np.mean([sum(x) > 0 for x in v]),
                        "first_useful": np.mean([bool(x) and x[0] == 1 for x in v]),
                        "chunks": np.mean([len(x) for x in v]), "tokens": d["tokens"].mean()})
                if per_seed:
                    m = pd.DataFrame(per_seed)
                    rows.append({"dataset": ds, "config": name, "lam": float(lam), "seeds": len(m),
                                 **m.mean().to_dict(), "AP_sd": m["AP"].std()})
    out = pd.DataFrame(rows)
    (V4 / "analysis").mkdir(parents=True, exist_ok=True)
    out.to_csv(V4 / "analysis" / "train_compare.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

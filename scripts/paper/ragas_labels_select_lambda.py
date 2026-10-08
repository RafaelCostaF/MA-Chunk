#!/usr/bin/env python3
"""RAGAS-aligned labels, budget matching (docs/PROTOCOL.md, section 9): for each dataset, variant and v3
lambda (0.3 and 1.0), choose among the variant's candidate lambdas the one whose mean number of
sent chunks on the test folds (training metric, no LLM) is closest to v3's, preferring candidates
within +-10%. Writes results/v4/a3/lambda_choice.json.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
TRAIN = RESULTS / "v4/train"
PAIRS = {"0.3": ["0.3", "0.4", "0.5"], "1.0": ["1.0", "1.25", "1.5"]}
VARIANTS = {"R6": "ma_msg_rw-marginal_ap_to-confidence_u-mix", "R7": "ma_msg_rw-marginal_ap_to-confidence_u-attr"}


def mean_chunks(files):
    return float(np.mean([pd.read_parquet(f, columns=["n_chunks"])["n_chunks"].mean() for f in files])) if files else np.nan


def v3_files(ds, lam):
    base = RESULTS / "ma_chunk" / ("" if ds == "crag" else f"{ds}_n1000") / "runs"
    fs = [str(base / f"ma_msg_lam{lam}_s0.parquet")] + glob.glob(str(RESULTS / f"validation/{ds}/runs/ma_msg_lam{lam}_s[12].parquet"))
    fs = [f for f in fs if Path(f).exists()]
    return fs if len(fs) == 3 else glob.glob(str(base / f"ma_msg_lam{lam}_s[0-2].parquet"))


def main():
    out = {}
    for ds in ["crag", "hotpotqa", "musique"]:
        for vname, prefix in VARIANTS.items():
            for ref, cands in PAIRS.items():
                target = mean_chunks(v3_files(ds, ref))
                stats = {l: mean_chunks(glob.glob(str(TRAIN / ds / "runs" / f"{prefix}_lam{l}_s[0-2].parquet"))) for l in cands}
                rel = {l: (c / target - 1) for l, c in stats.items() if np.isfinite(c)}
                within = {l: r for l, r in rel.items() if abs(r) <= 0.10}
                pool = within or rel
                choice = min(pool, key=lambda l: abs(pool[l]))
                out[f"{ds}|{vname}|{ref}"] = {"choice": choice, "v3_chunks": target, "chunks": stats,
                                              "rel_change": rel, "within_10pct": bool(within)}
                print(ds, vname, ref, "->", choice, {l: round(r, 3) for l, r in rel.items()}, "within" if within else "NONE within 10%")
    (RESULTS / "v4/a3").mkdir(parents=True, exist_ok=True)
    (RESULTS / "v4/a3/lambda_choice.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

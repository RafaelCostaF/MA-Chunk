#!/usr/bin/env python3
"""Reliability of the LLM judge and of the CRAG utility labels (inter-model agreement).

Both the end-task verdicts and the CRAG utility labels come from gpt-5-nano. Here a random
sample is re-judged / re-labelled by a second model (default gpt-4.1-nano) with exactly the same
prompts, and agreement is reported as raw agreement and Cohen's kappa (3 classes for verdicts,
2 for labels), plus the correlation of the per-item CRAG score (+1/-1/0).

Usage (from the repo root):
    OPENAI_MODEL=gpt-4.1-nano python scripts/paper/reliability.py --n 100 --out results/validation/analysis/reliability.json
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
from ma_chunk.reader import judge  # noqa: E402  (uses OPENAI_MODEL from the environment)
from ma_chunk.labels import label  # noqa: E402

EVALS = {"crag": RESULTS / "ma_chunk/eval/ma_round1_per_query.parquet",
         "hotpotqa": RESULTS / "ma_chunk/hotpotqa_n1000/eval/endtask_per_query.parquet",
         "musique": RESULTS / "ma_chunk/musique_n1000/eval/endtask_per_query.parquet"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="items per dataset (verdicts) / total labels x3")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    second = os.environ.get("OPENAI_MODEL", "?")
    rng = np.random.default_rng(0)
    report = {"second_model": second, "first_model": "gpt-5-nano", "verdicts": {}, "labels": {}}

    # ---- end-task verdicts
    all_rows = []
    for ds, path in EVALS.items():
        r = pd.read_parquet(path, columns=["query", "clean_answer", "response", "verdict"])
        r = r.drop_duplicates(["query", "response"])
        s = r.iloc[rng.choice(len(r), size=min(args.n, len(r)), replace=False)].copy()
        s["dataset"] = ds
        all_rows.append(s)
    sample = pd.concat(all_rows, ignore_index=True)
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        sample["verdict2"] = list(ex.map(lambda t: judge(*t), zip(sample["query"], sample["clean_answer"], sample["response"])))
    score = {"correct": 1, "incorrect": -1, "missing": 0}
    for ds, g in list(sample.groupby("dataset")) + [("all", sample)]:
        g = g[g["verdict2"].isin(score)]
        report["verdicts"][ds] = {
            "n": int(len(g)), "agreement": float((g["verdict"] == g["verdict2"]).mean()),
            "cohen_kappa": float(cohen_kappa_score(g["verdict"], g["verdict2"])),
            "crag_score_first": float(g["verdict"].map(score).mean()),
            "crag_score_second": float(g["verdict2"].map(score).mean()),
            "score_correlation": float(np.corrcoef(g["verdict"].map(score), g["verdict2"].map(score))[0, 1]),
        }

    # ---- CRAG utility labels (stratified: half useful, half not, per the first model)
    cand = pd.read_parquet(DATA / "crag/candidates.parquet", columns=["interaction_id", "chunk_id", "text"])
    lab = pd.read_parquet(DATA / "crag/labels.parquet")[["interaction_id", "chunk_id", "useful"]]
    gold = pd.read_parquet(DATA / "crag/questions.parquet")[["interaction_id", "query", "clean_answer"]] \
        .drop_duplicates("interaction_id")
    pool = cand.merge(lab, on=["interaction_id", "chunk_id"]).merge(gold, on="interaction_id")
    pool = pool[pool["useful"].isin([0, 1])]
    k = (3 * args.n) // 2
    pos = pool[pool.useful == 1].sample(n=min(k, (pool.useful == 1).sum()), random_state=0)
    neg = pool[pool.useful == 0].sample(n=min(k, (pool.useful == 0).sum()), random_state=0)
    ls = pd.concat([pos, neg], ignore_index=True)
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        ls["useful2"] = list(ex.map(label, zip(ls["query"], ls["clean_answer"], ls["text"])))
    ls = ls[ls["useful2"].isin([0, 1])]
    report["labels"] = {"n": int(len(ls)), "agreement": float((ls["useful"] == ls["useful2"]).mean()),
                        "cohen_kappa": float(cohen_kappa_score(ls["useful"], ls["useful2"])),
                        "second_positive_rate_on_first_positive": float(ls[ls.useful == 1]["useful2"].mean()),
                        "second_positive_rate_on_first_negative": float(ls[ls.useful == 0]["useful2"].mean()),
                        "note": "sample stratified 50/50 by the first model's label (base rate in the pool: 19.7% useful)"}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

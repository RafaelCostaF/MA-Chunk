#!/usr/bin/env python3
"""RAGAS evaluation of chunk selections that were already read by the common reader.

Joins, per (condition, interaction_id), the reader outputs (per-query parquets written by
ma_chunk.eval_reader: query, clean_answer, response, domain) with the contexts actually given to
the reader, in the order they were given (selection parquets: condition, interaction_id, chunks).
Then runs compute_ragas_metrics (ma_chunk/ragas_metrics.py: faithfulness, answer
relevancy, answer correctness, context precision, context recall; judge = OPENAI_MODEL,
gpt-5-nano, LLM/embedding calls cached) once per condition. Resumable: a condition whose
output exists is skipped.

Output: <out-dir>/<condition>.parquet with interaction_id, domain, condition, n_contexts,
response, the 5 RAGAS metrics.

Usage (from the repo root):
    python -m ma_chunk.ragas_eval --per-query results/crag/eval/ma_round1_per_query.parquet \
        --selections results/crag/baselines.parquet results/crag/runs/ma_msg_lam0.3_s0.parquet \
        --conditions ma_msg_lam0.3_s0 rrf_3 --out-dir results/ragas/crag
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ma_chunk.ragas_metrics import compute_ragas_metrics

METRICS = ["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"]


def load(per_query, selections, conditions, sample_ids=None):
    pq = pd.concat([pd.read_parquet(p) for p in per_query], ignore_index=True)
    keep = ["condition", "interaction_id", "query", "clean_answer", "response", "domain"]
    pq = pq[[c for c in keep if c in pq]].drop_duplicates(["condition", "interaction_id"])
    sel = pd.concat([pd.read_parquet(p, columns=["condition", "interaction_id", "chunks"]) for p in selections],
                    ignore_index=True).drop_duplicates(["condition", "interaction_id"])
    df = pq.merge(sel, on=["condition", "interaction_id"], how="left")
    if conditions:
        df = df[df["condition"].isin(conditions)]
    if sample_ids is not None:
        df = df[df["interaction_id"].isin(sample_ids)]
    missing = df["chunks"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} rows without contexts (selection parquet missing for some condition)")
    df["chunks"] = df["chunks"].apply(lambda x: [str(c) for c in x])
    df["response"] = df["response"].fillna("").astype(str)
    if "domain" not in df:
        df["domain"] = "-"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", nargs="+", required=True)
    ap.add_argument("--selections", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample", default=None, help="text file with one interaction_id per line (multi-hop sample)")
    args = ap.parse_args()

    sample = None
    if args.sample:
        sample = set(Path(args.sample).read_text().split())
    df = load(args.per_query, args.selections, args.conditions, sample)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cond, g in df.groupby("condition", sort=False):
        path = out / f"{cond}.parquet"
        if path.exists():
            print(f"skip {cond} (exists)")
            continue
        g = g.reset_index(drop=True)
        t0 = time.time()
        r = compute_ragas_metrics(g.rename(columns={"response": "llm_response", "chunks": "chunks_selected"})
                                  .assign(algo=cond))
        res = g[["interaction_id", "domain", "condition", "response"]].copy()
        res["n_contexts"] = g["chunks"].apply(len)
        for m in METRICS:
            res[m] = r[m].to_numpy()
        res.to_parquet(path)
        print(f"{cond}: n={len(res)} {time.time() - t0:.0f}s " +
              " ".join(f"{m}={np.nanmean(res[m]):.3f}" for m in METRICS), flush=True)


if __name__ == "__main__":
    main()

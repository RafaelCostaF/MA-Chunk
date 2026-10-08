"""End-task evaluation of chunk selections with the frozen reader + CRAG-style judge.

Uses the common reader prompt, judge, refusal rule and metrics
(ma_chunk/reader.py), so the numbers are directly comparable with it.

Input: one or more selection parquets with columns condition, interaction_id, chunks
(train_ma.py runs, baselines.py). Output: per-query results and a summary.

Usage (from the repo root):
    python -m ma_chunk.eval_reader \
        --selections results/crag/baselines.parquet results/crag/runs/ma_lam1.0_s0.parquet \
        --conditions rrf_3 topk_minilm_5 ma_lam1.0_s0 --out-dir results/crag/eval
"""

import argparse
import concurrent.futures as cf
import sys
from pathlib import Path

import pandas as pd

from ma_chunk.config import DATA, OPENAI_MODEL
from ma_chunk.reader import normalize, run_one, summarize, token_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="*", default=None, help="subset of conditions to evaluate")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="eval")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pool", default=None,
                        help="multi-hop pool parquet: questions/answers come from it (default: CRAG)")
    args = parser.parse_args()

    sel = pd.concat([pd.read_parquet(p) for p in args.selections], ignore_index=True)
    if args.conditions:
        sel = sel[sel["condition"].isin(args.conditions)]
    if args.pool:
        meta = pd.read_parquet(args.pool, columns=["interaction_id", "domain", "question_type", "query", "answers"])
        meta = meta.drop_duplicates("interaction_id")
        meta["answers"] = meta["answers"].apply(list)
        meta["clean_answer"] = meta["answers"].apply(lambda a: " / ".join(a))  # judge sees all accepted answers
    else:
        meta = pd.read_parquet(DATA / "crag" / "questions.parquet")[
            ["interaction_id", "domain", "question_type", "query", "clean_answer"]].drop_duplicates("interaction_id")
        meta["answers"] = meta["clean_answer"].apply(lambda a: [a])
    sel = sel.drop(columns=[c for c in ("domain", "question_type") if c in sel]).merge(meta, on="interaction_id")
    sel["chunks"] = sel["chunks"].apply(list)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        outputs = list(ex.map(run_one, sel.to_dict("records")))
    res = pd.concat([sel.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
    res["n_chunks"] = res["chunks"].apply(len)
    # Standard QA convention: best score over the accepted answers.
    res["f1"] = [max(token_f1(r, g) for g in gs) for r, gs in zip(res["response"], res["answers"])]
    res["em"] = [max(float(normalize(r) == normalize(g)) for g in gs) for r, gs in zip(res["response"], res["answers"])]
    res["model"] = OPENAI_MODEL

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.drop(columns=["chunks"]).to_parquet(out / f"{args.name}_per_query.parquet")
    summary = summarize(res)
    summary.to_csv(out / f"{args.name}_summary.csv", float_format="%.4f")
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()

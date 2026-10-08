"""Static selection baselines over the same candidate pool used by MA-Chunk.

    topk_<agent>_<k>  - the k best chunks of one retriever (bm25 / spacy / minilm);
    rrf_<k>           - the k best chunks of the reciprocal-rank fusion of the three;
    oracle            - upper bound: the best-ranked (RRF) chunk labelled useful, or the
                        RRF top-1 when none is useful (uses the labels -> NOT a real method).

Usage (from the repo root):
    python -m ma_chunk.baselines --out results/crag/baselines.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from ma_chunk.env_ma import AGENTS  # noqa: E402
from ma_chunk.train_ma import load_pool  # noqa: E402

KS = [1, 2, 3, 4, 5, 6, 8, 10]


def selection(group: pd.DataFrame, order_col: str, k: int, condition: str) -> dict:
    top = group.sort_values(order_col).head(k)
    return {"condition": condition, "interaction_id": group["interaction_id"].iloc[0],
            "chunks": top["text"].tolist(), "chunk_ids": top["chunk_id"].tolist(),
            "utility": coverage(top), "tokens": int(top["tokens"].sum()), "n_chunks": len(top)}


def coverage(selected: pd.DataFrame) -> float:
    need = int(selected["n_support"].iloc[0]) if "n_support" in selected and len(selected) else 1
    return min(1.0, selected["useful"].eq(1).sum() / max(1, need))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", default=None, help="multi-hop pool parquet (default: CRAG pool)")
    args = parser.parse_args()

    pool = load_pool(args.top_m, pool_path=args.pool)
    rows = []
    for _, g in pool.groupby("interaction_id", sort=False):
        for k in KS:
            for a in AGENTS:
                rows.append(selection(g, f"rank_{a}", k, f"topk_{a}_{k}"))
            rows.append(selection(g, "rank_rrf", k, f"rrf_{k}"))
        useful = g[g["useful"] == 1]
        need = int(g["n_support"].iloc[0]) if "n_support" in g else 1
        rows.append(selection(useful if len(useful) else g, "rank_rrf", max(1, need), "oracle"))
    df = pd.DataFrame(rows)
    df.to_parquet(args.out)
    print(df.groupby("condition")[["utility", "tokens", "n_chunks"]].mean().round(3).sort_values("utility").to_string())


if __name__ == "__main__":
    main()

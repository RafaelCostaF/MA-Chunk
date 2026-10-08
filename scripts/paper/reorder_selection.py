#!/usr/bin/env python3
"""R1 (no training): same chunks as a v3 selection, reordered before reading.

Pre-specified rule (docs/PROTOCOL.md, R1): each chunk is ranked by the highest per-query
min-max-normalized score it receives from any of the five retriever agents (computed over the
query's candidate pool); ties keep the original channel order. Only the order changes: the set of
chunks, and therefore tokens and utility, are identical to the input selection.

Output: a selection parquet (condition, interaction_id, chunks, chunk_ids, utility, tokens, n_chunks)
with conditions renamed <condition>_reord, readable by eval_reader.py and ragas_eval.py.

Usage (from the repo root):
    python scripts/paper/reorder_selection.py --selections results/ma_chunk/runs/ma_msg_lam0.3_s0.parquet \
        --out results/v4/r1/crag_reord.parquet [--pool <multi-hop candidates.parquet>]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
AGENTS = ["bm25", "spacy", "minilm", "e5", "ce"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default=str(DATA / "crag/candidates.parquet"))
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool, columns=["interaction_id", "chunk_id"] + [f"s_{a}" for a in AGENTS])
    for a in AGENTS:
        g = pool.groupby("interaction_id")[f"s_{a}"]
        lo, hi = g.transform("min"), g.transform("max")
        pool[f"n_{a}"] = (pool[f"s_{a}"] - lo) / (hi - lo + 1e-9)
    pool["key"] = pool[[f"n_{a}" for a in AGENTS]].max(axis=1)
    key = {(q, int(c)): k for q, c, k in zip(pool["interaction_id"], pool["chunk_id"], pool["key"])}

    sel = pd.concat([pd.read_parquet(p) for p in args.selections], ignore_index=True)
    rows = []
    for r in sel.itertuples():
        ids, texts = [int(c) for c in r.chunk_ids], list(r.chunks)
        order = sorted(range(len(ids)), key=lambda j: (-key[(r.interaction_id, ids[j])], j))
        rows.append({"condition": f"{r.condition}_reord", "interaction_id": r.interaction_id,
                     "chunks": [texts[j] for j in order], "chunk_ids": [ids[j] for j in order],
                     "utility": r.utility, "tokens": r.tokens, "n_chunks": len(ids)})
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    changed = np.mean([list(a) != list(b) for a, b in zip(sel["chunk_ids"], out["chunk_ids"])])
    print(f"wrote {len(out)} rows, {out.condition.nunique()} conditions; order changed in {changed:.1%} of the queries")


if __name__ == "__main__":
    main()

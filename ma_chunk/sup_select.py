"""H7 baseline: supervised selector with the same features, labels, folds and objective as MA-Chunk.

A classifier predicts P(useful | chunk features) from the union of the agents' top-M lists;
at test time a decision-theoretic greedy rule sends chunks by decreasing probability while the
expected marginal utility exceeds the token cost:
    E[U] after adding c = min(1, (E_covered + p_c) / need),   need = mean n_support of the train fold,
    send c  iff  E[U | +c] - E[U] >= lam * tokens(c) / 1000  (and budgets allow).
This is the strongest non-RL answer to "why reinforcement learning?": it optimizes the same
utility-minus-cost objective, but myopically and without coordination.

Features (per chunk): the five agents' scores, their log-ranks, per-query z-scores of the five
scores, agreement (#agents with the chunk in their top-M) and tokens.

Usage (from the repo root):
    python -m ma_chunk.sup_select --model logreg \
        --pool data/hotpotqa/candidates.parquet --out-dir results/validation/hotpotqa/runs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ma_chunk.env_ma import AGENTS  # noqa: E402
from ma_chunk.train_ma import load_pool  # noqa: E402

LAMS = [0.0, 0.1, 0.3, 0.6, 1.0, 2.0]


def featurize(pool: pd.DataFrame, top_m: int) -> pd.DataFrame:
    pool = pool[(pool[[f"rank_{a}" for a in AGENTS]] < top_m).any(axis=1)].copy()  # same lists the agents walk
    pool["agree"] = (pool[[f"rank_{a}" for a in AGENTS]] < top_m).sum(axis=1)
    for a in AGENTS:
        pool[f"lr_{a}"] = np.log1p(pool[f"rank_{a}"])
        pool[f"z_{a}"] = pool.groupby("interaction_id")[f"s_{a}"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
    feats = [f"s_{a}" for a in AGENTS] + [f"lr_{a}" for a in AGENTS] + [f"z_{a}" for a in AGENTS]
    pool[feats] = pool[feats].fillna(0.0)  # spaCy similarity is undefined for empty paragraphs
    return pool


def feature_cols(kind="full"):
    """full: everything; obs: feature parity with an MA+msg agent (the five scores it receives as
    messages, cross-agent agreement and the chunk's tokens - no per-query normalization or ranks)."""
    if kind == "obs":
        return [f"s_{a}" for a in AGENTS] + ["agree", "tokens"]
    return ([f"s_{a}" for a in AGENTS] + [f"lr_{a}" for a in AGENTS] + [f"z_{a}" for a in AGENTS]
            + ["agree", "tokens"])


def greedy(g: pd.DataFrame, p: np.ndarray, lam: float, need: float, max_chunks=10, budget=2000):
    order = np.argsort(-p)
    expected, sent, tokens = 0.0, [], 0
    for i in order:
        if len(sent) >= max_chunks or tokens >= budget:
            break
        gain = min(1.0, (expected + p[i]) / need) - min(1.0, expected / need)
        if gain < lam * g["tokens"].iloc[i] / 1000.0:
            continue  # a later, cheaper chunk may still be worth it
        sent.append(i)
        expected += p[i]
        tokens += int(g["tokens"].iloc[i])
    return sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=None)
    parser.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--features", choices=["full", "obs"], default="full")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = featurize(load_pool(args.top_m, pool_path=args.pool), args.top_m)
    if "n_support" not in pool:
        pool["n_support"] = 1
    meta = pool.drop_duplicates("interaction_id")[["interaction_id", "domain"]].reset_index(drop=True)
    X_cols = feature_cols(args.features)
    name = f"sup_{args.model}" + ("_obsfeat" if args.features == "obs" else "")

    for seed in args.seeds:
        rows = {lam: [] for lam in LAMS}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # same folds as train_ma.py
        for fold, (tr, te) in enumerate(skf.split(meta, meta["domain"])):
            tr_ids, te_ids = set(meta.loc[tr, "interaction_id"]), set(meta.loc[te, "interaction_id"])
            train = pool[pool["interaction_id"].isin(tr_ids)]
            model = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
                     if args.model == "logreg" else HistGradientBoostingClassifier(random_state=seed))
            model.fit(train[X_cols], (train["useful"] == 1).astype(int))
            need = float(train.drop_duplicates("interaction_id")["n_support"].mean())
            for qid, g in pool[pool["interaction_id"].isin(te_ids)].groupby("interaction_id", sort=False):
                g = g.reset_index(drop=True)
                p = model.predict_proba(g[X_cols])[:, 1]
                for lam in LAMS:
                    sent = greedy(g, p, lam, need)
                    n_sup = int(g["n_support"].iloc[0])
                    util = min(1.0, int(g["useful"].iloc[sent].eq(1).sum()) / max(1, n_sup)) if sent else 0.0
                    tok = int(g["tokens"].iloc[sent].sum()) if sent else 0
                    rows[lam].append({"condition": f"{name}_lam{lam}_s{seed}", "fold": fold,
                                      "interaction_id": qid, "chunks": g["text"].iloc[sent].tolist(),
                                      "chunk_ids": g["chunk_id"].iloc[sent].astype(int).tolist(),
                                      "utility": util, "tokens": tok, "team_reward": util - lam * tok / 1000.0,
                                      "n_chunks": len(sent)})
        for lam in LAMS:
            df = pd.DataFrame(rows[lam])
            tag = f"{name}_lam{lam}_s{seed}"
            df.to_parquet(out / f"{tag}.parquet")
            print(json.dumps({"tag": tag, "lam": lam, "seed": seed, "variant": name,
                              "team_reward": round(df["team_reward"].mean(), 4), "utility": round(df["utility"].mean(), 4),
                              "tokens": round(df["tokens"].mean(), 1), "n_chunks": round(df["n_chunks"].mean(), 3)}))


if __name__ == "__main__":
    main()

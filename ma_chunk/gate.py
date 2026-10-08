"""Metareasoning gate agent: learns, from the end-task reward only, whether to ANSWER or ABSTAIN.

A one-step decision per query (contextual bandit) trained with PPO. Reward = CRAG score of
the outcome: +1 correct, -1 incorrect, 0 missing if it answers; 0 if it abstains.
Observation = query-level evidence features available before the reader is called:
    best score of each retriever agent over the whole candidate pool;
    best score of each agent over the chunks actually sent (0 if none);
    number of chunks sent / 10 and tokens sent / 1000;
    agreement: largest number of agents ranking the same chunk in their top-3, / n_agents.
Evaluation is out-of-fold (5-fold CV stratified by domain, several seeds): the gate never
sees the reader outcome of a test query.

Usage (from the repo root):
    python -m ma_chunk.gate --eval results/crag/eval/ma_round1_per_query.parquet \
        --out results/crag/eval/gate_round1.csv
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from sklearn.model_selection import StratifiedKFold
from stable_baselines3 import PPO

from ma_chunk.env_ma import AGENTS  # noqa: E402
from ma_chunk.train_ma import load_pool  # noqa: E402



def threshold_gated_scores(x: pd.DataFrame, seeds=range(5)) -> np.ndarray:
    """Baseline gate: per-question CRAG score after a one-parameter abstention threshold on the best
    cross-encoder score (column qmax), chosen by 5-fold CV on the training folds; mean over seeds."""
    out = np.zeros((len(seeds), len(x)))
    for k, seed in enumerate(seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(x, x["domain"]):
            grid = np.quantile(x["qmax"].iloc[tr], np.linspace(0, 0.8, 41))
            best = max(grid, key=lambda t: np.where(x["qmax"].iloc[tr] < t, 0, x["score"].iloc[tr]).mean())
            out[k, te] = np.where(x["qmax"].iloc[te] < best, 0, x["score"].iloc[te])
    return out.mean(axis=0)

ANSWER, ABSTAIN = 0, 1


def features(pool: pd.DataFrame, results: pd.DataFrame) -> np.ndarray:
    by_q = {qid: g for qid, g in pool.groupby("interaction_id")}
    rows = []
    for r in results.itertuples():
        g = by_q.get(r.interaction_id)
        if g is None:
            rows.append(np.zeros(2 * len(AGENTS) + 3, dtype=np.float32))
            continue
        ids = set(getattr(r, "chunk_ids", []) if isinstance(getattr(r, "chunk_ids", None), (list, np.ndarray)) else [])
        sent = g[g["chunk_id"].isin(ids)]
        top3 = sum((g[f"rank_{a}"] < 3).astype(int) for a in AGENTS)
        rows.append(np.array(
            [g[f"s_{a}"].max() for a in AGENTS]
            + [sent[f"s_{a}"].max() if len(sent) else 0.0 for a in AGENTS]
            + [len(ids) / 10.0, float(r.input_tokens) / 1000.0, top3.max() / len(AGENTS)],
            dtype=np.float32))
    return np.nan_to_num(np.stack(rows))


class GateEnv(gym.Env):
    """One-step episodes over the training queries of a fold."""

    def __init__(self, X, score, seed=0):
        super().__init__()
        self.X, self.score = X, score
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-5, 5, shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = int(self.rng.integers(len(self.X)))
        return self.X[self.i], {}

    def step(self, action):
        reward = float(self.score[self.i]) if int(action) == ANSWER else 0.0
        return self.X[self.i], reward, True, False, {}


def gated_out_of_fold(X, score, domain, seeds=(0, 1, 2), timesteps=30_000):
    out = np.zeros((len(seeds), len(score)))
    abstain = np.zeros_like(out)
    for k, seed in enumerate(seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, domain):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6  # standardize with training statistics only
            env = GateEnv((X[tr] - mu) / sd, score[tr], seed=seed)
            model = PPO("MlpPolicy", env, seed=seed, n_steps=512, batch_size=128, gamma=0.0,
                        learning_rate=3e-4, ent_coef=0.0, policy_kwargs={"net_arch": [32]}, verbose=0, device="cpu")
            model.learn(total_timesteps=timesteps)
            act, _ = model.predict((X[te] - mu) / sd, deterministic=True)
            out[k, te] = np.where(act == ANSWER, score[te], 0.0)
            abstain[k, te] = (act == ABSTAIN)
    return out.mean(0), abstain.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", nargs="+", required=True, help="per-query eval parquets (written by ma_chunk.eval_reader)")
    parser.add_argument("--pool", default=None, help="multi-hop pool (default: CRAG pool)")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)

    pool = load_pool(8, pool_path=args.pool)
    res = pd.concat([pd.read_parquet(p) for p in args.eval], ignore_index=True)
    if args.conditions:
        res = res[res["condition"].isin(args.conditions)]
    rows = []
    for cond, x in res.groupby("condition"):
        x = x.reset_index(drop=True)
        score = x["verdict"].map({"correct": 1, "incorrect": -1}).fillna(0).to_numpy()
        gated, abstain_rate = gated_out_of_fold(features(pool, x), score, x["domain"].to_numpy())
        # Reference: one-parameter threshold policy on the query's best cross-encoder score,
        # also optimized for the same reward on the training folds only.
        thr = threshold_gated_scores(x.assign(score=score, qmax=x["interaction_id"].map(
            pool.groupby("interaction_id")["s_ce"].max())))
        rows.append({"condition": cond, "n": len(x), "score": score.mean(), "gated_score": gated.mean(),
                     "threshold_gated_score": thr.mean(), "abstain_rate": abstain_rate,
                     "input_tokens": x["input_tokens"].mean()})
        print(rows[-1])
    out = pd.DataFrame(rows).sort_values("gated_score", ascending=False)
    out.to_csv(args.out, index=False, float_format="%.4f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

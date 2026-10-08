"""Test-time robustness of trained MA-Chunk teams (no retraining): H5, H6a and H8 of
docs/PROTOCOL.md.

For every saved fold model (train_ma.py --save-models) it evaluates the out-of-fold queries under:
    H5  message noise:   Gaussian noise (sigma) on the other agents' scores in the observation;
    H6a agent failure:   one agent disabled at test time (never acts, sends no messages);
    H8  transfer:        the fold models of a source dataset applied to every query of another dataset.
Outputs one row per (dataset, lambda, seed, fold, condition) with the team reward, utility, tokens.

Usage (from the repo root):
    python -m ma_chunk.robustness --dataset hotpotqa --out results/validation/robustness_hotpotqa.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import PPO

from ma_chunk.env_ma import AGENTS  # noqa: E402
from ma_chunk.train_ma import load_pool, make_queries, rollout  # noqa: E402

from ma_chunk.config import DATA, RESULTS  # noqa: E402

VAL = RESULTS / "validation"
POOLS = {"crag": None,
         "hotpotqa": DATA / "hotpotqa" / "candidates.parquet",
         "musique": DATA / "musique" / "candidates.parquet"}
NOISE = [0.0, 0.05, 0.1, 0.2, 0.5]


def summarize(rows):
    df = pd.DataFrame(rows)
    return {"team_reward": df["team_reward"].mean(), "utility": df["utility"].mean(),
            "tokens": df["tokens"].mean(), "n": len(df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(POOLS), required=True)
    parser.add_argument("--variant", default="ma_msg")
    parser.add_argument("--lams", type=float, nargs="+", default=[0.3, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", required=True)
    parser.add_argument("--suffix", default="", help="tag suffix of the trained runs, e.g. _drop0.2 (failure-aware training)")
    parser.add_argument("--tests", nargs="+", default=["noise", "failure", "transfer"])
    parser.add_argument("--runs-root", default=None,
                        help="root with <dataset>/runs (default: results/validation)")
    parser.add_argument("--tag-prefix", default=None,
                        help="full tag before '_lam', e.g. ma_msg_drop0.2 (overrides --variant/--suffix)")
    args = parser.parse_args()
    root = Path(args.runs_root) if args.runs_root else VAL
    prefix = args.tag_prefix or f"{args.variant}{args.suffix}"
    torch.set_num_threads(1)

    ds = args.dataset
    pools = {d: load_pool(8, pool_path=p) for d, p in POOLS.items()}
    all_q = {d: {q.interaction_id: q for q in make_queries(pl, pl["interaction_id"].unique(), 8, AGENTS)}
             for d, pl in pools.items()}
    share = prefix.startswith("ma_msg")
    out = []
    for lam in args.lams:
        for seed in args.seeds:
            tag = f"{prefix}_lam{lam}_s{seed}"
            runs = pd.read_parquet(root / ds / "runs" / f"{tag}.parquet", columns=["interaction_id", "fold"])
            for fold, ids in runs.groupby("fold")["interaction_id"]:
                path = root / ds / "runs" / "models" / f"{tag}_f{fold}.zip"
                if not path.exists():
                    continue
                model = PPO.load(path, device="cpu")
                test_q = [all_q[ds][i] for i in ids]
                base = {"lam": lam, "token_budget": 2000, "max_chunks": 10, "share_scores": share, "seed": seed,
                        "presence_mask": "_drop" in prefix, "rich_obs": "_rich" in prefix}
                key = {"dataset": ds, "lam": lam, "seed": seed, "fold": fold}
                for sigma in ((NOISE if share else [0.0]) if "noise" in args.tests else [0.0]):
                    out.append({**key, "test": "noise", "level": sigma,
                                **summarize(rollout(model, test_q, {**base, "msg_noise": sigma}))})
                for agent in (AGENTS if "failure" in args.tests else []):
                    out.append({**key, "test": "failure", "level": agent,
                                **summarize(rollout(model, test_q, {**base, "disabled_agents": (agent,)}))})
                for target in (POOLS if "transfer" in args.tests else []):
                    if target != ds:
                        out.append({**key, "test": "transfer", "level": target,
                                    **summarize(rollout(model, list(all_q[target].values()), base))})
            print(f"{ds} {tag} done", flush=True)
    pd.DataFrame(out).to_csv(args.out, index=False, float_format="%.5f")


if __name__ == "__main__":
    main()

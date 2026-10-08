"""Trains MA-Chunk with parameter-shared PPO under K-fold cross-validation and writes the
test-fold selections (one row per (condition, interaction_id) with the chunks sent).

Variants:
    ma      - 5 retriever agents (bm25, spacy, minilm, e5, ce), shared policy (agent id in obs),
              each agent sees only its own retriever score;
    ma_msg  - same, plus explicit messages: each agent receives the other agents' scores
              for the chunk it is considering;
    single  - ablation: ONE agent over the reciprocal-rank-fused list of the retrievers
              (same features, same reward, no multi-agent decomposition).

Validation options (docs/PROTOCOL.md); defaults reproduce the main method:
    --reward-mode {marginal,terminal,individual}   (H1, H2)
    --turn-order {round_robin,random,confidence}   (H4)
    --with-random-agent                            (H6: sixth agent with random scores)
    --save-models                                  (needed by robustness.py: H5, H6, H8)
    --algo {ppo,a2c,dqn,recurrent_ppo}             (per-algorithm comparison; PPO is the validated default)
    --rich-obs                                     (per-query z-scores and log-ranks in the observation)
    --agent-dropout 0.2                            (failure-aware training, adds a presence mask)
    --pool data/<dataset>/candidates.parquet       (multi-hop pools; default: CRAG)
Precision-aware and RAGAS-aligned extensions (MA-Chunk-P):
    --reward-mode marginal_ap --ap-weight 1        (adds the RAGAS context-precision term to the reward)
    --utility {useful,attr,mix} --attr-labels F    (utility from usefulness labels, RAGAS attribution labels, or both)
    --support-floor K                              (utility saturates after K useful chunks)
Every run also stores learning curves and per-query multi-agent metrics.

Usage (from the repo root):
    python -m ma_chunk.train_ma --variant ma --lams 0.05 0.1 0.2 \
        --seeds 0 1 2 --out-dir results/crag/runs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, DQN, PPO

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from ma_chunk.env_ma import AGENTS, AGENTS_WITH_RANDOM, SEND, STOP, MAChunkEnv, QueryData  # noqa: E402

from ma_chunk.config import DATA  # noqa: E402

CRAG = DATA / "crag"  # default pool (candidates + usefulness labels); multi-hop pools via --pool


def load_pool(top_m: int, rrf_k: int = 60, pool_path=None, with_random: bool = False, attr_path=None) -> pd.DataFrame:
    """CRAG pool (candidates + LLM labels) by default; a multi-hop pool (multihop_data.py,
    gold labels and n_support included) when pool_path is given."""
    if pool_path:
        pool = pd.read_parquet(pool_path)
    else:
        pool = pd.read_parquet(CRAG / "candidates.parquet")
        labels = pd.read_parquet(CRAG / "labels.parquet")[["interaction_id", "chunk_id", "useful"]]
        pool = pool.merge(labels, on=["interaction_id", "chunk_id"], how="left")
    pool["useful"] = pool["useful"].fillna(0).clip(lower=0).astype(int)
    if attr_path:  # RAGAS-aligned attribution labels (ma_chunk.attr_labels)
        attr = pd.read_parquet(attr_path, columns=["interaction_id", "chunk_id", "attr_mask", "n_stmt"])
        pool = pool.merge(attr, on=["interaction_id", "chunk_id"], how="left")
        pool["attr_mask"] = pool["attr_mask"].fillna(0).astype(np.int64)
        pool["n_stmt"] = pool.groupby("interaction_id")["n_stmt"].transform("max").fillna(1).astype(int)
    # Fused single-agent view (reciprocal rank fusion of the three retrievers).
    rrf = sum(1.0 / (rrf_k + pool[f"rank_{a}"]) for a in AGENTS)
    pool["s_rrf"] = rrf / (len(AGENTS) / (rrf_k + 0.0))
    pool["rank_rrf"] = pool.groupby("interaction_id")["s_rrf"].rank(ascending=False, method="first").astype(int) - 1
    if with_random:  # H6: an agent whose scores are pure noise (fixed seed, independent of training seeds)
        pool["s_random"] = np.random.default_rng(20261002).random(len(pool)).astype(np.float32)
        pool["rank_random"] = pool.groupby("interaction_id")["s_random"].rank(
            ascending=False, method="first").astype(int) - 1
    return pool


def make_queries(pool: pd.DataFrame, ids, top_m: int, agents, universe=AGENTS) -> list:
    sub = pool[pool["interaction_id"].isin(set(ids))]
    return [QueryData(g, top_m if agents != ["rrf"] else top_m * len(AGENTS), agents, universe)
            for _, g in sub.groupby("interaction_id", sort=False)]


class CurveCallback(BaseCallback):
    """Learning curve: mean training-episode return over the last 100 episodes, sampled about
    every 2048 environment steps (works for on-policy and off-policy algorithms)."""

    def __init__(self, every: int = 2048):
        super().__init__()
        self.curve = []
        self.every = every
        self._last = 0

    def _record(self):
        buf = self.model.ep_info_buffer
        if buf and self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self.curve.append((int(self.num_timesteps), float(np.mean([e["r"] for e in buf]))))

    def _on_rollout_end(self):
        self._record()

    def _on_step(self):
        return True


ALGOS = ("ppo", "a2c", "dqn", "recurrent_ppo")


def make_model(algo: str, env, seed: int):
    """Same network size (two 64-unit layers) and gamma = 1 for every algorithm; the remaining
    hyperparameters are each library's defaults for discrete actions (documented in
    docs/ENVIRONMENT.md)."""
    common = {"seed": seed, "verbose": 0, "device": "cpu", "gamma": 1.0}
    if algo == "ppo":
        return PPO("MlpPolicy", env, n_steps=2048, batch_size=256, learning_rate=3e-4, ent_coef=0.01,
                   policy_kwargs={"net_arch": [64, 64]}, **common)
    if algo == "a2c":
        return A2C("MlpPolicy", env, n_steps=16, learning_rate=7e-4, ent_coef=0.01,
                   policy_kwargs={"net_arch": [64, 64]}, **common)
    if algo == "dqn":
        return DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=100_000, learning_starts=5_000,
                   batch_size=256, train_freq=4, target_update_interval=2_000, exploration_fraction=0.2,
                   exploration_final_eps=0.02, policy_kwargs={"net_arch": [64, 64]}, **common)
    if algo == "recurrent_ppo":
        return RecurrentPPO("MlpLstmPolicy", env, n_steps=2048, batch_size=256, learning_rate=3e-4, ent_coef=0.01,
                            policy_kwargs={"net_arch": [64, 64], "lstm_hidden_size": 64}, **common)
    raise ValueError(algo)


def rollout(model, queries, env_kwargs):
    """Deterministic test-fold episodes + per-query multi-agent metrics."""
    env = MAChunkEnv(queries, **env_kwargs)
    recurrent = model.__class__.__name__ == "RecurrentPPO"
    rows = []
    for q in queries:
        obs, _ = env.reset(options={"query": q})
        done = False
        lstm_state, episode_start = None, np.ones((1,), dtype=bool)
        while not done:
            if recurrent:  # the LSTM memory spans the whole episode (all agents' turns)
                action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_start,
                                                   deterministic=True)
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        sends = {a: sum(1 for ag, _, act in env.decisions if ag == a and act == SEND) for a in q.agents}
        stops = sum(1 for _, _, act in env.decisions if act == STOP)
        ch = env.channel
        if len(ch) > 1:
            sim = q.emb[ch] @ q.emb[ch].T
            iu = np.triu_indices(len(ch), 1)
            redundancy = float((sim[iu] > 0.9).mean())
        else:
            redundancy = np.nan
        rows.append({"interaction_id": q.interaction_id, "chunks": env.selected_texts(),
                     "chunk_ids": [int(q.chunk_ids[c]) for c in ch],
                     "utility": env.utility, "tokens": env.channel_tokens,
                     "team_reward": env.team_reward(), "ap": env.ap(), "n_useful": env.n_useful,
                     "utility_n1": min(1.0, env.n_useful / max(1, q.n_support)), "u_attr": env.u_attr(),
                     "team_reward_base": env.utility - env.lam * env.channel_tokens / 1000.0, "n_chunks": len(ch), "stops": stops,
                     "steps": env.steps, "first_useful_step": env.first_useful_step,
                     "redundancy": redundancy, "n_support": q.n_support,
                     **{f"sends_{a}": n for a, n in sends.items()},
                     **{f"credit_{a}": env.util_credit[j] for j, a in enumerate(q.agents)},
                     **{f"sent_{a}": [int(q.chunk_ids[c]) for c in env.sent_by[j]] for j, a in enumerate(q.agents)}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["ma", "ma_msg", "single"], default="ma")
    parser.add_argument("--lams", type=float, nargs="+", default=[0.1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--token-budget", type=int, default=2000)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pool", default=None, help="multi-hop pool parquet (default: CRAG pool)")
    parser.add_argument("--reward-mode", choices=["marginal", "terminal", "individual", "marginal_ap"], default="marginal")
    parser.add_argument("--ap-weight", type=float, default=1.0, help="weight of AP (context precision) in marginal_ap")
    parser.add_argument("--support-floor", type=int, default=1, help="utility saturates after max(n_support, K) useful chunks")
    parser.add_argument("--utility", choices=["useful", "attr", "mix"], default="useful", help="utility labels: useful chunks, RAGAS attribution, or their mean")
    parser.add_argument("--attr-labels", default=None, help="attribution labels parquet (required for --utility attr/mix)")
    parser.add_argument("--turn-order", choices=["round_robin", "random", "confidence"], default="round_robin")
    parser.add_argument("--with-random-agent", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--rich-obs", action="store_true", help="rich observations: per-query z-scores and log-ranks")
    parser.add_argument("--algo", choices=list(ALGOS), default="ppo")
    parser.add_argument("--agent-dropout", type=float, default=0.0,
                        help="failure-aware training: per-episode failure probability of each agent during training (adds a presence mask)")
    args = parser.parse_args()
    torch.set_num_threads(1)  # tiny MLP: many threads only add contention; parallelize runs instead

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = AGENTS_WITH_RANDOM if args.with_random_agent else AGENTS
    agents = ["rrf"] if args.variant == "single" else universe
    if args.utility != "useful" and not args.attr_labels:
        parser.error("--utility attr/mix needs --attr-labels")
    pool = load_pool(args.top_m, pool_path=args.pool, with_random=args.with_random_agent, attr_path=args.attr_labels)
    suffix = "".join([
        f"_rw-{args.reward_mode}" if args.reward_mode != "marginal" else "",
        f"_to-{args.turn_order}" if args.turn_order != "round_robin" else "",
        "_rand" if args.with_random_agent else "",
        f"_m{args.top_m}" if args.top_m != 8 else "",
        "_rich" if args.rich_obs else "",
        f"_drop{args.agent_dropout}" if args.agent_dropout > 0 else "",
        f"_algo-{args.algo}" if args.algo != "ppo" else "",
        f"_ap{args.ap_weight:g}" if args.reward_mode == "marginal_ap" and args.ap_weight != 1.0 else "",
        f"_sup{args.support_floor}" if args.support_floor != 1 else "",
        f"_u-{args.utility}" if args.utility != "useful" else "",
    ])
    meta = pool.drop_duplicates("interaction_id")[["interaction_id", "domain"]].reset_index(drop=True)

    for lam in args.lams:
        for seed in args.seeds:
            tag = f"{args.variant}{suffix}_lam{lam}_s{seed}"
            env_kwargs = {"lam": lam, "token_budget": args.token_budget, "max_chunks": args.max_chunks,
                          "share_scores": args.variant == "ma_msg", "reward_mode": args.reward_mode,
                          "turn_order": args.turn_order, "universe": universe, "rich_obs": args.rich_obs,
                          "presence_mask": args.agent_dropout > 0, "ap_weight": args.ap_weight,
                          "support_floor": args.support_floor, "utility_mode": args.utility}
            curves = {}
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            rows, t0 = [], time.time()
            for fold, (tr, te) in enumerate(skf.split(meta, meta["domain"])):
                train_q = make_queries(pool, meta.loc[tr, "interaction_id"], args.top_m, agents, universe)
                test_q = make_queries(pool, meta.loc[te, "interaction_id"], args.top_m, agents, universe)
                env = MAChunkEnv(train_q, seed=seed, agent_dropout=args.agent_dropout, **env_kwargs)
                model = make_model(args.algo, env, seed)
                cb = CurveCallback(every=max(1, min(2048, args.timesteps // 50)))  # ~50+ points per fold
                model.learn(total_timesteps=args.timesteps, callback=cb)
                curves[fold] = cb.curve
                if args.save_models:
                    (out_dir / "models").mkdir(exist_ok=True)
                    model.save(out_dir / "models" / f"{tag}_f{fold}.zip")
                # Test-fold evaluation always uses the team reward's marginal bookkeeping for metrics;
                # the training reward mode does not change how outcomes are measured.
                for r in rollout(model, test_q, env_kwargs):
                    rows.append({"condition": tag, "fold": fold, **r})
            df = pd.DataFrame(rows)
            df.to_parquet(out_dir / f"{tag}.parquet")
            (out_dir / "curves").mkdir(exist_ok=True)
            (out_dir / "curves" / f"{tag}.json").write_text(json.dumps(curves))
            summary = {"tag": tag, "lam": lam, "seed": seed, "variant": args.variant,
                       "reward_mode": args.reward_mode, "turn_order": args.turn_order,
                       "with_random_agent": args.with_random_agent, "top_m": args.top_m, "rich_obs": args.rich_obs, "agent_dropout": args.agent_dropout, "algo": args.algo,
                       "team_reward": df["team_reward"].mean(),
                       "utility": df["utility"].mean(), "tokens": df["tokens"].mean(), "ap": df["ap"].mean(),
                       "team_reward_base": df["team_reward_base"].mean(), "ap_weight": args.ap_weight,
                       "u_attr": df["u_attr"].mean(), "utility_mode": args.utility,
                       "n_chunks": df["n_chunks"].mean(), "n_chunks_std": df["n_chunks"].std(),
                       "minutes": (time.time() - t0) / 60}
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()}))


if __name__ == "__main__":
    main()

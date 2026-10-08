"""Property tests for the MA-Chunk environment on every pool (random policies).

Checks, for thousands of random episodes:
  1. telescoping credit: sum of step rewards == U(S_T) - lam * tokens(S_T) / 1000;
  2. U(S_T) equals the covered fraction recomputed from the labels;
  3. the channel never holds duplicates and respects the chunk / token budget;
  4. a STOPped agent never acts again;
  5. per-agent credit sums to the team reward.

Usage (from the repo root):
    python -m pytest tests/   (or: python tests/test_env.py)
"""

import sys
from pathlib import Path

import numpy as np

from ma_chunk.env_ma import AGENTS, AGENTS_WITH_RANDOM, SEND, STOP, MAChunkEnv, obs_dim  # noqa: E402
from ma_chunk.config import DATA
from ma_chunk.train_ma import load_pool, make_queries


def check_pool(name, pool, episodes=3000, lam=0.5, seed=0, reward_mode="marginal",
               turn_order="round_robin", disabled=(), universe=AGENTS, msg_noise=0.0, rich=False, dropout=0.0):
    queries = make_queries(pool, pool["interaction_id"].unique(), 8, universe, universe)
    env = MAChunkEnv(queries, lam=lam, token_budget=2000, max_chunks=10, seed=seed, share_scores=True,
                     reward_mode=reward_mode, turn_order=turn_order, disabled_agents=disabled,
                     universe=universe, msg_noise=msg_noise, rich_obs=rich,
                     agent_dropout=dropout, presence_mask=dropout > 0)
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        obs, _ = env.reset()
        assert obs.shape == (obs_dim(universe, rich, dropout > 0),)
        assert len(env.disabled) < len(env.q.agents), "every agent failed"
        disabled = tuple(env.disabled)
        total, credit, stopped, done, selfish = 0.0, {}, set(), False, 0.0
        while not done:
            agent = env.q.agents[env.turn]
            assert agent not in stopped, "a stopped agent acted again"
            assert agent not in disabled, "a failed agent acted"
            for d in disabled:  # failed agents send no messages
                assert obs[len(universe) + universe.index(d)] == 0.0
            action = int(rng.choice(3, p=[0.45, 0.45, 0.10]))
            c = env._current(env.turn)
            if action == SEND:
                selfish += int(env.q.useful[c] == 1) / max(1, env.q.n_support) - lam * env.q.tokens[c] / 1000.0
            obs, r, done, _, _ = env.step(action)
            total += r
            credit[agent] = credit.get(agent, 0.0) + r
            if action == STOP:
                stopped.add(agent)
        q, s = env.q, env.channel
        expected_u = min(1.0, sum(int(q.useful[c] == 1) for c in s) / max(1, q.n_support))
        team = expected_u - lam * sum(int(q.tokens[c]) for c in s) / 1000.0
        assert abs(env.utility - expected_u) < 1e-9, "utility differs from recomputed coverage"
        if reward_mode in ("marginal", "terminal"):
            assert abs(total - team) < 1e-6, f"{reward_mode}: return {total} != team reward {team}"
            assert abs(sum(credit.values()) - team) < 1e-6, "per-agent credit does not sum to team reward"
        else:  # individual: each agent is paid for its own chunks, regardless of the team
            assert abs(total - selfish) < 1e-6, f"individual: return {total} != selfish sum {selfish}"
        assert abs(sum(env.util_credit) - env.utility) < 1e-9, "utility credit does not telescope"
        assert abs(env.team_reward() - team) < 1e-9
        assert len(s) == len(set(s)), "duplicate chunk in channel"
        assert len(s) <= env.max_chunks, "chunk budget exceeded"
        last = int(q.tokens[s[-1]]) if s else 0
        assert env.channel_tokens - last < env.token_budget, "token budget exceeded before the last send"
        sends = sum(1 for _, _, a in env.decisions if a == SEND)
        assert sends == len(s), "SEND count does not match channel size"
    print(f"[ok] {name} reward={reward_mode} order={turn_order} disabled={list(disabled)} "
          f"agents={len(universe)} noise={msg_noise}: {episodes} random episodes, {len(queries)} queries")


def main():
    pools = {"crag": load_pool(8, with_random=True)}
    for ds in ("hotpotqa", "musique"):
        path = DATA / ds / "candidates.parquet"
        if path.exists():
            pools[ds] = load_pool(8, pool_path=path, with_random=True)
    for name, pool in pools.items():
        for mode in ("marginal", "terminal", "individual"):
            check_pool(name, pool, episodes=1000, reward_mode=mode)
        for order in ("random", "confidence"):
            check_pool(name, pool, episodes=1000, turn_order=order)
        check_pool(name, pool, episodes=1000, disabled=("ce", "e5"), msg_noise=0.2)
        check_pool(name, pool, episodes=1000, universe=AGENTS_WITH_RANDOM, turn_order="confidence")
        check_pool(name, pool, episodes=1000, rich=True, disabled=("bm25",))
        check_pool(name, pool, episodes=1000, dropout=0.5)


def test_environment_properties():
    main()


if __name__ == "__main__":
    main()

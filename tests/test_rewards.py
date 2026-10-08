"""Property tests of the precision-aware reward (reward_mode="marginal_ap") and of the RAGAS-aligned utilities.

    1. Telescoping: under random policies the sum of per-step rewards equals the team objective
       U(S) + beta * AP(S) - lambda * tokens / 1000 (Proposition 1 extends to AP).
    2. env.ap() equals RAGAS 0.2.15's own context-precision formula
       (LLMContextPrecisionWithReference._calculate_average_precision) applied to the labels of the
       channel, in channel order.
    3. RAGAS-aligned utilities ("attr", "mix") with random synthetic attribution masks: telescoping
       and U_attr = fraction of reference sentences covered by the channel, recomputed from scratch.

Usage (from the repo root):
    python -m pytest tests/   (or: python tests/test_rewards.py)
"""

import sys
from pathlib import Path

import numpy as np

from ma_chunk.env_ma import AGENTS, SEND, STOP, MAChunkEnv  # noqa: E402
from ma_chunk.config import DATA
from ma_chunk.train_ma import load_pool, make_queries


class _V:
    def __init__(self, v):
        self.verdict = v


def ragas_ap(labels):
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    return LLMContextPrecisionWithReference()._calculate_average_precision([_V(int(x)) for x in labels])


def random_episode(env, rng, q, p_send=0.5, p_stop=0.05):
    env.reset(options={"query": q})
    total, done, actions = 0.0, False, []
    while not done:
        u = rng.random()
        a = STOP if u < p_stop else (SEND if u < p_stop + p_send else 1)
        actions.append(a)
        _, r, done, _, info = env.step(a)
        total += r
    return total, info, actions


def main():
    rng = np.random.default_rng(0)
    pools = {"crag": None, "hotpotqa": str(DATA / "hotpotqa" / "candidates.parquet"),
             "musique": str(DATA / "musique" / "candidates.parquet")}
    for name, path in pools.items():
        pool = load_pool(8, pool_path=path)
        ids = pool["interaction_id"].unique()[:300]
        qs = make_queries(pool, ids, 8, AGENTS, AGENTS)
        for beta in (0.5, 1.0, 2.0):
            env = MAChunkEnv(qs, lam=0.5, seed=0, share_scores=True, reward_mode="marginal_ap", ap_weight=beta)
            n_ap_checked = 0
            for k in range(600):
                q = qs[k % len(qs)]
                total, info, _ = random_episode(env, rng, q)
                target = env.utility + beta * env.ap() - 0.5 * env.channel_tokens / 1000.0
                assert abs(total - target) < 1e-9, (name, beta, total, target)
                assert abs(info["team_reward"] - target) < 1e-9
                labels = [int(q.useful[c] == 1) for c in env.channel]
                if labels:
                    expected = ragas_ap(labels) if sum(labels) else 0.0
                    assert abs(env.ap() - expected) < 1e-6, (labels, env.ap(), expected)
                    n_ap_checked += 1
            print(f"[ok] {name} beta={beta}: telescoping + RAGAS AP on {n_ap_checked} episodes")
        # RAGAS-aligned utilities on synthetic masks
        for q in qs:
            q.n_stmt = int(rng.integers(1, 3))
            q.attr_mask = rng.integers(0, 2 ** q.n_stmt, size=len(q.chunk_ids)) * (rng.random(len(q.chunk_ids)) < 0.3)
        for umode in ("attr", "mix"):
            env = MAChunkEnv(qs, lam=0.4, seed=2, share_scores=True, reward_mode="marginal_ap", ap_weight=1.0,
                             turn_order="confidence", utility_mode=umode)
            for k in range(600):
                q = qs[k % len(qs)]
                total, info, _ = random_episode(env, rng, q)
                cov = 0
                for c in env.channel:
                    cov |= int(q.attr_mask[c])
                u_attr = bin(cov).count("1") / q.n_stmt
                u_useful = min(1.0, sum(int(q.useful[c] == 1) for c in env.channel) / max(1, q.n_support))
                u = u_attr if umode == "attr" else 0.5 * (u_useful + u_attr)
                assert abs(env.u_attr() - u_attr) < 1e-12 and abs(env.utility - u) < 1e-12
                target = u + env.ap() - 0.4 * env.channel_tokens / 1000.0
                assert abs(total - target) < 1e-9, (umode, total, target)
            print(f"[ok] {name} utility={umode}: telescoping + coverage recomputed (600 episodes)")


def test_reward_properties():
    main()


if __name__ == "__main__":
    main()

"""MA-Chunk environment: cooperative, cost-aware multi-agent context curation.

Dec-POMDP, executed as turn-based (agent-environment cycle) episodes so that a single
parameter-shared policy can be trained with standard PPO (the acting agent's id is part of
its observation). One episode = one query.

Agents: one per retriever (bm25, spacy, minilm, e5, ce = cross-encoder). Agent i walks down its own top-M list
and, at each of its turns, looks at its current candidate chunk and chooses
    SKIP (0) - move to its next candidate,
    SEND (1) - write the chunk to the shared context channel read by the frozen LLM reader,
    STOP (2) - stop contributing for this query (metareasoning: "enough evidence").
Chunks already in the channel (sent by another agent) are skipped automatically.
The episode ends when every agent has stopped or exhausted its list, or when the channel
reaches the token budget / max number of chunks.

Local observation of the acting agent (partial: it sees its own retriever's signal, plus
what the channel reveals about the others):
    agent one-hot; own score; own rank / M; agreement (fraction of the other agents that
    also rank this chunk in their top-M - learned through the channel); redundancy (max
    MiniLM cosine with chunks already in the channel); channel tokens / budget; channel
    size / max chunks; own sends / max chunks; others' sends / max chunks; list progress;
    chunk tokens / 512.

Team reward: U(S) - lam * tokens(S) / 1000, where U(S) is the covered fraction of the
query's supporting evidence, min(1, #useful chunks in S / n_support). On CRAG n_support = 1
(U = 1 if the channel contains at least one chunk labelled useful by the LLM judge,
labels.py); on multi-hop QA n_support is the number of gold supporting paragraphs. It is paid as marginal
contributions at each SEND (U(S + c) - U(S) - lam * tokens(c) / 1000), which telescope to
the team reward, so each agent is credited exactly with what its message added
(a sequential difference reward). No API call is needed during training.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

AGENTS = ["bm25", "spacy", "minilm", "e5", "ce"]
# Robustness test (H6): a sixth agent whose "retriever" is a random score over the pool.
AGENTS_WITH_RANDOM = AGENTS + ["random"]
SKIP, SEND, STOP = 0, 1, 2
OBS_DIM = 2 * len(AGENTS) + 11  # default observation size (5-agent universe)

REWARD_MODES = ("marginal", "terminal", "individual", "marginal_ap")
TURN_ORDERS = ("round_robin", "random", "confidence")


def obs_dim(universe, rich: bool = False, presence: bool = False) -> int:
    return 2 * len(universe) + 11 + (2 * len(universe) if rich else 0) + (len(universe) if presence else 0)


class QueryData:
    """Pool of one query: per-agent ordered candidate lists + chunk features."""

    def __init__(self, pool: pd.DataFrame, top_m: int, agents=AGENTS, universe=AGENTS):
        pool = pool.reset_index(drop=True)
        self.interaction_id = pool["interaction_id"].iloc[0]
        self.chunk_ids = pool["chunk_id"].to_numpy()
        self.texts = pool["text"].tolist()
        self.tokens = pool["tokens"].to_numpy()
        self.useful = pool["useful"].to_numpy() if "useful" in pool else np.zeros(len(pool), dtype=int)
        self.n_support = int(pool["n_support"].iloc[0]) if "n_support" in pool else 1
        # RAGAS-aligned attribution labels (bitmask of reference sentences that
        # each chunk supports) and the number of reference sentences; absent -> zeros / 1.
        self.attr_mask = pool["attr_mask"].fillna(0).astype(np.int64).to_numpy() if "attr_mask" in pool \
            else np.zeros(len(pool), dtype=np.int64)
        self.n_stmt = int(pool["n_stmt"].max()) if "n_stmt" in pool and pool["n_stmt"].notna().any() else 1
        self.emb = np.stack(pool["emb"].to_numpy()).astype(np.float32)
        self.agents = list(agents)
        self.scores = {a: pool[f"s_{a}"].to_numpy(dtype=np.float32) for a in set(self.agents) | set(universe)}
        in_top = {a: pool[f"rank_{a}"].to_numpy() < top_m for a in self.agents}
        self.lists = {
            a: [int(i) for i in np.argsort(pool[f"rank_{a}"].to_numpy(), kind="stable") if in_top[a][i]]
            for a in self.agents
        }
        # Per-agent min-max of its own list scores (used by the "confidence" turn order).
        self.score_range = {
            a: (float(self.scores[a][self.lists[a]].min()), float(self.scores[a][self.lists[a]].max()))
            if self.lists[a] else (0.0, 1.0) for a in self.agents
        }
        self.n_in_top = sum(in_top[a].astype(int) for a in self.agents)
        self.top_m = top_m
        # Rich observation: per-query z-score and log-rank of every retriever's score.
        self.z = {a: np.nan_to_num((self.scores[a] - self.scores[a].mean()) / (self.scores[a].std() + 1e-6))
                  for a in self.scores}
        self.log_rank = {a: np.log1p(pool[f"rank_{a}"].to_numpy(dtype=np.float32)) / np.log1p(1000.0)
                         for a in self.scores if f"rank_{a}" in pool}


class MAChunkEnv(gym.Env):
    """Options (defaults reproduce the main method exactly):
        reward_mode  - "marginal": team reward paid as each SEND's marginal contribution (proposed);
                       "terminal": team reward R(S_T) paid once at the end (no decomposition);
                       "individual": each SEND pays the sender for its own chunk's utility,
                       1[useful]/n_support - cost, ignoring what the team already has (selfish).
                       "marginal_ap" (precision-aware): team objective U(S) + ap_weight * AP(S) - cost, paid as
                       marginal contributions; AP is the average precision of the channel order
                       computed exactly as RAGAS context precision, with the utility labels as verdicts.
        turn_order   - "round_robin" (proposed), "random" (uniform among agents with a candidate),
                       "confidence" (the agent whose current candidate has the highest
                       min-max-normalized own score speaks next).
        msg_noise    - std of Gaussian noise added to the messages (other agents' scores) at
                       observation time; robustness test only.
        disabled_agents - agents that fail: they never act and send no messages.
        universe     - agent universe defining the observation layout (AGENTS or AGENTS_WITH_RANDOM).
    """

    def __init__(self, queries: list[QueryData], lam: float = 0.1, token_budget: int = 2000,
                 max_chunks: int = 10, seed: int | None = None, sequential: bool = False,
                 share_scores: bool = False, reward_mode: str = "marginal",
                 turn_order: str = "round_robin", msg_noise: float = 0.0,
                 disabled_agents=(), universe=AGENTS, rich_obs: bool = False,
                 agent_dropout: float = 0.0, presence_mask: bool = False, ap_weight: float = 1.0,
                 support_floor: int = 1, utility_mode: str = "useful"):
        super().__init__()
        assert reward_mode in REWARD_MODES and turn_order in TURN_ORDERS
        # share_scores: agents broadcast their retriever score for the chunk under
        # consideration (explicit communication); otherwise each agent only sees its own.
        self.share_scores = share_scores
        self.queries = queries
        self.lam = lam
        self.token_budget = token_budget
        self.max_chunks = max_chunks
        self.sequential = sequential
        self.reward_mode = reward_mode
        self.ap_weight = ap_weight if reward_mode == "marginal_ap" else 0.0
        # Utility saturates only after max(n_support, support_floor) useful chunks; with
        # support_floor=2 on CRAG a second useful chunk still has value (RAGAS context recall).
        self.support_floor = support_floor
        # "useful" = coverage of useful chunks (default); "attr" = fraction of
        # reference sentences attributed to the channel (RAGAS context-recall labels); "mix" = mean of both.
        assert utility_mode in ("useful", "attr", "mix")
        self.utility_mode = utility_mode
        self.turn_order = turn_order
        self.msg_noise = msg_noise
        self.disabled = set(disabled_agents)
        self.universe = list(universe)
        self.rich_obs = rich_obs
        # Failure-aware training. agent_dropout = per-episode probability that each agent
        # fails (at least one stays); presence_mask adds one "agent available" bit per agent so
        # that absence is not confused with a low score.
        self.agent_dropout = agent_dropout
        self.presence_mask = presence_mask
        self._fixed_disabled = set(disabled_agents)
        self._drop_rng = np.random.default_rng(None if seed is None else seed + 15485863)
        self._rng = np.random.default_rng(seed)
        self._order_rng = np.random.default_rng(None if seed is None else seed + 7919)
        self._noise_rng = np.random.default_rng(None if seed is None else seed + 104729)
        self._next = 0
        dim = obs_dim(self.universe, rich_obs, presence_mask)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(dim,), dtype=np.float32) \
            if (rich_obs or presence_mask) else spaces.Box(-1.0, 2.0, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    # --- episode bookkeeping -------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "query" in options:
            self.q = options["query"]
        elif self.sequential:
            self.q = self.queries[self._next % len(self.queries)]
            self._next += 1
        else:
            self.q = self.queries[int(self._rng.integers(len(self.queries)))]
        q = self.q
        self.n_agents = len(q.agents)
        self.disabled = set(self._fixed_disabled)
        if self.agent_dropout > 0:
            drop = [a for a in q.agents if self._drop_rng.random() < self.agent_dropout]
            if len(set(drop) | self.disabled) >= len(q.agents):  # keep at least one agent alive
                drop = drop[:-1]
            self.disabled |= set(drop)
        self.ptr = [0] * self.n_agents
        self.active = [len(q.lists[a]) > 0 and a not in self.disabled for a in q.agents]
        self.sent_by = [[] for _ in range(self.n_agents)]
        self.util_credit = [0.0] * self.n_agents  # utility increments produced by each agent
        self.channel: list[int] = []
        self.channel_tokens = 0
        self.utility = 0.0
        self.n_useful = 0
        self.ap_num = 0.0  # sum over useful positions k of precision@k (RAGAS context precision numerator)
        self.attr_cov = 0  # bitmask of reference sentences attributed to some chunk in the channel
        self.turn = 0
        self.steps = 0
        self.first_useful_step = None
        self.decisions = []  # (agent, chunk, action) log for analysis
        self._select_turn(start=0)
        return self._obs(), {}

    def _current(self, i):
        a = self.q.agents[i]
        return self.q.lists[a][self.ptr[i]] if self.ptr[i] < len(self.q.lists[a]) else None

    def _skip_sent(self, i):
        while self.active[i]:
            c = self._current(i)
            if c is None:
                self.active[i] = False
            elif c in self.channel:
                self.ptr[i] += 1
            else:
                return

    def _advance_to_valid(self):
        """Moves self.turn to the next agent (round-robin) that still has a candidate."""
        for _ in range(self.n_agents):
            self._skip_sent(self.turn)
            if self.active[self.turn]:
                return True
            self.turn = (self.turn + 1) % self.n_agents
        return False

    def _select_turn(self, start: int) -> bool:
        """Chooses who speaks next; returns False when nobody can."""
        if self.turn_order == "round_robin":
            self.turn = start
            return self._advance_to_valid()
        for i in range(self.n_agents):
            self._skip_sent(i)
        valid = [i for i in range(self.n_agents) if self.active[i]]
        if not valid:
            return False
        if self.turn_order == "random":
            self.turn = int(self._order_rng.choice(valid))
        else:  # confidence
            def conf(i):
                a = self.q.agents[i]
                lo, hi = self.q.score_range[a]
                return (self.q.scores[a][self._current(i)] - lo) / (hi - lo + 1e-9)
            self.turn = max(valid, key=conf)
        return True

    def _done(self):
        return (not any(self.active) or self.channel_tokens >= self.token_budget
                or len(self.channel) >= self.max_chunks)

    def u_useful(self) -> float:
        return min(1.0, self.n_useful / max(1, self.q.n_support, self.support_floor))

    def u_attr(self) -> float:
        return bin(self.attr_cov).count("1") / max(1, self.q.n_stmt)

    def _utility(self) -> float:
        if self.utility_mode == "useful":
            return self.u_useful()
        if self.utility_mode == "attr":
            return self.u_attr()
        return 0.5 * (self.u_useful() + self.u_attr())

    def ap(self) -> float:
        """Average precision of the channel order with the labels as verdicts (= RAGAS context
        precision formula; 0 when no useful chunk was sent)."""
        return self.ap_num / self.n_useful if self.n_useful else 0.0

    def team_reward(self) -> float:
        return self.utility + self.ap_weight * self.ap() - self.lam * self.channel_tokens / 1000.0

    # --- dynamics ------------------------------------------------------------
    def step(self, action):
        action = int(action)
        i = self.turn
        c = self._current(i)
        reward = 0.0
        self.steps += 1
        if action == SEND:
            useful = int(self.q.useful[c] == 1)
            old_ap = self.ap()
            self.n_useful += useful
            self.attr_cov |= int(self.q.attr_mask[c])
            new_utility = self._utility()
            cost = self.lam * self.q.tokens[c] / 1000.0
            if useful:
                self.ap_num += self.n_useful / (len(self.channel) + 1)
            if self.reward_mode == "marginal":
                reward = (new_utility - self.utility) - cost
            elif self.reward_mode == "marginal_ap":
                new_ap = self.ap_num / self.n_useful if self.n_useful else 0.0
                reward = (new_utility - self.utility) + self.ap_weight * (new_ap - old_ap) - cost
            elif self.reward_mode == "individual":
                reward = useful / max(1, self.q.n_support, self.support_floor) - cost
            self.util_credit[i] += new_utility - self.utility
            if useful and self.first_useful_step is None:
                self.first_useful_step = self.steps
            self.utility = new_utility
            self.channel.append(c)
            self.channel_tokens += int(self.q.tokens[c])
            self.sent_by[i].append(c)
            self.ptr[i] += 1
        elif action == SKIP:
            self.ptr[i] += 1
        else:
            self.active[i] = False
        self.decisions.append((self.q.agents[i], int(self.q.chunk_ids[c]), action))

        terminated = self._done() or not self._select_turn(start=(i + 1) % self.n_agents)
        info = {}
        if terminated:
            if self.reward_mode == "terminal":
                reward = self.team_reward()
            info = {"utility": self.utility, "n_chunks": len(self.channel), "tokens": self.channel_tokens,
                    "team_reward": self.team_reward(), "ap": self.ap()}
        return self._obs(), float(reward), terminated, False, info

    def _obs(self):
        uni = self.universe
        obs = np.zeros(obs_dim(uni, self.rich_obs, self.presence_mask), dtype=np.float32)
        i = self.turn
        c = self._current(i) if self.active[i] else None
        if c is None:
            return obs
        q = self.q
        a = q.agents[i]
        k = len(uni)
        if a in uni:
            obs[uni.index(a)] = 1.0
        # Retriever-score slots: own score always; the others' only as messages (share_scores)
        # or for the centralized single agent over the fused list. Failed agents send nothing.
        for j, b in enumerate(uni):
            if b in self.disabled:
                continue
            if b == a or self.share_scores or a not in uni:
                value = float(q.scores[b][c])
                if b != a and self.msg_noise > 0:
                    value += float(self._noise_rng.normal(0.0, self.msg_noise))
                obs[k + j] = float(np.clip(value, -1, 1))
        k = 2 * len(uni)
        redundancy = float(np.max(q.emb[self.channel] @ q.emb[c])) if self.channel else 0.0
        others_sent = sum(len(s) for j, s in enumerate(self.sent_by) if j != i)
        base = 2 * len(uni) + 11
        if self.rich_obs:  # same visibility rule as the score slots (own always; others as messages)
            for j, b in enumerate(uni):
                if b in self.disabled or not (b == a or self.share_scores or a not in uni):
                    continue
                obs[base + j] = float(np.clip(q.z[b][c], -5, 5))
                obs[base + len(uni) + j] = float(q.log_rank[b][c]) if b in q.log_rank else 0.0
        if self.presence_mask:
            off = base + (2 * len(uni) if self.rich_obs else 0)
            for j, b in enumerate(uni):
                obs[off + j] = 0.0 if b in self.disabled else 1.0
        obs[k:base] = [
            float(np.clip(q.scores[a][c], -1, 1)),
            self.ptr[i] / q.top_m,
            (q.n_in_top[c] - 1) / max(1, self.n_agents - 1),
            redundancy,
            min(1.0, self.channel_tokens / self.token_budget),
            len(self.channel) / self.max_chunks,
            len(self.sent_by[i]) / self.max_chunks,
            others_sent / self.max_chunks,
            self.ptr[i] / max(1, len(q.lists[a])),
            min(2.0, q.tokens[c] / 512.0),
            float(not self.channel),
        ]
        return obs

    # --- helpers -------------------------------------------------------------
    def selected_texts(self):
        return [self.q.texts[c] for c in self.channel]


def load_queries(candidates_path, labels_path, top_m: int, agents=AGENTS) -> dict:
    pool = pd.read_parquet(candidates_path)
    if labels_path:
        labels = pd.read_parquet(labels_path)[["interaction_id", "chunk_id", "useful"]]
        pool = pool.merge(labels, on=["interaction_id", "chunk_id"], how="left")
        pool["useful"] = pool["useful"].fillna(0).clip(lower=0).astype(int)
    return {qid: QueryData(g, top_m, agents) for qid, g in pool.groupby("interaction_id", sort=False)}

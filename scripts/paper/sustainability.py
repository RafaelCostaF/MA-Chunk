#!/usr/bin/env python3
"""Computational cost and sustainability (CodeCarbon) of MA-Chunk and its comparators.

Same CodeCarbon configuration as the prior single-agent pipeline (EmissionsTracker(tracking_mode="process"),
the prior single-agent pipeline), so the numbers are comparable. On this machine RAPL counters are not
readable without root, so CodeCarbon estimates CPU power from the processor TDP and the
process's CPU share; GPU energy is read from NVML for the whole device (shared machine: an
upper bound). See docs/REPRODUCE.md for the caveats.

Modes (one JSON line per measurement, appended to --out):
    train   - 5-fold CV training of one (dataset, variant, algorithm) at lambda=1, seed 0
              (same settings as the experiments), tracked as one block; then the deterministic
              test-fold rollouts of the trained models, tracked separately (per-query inference).
    sup     - the supervised selector (gradient boosting): fit on 4 folds + greedy selection.
    scoring - candidate scoring by the five retrievers on the GPU for a sample of questions.

Usage (from the repo root):
    python scripts/paper/sustainability.py train --dataset hotpotqa --variant ma_msg --algo ppo --out costs.jsonl
    python scripts/paper/sustainability.py sup --dataset hotpotqa --out costs.jsonl
    python scripts/paper/sustainability.py scoring --dataset crag --n 20 --out costs.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from sklearn.model_selection import StratifiedKFold

from ma_chunk.config import DATA, REPO, RESULTS  # noqa: E402
REPO_ROOT = REPO
POOLS = {"crag": None,
         "hotpotqa": DATA / "hotpotqa/candidates.parquet",
         "musique": DATA / "musique/candidates.parquet"}


def tracker(gpu_ids=None):
    kw = {"gpu_ids": gpu_ids} if gpu_ids is not None else {}
    return EmissionsTracker(log_level="error", save_to_file=False, tracking_mode="process", **kw)


def measure(fn, uses_gpu=False):
    """CPU-only jobs: CodeCarbon reads GPU power for the whole device(s) even in process mode, which
    on a shared machine charges other users' GPU load to us. For them we report CPU + RAM energy and
    scale CO2 by the same carbon intensity CodeCarbon applied. GPU jobs track only GPU 0 (the device
    they use); since that device is shared, their GPU energy is an upper bound."""
    t = tracker(gpu_ids=[0] if uses_gpu else None)
    t.start()
    t0 = time.perf_counter()
    out = fn()
    seconds = time.perf_counter() - t0
    kg = float(t.stop() or 0.0)
    d = t.final_emissions_data
    total = float(d.energy_consumed)
    energy = total if uses_gpu else float(d.cpu_energy) + float(d.ram_energy)
    co2 = kg if uses_gpu else (kg * energy / total if total > 0 else 0.0)
    return out, {"seconds": seconds, "energy_kwh": energy, "co2_kg": co2,
                 "cpu_energy_kwh": float(d.cpu_energy), "ram_energy_kwh": float(d.ram_energy),
                 "gpu_energy_kwh": float(d.gpu_energy) if uses_gpu else 0.0,
                 "raw_total_energy_kwh": total, "carbon_intensity_kg_per_kwh": kg / total if total > 0 else None,
                 "country": d.country_iso_code, "cpu_model": d.cpu_model, "gpu_model": d.gpu_model}


def mode_train(args):
    from ma_chunk.env_ma import AGENTS, MAChunkEnv
    from ma_chunk.train_ma import load_pool, make_model, make_queries, rollout
    torch.set_num_threads(1)
    pool = load_pool(8, pool_path=POOLS[args.dataset])
    meta = pool.drop_duplicates("interaction_id")[["interaction_id", "domain"]].reset_index(drop=True)
    agents = ["rrf"] if args.variant == "single" else AGENTS
    env_kwargs = {"lam": 1.0, "token_budget": 2000, "max_chunks": 10, "share_scores": args.variant == "ma_msg"}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(meta, meta["domain"]))
    data = [(make_queries(pool, meta.loc[tr, "interaction_id"], 8, agents),
             make_queries(pool, meta.loc[te, "interaction_id"], 8, agents)) for tr, te in folds]

    def train_all():
        models = []
        for train_q, _ in data:
            model = make_model(args.algo, MAChunkEnv(train_q, seed=0, **env_kwargs), 0)
            model.learn(total_timesteps=args.timesteps)
            models.append(model)
        return models

    models, train_cost = measure(train_all)
    n_test = sum(len(te) for _, te in data)
    rows, infer_cost = measure(lambda: [r for m, (_, te) in zip(models, data) for r in rollout(m, te, env_kwargs)])
    team = float(np.mean([r["team_reward"] for r in rows]))
    return {"mode": "train", "dataset": args.dataset, "variant": args.variant, "algo": args.algo,
            "timesteps_per_fold": args.timesteps, "folds": 5, "team_reward": team,
            **{f"train_{k}": v for k, v in train_cost.items() if k not in ("country", "cpu_model", "gpu_model")},
            "n_test_queries": n_test,
            "infer_seconds_per_query": infer_cost["seconds"] / n_test,
            "infer_energy_kwh_per_query": infer_cost["energy_kwh"] / n_test,
            "infer_co2_kg_per_query": infer_cost["co2_kg"] / n_test,
            "country": train_cost["country"], "cpu_model": train_cost["cpu_model"],
            "carbon_intensity_kg_per_kwh": train_cost["carbon_intensity_kg_per_kwh"]}


def mode_sup(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sup_select import feature_cols, featurize, greedy
    from ma_chunk.train_ma import load_pool
    pool = featurize(load_pool(8, pool_path=POOLS[args.dataset]), 8)
    if "n_support" not in pool:
        pool["n_support"] = 1
    meta = pool.drop_duplicates("interaction_id")[["interaction_id", "domain"]].reset_index(drop=True)
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(meta, meta["domain"]))
    train = pool[pool["interaction_id"].isin(set(meta.loc[tr, "interaction_id"]))]
    test = pool[pool["interaction_id"].isin(set(meta.loc[te, "interaction_id"]))]
    cols = feature_cols("full")
    model, fit_cost = measure(lambda: HistGradientBoostingClassifier(random_state=0).fit(
        train[cols], (train["useful"] == 1).astype(int)))
    need = float(train.drop_duplicates("interaction_id")["n_support"].mean())
    groups = [g.reset_index(drop=True) for _, g in test.groupby("interaction_id")]
    _, infer_cost = measure(lambda: [greedy(g, model.predict_proba(g[cols])[:, 1], 1.0, need) for g in groups])
    return {"mode": "sup", "dataset": args.dataset, "variant": "supervised_gbm", "algo": "gbm",
            **{f"train_{k}": v for k, v in fit_cost.items() if k not in ("country", "cpu_model", "gpu_model")},
            "n_test_queries": len(groups),
            "infer_seconds_per_query": infer_cost["seconds"] / len(groups),
            "infer_energy_kwh_per_query": infer_cost["energy_kwh"] / len(groups),
            "infer_co2_kg_per_query": infer_cost["co2_kg"] / len(groups)}


def mode_scoring(args):
    if args.dataset == "crag":
        from ma_chunk.config import build
        data = pd.read_parquet(DATA / "crag" / "crag_sample.parquet").head(args.n)
        _, cost = measure(lambda: build(data, 8), uses_gpu=True)
    else:
        import multihop_data as mh
        qs = mh.hotpot_questions(args.n, 0) if args.dataset == "hotpotqa" else mh.musique_questions(args.n, 0, 20)
        _, cost = measure(lambda: mh.build_pool(qs, args.dataset), uses_gpu=True)
    return {"mode": "scoring", "dataset": args.dataset, "variant": "5_retrievers", "algo": "-", "n_queries": args.n,
            "seconds_per_query": cost["seconds"] / args.n, "energy_kwh_per_query": cost["energy_kwh"] / args.n,
            "gpu_energy_kwh_per_query": cost["gpu_energy_kwh"] / args.n, "co2_kg_per_query": cost["co2_kg"] / args.n,
            "gpu_model": cost["gpu_model"], "carbon_intensity_kg_per_kwh": cost["carbon_intensity_kg_per_kwh"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "sup", "scoring"])
    parser.add_argument("--dataset", choices=list(POOLS), required=True)
    parser.add_argument("--variant", default="ma_msg", choices=["ma_msg", "ma", "single"])
    parser.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "dqn", "recurrent_ppo"])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n", type=int, default=20, help="scoring: number of questions/pages")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    row = {"train": mode_train, "sup": mode_sup, "scoring": mode_scoring}[args.mode](args)
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()

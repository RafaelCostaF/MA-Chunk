"""Labels the utility of every pooled candidate chunk with the LLM judge (offline, run once).

useful = 1 if the passage contains information that supports the gold answer (or, for
false-premise questions, shows the premise is false). Training then needs no API calls:
the team reward of any chunk subset is computed from these labels.

Usage (from the repo root):
    python -m ma_chunk.labels \
        --candidates data/crag/candidates.parquet --out data/crag/labels.parquet
"""

import argparse
import concurrent.futures as cf
import re
import sys
from pathlib import Path

import pandas as pd

from ma_chunk.config import DATA, OPENAI_MODEL
from ma_chunk.llm import _chat

PROMPT = """Question: {query}
Ground-truth answer: {gold}

Passage:
\"\"\"{passage}\"\"\"

Does the passage contain information that supports the ground-truth answer, so that a
reader could answer the question correctly from it (alone or combined with other
passages)? If the ground truth says the question has a false premise, answer yes only if
the passage shows the premise is false. Topical overlap without the needed fact is NOT
enough.

Return only JSON: {{"useful": 1}} or {{"useful": 0}}"""


def label(args) -> int:
    query, gold, passage = args
    try:
        content, _, _ = _chat(PROMPT.format(query=query, gold=gold, passage=passage),
                              "You are a strict and precise annotator.")
        match = re.search(r'"useful"\s*:\s*([01])', content)
        return int(match.group(1)) if match else -1
    except Exception as e:  # noqa: BLE001
        print(f"[label error] {e}")
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reuse", default=None, help="existing labels parquet: only unlabelled chunks are sent")
    args = parser.parse_args()

    pool = pd.read_parquet(args.candidates, columns=["interaction_id", "chunk_id", "text"])
    gold = pd.read_parquet(DATA / "crag" / "questions.parquet")[
        ["interaction_id", "query", "clean_answer"]].drop_duplicates("interaction_id")
    pool = pool.merge(gold, on="interaction_id", how="left")

    pool["useful"] = -2  # not labelled yet
    pool["label_model"] = OPENAI_MODEL
    if args.reuse:
        old = pd.read_parquet(args.reuse)[["interaction_id", "chunk_id", "useful", "label_model"]]
        pool = pool.drop(columns=["useful", "label_model"]).merge(
            old, on=["interaction_id", "chunk_id"], how="left")
        pool["useful"] = pool["useful"].fillna(-2).astype(int)
        pool["label_model"] = pool["label_model"].fillna(OPENAI_MODEL)
    todo = pool.index[pool["useful"].isin([-2, -1])]
    print(f"labelling {len(todo)} of {len(pool)} chunks")
    jobs = list(zip(pool.loc[todo, "query"], pool.loc[todo, "clean_answer"], pool.loc[todo, "text"]))
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        pool.loc[todo, "useful"] = list(ex.map(label, jobs))
    pool[["interaction_id", "chunk_id", "useful", "label_model"]].to_parquet(args.out)

    per_query = pool.groupby("interaction_id")["useful"].apply(lambda u: (u == 1).any())
    print(f"labels: {(pool.useful == 1).mean():.3f} useful, {(pool.useful == -1).sum()} failed; "
          f"queries with >=1 useful chunk in the pool: {per_query.mean():.3f}")


if __name__ == "__main__":
    main()

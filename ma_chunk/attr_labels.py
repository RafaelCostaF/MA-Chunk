"""RAGAS-aligned attribution labels for candidate chunks.

For each candidate chunk, the judge (OPENAI_MODEL, gpt-5-nano; calls cached by llm_cache) decides,
sentence by sentence, whether the reference answer can be attributed to that chunk, using the
literal instruction (and the example) of ragas 0.2.15 ContextRecallClassificationPrompt, the
prompt behind RAGAS context recall. The reference is split into sentences deterministically, so
every chunk of a question is judged against the same fixed list. Reference = what RAGAS uses:
CRAG clean_answer; multi-hop " / ".join(answers).

Output parquet: interaction_id, chunk_id, n_stmt, attr (list of 0/1 per sentence, -1 on parse
failure), attr_mask (bitmask of attributed sentences), attr_any.

Usage (from the repo root):
    python -m ma_chunk.attr_labels --dataset crag --out data/crag/attr_labels.parquet
    # pilot: only the chunks of some selections, on a subset of questions
    ... --only-selections <runs parquet> --ids <file with interaction ids>
"""

import argparse
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ma_chunk.config import DATA
from ma_chunk.llm import _chat

POOLS = {ds: DATA / ds / "candidates.parquet" for ds in ("crag", "hotpotqa", "musique")}

INSTRUCTION = ("Given a context, and an answer, analyze each sentence in the answer and classify if the sentence "
               "can be attributed to the given context or not. Use only 'Yes' (1) or 'No' (0) as a binary "
               "classification. Output json with reason.")
EXAMPLE = """Example
question: What can you tell me about albert Albert Einstein?
context: Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics'.
answer sentences:
1. Albert Einstein born in 14 March 1879 was German-born theoretical physicist.
2. He published 4 papers in 1905.
output: {"classifications": [{"statement": "Albert Einstein born in 14 March 1879 was German-born theoretical physicist.", "reason": "The date of birth of Einstein is mentioned clearly in the context.", "attributed": 1}, {"statement": "He published 4 papers in 1905.", "reason": "There is no mention about papers he wrote in the given context.", "attributed": 0}]}"""


def sentences(reference: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(reference).strip()) if s.strip()]
    return parts or [str(reference).strip()]


def classify(args):
    question, context, sents = args
    listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    prompt = (f"{INSTRUCTION}\n\n{EXAMPLE}\n\nNow classify, in order, exactly the {len(sents)} answer sentence(s) below.\n"
              f"question: {question}\ncontext: {context}\nanswer sentences:\n{listing}\noutput:")
    try:
        content, _, _ = _chat(prompt, "You are a careful evaluator. Return only JSON.")
        m = re.search(r"\{.*\}", content, re.S)
        cls = json.loads(m.group(0))["classifications"]
        vals = [int(bool(int(c.get("attributed", 0)))) for c in cls][:len(sents)]
        return vals if len(vals) == len(sents) else [-1] * len(sents)
    except Exception:  # noqa: BLE001
        return [-1] * len(sents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(POOLS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-selections", nargs="*", default=None, help="label only chunks sent in these runs")
    ap.add_argument("--ids", default=None, help="file with interaction ids to restrict to")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    if args.dataset == "crag":
        pool = pd.read_parquet(POOLS["crag"], columns=["interaction_id", "chunk_id", "text"])
        gold = pd.read_parquet(DATA / "crag" / "questions.parquet")[
            ["interaction_id", "query", "clean_answer"]].drop_duplicates("interaction_id")
        pool = pool.merge(gold, on="interaction_id")
        pool["reference"] = pool["clean_answer"]
    else:
        pool = pd.read_parquet(POOLS[args.dataset], columns=["interaction_id", "chunk_id", "text", "query", "answers"])
        pool["reference"] = pool["answers"].apply(lambda a: " / ".join(list(a)))
    if args.ids:
        ids = set(Path(args.ids).read_text().split())
        pool = pool[pool["interaction_id"].isin(ids)]
    if args.only_selections:
        sel = pd.concat([pd.read_parquet(p, columns=["interaction_id", "chunk_ids"]) for p in args.only_selections])
        keep = {(q, int(c)) for q, ids in zip(sel["interaction_id"], sel["chunk_ids"]) for c in ids}
        pool = pool[[(q, int(c)) in keep for q, c in zip(pool["interaction_id"], pool["chunk_id"])]]
    pool = pool.reset_index(drop=True)
    sents = [sentences(r) for r in pool["reference"]]
    print(f"{args.dataset}: labelling {len(pool)} chunks of {pool['interaction_id'].nunique()} questions", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(classify, zip(pool["query"], pool["text"], sents)))
    out = pool[["interaction_id", "chunk_id"]].copy()
    out["n_stmt"] = [len(s) for s in sents]
    out["attr"] = res
    out["attr_mask"] = [sum(1 << i for i, v in enumerate(r) if v == 1) for r in res]
    out["attr_any"] = [int(any(v == 1 for v in r)) for r in res]
    out["failed"] = [int(any(v == -1 for v in r)) for r in res]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"done: attributed {out['attr_any'].mean():.3f}, failed {out['failed'].mean():.3f}, "
          f"multi-sentence refs {np.mean(out['n_stmt'] > 1):.3f}")


if __name__ == "__main__":
    main()

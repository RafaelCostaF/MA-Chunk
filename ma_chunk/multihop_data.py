"""Builds MA-Chunk candidate pools for multi-hop QA (HotpotQA, MuSiQue), with gold labels.

Same schema as candidates.parquet (+ useful, n_support), so the environment, training and
baselines run unchanged. Utility labels come from the datasets' gold annotations (no LLM):
    hotpotqa - distractor setting: the 10 paragraphs given with each question are the
               candidates; useful = paragraph title is in supporting_facts;
    musique  - closed corpus: every support paragraph of the train+dev questions forms the
               corpus; each question's candidates are its top-N paragraphs by BM25 over that
               corpus (gold paragraphs are NOT injected); useful = paragraph is one of the
               question's support paragraphs.
n_support = number of supporting paragraphs; the environment's utility is the covered fraction.

Usage (from the repo root):
    python -m ma_chunk.multihop_data --dataset hotpotqa --n 200 \
        --out data/hotpotqa/candidates.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ma_chunk.config import CROSS_ENCODER, E5, HF_CACHE, MINILM
from ma_chunk.text import _tokenize, compute_similarities, get_nlp


AGENTS = ["bm25", "spacy", "minilm", "e5", "ce"]
DS_CACHE = None  # Hugging Face default (set HF_HOME to move it)


def load(name, split):
    return load_dataset("RUC-NLPIR/FlashRAG_datasets", name, split=split, cache_dir=DS_CACHE)


def hotpot_questions(n, seed):
    d = load("hotpotqa", "dev").shuffle(seed=seed)
    out = []
    for x in d:
        if len(out) == n:
            break
        ctx, sup = x["metadata"]["context"], set(x["metadata"]["supporting_facts"]["title"])
        if not ctx["title"]:  # a few FlashRAG items come without their paragraphs
            continue
        paras = [f"{t}: {''.join(s)}" for t, s in zip(ctx["title"], ctx["sentences"])]
        useful = [int(t in sup) for t in ctx["title"]]
        out.append({"id": x["id"], "question": x["question"], "answers": x["golden_answers"],
                    "type": x["metadata"]["type"], "paragraphs": paras, "useful": useful,
                    "n_support": len(sup)})
    return out


def musique_questions(n, seed, n_candidates):
    corpus, key = [], {}
    for split in ("train", "dev"):
        for x in load("musique", split):
            for step in x["metadata"]["question_decomposition"]:
                p = step["support_paragraph"]
                text = f"{p['title']}: {p['paragraph_text']}"
                if text not in key:
                    key[text] = len(corpus)
                    corpus.append(text)
    print(f"musique corpus: {len(corpus)} paragraphs")
    bm25 = BM25Okapi([_tokenize(t) for t in corpus])

    d = load("musique", "dev").shuffle(seed=seed)
    out = []
    for x in d:
        if not x["metadata"]["answerable"]:
            continue
        gold = {key[f"{s['support_paragraph']['title']}: {s['support_paragraph']['paragraph_text']}"]
                for s in x["metadata"]["question_decomposition"]}
        top = np.argsort(-bm25.get_scores(_tokenize(x["question"])))[:n_candidates]
        out.append({"id": x["id"], "question": x["question"], "answers": x["golden_answers"],
                    "type": f"{len(gold)}hop", "paragraphs": [corpus[i] for i in top],
                    "useful": [int(i in gold) for i in top], "n_support": len(gold)})
        if len(out) == n:
            break
    return out


def build_pool(questions, dataset):
    encoder = SentenceTransformer(MINILM, device="cuda", cache_folder=HF_CACHE)
    e5 = SentenceTransformer(E5, device="cuda", cache_folder=HF_CACHE)
    reranker = CrossEncoder(CROSS_ENCODER, device="cuda", cache_folder=HF_CACHE)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    nlp = get_nlp()
    rows = []
    for q in questions:
        paras = q["paragraphs"]
        bm25 = np.array(BM25Okapi([_tokenize(p) for p in paras]).get_scores(_tokenize(q["question"])))
        bm25 = bm25 / bm25.max() if bm25.max() > 0 else np.zeros(len(paras))
        emb = encoder.encode(paras, normalize_embeddings=True)
        scores = {
            "bm25": bm25,
            "spacy": compute_similarities(nlp, q["question"], paras),
            "minilm": emb @ encoder.encode([q["question"]], normalize_embeddings=True)[0],
            "e5": e5.encode([f"passage: {p}" for p in paras], normalize_embeddings=True)
                  @ e5.encode([f"query: {q['question']}"], normalize_embeddings=True)[0],
            "ce": 1.0 / (1.0 + np.exp(-np.asarray(reranker.predict([(q["question"], p) for p in paras])))),
        }
        ranks = {}
        for a in AGENTS:
            ranks[a] = np.empty(len(paras), dtype=int)
            ranks[a][np.argsort(-scores[a], kind="stable")] = np.arange(len(paras))
        for c, text in enumerate(paras):
            rows.append({
                "interaction_id": q["id"], "domain": dataset, "question_type": q["type"],
                "query": q["question"], "answers": list(q["answers"]),
                "chunk_id": c, "text": text, "tokens": len(tokenizer.encode(text)),
                **{f"s_{a}": float(scores[a][c]) for a in AGENTS},
                **{f"rank_{a}": int(ranks[a][c]) for a in AGENTS},
                "n_page_chunks": len(paras), "emb": emb[c].astype(np.float32).tolist(),
                "useful": int(q["useful"][c]), "n_support": int(q["n_support"]),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hotpotqa", "musique"], required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-candidates", type=int, default=20, help="musique: BM25 candidates per question")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    questions = (hotpot_questions(args.n, args.seed) if args.dataset == "hotpotqa"
                 else musique_questions(args.n, args.seed, args.n_candidates))
    pool = build_pool(questions, args.dataset)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(args.out)
    per_q = pool.groupby("interaction_id")
    cover = (per_q["useful"].sum() / per_q["n_support"].first()).clip(upper=1)
    print(f"{args.dataset}: {per_q.ngroups} questions, {len(pool)} candidates, "
          f"{pool.useful.mean():.3f} useful, max reachable coverage {cover.mean():.3f}")


if __name__ == "__main__":
    main()

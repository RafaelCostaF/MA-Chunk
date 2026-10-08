"""Builds the per-query candidate pool for MA-Chunk (offline, run once).

Each agent is a retriever with its own view of the page: BM25 (sparse), spaCy
en_core_web_md cosine (static word vectors, the signal of a prior single-agent RL selector), MiniLM
(all-MiniLM-L6-v2, dense bi-encoder), E5 (intfloat/e5-base-v2, the dense retriever
family used by Search-R1) and a cross-encoder reranker (ms-marco-MiniLM-L-6-v2,
sigmoid of the logit). Every agent ranks ALL chunks of the page and proposes its own
top-M; the pool is the union of the top-M lists.

For each (query, pooled chunk) we store the three scores, the three page-level
ranks, the token count (tiktoken cl100k_base) and the MiniLM embedding (used by
the environment for the redundancy feature).

Usage (from the repo root):
    python -m ma_chunk.candidates --top-m 8 \
        --out data/crag/candidates.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from sentence_transformers import CrossEncoder, SentenceTransformer

from ma_chunk.config import CROSS_ENCODER, DATA, E5, HF_CACHE, MINILM
from ma_chunk.text import chunks_and_scores_for_text

AGENTS = ["bm25", "spacy", "minilm", "e5", "ce"]


def build(dataset: pd.DataFrame, top_m: int) -> pd.DataFrame:
    encoder = SentenceTransformer(MINILM, device="cuda", cache_folder=HF_CACHE)
    e5 = SentenceTransformer(E5, device="cuda", cache_folder=HF_CACHE)
    reranker = CrossEncoder(CROSS_ENCODER, device="cuda", cache_folder=HF_CACHE)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    rows = []
    for i, r in enumerate(dataset.itertuples()):
        chunks, spacy_sim, bm25 = chunks_and_scores_for_text(r.query, r.page_results_text)
        if not chunks:
            continue
        query_emb = encoder.encode([r.query], normalize_embeddings=True)[0]
        chunk_emb = encoder.encode(chunks, batch_size=256, normalize_embeddings=True)
        minilm = chunk_emb @ query_emb
        e5_query = e5.encode([f"query: {r.query}"], normalize_embeddings=True)[0]
        e5_chunks = e5.encode([f"passage: {c}" for c in chunks], batch_size=256, normalize_embeddings=True)
        ce_logits = reranker.predict([(r.query, c) for c in chunks], batch_size=512)

        scores = {"bm25": np.asarray(bm25), "spacy": np.asarray(spacy_sim), "minilm": minilm,
                  "e5": e5_chunks @ e5_query, "ce": 1.0 / (1.0 + np.exp(-np.asarray(ce_logits)))}
        ranks = {a: np.empty(len(chunks), dtype=int) for a in AGENTS}
        pool = set()
        for a in AGENTS:
            order = np.argsort(-scores[a], kind="stable")
            ranks[a][order] = np.arange(len(chunks))
            pool.update(order[:top_m].tolist())

        for c in sorted(pool):
            rows.append({
                "interaction_id": r.interaction_id, "domain": r.domain, "question_type": r.question_type,
                "chunk_id": int(c), "text": chunks[c], "tokens": len(tokenizer.encode(chunks[c])),
                **{f"s_{a}": float(scores[a][c]) for a in AGENTS},
                **{f"rank_{a}": int(ranks[a][c]) for a in AGENTS},
                "n_page_chunks": len(chunks), "emb": chunk_emb[c].astype(np.float32).tolist(),
            })
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(dataset)} queries")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATA / "crag" / "crag_sample.parquet"))
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = pd.read_parquet(args.dataset)
    pool = build(dataset, args.top_m)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(args.out)
    per_query = pool.groupby("interaction_id").size()
    print(f"{len(pool)} pooled chunks for {per_query.size} queries "
          f"(mean {per_query.mean():.1f} per query, top-{args.top_m} per agent)")


if __name__ == "__main__":
    main()

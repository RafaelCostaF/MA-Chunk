"""Text utilities shared by the candidate builders: chunking and the two cheap retrieval signals
(spaCy static word vectors and BM25). Results are cached on disk (joblib) because CRAG pages have
about 1,450 chunks each."""

from __future__ import annotations

import re

import numpy as np
from joblib import Memory
from rank_bm25 import BM25Okapi

from ma_chunk.config import CACHE, CHUNK_MAX_LENGTH

_disk_cache = Memory(location=str(CACHE / "text_scores"), verbose=0)
_NLP = None


def get_nlp():
    """spaCy en_core_web_md with every pipe disabled: Doc.vector is the mean of static word vectors,
    which does not need the tagger/parser (about 25x faster on long pages)."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_md")
        _NLP.disable_pipes(_NLP.pipe_names)
    return _NLP


def compute_similarities(nlp, query: str, chunks: list[str]) -> np.ndarray:
    """Cosine similarity between the mean word vectors of the query and of each chunk."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    query_doc = nlp(query)
    if not query_doc.vector_norm:
        return np.zeros(len(chunks), dtype=np.float32)
    docs = nlp.pipe(chunks, batch_size=200)
    return np.array([query_doc.similarity(d) if d.vector_norm else 0.0 for d in docs], dtype=np.float32)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def compute_bm25_scores(query: str, chunks: list[str]) -> np.ndarray:
    """BM25 between the query and each chunk, normalized by the page's maximum (0..1)."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    bm25 = BM25Okapi([_tokenize(c) for c in chunks])
    scores = np.array(bm25.get_scores(_tokenize(query)), dtype=np.float32)
    max_score = scores.max()
    if max_score <= 0:
        return np.zeros(len(chunks), dtype=np.float32)
    return scores / max_score


def chunk_text(text: str, max_length: int = CHUNK_MAX_LENGTH) -> list[str]:
    """Greedy sentence packing into chunks of at most max_length characters."""
    sentences = str(text).split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_length:
            current += sentence + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks


@_disk_cache.cache
def chunks_and_scores_for_text(query: str, page_text: str):
    """Chunks a page and scores every chunk with spaCy cosine and BM25 (disk-cached)."""
    chunks = chunk_text(page_text)
    return chunks, compute_similarities(get_nlp(), query, chunks), compute_bm25_scores(query, chunks)

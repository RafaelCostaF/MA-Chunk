"""Single LLM client (reader, judge and labellers) with a local SQLite cache.

Identical prompts are answered from the cache, so re-running an evaluation costs nothing and is
deterministic. RAGAS calls go through the same cache (see ragas_metrics.py). Reasoning models
(gpt-5*, o-series) only accept the default temperature and spend tokens on hidden reasoning, so
they get a larger token budget and `reasoning_effort`.
"""

from __future__ import annotations

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI

from ma_chunk.config import CACHE, OPENAI_MODEL, OPENAI_REASONING_EFFORT, is_reasoning_model, openai_api_key

LLM_CACHE_DIR = CACHE / "llm"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class _ConcurrentSafeSQLiteCache(SQLiteCache):
    """SQLiteCache.update is select-then-insert; when two parallel jobs send the same prompt the
    second insert raises IntegrityError although the entry is already stored, so it is ignored."""

    def update(self, prompt, llm_string, return_val):
        from sqlalchemy.exc import IntegrityError

        try:
            super().update(prompt, llm_string, return_val)
        except IntegrityError:
            pass


set_llm_cache(_ConcurrentSafeSQLiteCache(database_path=str(LLM_CACHE_DIR / "chat_cache.sqlite")))

_llm = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        if is_reasoning_model(OPENAI_MODEL):
            _llm = ChatOpenAI(model=OPENAI_MODEL, api_key=openai_api_key(), max_tokens=4000,
                              reasoning_effort=OPENAI_REASONING_EFFORT)
        else:
            _llm = ChatOpenAI(model=OPENAI_MODEL, api_key=openai_api_key(), max_tokens=500, temperature=0.2)
    return _llm


def _chat(prompt: str, system: str) -> tuple[str, int, int]:
    response = get_llm().invoke([{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    usage = response.usage_metadata or {}
    return (response.content or "").strip(), usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def get_response_from_llm(query: str, chunks) -> tuple[str, int, int]:
    """The frozen reader: answers only from the chunks it is given (same prompt for every method)."""
    prompt = (
        "You are an intelligent assistant that answers questions exclusively based on the "
        "information provided below.\n\n"
        f"User query:\n{query}\n\n"
        f"Available sources (chunks):\n{chunks}\n\n"
        "Respond clearly, objectively, and only using the sources. Return ONLY the answer, "
        "without any additional explanations or context.\n\n"
        "If there's no answer in the sources, return an empty string.\n\n"
        "Answer:"
    )
    try:
        return _chat(prompt, "You are a helpful and concise assistant.")
    except Exception as e:  # noqa: BLE001
        print(f"[LLM error] reader: {e}")
        return "", 0, 0


def cached_embeddings(embeddings):
    """Wraps a langchain embeddings object with an on-disk cache (used by RAGAS answer relevancy)."""
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain.storage import LocalFileStore

    store = LocalFileStore(str(LLM_CACHE_DIR / "embeddings"))
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=getattr(embeddings, "model", "openai-embeddings"), query_embedding_cache=True)

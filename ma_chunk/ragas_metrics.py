"""RAGAS (0.2.x) metrics with the common judge: faithfulness, answer relevancy, answer correctness,
context precision and context recall. LLM and embedding calls go through the local cache."""

import ast

import pandas as pd

from ma_chunk.config import OPENAI_MODEL, OPENAI_REASONING_EFFORT, is_reasoning_model, openai_api_key
from ma_chunk.llm import cached_embeddings


def as_list(x):
    """Contexts may arrive as lists, numpy arrays or their string repr; RAGAS needs a list of strings."""
    if x is None:
        return []
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return [x]
    return [str(c) for c in list(x)]


class _IgnoreTemperatureChatOpenAI:
    """Mixin that ignores attempts to set 'temperature' - needed only for
    reasoning models (o1/o3/gpt-5*), which accept only the default
    temperature. RAGAS forces `langchain_llm.temperature = 1e-8` before every
    call (ragas/llms/base.py, LangchainLLMWrapper.generate) to try to make
    the judge deterministic - this breaks with those models (observed: a 400
    BadRequestError on every call without this mixin). "Normal" models such
    as gpt-4.1-nano don't need this."""

    def __setattr__(self, name, value):
        if name == "temperature":
            value = None
        super().__setattr__(name, value)


def _make_evaluator_chat_openai(model: str, api_key: str):
    from langchain_openai import ChatOpenAI

    if is_reasoning_model(model):
        cls = type("IgnoreTemperatureChatOpenAI", (_IgnoreTemperatureChatOpenAI, ChatOpenAI), {})
        return cls(model=model, api_key=api_key, reasoning_effort=OPENAI_REASONING_EFFORT)
    return ChatOpenAI(model=model, api_key=api_key)


def compute_ragas_metrics(df: pd.DataFrame) -> pd.DataFrame:
    from datasets import Dataset
    from langchain_openai import OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness

    evaluator_llm = LangchainLLMWrapper(_make_evaluator_chat_openai(OPENAI_MODEL, openai_api_key()))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        cached_embeddings(OpenAIEmbeddings(api_key=openai_api_key()))
    )

    # RAGAS 0.2 column names: the evaluated answer must be "response" and the gold answer
    # "reference" (mapping the gold answer to "answer" would make RAGAS evaluate the gold itself).
    ragas_df = df[["clean_answer", "llm_response", "chunks_selected", "query"]].copy()
    ragas_df["chunks_selected"] = ragas_df["chunks_selected"].apply(as_list)
    ragas_df["llm_response"] = ragas_df["llm_response"].fillna("")
    ragas_df = ragas_df.rename(columns={
        "clean_answer": "reference",
        "llm_response": "response",
        "chunks_selected": "retrieved_contexts",
        "query": "user_input",
    })

    dataset = Dataset.from_pandas(ragas_df)
    print(f"Computing RAGAS metrics ({OPENAI_MODEL}) on {len(ragas_df)} rows...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    ragas_out = result.to_pandas().reset_index(drop=True)
    ragas_out["domain"] = df["domain"].values
    ragas_out["algo"] = df["algo"].values
    return ragas_out

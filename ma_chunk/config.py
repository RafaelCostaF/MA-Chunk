"""Central configuration: paths, LLM settings and model names.

Every location can be overridden with an environment variable (or a `.env` file at the
repository root), so nothing in the code depends on a particular machine:

    MA_CHUNK_DATA      data directory (candidate pools, labels, CRAG sample)   default: <repo>/data
    MA_CHUNK_RESULTS   results directory (runs, evaluations, analyses)         default: <repo>/results
    MA_CHUNK_CACHE     cache directory (LLM/embedding cache, text scores)      default: <repo>/.cache
    HF_HOME            Hugging Face cache (models and datasets)                default: Hugging Face's own
    OPENAI_API_KEY     only needed for labelling, reading/judging and RAGAS (never for training)
    OPENAI_MODEL       reader / judge / labeller model                         default: gpt-5-nano
    OPENAI_REASONING_EFFORT                                                    default: minimal
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

REPO = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(REPO / ".env")

DATA = Path(os.environ.get("MA_CHUNK_DATA", REPO / "data"))
RESULTS = Path(os.environ.get("MA_CHUNK_RESULTS", REPO / "results"))
CACHE = Path(os.environ.get("MA_CHUNK_CACHE", REPO / ".cache"))
HF_CACHE = os.environ.get("HF_HOME")  # None -> Hugging Face default location

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-nano")
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "minimal")
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

CHUNK_MAX_LENGTH = 500  # characters per chunk (CRAG pages are split by sentences up to this length)

# Retriever models of the five agents (bm25 and spaCy need no download besides en_core_web_md).
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
E5 = "intfloat/e5-base-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def is_reasoning_model(model: str = OPENAI_MODEL) -> bool:
    return model.lower().startswith(_REASONING_MODEL_PREFIXES)


def openai_api_key() -> str:
    """Returns the API key, failing with a clear message only when an LLM is actually needed."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in <repo>/.env (see .env.example) or export it. "
                           "It is only needed for labelling, end-task evaluation and RAGAS, not for training.")
    return key

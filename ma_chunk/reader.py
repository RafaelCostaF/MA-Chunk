"""Common reader and CRAG-style judge used to evaluate every method (including baselines).

The reader (ma_chunk.llm.get_response_from_llm) answers from the chunks it is given; the judge
labels the answer correct / incorrect / missing against the gold answer, with a deterministic
refusal rule so that "the sources do not mention X" is always counted as missing. CRAG score =
%correct - %incorrect. SQuAD-style token F1 and exact match are reported as well.
"""

import re
import string
from collections import Counter

import pandas as pd

from ma_chunk.llm import _chat, get_response_from_llm

JUDGE_PROMPT = """You are grading an answer to a question against the ground-truth answer.

Question: {query}
Ground-truth answer: {gold}
Candidate answer: {response}

Classify the candidate answer:
- "correct": it states the ground-truth answer (paraphrases and extra harmless detail are fine);
- "missing": it is empty, says it does not know, or says the information is not available /
  not provided / not in the sources (e.g. "The provided sources do not contain the answer",
  "The sources discuss X but do not mention Y") - even if it adds unrelated details;
- "incorrect": anything else (wrong, contradictory or hallucinated).
If the ground truth says the question has a false premise / is invalid, the candidate is
"correct" only if it points out the false premise.

Return only JSON: {{"verdict": "correct" | "incorrect" | "missing"}}"""


def normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def clean_text(response) -> str:
    """The reader is told to return an empty string when there is no answer; it sometimes
    returns the literal quotes instead."""
    text = str(response or "").strip()
    return "" if text in ('""', "''") else text


# Deterministic refusal detection (the LLM judge with minimal reasoning sometimes
# labels "the sources do not provide X" as incorrect instead of missing).
REFUSAL = re.compile(
    r"\b(do(es)? not|don't|doesn't|did not|cannot|can't|unable to)\s+"
    r"(provide|contain|mention|include|specify|state|give|list|say|find|determine|answer)\b"
    r"|\bno (specific |relevant |explicit )?(information|data|mention|answer|details?)\b"
    r"|\bnot (explicitly |directly )?(provided|available|mentioned|specified|stated|included|found|given)\b",
    re.IGNORECASE,
)


def is_refusal(response: str) -> bool:
    return bool(REFUSAL.search(str(response)))


def judge(query: str, gold: str, response: str) -> str:
    if not str(response).strip() or is_refusal(response):
        return "missing"
    try:
        content, _, _ = _chat(JUDGE_PROMPT.format(query=query, gold=gold, response=response),
                              "You are a strict and fair grader.")
        match = re.search(r'"verdict"\s*:\s*"(correct|incorrect|missing)"', content)
        return match.group(1) if match else "invalid"
    except Exception as e:  # noqa: BLE001
        print(f"[judge error] {e}")
        return "invalid"


def run_one(row) -> dict:
    response, tokens_in, tokens_out = get_response_from_llm(row["query"], row["chunks"])
    response = clean_text(response)
    verdict = judge(row["query"], row["clean_answer"], response)
    return {"response": response, "input_tokens": tokens_in, "output_tokens": tokens_out, "verdict": verdict}


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby("condition")
    summary = pd.DataFrame({
        "n": g.size(),
        "correct": g["verdict"].apply(lambda v: (v == "correct").mean()),
        "incorrect": g["verdict"].apply(lambda v: (v == "incorrect").mean()),
        "missing": g["verdict"].apply(lambda v: (v == "missing").mean()),
        "invalid": g["verdict"].apply(lambda v: (v == "invalid").mean()),
        "token_f1": g["f1"].mean(),
        "em": g["em"].mean(),
        "n_chunks": g["n_chunks"].mean(),
        "input_tokens": g["input_tokens"].mean(),
    })
    summary.insert(1, "crag_score", summary["correct"] - summary["incorrect"])
    return summary.sort_values("crag_score", ascending=False)

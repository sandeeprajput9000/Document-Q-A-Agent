"""Benchmark metrics, strict Watsonx judging, and retrieval heuristic scoring."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import invoke_with_retry

logger = logging.getLogger(__name__)
_TERMS = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}")


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load final benchmark questions and validate required fields."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("Benchmark JSON must contain a list")
    required = {"id", "question", "ground_truth_answer", "source_file", "source_page"}
    seen_ids: set[str] = set()
    for number, item in enumerate(questions, start=1):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Question {number} is missing fields: {sorted(missing)}")
        extra = item.keys() - required
        if extra:
            raise ValueError(f"Question {number} has unexpected fields: {sorted(extra)}")
        question_id = str(item["id"])
        if question_id in seen_ids:
            raise ValueError(f"Duplicate benchmark question id: {question_id}")
        seen_ids.add(question_id)
        if not all(str(item[field]).strip() for field in ("id", "question", "ground_truth_answer", "source_file")):
            raise ValueError(f"Question {number} contains an empty required value")
        if int(item["source_page"]) < 1:
            raise ValueError(f"Question {number} source_page must be 1-based and positive")
    return questions


def retrieval_heuristic(
    question: dict[str, Any], answer: str, sources: list[dict[str, Any]]
) -> bool:
    """Require a source-page hit and meaningful ground-truth term overlap."""
    expected_file = str(question["source_file"]).casefold()
    expected_page = int(question["source_page"])
    page_hit = any(
        str(source.get("source_file", "")).casefold() == expected_file
        and int(source.get("source_page", -1)) == expected_page
        for source in sources
    )
    expected_terms = {term.casefold() for term in _TERMS.findall(str(question["ground_truth_answer"]))}
    if not expected_terms:
        return page_hit
    answer_terms = {term.casefold() for term in _TERMS.findall(answer)}
    overlap = len(expected_terms & answer_terms) / len(expected_terms)
    return page_hit and overlap >= 0.35


def _llm_judge(llm: Any, question: dict[str, Any], answer: str) -> bool | None:
    """Ask Watsonx for a strict binary answer comparison."""
    try:
        result = invoke_with_retry(
            lambda: llm.invoke(
                [
                    SystemMessage(
                        content="Compare an answer with the ground truth. Reply with exactly CORRECT or INCORRECT."
                    ),
                    HumanMessage(
                        content=(
                            f"Question: {question['question']}\n"
                            f"Ground truth: {question['ground_truth_answer']}\n"
                            f"Candidate answer: {answer}"
                        )
                    ),
                ]
            )
        )
        verdict = str(result.content).strip().upper()
        if verdict.startswith("CORRECT"):
            return True
        if verdict.startswith("INCORRECT"):
            return False
        logger.warning("Judge returned an unrecognized verdict: %s", verdict)
    except Exception:
        logger.exception("LLM-as-judge failed for question %s", question.get("id"))
    return None


def evaluate_questions(
    pipeline: Any,
    questions: list[dict[str, Any]],
    *,
    use_llm_judge: bool | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a question set and return aggregate and per-question metrics."""
    if len(questions) != 50:
        raise ValueError(
            f"Expected exactly 50 manually verified benchmark questions, got {len(questions)}"
        )
    judge_enabled = pipeline.watsonx_enabled if use_llm_judge is None else use_llm_judge
    rows: list[dict[str, Any]] = []
    for question in questions:
        output = pipeline.ask(str(question["question"]))
        heuristic = retrieval_heuristic(question, output["answer"], output["sources"])
        judged = _llm_judge(pipeline.llm, question, output["answer"]) if judge_enabled else None
        rows.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "heuristic_correct": heuristic,
                "llm_judge_correct": judged,
                "latency_s": float(output["latency_s"]),
                "answer": output["answer"],
                "expected_source_file": question["source_file"],
                "expected_source_page": int(question["source_page"]),
                "retrieved_sources": json.dumps(output["sources"], ensure_ascii=False),
            }
        )

    latencies = np.asarray([row["latency_s"] for row in rows], dtype=np.float64)
    heuristic_scores = np.asarray([row["heuristic_correct"] for row in rows], dtype=np.float64)
    valid_judgments = [row["llm_judge_correct"] for row in rows if row["llm_judge_correct"] is not None]
    judged_scores = np.asarray(valid_judgments, dtype=np.float64)
    metrics = {
        "question_count": len(rows),
        "heuristic_accuracy": float(np.mean(heuristic_scores)),
        "heuristic_accuracy_std": float(np.std(heuristic_scores, ddof=1)) if len(rows) > 1 else 0.0,
        "llm_judge_accuracy": float(np.mean(judged_scores)) if len(valid_judgments) else None,
        "llm_judge_accuracy_std": float(np.std(judged_scores, ddof=1)) if len(valid_judgments) > 1 else (0.0 if valid_judgments else None),
        "llm_judged_count": len(valid_judgments),
        "latency_mean_s": float(np.mean(latencies)),
        "latency_std_s": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
    }
    return metrics, rows


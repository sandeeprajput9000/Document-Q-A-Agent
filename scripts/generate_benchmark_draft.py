"""Draft source-linked Q&A items with Watsonx for manual review."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract one JSON object from a model response."""
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    match = re.search(r"\{.*\}", candidate, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group())
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/benchmark/questions_draft.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    random.seed(args.seed)
    pipeline = RAGPipeline(config_path=args.config)
    if not pipeline.watsonx_enabled:
        logger.error("Watsonx credentials are required to draft questions; set .env and retry")
        return 2
    pipeline.build_index()
    assert pipeline.vector_store is not None
    chunks = list(pipeline.vector_store.documents)
    if not chunks:
        logger.error("The FAISS index has no chunks")
        return 2
    random.shuffle(chunks)
    draft: list[dict[str, Any]] = []
    attempts = 0
    while len(draft) < args.count and attempts < args.count * 3:
        document = chunks[attempts % len(chunks)]
        attempts += 1
        response = pipeline.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Create one concise factual question and its answer that can be answered "
                        "only from the supplied passage. Return a JSON object with exactly the "
                        "keys question and ground_truth_answer. Do not add markdown."
                    )
                ),
                HumanMessage(content=document.page_content[:6000]),
            ]
        )
        pair = parse_json_object(str(response.content))
        if not pair or not pair.get("question") or not pair.get("ground_truth_answer"):
            logger.warning("Could not parse draft %d from model output", attempts)
            continue
        source_file = str(document.metadata.get("source", "unknown"))
        source_page = int(document.metadata.get("page", 0))
        draft.append(
            {
                "id": f"draft-{len(draft) + 1:03d}",
                "question": str(pair["question"]).strip(),
                "ground_truth_answer": str(pair["ground_truth_answer"]).strip(),
                "source_file": source_file,
                "source_page": source_page,
                "manual_review_required": True,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d unverified draft questions to %s", len(draft), args.output)
    if len(draft) != args.count:
        logger.warning("Requested %d drafts but generated %d", args.count, len(draft))
        return 1
    logger.warning("All draft answers must be verified against the cited page before benchmarking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""IBM watsonx.ai chat model setup and a small deterministic offline fallback."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)
_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "based", "does", "for", "from",
    "how", "into", "its", "that", "the", "their", "these", "this", "what", "when",
    "where", "which", "with", "would",
}


def _normalize_watsonx_tool_arguments(response: dict[str, Any]) -> dict[str, Any]:
    """Unwrap Granite tool arguments when the Chat API JSON-encodes them twice.

    Some watsonx Chat API responses contain a JSON string inside the standard
    ``function.arguments`` JSON string. LangChain expects that field to decode
    directly to an object, so normalize the response before ChatWatsonx parses it.
    """
    normalized = copy.deepcopy(response)
    for choice in normalized.get("choices", []):
        message = choice.get("message", {})
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            candidate = arguments
            for _ in range(2):
                try:
                    decoded = json.loads(candidate)
                except json.JSONDecodeError:
                    break
                if isinstance(decoded, dict):
                    if candidate != arguments:
                        function["arguments"] = json.dumps(decoded, ensure_ascii=False)
                    break
                if isinstance(decoded, str):
                    candidate = decoded
                    continue
                break
    return normalized


class FakeLLM(BaseChatModel):
    """Offline chat model for tests and local retrieval-only evaluation.

    It uses the first retrieved passage in a grounded prompt. It does not claim
    that a model judged the answer and should not be used as a benchmark judge.
    """

    fixed_response: str | None = None

    @property
    def _llm_type(self) -> str:
        return "document-qa-fake"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"fixed_response": self.fixed_response}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.fixed_response is not None:
            content = self.fixed_response
        else:
            prompt = "\n".join(str(message.content) for message in messages)
            question_match = re.search(r"Question:\s*(.+?)(?:\n\nRetrieved context:|$)", prompt, re.S)
            question_terms = {
                token.casefold()
                for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", question_match.group(1) if question_match else "")
                if token.casefold() not in _STOP_WORDS
            }
            contexts = re.findall(
                r"\[Source: ([^\]]+)\]\s*\n(.+?)(?=\n\n\[Source:|$)", prompt, re.S
            )
            if not contexts:
                content = "not found in documents"
            else:
                candidates: list[tuple[int, str, str]] = []
                for source, passage in contexts:
                    sentences = re.split(r"(?<=[.!?])\s+|\n+", passage)
                    for sentence_index, sentence in enumerate(sentences):
                        sentence = " ".join(sentence.split()).strip()
                        if len(sentence) < 25:
                            continue
                        if sentence[0].islower() and sentence_index > 0:
                            sentence = f"{sentences[sentence_index - 1]} {sentence}".strip()
                        terms = {
                            token.casefold()
                            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", sentence)
                        }
                        overlap = len(question_terms & terms)
                        if overlap:
                            candidates.append((overlap, source, sentence[:500]))
                if candidates:
                    _, source, excerpt = max(candidates, key=lambda item: (item[0], -len(item[2])))
                else:
                    source, passage = contexts[0]
                    excerpt = " ".join(passage.split())[:350]
                content = f"{excerpt} [{source}]"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def watsonx_credentials_present() -> bool:
    """Whether all required IBM watsonx.ai values are configured."""
    load_dotenv(override=False)
    names = ("WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID")
    return all(os.getenv(name, "").strip() for name in names)


def invoke_with_retry(call: Any, attempts: int = 3, base_seconds: float = 1.0) -> Any:
    """Retry a remote invocation with bounded exponential backoff."""
    if attempts <= 0 or base_seconds < 0:
        raise ValueError("attempts must be positive and base_seconds cannot be negative")
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt + 1 >= attempts:
                raise
            delay = base_seconds * (2**attempt)
            logger.warning("Model request failed; retrying in %.2f seconds", delay, exc_info=True)
            time.sleep(delay)
    raise RuntimeError("unreachable retry state")


def create_chat_model(
    model_id: str = "ibm/granite-4-h-small",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[Any, bool]:
    """Create ChatWatsonx when credentials exist, otherwise return FakeLLM."""
    load_dotenv(override=False)
    if not watsonx_credentials_present():
        logger.warning("watsonx credentials are incomplete; using the offline FakeLLM")
        return FakeLLM(), False

    try:
        from langchain_ibm import ChatWatsonx

        class _CompatibleChatWatsonx(ChatWatsonx):
            """Normalize watsonx tool arguments before LangChain message parsing."""

            def _create_chat_result(
                self,
                response: dict[str, Any],
                generation_info: dict[str, Any] | None = None,
            ) -> ChatResult:
                return super()._create_chat_result(
                    _normalize_watsonx_tool_arguments(response), generation_info
                )

        model = _CompatibleChatWatsonx(
            model_id=os.getenv("WATSONX_MODEL_ID", model_id),
            url=os.environ["WATSONX_URL"],
            apikey=os.environ["WATSONX_APIKEY"],
            project_id=os.environ["WATSONX_PROJECT_ID"],
            params={"max_new_tokens": max_new_tokens, "temperature": temperature},
        )
        return model, True
    except Exception:
        logger.exception("Could not initialize ChatWatsonx")
        raise


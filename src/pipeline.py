"""End-to-end RAG pipeline for building indexes and asking questions."""

from __future__ import annotations

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent import SYSTEM_PROMPT, create_document_agent
from src.chunking import chunk_documents
from src.config import Settings, load_settings
from src.embeddings import HuggingFaceEmbedder
from src.ingestion import load_pdf_documents
from src.llm import FakeLLM, create_chat_model, invoke_with_retry
from src.retriever import Retriever
from src.tools import create_document_tools
from src.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Build/load a configured index and return grounded answers with citations."""

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        *,
        settings: Settings | None = None,
        embedder: Any | None = None,
        llm: Any | None = None,
        vector_store: FAISSVectorStore | None = None,
        watsonx_enabled: bool | None = None,
    ) -> None:
        self.settings = settings or load_settings(config_path)
        self.embedder = embedder
        if llm is None:
            self.llm, self.watsonx_enabled = create_chat_model(
                model_id="ibm/granite-3-8b-instruct",
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
            )
        else:
            self.llm = llm
            self.watsonx_enabled = watsonx_enabled if watsonx_enabled is not None else not isinstance(llm, FakeLLM)
        self.vector_store = vector_store
        self.retriever: Retriever | None = None
        self._agent: Any | None = None

    @property
    def index_directory(self) -> Path:
        """Return a cache directory keyed by chunking and embedding settings."""
        config_key = FAISSVectorStore.config_key(
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            self.settings.embedding_model,
            self.settings.embedding_max_length,
        )
        pdf_dir = self.settings.path("pdf_dir")
        corpus = "|".join(
            f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
            for path in sorted(pdf_dir.glob("*.pdf"))
        ) if pdf_dir.exists() else ""
        key = hashlib.sha256(f"{config_key}|{corpus}".encode("utf-8")).hexdigest()[:16]
        return self.settings.path("index_dir") / key

    def _get_embedder(self) -> Any:
        if self.embedder is None:
            self.embedder = HuggingFaceEmbedder(
                model_name=self.settings.embedding_model,
                batch_size=self.settings.embedding_batch_size,
                max_length=self.settings.embedding_max_length,
                cache_dir=self.settings.path("model_cache_dir"),
            )
        return self.embedder

    def build_index(self, force: bool = False) -> dict[str, int | str]:
        """Load PDFs, chunk pages, embed batches, and save a FAISS index."""
        if self.index_directory.exists() and not force:
            try:
                self.vector_store = FAISSVectorStore.load(self.index_directory)
                return {"pages": 0, "chunks": len(self.vector_store.documents), "index": str(self.index_directory)}
            except FileNotFoundError:
                logger.info("Found incomplete cached index at %s; rebuilding", self.index_directory)
        pages = load_pdf_documents(self.settings.path("pdf_dir"))
        if not pages:
            raise ValueError(f"No readable PDF pages found in {self.settings.path('pdf_dir')}")
        chunks = chunk_documents(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        embedder = self._get_embedder()
        vectors = embedder.embed_documents([chunk.page_content for chunk in chunks])
        self.vector_store = FAISSVectorStore.from_embeddings(vectors, chunks)
        self.index_directory.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(self.index_directory)
        self.retriever = None
        logger.info("Indexed %d pages into %d chunks", len(pages), len(chunks))
        return {"pages": len(pages), "chunks": len(chunks), "index": str(self.index_directory)}

    def _ensure_ready(self) -> None:
        if self.vector_store is None:
            if self.index_directory.exists():
                self.vector_store = FAISSVectorStore.load(self.index_directory)
            else:
                self.build_index()
        if self.retriever is None:
            self.retriever = Retriever(self.vector_store, self._get_embedder(), self.settings.top_k)

    @staticmethod
    def _collect_sources(messages: list[Any]) -> list[dict[str, Any]]:
        sources: dict[tuple[str, int], dict[str, Any]] = {}
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            try:
                payload = json.loads(str(message.content))
            except (TypeError, json.JSONDecodeError):
                continue
            for item in payload.get("results", []):
                name = str(item.get("source_file", "unknown"))
                page = int(item.get("source_page", 0))
                sources[(name, page)] = {"source_file": name, "source_page": page}
        return list(sources.values())

    def _ask_offline(self, question: str) -> tuple[str, list[dict[str, Any]], list[str]]:
        assert self.retriever is not None
        hits = self.retriever.search(question, self.settings.top_k)
        if not hits:
            return "not found in documents", [], []
        context = "\n\n".join(
            f"[Source: {hit.source['source_file']}, page {hit.source['source_page']}]\n{hit.document.page_content}"
            for hit in hits
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nRetrieved context:\n{context}"),
        ]
        response = invoke_with_retry(
            lambda: self.llm.invoke(messages),
            attempts=self.settings.retry_attempts,
            base_seconds=self.settings.retry_base_seconds,
        )
        answer = str(response.content).strip()
        sources = [
            {"source_file": hit.source["source_file"], "source_page": hit.source["source_page"]}
            for hit in hits
        ]
        return answer or "not found in documents", sources, []

    def ask(self, question: str) -> dict[str, Any]:
        """Answer a question and return answer, sources, latency, and tool calls."""
        if not question.strip():
            raise ValueError("question must not be empty")
        self._ensure_ready()
        start = time.perf_counter()
        if self.watsonx_enabled:
            assert self.retriever is not None
            tools = create_document_tools(self.retriever)
            if self._agent is None:
                self._agent = create_document_agent(self.llm, tools)
            state = invoke_with_retry(
                lambda: self._agent.invoke({"messages": [HumanMessage(content=question)]}),
                attempts=self.settings.retry_attempts,
                base_seconds=self.settings.retry_base_seconds,
            )
            messages = state.get("messages", [])
            answer_messages = [
                message
                for message in messages
                if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
            ]
            answer = str(answer_messages[-1].content).strip() if answer_messages else "not found in documents"
            sources = self._collect_sources(messages)
            tool_calls = [
                str(call.get("name"))
                for message in messages
                if isinstance(message, AIMessage)
                for call in (getattr(message, "tool_calls", None) or [])
            ]
            if not sources and answer != "not found in documents" and "get_document_list" not in tool_calls:
                logger.warning("Agent answered without a source-bearing tool result; discarding answer")
                answer = "not found in documents"
        else:
            answer, sources, tool_calls = self._ask_offline(question)
        latency = time.perf_counter() - start
        return {
            "answer": answer,
            "sources": sources,
            "latency_s": latency,
            "tool_calls": tool_calls,
        }


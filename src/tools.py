"""LangChain tools that expose indexed document search and summaries."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from src.retriever import Retriever


def create_document_tools(retriever: Retriever) -> list[Any]:
    """Build search, document-list, and per-document summary tools."""

    @tool
    def search_documents(query: str, k: int | None = None) -> str:
        """Search indexed PDF passages for a query. Returns text with source pages."""
        hits = retriever.search(query, k)
        results = [
            {
                **hit.source,
                "text": hit.document.page_content,
            }
            for hit in hits
        ]
        return json.dumps({"results": results}, ensure_ascii=False)

    @tool
    def get_document_list() -> str:
        """List PDF filenames available in the document index."""
        return json.dumps({"documents": retriever.document_list()}, ensure_ascii=False)

    @tool
    def summarize_document(filename: str) -> str:
        """Return indexed passages from a named PDF to support a source-grounded summary."""
        documents = retriever.chunks_for_document(filename)
        results = [
            {
                "source_file": str(document.metadata.get("source", "unknown")),
                "source_page": int(document.metadata.get("page", 0)),
                "text": document.page_content,
            }
            for document in documents[:12]
        ]
        return json.dumps({"results": results}, ensure_ascii=False)

    return [search_documents, get_document_list, summarize_document]


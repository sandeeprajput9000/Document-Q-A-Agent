"""Top-k retrieval and source formatting over a FAISS vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True)
class SearchHit:
    """A retrieved text chunk and cosine-like score."""

    document: Document
    score: float

    @property
    def source(self) -> dict[str, Any]:
        """Expose source and page in the public response shape."""
        return {
            "source_file": str(self.document.metadata.get("source", "unknown")),
            "source_page": int(self.document.metadata.get("page", 0)),
            "score": self.score,
        }


class Retriever:
    """Embed queries and rank chunks from the loaded vector store."""

    def __init__(self, vector_store: Any, embedder: Any, top_k: int = 2) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def search(self, query: str, k: int | None = None) -> list[SearchHit]:
        """Return top-k chunks and scores for a natural-language query."""
        if not query.strip():
            return []
        count = k or self.top_k
        return [
            SearchHit(document=document, score=score)
            for document, score in self.vector_store.search(self.embedder.embed_query(query), count)
        ]

    def document_list(self) -> list[str]:
        """List unique source names present in the index."""
        return sorted({str(doc.metadata.get("source", "unknown")) for doc in self.vector_store.documents})

    def chunks_for_document(self, filename: str) -> list[Document]:
        """Get all indexed chunks whose source filename matches exactly."""
        return [
            document
            for document in self.vector_store.documents
            if str(document.metadata.get("source", "")).casefold() == filename.casefold()
        ]


"""FAISS IndexFlatIP storage with a JSON metadata sidecar."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """A normalized-inner-product FAISS index and ordered document metadata."""

    def __init__(self, index: Any, documents: list[Document]) -> None:
        if index.ntotal != len(documents):
            raise ValueError("FAISS index and document metadata counts differ")
        self.index = index
        self.documents = documents

    @staticmethod
    def config_key(
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_max_length: int = 512,
    ) -> str:
        """Return a stable compact cache key for index-affecting settings."""
        raw = f"{chunk_size}|{chunk_overlap}|{embedding_model}|{embedding_max_length}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    @classmethod
    def from_embeddings(cls, vectors: np.ndarray, documents: list[Document]) -> "FAISSVectorStore":
        """Build an inner-product FAISS index from unit-normalized vectors."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise ValueError("Expected one 2D embedding vector per document")
        if not len(documents):
            raise ValueError("Cannot build an index without chunks")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(~np.isfinite(vectors)) or np.any(norms < 1e-8):
            raise ValueError("Embedding vectors must be finite and non-zero")
        vectors = vectors / norms[:, None]
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors))
        return cls(index=index, documents=documents)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Document, float]]:
        """Search the index and return up to k document/inner-product pairs."""
        if k <= 0:
            raise ValueError("k must be positive")
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.index.d:
            raise ValueError(f"Query dimension {query.shape[1]} != index dimension {self.index.d}")
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores, ids = self.index.search(np.ascontiguousarray(query), min(k, len(self.documents)))
        return [
            (self.documents[int(index)], float(score))
            for index, score in zip(ids[0], scores[0])
            if index >= 0
        ]

    def save(self, directory: str | Path) -> None:
        """Persist FAISS vectors and safe JSON metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        payload = [
            {"page_content": document.page_content, "metadata": document.metadata}
            for document in self.documents
        ]
        (directory / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved FAISS index with %d chunks to %s", len(self.documents), directory)

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        """Load a FAISS index and its JSON metadata sidecar."""
        directory = Path(directory)
        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete FAISS index in {directory}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        documents = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in raw]
        return cls(faiss.read_index(str(index_path)), documents)


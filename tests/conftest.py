"""Shared deterministic helpers for offline tests."""

from __future__ import annotations

import numpy as np


class StaticEmbedder:
    """Simple unit-vector embedder used to keep tests offline."""

    def __init__(self, query_vector: list[float] | None = None) -> None:
        self.query_vector = np.asarray(query_vector or [1.0, 0.0], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.tile(self.query_vector, (len(texts), 1))

    def embed_query(self, text: str) -> np.ndarray:
        return self.query_vector.copy()


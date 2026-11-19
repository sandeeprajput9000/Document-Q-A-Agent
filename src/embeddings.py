"""Batched Hugging Face PyTorch embeddings with mean pooling and normalization."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
SUPPORTED_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
)


@lru_cache(maxsize=3)
def _load_encoder(model_name: str, cache_dir: str | None, device_name: str) -> tuple[Any, Any]:
    """Keep the three configured model/tokenizer pairs warm in process memory."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=True
        )
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    except OSError:
        logger.info("Embedding model %s is not cached; downloading model files", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(torch.device(device_name))
    model.eval()
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states, ignoring tokenizer padding positions."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def normalize_embeddings(vectors: torch.Tensor) -> torch.Tensor:
    """Return unit-length vectors; zero vectors remain finite zeros."""
    return functional.normalize(vectors, p=2, dim=1, eps=1e-12)


class HuggingFaceEmbedder:
    """Download/cache a transformer encoder and embed strings in batches."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        max_length: int = 512,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cache = str(cache_dir) if cache_dir else None
        if tokenizer is None and model is None:
            self.tokenizer, self.model = _load_encoder(model_name, cache, str(self.device))
        else:
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
            self.model = model or AutoModel.from_pretrained(model_name, cache_dir=cache)
            self.model.to(self.device)
            self.model.eval()
        config = getattr(self.model, "config", None)
        self.dimension = int(getattr(config, "hidden_size", 0)) or None
        logger.info("Loaded embedding model %s on %s", model_name, self.device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed text batches as normalized float32 vectors."""
        if not texts:
            dimension = self.dimension or 0
            return np.empty((0, dimension), dtype=np.float32)
        batches: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            sample_mapping = encoded.pop("overflow_to_sample_mapping", None)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
                pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
                token_counts = encoded["attention_mask"].sum(dim=1).to(pooled.dtype)
                local_mapping = (
                    sample_mapping.to(self.device)
                    if sample_mapping is not None
                    else torch.arange(len(texts[start : start + self.batch_size]), device=self.device)
                )
                aggregate = torch.zeros(
                    (len(texts[start : start + self.batch_size]), pooled.shape[1]),
                    dtype=pooled.dtype,
                    device=self.device,
                )
                aggregate.index_add_(0, local_mapping, pooled * token_counts.unsqueeze(1))
                counts = torch.zeros(aggregate.shape[0], dtype=pooled.dtype, device=self.device)
                counts.index_add_(0, local_mapping, token_counts)
                pooled_texts = aggregate / counts.clamp(min=1e-9).unsqueeze(1)
                batches.append(normalize_embeddings(pooled_texts).cpu())
        vectors = torch.cat(batches, dim=0).numpy().astype(np.float32, copy=False)
        self.dimension = vectors.shape[1]
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query and return its normalized vector."""
        return self.embed_documents([text])[0]


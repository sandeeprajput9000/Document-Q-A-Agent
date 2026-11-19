"""Configuration loading and project path resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    """Runtime settings, with paths resolved relative to the project root."""

    project_root: Path
    chunk_size: int = 1024
    chunk_overlap: int = 0
    top_k: int = 2
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_max_length: int = 512
    max_new_tokens: int = 256
    temperature: float = 0.0
    retry_attempts: int = 3
    retry_base_seconds: float = 1.0
    pdf_dir: Path = Path("data/pdfs")
    index_dir: Path = Path("data/index")
    benchmark_file: Path = Path("data/benchmark/questions.json")
    results_dir: Path = Path("results")
    model_cache_dir: Path = Path("data/cache/models")

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size - 1")
        if self.top_k <= 0 or self.embedding_batch_size <= 0 or self.embedding_max_length <= 0:
            raise ValueError("top_k, embedding_batch_size, and embedding_max_length must be positive")
        if self.max_new_tokens <= 0 or self.retry_attempts <= 0:
            raise ValueError("max_new_tokens and retry_attempts must be positive")

    def path(self, name: str) -> Path:
        """Return a configured path, anchored at the project root if relative."""
        value = getattr(self, name)
        return value if value.is_absolute() else self.project_root / value

    def with_overrides(self, **overrides: Any) -> "Settings":
        """Return a validated settings copy for experiments."""
        return replace(self, **overrides)


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    """Load YAML settings and resolve project root from the config file."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    paths = raw.pop("paths", {})
    allowed = {field for field in Settings.__dataclass_fields__ if field != "project_root"}
    values = {key: value for key, value in raw.items() if key in allowed}
    path_keys = {
        "pdf_dir": "pdf_dir",
        "index_dir": "index_dir",
        "benchmark_file": "benchmark_file",
        "results_dir": "results_dir",
        "model_cache_dir": "model_cache_dir",
    }
    values.update({key: paths[src] for key, src in path_keys.items() if src in paths})
    values = {key: Path(value) if key in path_keys else value for key, value in values.items()}
    return Settings(project_root=path.parent, **values)


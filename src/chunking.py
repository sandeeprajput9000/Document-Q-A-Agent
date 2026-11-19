"""Token-aware recursive text chunking with overlap and source preservation."""

from __future__ import annotations

from functools import lru_cache

import tiktoken
from langchain_core.documents import Document


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    """Count tokens with the stable cl100k_base encoding."""
    return len(_encoder().encode(text, disallowed_special=()))


def _token_windows(text: str, size: int, overlap: int) -> list[str]:
    encoding = _encoder()
    token_ids = encoding.encode(text, disallowed_special=())
    stride = max(1, size - overlap)
    return [encoding.decode(token_ids[start : start + size]) for start in range(0, len(token_ids), stride)]


def _recursive_split(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    if token_count(text) <= size:
        return [text] if text else []
    if not separators or separators[-1] == "":
        return _token_windows(text, size, 0)

    separator = separators[0]
    parts = text.split(separator)
    if len(parts) == 1:
        return _recursive_split(text, size, separators[1:])

    merged: list[str] = []
    current = ""
    for index, part in enumerate(parts):
        candidate = part if not current else current + separator + part
        if current and token_count(candidate) > size:
            merged.extend(_recursive_split(current, size, separators[1:]))
            current = part
        else:
            current = candidate
        if index == len(parts) - 1 and current:
            merged.extend(_recursive_split(current, size, separators[1:]))
    return merged


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text near natural boundaries, enforcing token limits and overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1")
    if not text.strip():
        return []

    body_size = chunk_size - chunk_overlap
    pieces = _recursive_split(text.strip(), body_size, ("\n\n", "\n", ". ", " ", ""))
    chunks: list[str] = []
    prior_tokens: list[int] = []
    encoding = _encoder()
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        piece_ids = encoding.encode(piece, disallowed_special=())
        if chunks and chunk_overlap:
            prefix_ids = prior_tokens[-chunk_overlap:]
            combined = piece
            while prefix_ids:
                prefix = encoding.decode(prefix_ids)
                combined = f"{prefix} {piece}"
                if token_count(combined) <= chunk_size:
                    break
                prefix_ids = prefix_ids[1:]
            if prefix_ids:
                piece = combined
                piece_ids = encoding.encode(piece, disallowed_special=())
        chunks.append(piece)
        prior_tokens = piece_ids
    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Chunk pages independently and keep their source filename and page."""
    chunks: list[Document] = []
    for document in documents:
        for chunk_index, text in enumerate(
            split_text(document.page_content, chunk_size, chunk_overlap)
        ):
            metadata = dict(document.metadata)
            metadata["chunk"] = chunk_index
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


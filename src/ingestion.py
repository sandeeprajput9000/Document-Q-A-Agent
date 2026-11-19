"""PDF loading, text cleanup, and page-level source metadata."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_SPACE = re.compile(r"[ \t\f\v]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize PDF extraction artifacts while retaining paragraph breaks."""
    text = _CONTROL_CHARS.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


def load_pdf_documents(pdf_dir: str | Path) -> list[Document]:
    """Load readable PDF pages as LangChain Documents with 1-based page numbers."""
    root = Path(pdf_dir)
    if not root.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {root}")
    documents: list[Document] = []
    paths = sorted(root.glob("*.pdf"))
    for pdf_path in paths:
        try:
            for loaded_page in PyPDFLoader(str(pdf_path)).lazy_load():
                text = clean_text(loaded_page.page_content or "")
                if text:
                    metadata = loaded_page.metadata
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": int(metadata.get("page", 0)) + 1,
                            },
                        )
                    )
        except Exception as exc:  # One malformed PDF should not block the corpus.
            logger.warning("Skipping unreadable PDF %s: %s", pdf_path.name, exc)
    logger.info("Loaded %d non-empty pages from %d PDFs", len(documents), len(paths))
    if paths and not documents:
        raise ValueError(f"No extractable text found in PDFs under {root}")
    return documents


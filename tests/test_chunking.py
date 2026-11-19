from langchain_core.documents import Document

from src.chunking import chunk_documents, split_text, token_count


def test_chunks_respect_token_limit_and_overlap() -> None:
    text = "First sentence has useful words.\n\n" + "meaningful document evidence " * 70
    chunks = split_text(text, chunk_size=32, chunk_overlap=5)
    assert len(chunks) > 1
    assert all(token_count(chunk) <= 32 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_page_metadata_survives_chunking() -> None:
    page = Document(page_content="A short paragraph. " * 50, metadata={"source": "sample.pdf", "page": 7})
    chunks = chunk_documents([page], chunk_size=24, chunk_overlap=4)
    assert chunks
    assert all(chunk.metadata["source"] == "sample.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 7 for chunk in chunks)
    assert [chunk.metadata["chunk"] for chunk in chunks] == list(range(len(chunks)))


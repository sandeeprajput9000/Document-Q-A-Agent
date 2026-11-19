import numpy as np
from langchain_core.documents import Document

from src.vector_store import FAISSVectorStore


def test_faiss_round_trip_and_search(tmp_path) -> None:
    documents = [
        Document(page_content="Relevant passage", metadata={"source": "one.pdf", "page": 2}),
        Document(page_content="Other passage", metadata={"source": "two.pdf", "page": 1}),
    ]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = FAISSVectorStore.from_embeddings(vectors, documents)
    store.save(tmp_path)
    loaded = FAISSVectorStore.load(tmp_path)
    hits = loaded.search(np.asarray([0.9, 0.1], dtype=np.float32), 1)
    assert len(hits) == 1
    assert hits[0][0].page_content == "Relevant passage"
    assert hits[0][0].metadata == {"source": "one.pdf", "page": 2}
    assert hits[0][1] > 0.9


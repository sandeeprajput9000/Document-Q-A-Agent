import numpy as np
from langchain_core.documents import Document

from src.config import Settings
from src.llm import FakeLLM
from src.pipeline import RAGPipeline
from src.vector_store import FAISSVectorStore


class StaticEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def test_pipeline_returns_answer_sources_latency_and_tool_calls(tmp_path) -> None:
    documents = [
        Document(
            page_content="The report identifies retrieval as a method for grounding language model answers.",
            metadata={"source": "report.pdf", "page": 4},
        )
    ]
    store = FAISSVectorStore.from_embeddings(np.asarray([[1.0, 0.0]], dtype=np.float32), documents)
    settings = Settings(project_root=tmp_path)
    pipeline = RAGPipeline(
        settings=settings,
        embedder=StaticEmbedder(),
        llm=FakeLLM(),
        vector_store=store,
        watsonx_enabled=False,
    )
    result = pipeline.ask("How does the report ground answers?")
    assert "retrieval" in result["answer"]
    assert result["sources"] == [{"source_file": "report.pdf", "source_page": 4}]
    assert result["latency_s"] >= 0
    assert result["tool_calls"] == []


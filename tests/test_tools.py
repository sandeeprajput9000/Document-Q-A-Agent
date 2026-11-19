import numpy as np
from langchain_core.documents import Document

from src.retriever import Retriever
from src.tools import create_document_tools
from src.vector_store import FAISSVectorStore


class StaticEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_tool_schemas_expose_expected_arguments() -> None:
    documents = [Document(page_content="Some text", metadata={"source": "doc.pdf", "page": 1})]
    store = FAISSVectorStore.from_embeddings(np.asarray([[1.0, 0.0]], dtype=np.float32), documents)
    retriever = Retriever(store, StaticEmbedder(), top_k=2)
    tools = create_document_tools(retriever)
    assert {tool.name for tool in tools} == {"search_documents", "get_document_list", "summarize_document"}
    schemas = {tool.name: set(tool.args) for tool in tools}
    assert schemas["search_documents"] >= {"query"}
    assert schemas["get_document_list"] == set()
    assert schemas["summarize_document"] >= {"filename"}


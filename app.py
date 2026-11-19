"""Optional Streamlit chat UI for the Document Q&A Agent."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="Document Q&A Agent", page_icon="📄", layout="wide")
st.title("Document Q&A Agent")
st.caption("Ask questions about PDFs in data/pdfs. Answers include source pages.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(config_path=PROJECT_ROOT / "config.yaml")
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Index")
    if st.button("Build or load index"):
        with st.spinner("Reading PDFs and preparing the index…"):
            summary = st.session_state.pipeline.build_index()
        st.success(f"Ready: {summary['chunks']} chunks")
    st.caption("Put PDFs in data/pdfs, then build the index.")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = st.session_state.pipeline.ask(question)
            st.markdown(result["answer"])
            if result["sources"]:
                citations = ", ".join(
                    f"[{source['source_file']}, page {source['source_page']}]"
                    for source in result["sources"]
                )
                st.caption(f"Sources: {citations} · {result['latency_s']:.2f}s")
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        except Exception as exc:
            st.error(f"Could not answer this question: {exc}")


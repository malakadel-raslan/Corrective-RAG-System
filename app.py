import os
import tempfile

import streamlit as st

from src import config
from src.ingest import ingest_files
from src.graph import run_corrective_rag

st.set_page_config(page_title="Corrective RAG", page_icon="🧠", layout="wide")

st.title("🧠 Corrective RAG System")
st.caption(
    "Upload documents, then ask questions. Retrieved context is graded for "
    "relevance and the query is rewritten and re-retrieved automatically "
    "when the first pass isn't good enough."
)

if not config.OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Copy `.env.example` to `.env` and add your key."
    )
    st.stop()

# --- Sidebar: document ingestion ---
with st.sidebar:
    st.header("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Ingest documents", disabled=not uploaded_files):
        with st.spinner("Loading, cleaning, chunking, and embedding..."):
            os.makedirs(config.DATA_DIR, exist_ok=True)
            paths = []
            for f in uploaded_files:
                path = os.path.join(config.DATA_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(path)

            n_chunks = ingest_files(paths)
            st.success(f"Ingested {len(paths)} file(s) into {n_chunks} chunks.")

    st.divider()
    st.caption(
        f"LLM: `{config.LLM_MODEL}` · Embeddings: `{config.EMBEDDING_MODEL}`\n\n"
        f"Web fallback: {'enabled' if config.ENABLE_WEB_FALLBACK else 'disabled (no TAVILY_API_KEY)'}"
    )

# --- Main: Q&A ---
if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your documents...")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"]:
            st.caption("Sources: " + ", ".join(entry["sources"]))
        meta = []
        if entry["rewrite_count"]:
            meta.append(f"query rewritten {entry['rewrite_count']}x")
        if entry["used_web_fallback"]:
            meta.append("used web fallback")
        if meta:
            st.caption("⚙️ " + " · ".join(meta))

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if not os.path.exists(config.VECTORSTORE_DIR) or not os.listdir(config.VECTORSTORE_DIR):
            st.warning("No documents ingested yet. Upload and ingest documents in the sidebar first.")
        else:
            with st.spinner("Retrieving, grading, and generating..."):
                result = run_corrective_rag(question)

            st.write(result["answer"])
            if result["sources"]:
                st.caption("Sources: " + ", ".join(result["sources"]))
            meta = []
            if result["rewrite_count"]:
                meta.append(f"query rewritten {result['rewrite_count']}x")
            if result["used_web_fallback"]:
                meta.append("used web fallback")
            if meta:
                st.caption("⚙️ " + " · ".join(meta))

            st.session_state.history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "rewrite_count": result["rewrite_count"],
                "used_web_fallback": result["used_web_fallback"],
            })

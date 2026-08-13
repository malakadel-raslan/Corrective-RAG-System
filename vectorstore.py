"""
Vector store layer: embed chunks and store/retrieve them via ChromaDB.
Embeddings run locally (sentence-transformers) so no API key is needed
for this part of the pipeline.
"""
import os
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "corrective_rag_docs"

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def add_documents(documents: List[Document]) -> int:
    """Embed and store new chunks. Returns number of chunks added."""
    if not documents:
        return 0
    store = get_vectorstore()
    store.add_documents(documents)
    return len(documents)


def get_retriever(k: int = 4):
    store = get_vectorstore()
    return store.as_retriever(search_kwargs={"k": k})


def clear_vectorstore():
    """Wipe the collection, e.g. when the user wants to start fresh."""
    store = get_vectorstore()
    ids = store.get()["ids"]
    if ids:
        store.delete(ids=ids)


def collection_size() -> int:
    store = get_vectorstore()
    return len(store.get()["ids"])

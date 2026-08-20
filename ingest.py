"""
Ingestion pipeline for the Corrective RAG system.

Loads documents from a folder (or a single uploaded file), cleans and splits
them into chunks, embeds them, and stores them in a persistent Chroma
vector database.
"""
import os
import re
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config


def load_document(file_path: str) -> List[Document]:
    """Load a single document based on its file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def clean_text(text: str) -> str:
    """Basic text cleaning: collapse whitespace, strip odd control chars."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def load_and_clean_documents(paths: List[str]) -> List[Document]:
    """Load a list of file paths and clean their text content."""
    all_docs: List[Document] = []
    for path in paths:
        docs = load_document(path)
        for d in docs:
            d.page_content = clean_text(d.page_content)
            d.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)
    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed chunks and persist them to the Chroma vector store."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTORSTORE_DIR,
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing persisted Chroma vector store."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.VECTORSTORE_DIR,
    )


def ingest_files(paths: List[str]) -> int:
    """Full pipeline: load -> clean -> split -> embed -> store.

    Returns the number of chunks stored.
    """
    documents = load_and_clean_documents(paths)
    chunks = split_documents(documents)
    build_vectorstore(chunks)
    return len(chunks)


if __name__ == "__main__":
    # CLI usage: ingest every file in the data/ directory
    files = [
        os.path.join(config.DATA_DIR, f)
        for f in os.listdir(config.DATA_DIR)
        if os.path.isfile(os.path.join(config.DATA_DIR, f))
    ]
    if not files:
        print(f"No files found in {config.DATA_DIR}/. Add documents there first.")
    else:
        n = ingest_files(files)
        print(f"Ingested {len(files)} file(s) into {n} chunks.")

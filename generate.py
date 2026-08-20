"""
Answer generation — strictly grounded in verified (graded-relevant) context.
"""
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.grader import get_llm

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant answering questions using ONLY the provided "
     "context. Do not use any outside knowledge.\n"
     "If the context does not contain enough information to answer the "
     "question, respond exactly with: "
     "\"I don't have enough verified information to answer this question.\"\n"
     "Keep the answer concise (max 4 sentences) and factual."),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])


def format_context(documents: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def generate_answer(question: str, documents: List[Document]) -> str:
    if not documents:
        return "I don't have enough verified information to answer this question."

    llm = get_llm(temperature=0.0)
    chain = _GENERATE_PROMPT | llm
    context = format_context(documents)
    result = chain.invoke({"context": context, "question": question})
    return result.content.strip()


def get_sources(documents: List[Document]) -> List[str]:
    """Unique list of source filenames used to produce the answer."""
    seen = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.append(source)
    return seen

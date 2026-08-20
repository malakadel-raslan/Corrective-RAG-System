"""
Grading and query-rewriting logic — the "corrective" part of Corrective RAG.

- grade_documents: judges each retrieved chunk as relevant/irrelevant to the
  question using a structured LLM call.
- rewrite_query: asks the LLM to rephrase the question for better retrieval
  when the current chunks aren't good enough.
"""
from typing import List, Literal

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src import config


class GradeResult(BaseModel):
    """Structured grade for a single retrieved chunk."""
    binary_score: Literal["yes", "no"] = Field(
        description="Is the document relevant to the question? 'yes' or 'no'."
    )


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY,
    )


_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a grader assessing the relevance of a retrieved document to a "
     "user question.\n"
     "If the document contains keywords or semantic meaning related to the "
     "question, grade it as relevant.\n"
     "This does not need to be a strict test — the goal is to filter out "
     "clearly irrelevant retrievals, not to be overly strict.\n"
     "Give a binary score 'yes' or 'no'."),
    ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
])


def grade_documents(
    question: str, documents: List[Document]
) -> tuple[List[Document], float]:
    """Grade each document as relevant/irrelevant.

    Returns (relevant_documents, relevance_ratio).
    """
    llm = get_llm().with_structured_output(GradeResult)
    chain = _GRADE_PROMPT | llm

    relevant_docs: List[Document] = []
    for doc in documents:
        result: GradeResult = chain.invoke(
            {"document": doc.page_content, "question": question}
        )
        if result.binary_score == "yes":
            relevant_docs.append(doc)

    ratio = len(relevant_docs) / len(documents) if documents else 0.0
    return relevant_docs, ratio


_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query rewriter that converts an input question into a "
     "better version optimized for vector-store retrieval.\n"
     "Look at the original question and try to reason about the underlying "
     "semantic intent, then produce a single improved, more specific search "
     "query. Return ONLY the rewritten query, nothing else."),
    ("human", "Original question: {question}"),
])


def rewrite_query(question: str) -> str:
    """Rewrite the question to improve retrieval quality."""
    llm = get_llm(temperature=0.3)
    chain = _REWRITE_PROMPT | llm
    result = chain.invoke({"question": question})
    return result.content.strip()

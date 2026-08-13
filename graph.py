"""
Corrective RAG pipeline built as a LangGraph state machine.

Flow:
    retrieve -> grade_documents -> (relevant enough?) -> generate
                                 -> (weak/irrelevant)  -> rewrite_query -> retrieve (loop, capped)
                                 -> (retries exhausted) -> generate_unanswered

Every step is logged into `trace` so the UI can show *why* the system
did what it did (retrieved N chunks, graded them, rewrote the query, etc).
"""
import os
from typing import List, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from . import llm
from .vectorstore import get_retriever

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 4))
MAX_REWRITE_ATTEMPTS = int(os.getenv("MAX_REWRITE_ATTEMPTS", 2))
RELEVANCE_THRESHOLD = int(os.getenv("RELEVANCE_THRESHOLD", 1))  # min "relevant" chunks needed


class CRAGState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    graded_relevant: List[Document]
    generation: str
    attempts: int
    trace: List[str]


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def retrieve(state: CRAGState) -> CRAGState:
    retriever = get_retriever(k=RETRIEVAL_TOP_K)
    docs = retriever.invoke(state["question"])
    trace = state["trace"] + [
        f"Retrieved {len(docs)} chunk(s) for query: \"{state['question']}\""
    ]
    return {**state, "documents": docs, "trace": trace}


GRADE_SYSTEM_PROMPT = """You are a strict relevance grader for a retrieval-augmented \
generation system. Given a user question and a retrieved text chunk, decide if the \
chunk contains information that helps answer the question.

Respond ONLY with JSON: {"relevant": true} or {"relevant": false}. No explanation."""


def grade_documents(state: CRAGState) -> CRAGState:
    relevant_docs = []
    grades = []
    for doc in state["documents"]:
        user_prompt = (
            f"Question: {state['original_question']}\n\n"
            f"Retrieved chunk:\n{doc.page_content}"
        )
        try:
            result = llm.chat_json(GRADE_SYSTEM_PROMPT, user_prompt)
            is_relevant = bool(result.get("relevant", False))
        except Exception:
            # if grading fails, don't silently drop the chunk from evaluation -
            # treat as irrelevant so the system stays conservative
            is_relevant = False
        grades.append(is_relevant)
        if is_relevant:
            relevant_docs.append(doc)

    trace = state["trace"] + [
        f"Graded chunks: {sum(grades)}/{len(grades)} marked relevant"
    ]
    return {**state, "graded_relevant": relevant_docs, "trace": trace}


def decide_next_step(state: CRAGState) -> str:
    enough_relevant = len(state["graded_relevant"]) >= RELEVANCE_THRESHOLD
    if enough_relevant:
        return "generate"
    if state["attempts"] < MAX_REWRITE_ATTEMPTS:
        return "rewrite_query"
    return "generate_unanswered"


REWRITE_SYSTEM_PROMPT = """You rewrite search queries to improve retrieval from a \
document vector store. The previous query returned weak or irrelevant results. \
Rewrite it to be clearer, broader, or use different phrasing/synonyms that might \
match how the answer is worded in the source documents.

Respond ONLY with JSON: {"rewritten_query": "..."}"""


def rewrite_query(state: CRAGState) -> CRAGState:
    user_prompt = f"Original question: {state['original_question']}\nCurrent query: {state['question']}"
    try:
        result = llm.chat_json(REWRITE_SYSTEM_PROMPT, user_prompt)
        new_query = result.get("rewritten_query", state["question"])
    except Exception:
        new_query = state["question"]

    trace = state["trace"] + [f"Query rewritten to: \"{new_query}\""]
    return {
        **state,
        "question": new_query,
        "attempts": state["attempts"] + 1,
        "trace": trace,
    }


GENERATE_SYSTEM_PROMPT = """You are a course assistant. Answer the user's question \
using ONLY the provided context chunks. Every claim must be traceable to the context.

Rules:
- If the context fully supports an answer, answer clearly and cite sources by filename.
- If the context is insufficient, say so explicitly instead of guessing or using \
outside knowledge.
- Never fabricate information not present in the context."""


def generate(state: CRAGState) -> CRAGState:
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in state["graded_relevant"]
    )
    user_prompt = f"Question: {state['original_question']}\n\nContext:\n{context}"
    answer = llm.chat(GENERATE_SYSTEM_PROMPT, user_prompt, temperature=0.2)

    trace = state["trace"] + ["Generated answer from verified context"]
    return {**state, "generation": answer, "trace": trace}


def generate_unanswered(state: CRAGState) -> CRAGState:
    answer = (
        "I don't have enough verified information in the provided documents to "
        "answer this question confidently. Try rephrasing it or uploading a "
        "document that covers this topic."
    )
    trace = state["trace"] + [
        f"Retries exhausted ({state['attempts']}/{MAX_REWRITE_ATTEMPTS}) with no "
        "sufficiently relevant context — declining to answer"
    ]
    return {**state, "generation": answer, "trace": trace}


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(CRAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("generate_unanswered", generate_unanswered)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "generate_unanswered": "generate_unanswered",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("generate_unanswered", END)

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def answer_question(question: str) -> CRAGState:
    graph = get_graph()
    initial_state: CRAGState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "graded_relevant": [],
        "generation": "",
        "attempts": 0,
        "trace": [],
    }
    return graph.invoke(initial_state)

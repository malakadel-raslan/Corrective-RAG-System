"""
The Corrective RAG graph.

Flow:
  retrieve -> grade -> [relevant enough?]
      -> yes: generate -> END
      -> no:  rewrite_query -> retrieve (loop, up to MAX_REWRITE_ATTEMPTS)
              -> if still not enough and web fallback enabled: web_search -> generate
              -> otherwise: generate with whatever was found (or "don't know")
"""
from typing import List, TypedDict, Optional

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from src import config
from src.ingest import load_vectorstore
from src.grader import grade_documents, rewrite_query
from src.generate import generate_answer, get_sources


class GraphState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    relevance_ratio: float
    rewrite_count: int
    used_web_fallback: bool
    answer: Optional[str]
    sources: List[str]


def _get_retriever():
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})


def retrieve_node(state: GraphState) -> GraphState:
    retriever = _get_retriever()
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}


def grade_node(state: GraphState) -> GraphState:
    relevant_docs, ratio = grade_documents(state["original_question"], state["documents"])
    return {**state, "documents": relevant_docs, "relevance_ratio": ratio}


def rewrite_node(state: GraphState) -> GraphState:
    new_question = rewrite_query(state["question"])
    return {
        **state,
        "question": new_question,
        "rewrite_count": state["rewrite_count"] + 1,
    }


def web_search_node(state: GraphState) -> GraphState:
    """Optional fallback: search the web if local docs still don't cover it."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        results = client.search(state["original_question"], max_results=3)
        web_docs = [
            Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", "web")},
            )
            for r in results.get("results", [])
        ]
        return {**state, "documents": web_docs, "used_web_fallback": True}
    except Exception:
        # If web search fails for any reason, fall back to empty context
        return {**state, "documents": [], "used_web_fallback": True}


def generate_node(state: GraphState) -> GraphState:
    answer = generate_answer(state["original_question"], state["documents"])
    sources = get_sources(state["documents"])
    return {**state, "answer": answer, "sources": sources}


def decide_after_grade(state: GraphState) -> str:
    """Route: good enough -> generate. Not enough -> rewrite (if attempts left)
    or fall back (web search or generate with what we have)."""
    if state["relevance_ratio"] >= config.RELEVANCE_THRESHOLD:
        return "generate"

    if state["rewrite_count"] < config.MAX_REWRITE_ATTEMPTS:
        return "rewrite"

    if config.ENABLE_WEB_FALLBACK and not state["used_web_fallback"]:
        return "web_search"

    return "generate"  # give up gracefully; generate_answer handles empty docs


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        decide_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_corrective_rag(question: str) -> GraphState:
    """Convenience entry point: runs the full corrective RAG flow for a question."""
    graph = build_graph()
    initial_state: GraphState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "relevance_ratio": 0.0,
        "rewrite_count": 0,
        "used_web_fallback": False,
        "answer": None,
        "sources": [],
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    q = input("Ask a question: ")
    result = run_corrective_rag(q)
    print("\nAnswer:", result["answer"])
    print("Sources:", result["sources"])
    print("Rewrite attempts used:", result["rewrite_count"])
    print("Web fallback used:", result["used_web_fallback"])

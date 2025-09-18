import re
from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain.schema import Document

class AgentState(TypedDict, total=False):
    question: str
    route: str
    retrieved_docs: List[Document]
    answer: str
    attempts: int
    retry: bool

def _format_context(docs: List[Document], max_snippets: int = 5) -> str:
    """Creating context with source info for LLM."""
    chunks = []
    for d in docs[:max_snippets]:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "")
        header = f"[{src} - page {page}]"
        chunks.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(chunks)

def normalize_query(q: str) -> str:
    out = q
    out = re.sub(r"(\d)\.\s*E\s*(\d+)", r"\1.E\2", out, flags=re.IGNORECASE)
    out = re.sub(r"(\d)\s+E\s*(\d+)", r"\1.E\2", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+zur Behandlung", "", out, flags=re.IGNORECASE)
    return out.strip()

def build_langgraph_agent(retriever, llm, max_retries: int = 2):
    """LangGraph agent."""

    # ---------- Nodes ----------
    def router_node(state: AgentState) -> Dict[str, Any]:
        q = state["question"]
        router_prompt = f"""
Classify if the following question requires LOOKUP in the provided PDF corpus or can be answered DIRECTLY from general knowledge.

Answer with a single uppercase token: LOOKUP or DIRECT.

Question: "{q}"
"""
        resp = llm.invoke(router_prompt)
        token = (resp.content or "").strip().upper()
        route = "retriever" if "LOOKUP" in token else "llm_direct"
        return {"route": route}

    def retriever_node(state: AgentState) -> Dict[str, Any]:
        q = normalize_query(state["question"])
        docs = retriever.get_relevant_documents(q)
        return {"retrieved_docs": docs}

    def llm_answer_node(state: AgentState) -> Dict[str, Any]:
        context = ""
        if state.get("retrieved_docs"):
            context = _format_context(state["retrieved_docs"], max_snippets=5)

        prompt = f"""
You are an expert assistant. Use the provided context (if any) to answer the question.
Cite context inline as [filename - page] if used.
If context is missing or insufficient, answer with your general knowledge
and indicate clearly that context data from PDFs is insufficient.

Question:
{state['question']}

Context:
{context}

Answer concisely with citations if possible.
"""
        resp = llm.invoke(prompt)
        answer = (resp.content or "").strip()
        return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

    def self_correction_node(state: AgentState) -> Dict[str, Any]:
        attempts = state.get("attempts", 0)
        if attempts >= max_retries:
            return {"retry": False}

        # Ask LLM if retry is needed before rewriting
        decision_prompt = f"""
You are reviewing a Q&A system output.

Question: {state['question']}
Answer: {state['answer']}

Decide if the answer is insufficient due to missing context (not enough PDF info).
Respond with only YES (retry with improved query) or NO (stop).
"""
        decision = llm.invoke(decision_prompt).content.strip().upper()

        if "NO" in decision:
            return {"retry": False}

        # If YES, rewrite the query
        reform_prompt = f"""
Rewrite the following question as an improved search query to maximize recall in a vector database.
Include likely synonyms, abbreviations, German/English variants.
Return only the improved query.

Original: {state['question']}
"""
        reform = llm.invoke(reform_prompt).content.strip()
        improved_q = reform if reform else state['question']
        return {"retry": True, "question": improved_q}

    # ---------- Graph ----------
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("llm_answer", llm_answer_node)
    workflow.add_node("llm_direct", llm_answer_node)
    workflow.add_node("self_correction", self_correction_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"retriever": "retriever", "llm_direct": "llm_direct"},
    )

    workflow.add_edge("retriever", "llm_answer")
    workflow.add_edge("llm_direct", "llm_answer")

    workflow.add_conditional_edges(
        "llm_answer",
        lambda s: "self_correction" if s.get("attempts", 0) < max_retries else END,
        {"self_correction": "self_correction", END: END},
    )

    workflow.add_conditional_edges(
        "self_correction",
        lambda s: "retriever" if s.get("retry") else END,
        {"retriever": "retriever", END: END},
    )

    return workflow.compile()

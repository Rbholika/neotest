"""
agents/rag_agent.py — Retrieval-Augmented Generation Agent
===========================================================
LangGraph StateGraph nodes:
    process_query → security_check → retrieve_context
        → generate_response → validate_output → export

Usage:
    from agents.rag_agent import RAGAgent
    agent = RAGAgent()
    result = agent.invoke({
        "query":      "What was the root cause of the incident?",
        "collection": "incidents",
        "export":     True,           # optional
        "export_fmt": "pdf",          # "pdf" | "docx" | "md"
    })
"""

import os
import json
from typing import Any, Dict, Optional, List, TypedDict

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma

from agents.base_agent import BaseAgent, AgentResponse
from core.logger import get_logger
from core.security import guardrails
from config import CONFIG

logger = get_logger("agent.rag")


# ── LangGraph State ───────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query:          str
    collection:     str
    context_docs:   List[str]
    sources:        List[str]
    response:       str
    export:         bool
    export_fmt:     str
    export_path:    Optional[str]
    error:          Optional[str]


# ── Agent ─────────────────────────────────────────────────────────────────────

class RAGAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="rag")

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_process_query(self, state: RAGState) -> RAGState:
        query = state.get("query", "").strip()
        if not query:
            state["error"] = "Query is empty."
        else:
            state["query"] = query
            logger.info(f"Processing query: {query[:80]}…")
        return state

    def _node_security(self, state: RAGState) -> RAGState:
        result = guardrails.check_input(state["query"])
        if not result.passed:
            state["error"] = "Security check failed: " + "; ".join(result.violations)
        else:
            state["query"] = result.sanitised_text or state["query"]
        return state

    def _node_retrieve(self, state: RAGState) -> RAGState:
        collection  = state.get("collection", "default")
        persist_dir = os.path.join(CONFIG.CHROMA_DIR, collection)

        if not os.path.exists(persist_dir):
            state["error"] = (
                f"Collection '{collection}' not found. "
                "Please upload documents first."
            )
            return state

        try:
            embeddings = self._get_embeddings()
            vs = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=collection,
            )
            results = vs.similarity_search_with_relevance_scores(
                state["query"], k=CONFIG.RETRIEVAL_K
            )
            docs    = [doc.page_content for doc, score in results if score > 0.3]
            sources = [
                doc.metadata.get("source", "unknown")
                for doc, score in results if score > 0.3
            ]
            state["context_docs"] = docs
            state["sources"]      = list(dict.fromkeys(sources))   # deduplicate
            logger.info(
                f"Retrieved {len(docs)} relevant chunks from '{collection}'"
            )
        except Exception as exc:
            state["error"] = f"Retrieval error: {exc}"

        return state

    def _node_generate(self, state: RAGState) -> RAGState:
        docs   = state.get("context_docs", [])
        query  = state["query"]

        if not docs:
            context_text = "No relevant context found in the knowledge base."
        else:
            context_text = "\n\n---\n\n".join(
                f"[Source {i+1}] {doc}" for i, doc in enumerate(docs)
            )

        prompt = (
            f"Context retrieved from the knowledge base:\n\n{context_text}\n\n"
            f"---\n\nUser question: {query}\n\n"
            "Instructions:\n"
            "  • Answer ONLY using the context above.\n"
            "  • If the context doesn't contain the answer, say so explicitly.\n"
            "  • Cite sources by [Source N] numbers.\n"
            "  • Be concise and structured.\n"
            "  • Do NOT hallucinate or add facts not present in the context.\n"
        )

        sys_prompt = self._system_prompt(
            "RAG-powered AI analyst",
            (
                "You must ground every claim in the provided context. "
                "Never fabricate information. "
                "If uncertain, say 'Based on available context, ...' or "
                "'The context does not address this.'"
            ),
        )

        try:
            response         = self._llm_call(sys_prompt, prompt, check_input=False)
            state["response"] = response
        except Exception as exc:
            state["error"] = f"Generation failed: {exc}"

        return state

    def _node_validate_output(self, state: RAGState) -> RAGState:
        response = state.get("response", "")
        if not response:
            state["error"] = "Empty response generated."
            return state

        # Output guardrails — mask any leaked PII
        out_result      = guardrails.check_output(response)
        state["response"] = out_result.sanitised_text or response
        return state

    def _node_export(self, state: RAGState) -> RAGState:
        if not state.get("export"):
            return state

        from tools.export_tool import ExportTool
        fmt  = state.get("export_fmt", "pdf").lower()
        tool = ExportTool()

        content = {
            "title":   "RAG Analysis Report",
            "query":   state["query"],
            "sources": state.get("sources", []),
            "response": state.get("response", ""),
        }

        try:
            if fmt == "pdf":
                path = tool.to_pdf(content, filename="rag_report")
            elif fmt == "docx":
                path = tool.to_docx(content, filename="rag_report")
            else:
                path = tool.to_markdown(content, filename="rag_report")

            state["export_path"] = path
            logger.info(f"RAG report exported → {path}")
        except Exception as exc:
            logger.error(f"Export failed: {exc}")

        return state

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route(self, state: RAGState) -> str:
        return "error_end" if state.get("error") else "next"

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(RAGState)

        g.add_node("process_query",    self._node_process_query)
        g.add_node("security_check",   self._node_security)
        g.add_node("retrieve_context", self._node_retrieve)
        g.add_node("generate_response",self._node_generate)
        g.add_node("validate_output",  self._node_validate_output)
        g.add_node("export_result",    self._node_export)
        g.add_node("error_end",        lambda s: s)

        g.set_entry_point("process_query")

        steps = [
            ("process_query",    "security_check"),
            ("security_check",   "retrieve_context"),
            ("retrieve_context", "generate_response"),
            ("generate_response","validate_output"),
            ("validate_output",  "export_result"),
        ]
        for src, nxt in steps:
            g.add_conditional_edges(
                src, self._route,
                {"next": nxt, "error_end": "error_end"},
            )

        g.add_edge("export_result", END)
        g.add_edge("error_end",     END)
        return g.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(self, inputs: Dict[str, Any]) -> AgentResponse:
        """
        inputs = {
            "query":      str,
            "collection": str,       # Chroma collection name
            "export":     bool,      # optional, default False
            "export_fmt": str,       # "pdf" | "docx" | "md"
        }
        """
        resp = AgentResponse(agent_name=self.name)
        try:
            initial: RAGState = {
                "query":        inputs.get("query", ""),
                "collection":   inputs.get("collection", "default"),
                "context_docs": [],
                "sources":      [],
                "response":     "",
                "export":       inputs.get("export", False),
                "export_fmt":   inputs.get("export_fmt", "pdf"),
                "export_path":  None,
                "error":        None,
            }
            final = self._graph.invoke(initial)

            if final.get("error"):
                return resp.fail(final["error"])

            resp.output      = final["response"]
            resp.raw_text    = final["response"]
            resp.metadata    = {
                "sources":      final.get("sources", []),
                "chunks_used":  len(final.get("context_docs", [])),
                "collection":   final["collection"],
                "export_path":  final.get("export_path"),
            }
            if final.get("export_path"):
                resp.export_paths = [final["export_path"]]

        except Exception as exc:
            return resp.fail(str(exc))
        return resp

"""
orchestrator/multi_agent.py — LangGraph Multi-Agent Orchestrator
================================================================
Routes user requests to the appropriate agent(s) and synthesises results.

Routing logic (AI-driven):
    "generate data" / "synthetic"         → SyntheticDataAgent
    "upload" / "ingest" / "add document"  → FileUploadAgent
    "analyse" / "query" / "what" / "why"  → RAGAgent
    "visualise" / "chart" / "plot"        → VisualizationAgent
    "full pipeline"                        → all agents in order

Usage:
    from orchestrator.multi_agent import MultiAgentOrchestrator
    orch = MultiAgentOrchestrator()
    result = orch.run("Generate 30 synthetic incident records and visualise them")
"""

import json
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.synthetic_data_agent import SyntheticDataAgent
from agents.file_upload_agent    import FileUploadAgent
from agents.rag_agent            import RAGAgent
from agents.visualization_agent  import VisualizationAgent
from agents.base_agent           import AgentResponse, BaseAgent
from core.logger                 import get_logger
from core.security               import guardrails
from config                      import CONFIG

logger = get_logger("orchestrator")


# ── Orchestrator State ────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    user_input:        str
    context:           Dict[str, Any]     # extra params from caller
    route_plan:        List[str]          # ordered list of agent names
    agent_results:     Dict[str, Any]     # name → AgentResponse
    synthesised_output: str
    error:             Optional[str]


# ── Agent Registry ────────────────────────────────────────────────────────────

AGENT_MAP: Dict[str, type] = {
    "synthetic_data":  SyntheticDataAgent,
    "file_upload":     FileUploadAgent,
    "rag":             RAGAgent,
    "visualization":   VisualizationAgent,
}

ROUTE_KEYWORDS: Dict[str, List[str]] = {
    "synthetic_data":  ["synthetic", "generate data", "fake data",
                        "sample data", "mock data", "create records"],
    "file_upload":     ["upload", "ingest", "add document", "load file",
                        "import file", "add pdf", "index"],
    "rag":             ["analyse", "analyze", "query", "search", "find",
                        "what", "why", "how", "explain", "summarise",
                        "summarize", "tell me", "describe"],
    "visualization":   ["visualise", "visualize", "chart", "plot", "graph",
                        "show", "display", "diagram", "dashboard"],
}


class MultiAgentOrchestrator:

    def __init__(self, problem_statement: str = ""):
        self.problem_statement = problem_statement
        self._agents: Dict[str, BaseAgent] = {}
        self._graph  = self._build_graph()

    def _get_agent(self, name: str) -> BaseAgent:
        if name not in self._agents:
            cls = AGENT_MAP.get(name)
            if cls:
                self._agents[name] = cls()
        return self._agents.get(name)

    # ── LangGraph nodes ───────────────────────────────────────────────────────

    def _node_security_gate(self, state: OrchestratorState) -> OrchestratorState:
        result = guardrails.check_input(state["user_input"])
        if not result.passed:
            state["error"] = (
                "Request blocked by security: "
                + "; ".join(result.violations)
            )
        else:
            state["user_input"] = result.sanitised_text or state["user_input"]
        return state

    def _node_route(self, state: OrchestratorState) -> OrchestratorState:
        user_input = state["user_input"].lower()

        # Check for explicit full-pipeline request
        if any(kw in user_input for kw in ["full pipeline", "end to end", "everything"]):
            state["route_plan"] = ["synthetic_data", "rag", "visualization"]
            return state

        # AI-assisted routing for ambiguous inputs
        plan = self._ai_route(user_input)
        if not plan:
            # Keyword fallback
            plan = []
            for agent, keywords in ROUTE_KEYWORDS.items():
                if any(kw in user_input for kw in keywords):
                    plan.append(agent)

        # Default to RAG if nothing matched
        if not plan:
            plan = ["rag"]

        state["route_plan"] = plan
        logger.info(f"Route plan: {plan}")
        return state

    def _ai_route(self, user_input: str) -> List[str]:
        """Ask the LLM to select agents for this request."""
        try:
            agent_keys = list(AGENT_MAP.keys())
            prompt = (
                f"User request: '{user_input}'\n\n"
                f"Available agents: {agent_keys}\n"
                "Agents:\n"
                "  synthetic_data  — generates synthetic/fake data\n"
                "  file_upload     — uploads and indexes documents\n"
                "  rag             — answers questions from documents\n"
                "  visualization   — creates charts and graphs\n\n"
                "Return a JSON array of agent names to execute IN ORDER.\n"
                "Example: [\"rag\", \"visualization\"]\n"
                "Return ONLY valid JSON. No explanation."
            )
            llm  = CONFIG.get_llm()
            from langchain_core.messages import HumanMessage, SystemMessage
            resp = llm.invoke([
                SystemMessage(content="You are a request routing assistant. Return only JSON."),
                HumanMessage(content=prompt),
            ])
            raw  = resp.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            plan = json.loads(raw)
            valid = [a for a in plan if a in AGENT_MAP]
            return valid if valid else []
        except Exception as exc:
            logger.warning(f"AI routing failed, using keyword fallback: {exc}")
            return []

    def _node_execute_agents(self, state: OrchestratorState) -> OrchestratorState:
        plan    = state.get("route_plan", [])
        context = state.get("context", {})
        results = {}

        for agent_name in plan:
            agent = self._get_agent(agent_name)
            if agent is None:
                logger.warning(f"Agent '{agent_name}' not found — skipping")
                continue

            logger.info(f"Executing agent: {agent_name}")
            try:
                # Build inputs from context + previous results
                inputs = self._build_agent_inputs(
                    agent_name, context, results, state["user_input"]
                )
                result = agent.invoke(inputs)
                results[agent_name] = {
                    "success":      result.success,
                    "output":       result.output,
                    "raw_text":     result.raw_text,
                    "metadata":     result.metadata,
                    "export_paths": result.export_paths,
                    "errors":       result.errors,
                    "warnings":     result.warnings,
                }
                logger.info(
                    f"Agent '{agent_name}' completed "
                    f"(success={result.success})"
                )
            except Exception as exc:
                logger.error(f"Agent '{agent_name}' crashed: {exc}")
                results[agent_name] = {"success": False, "errors": [str(exc)]}

        state["agent_results"] = results
        return state

    def _node_synthesise(self, state: OrchestratorState) -> OrchestratorState:
        results = state.get("agent_results", {})
        parts   = []

        for agent_name, res in results.items():
            if not res.get("success"):
                parts.append(
                    f"**{agent_name}** failed: "
                    + "; ".join(res.get("errors", ["unknown error"]))
                )
                continue

            parts.append(f"**{agent_name.replace('_', ' ').title()} Results:**")
            if res.get("raw_text"):
                parts.append(res["raw_text"][:1000])    # cap for display
            if res.get("export_paths"):
                parts.append("Exported to: " + ", ".join(res["export_paths"]))
            if res.get("warnings"):
                parts.append("⚠️ " + "; ".join(res["warnings"]))
            parts.append("")

        state["synthesised_output"] = "\n".join(parts)
        return state

    # ── Input builder ─────────────────────────────────────────────────────────

    def _build_agent_inputs(
        self,
        agent_name: str,
        context: Dict[str, Any],
        prev_results: Dict[str, Any],
        user_input: str,
    ) -> Dict[str, Any]:
        """
        Construct the inputs dict for each agent, pulling from context
        and chaining outputs from previous agents.
        """
        base = dict(context)
        base.setdefault("problem_statement", self.problem_statement or user_input)

        if agent_name == "synthetic_data":
            base.setdefault("schema",     context.get("schema", {
                "fields": [
                    {"name": "id",          "type": "integer"},
                    {"name": "timestamp",   "type": "datetime"},
                    {"name": "severity",    "type": "string"},
                    {"name": "description", "type": "text"},
                    {"name": "status",      "type": "string"},
                ],
                "num_rows": context.get("num_rows", 20),
            }))
            base.setdefault("output_format", "json")

        elif agent_name == "file_upload":
            base.setdefault("file_path",  context.get("file_path", ""))
            base.setdefault("collection", context.get("collection", "default"))

        elif agent_name == "rag":
            base.setdefault("query",      context.get("query", user_input))
            base.setdefault("collection", context.get("collection", "default"))
            base.setdefault("export",     context.get("export", False))
            base.setdefault("export_fmt", context.get("export_fmt", "pdf"))

        elif agent_name == "visualization":
            # Chain data from synthetic agent if available
            synth = prev_results.get("synthetic_data", {})
            if synth.get("success") and synth.get("output"):
                base.setdefault("data", synth["output"])
            base.setdefault("title",      context.get("title", "Analysis Chart"))
            base.setdefault("description", user_input)
            base.setdefault("export_fmt", "html")

        return base

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route_after_security(self, state: OrchestratorState) -> str:
        return "error_end" if state.get("error") else "route"

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(OrchestratorState)

        g.add_node("security_gate",    self._node_security_gate)
        g.add_node("route",            self._node_route)
        g.add_node("execute_agents",   self._node_execute_agents)
        g.add_node("synthesise",       self._node_synthesise)
        g.add_node("error_end",        lambda s: s)

        g.set_entry_point("security_gate")
        g.add_conditional_edges(
            "security_gate",
            self._route_after_security,
            {"route": "route", "error_end": "error_end"},
        )
        g.add_edge("route",          "execute_agents")
        g.add_edge("execute_agents", "synthesise")
        g.add_edge("synthesise",     END)
        g.add_edge("error_end",      END)

        return g.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the multi-agent pipeline.

        Args:
            user_input: Natural language request
            context:    Optional extra params passed to agents
                        (file_path, collection, schema, query, etc.)

        Returns:
            {
                "success":           bool,
                "synthesised_output": str,
                "agent_results":     dict,
                "route_plan":        list,
                "error":             str | None,
            }
        """
        initial: OrchestratorState = {
            "user_input":         user_input,
            "context":            context or {},
            "route_plan":         [],
            "agent_results":      {},
            "synthesised_output": "",
            "error":              None,
        }
        final = self._graph.invoke(initial)

        return {
            "success":            not bool(final.get("error")),
            "synthesised_output": final.get("synthesised_output", ""),
            "agent_results":      final.get("agent_results", {}),
            "route_plan":         final.get("route_plan", []),
            "error":              final.get("error"),
        }

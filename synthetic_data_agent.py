"""
agents/synthetic_data_agent.py — Synthetic Data Generator
==========================================================
LangGraph StateGraph nodes:
    parse_request → security_check → generate_data
        → validate_data → [retry | export_data]

Usage:
    from agents.synthetic_data_agent import SyntheticDataAgent
    agent = SyntheticDataAgent()
    result = agent.invoke({
        "problem_statement": "...",
        "schema": {"fields": [...], "num_rows": 50},
        "output_format": "csv",   # or "json"
    })
"""

import json
import csv
import io
import os
from typing import Any, Dict, Optional, TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from agents.base_agent import BaseAgent, AgentResponse
from core.logger import get_logger
from core.security import guardrails
from config import CONFIG

logger = get_logger("agent.synthetic_data")


# ── LangGraph State ───────────────────────────────────────────────────────────

class SyntheticDataState(TypedDict):
    problem_statement: str
    schema:            Dict[str, Any]
    output_format:     str                    # "csv" | "json"
    num_rows:          int
    generated_data:    Optional[List[Dict]]
    validation_errors: List[str]
    retry_count:       int
    export_path:       Optional[str]
    error:             Optional[str]


# ── Agent ─────────────────────────────────────────────────────────────────────

class SyntheticDataAgent(BaseAgent):

    MAX_RETRIES = 2

    def __init__(self):
        super().__init__(name="synthetic_data")

    # ── LangGraph nodes ───────────────────────────────────────────────────────

    def _node_parse(self, state: SyntheticDataState) -> SyntheticDataState:
        logger.info("Parsing request schema …")
        schema = state.get("schema", {})
        if not schema.get("fields"):
            state["error"] = "Schema must include 'fields' list."
        if not state.get("num_rows"):
            state["num_rows"] = schema.get("num_rows", 20)
        state["output_format"] = state.get("output_format", "json").lower()
        state["retry_count"]   = 0
        state["validation_errors"] = []
        return state

    def _node_security(self, state: SyntheticDataState) -> SyntheticDataState:
        text = state.get("problem_statement", "")
        result = guardrails.check_input(text)
        if not result.passed:
            state["error"] = "Security check failed: " + "; ".join(result.violations)
        else:
            state["problem_statement"] = result.sanitised_text or text
        return state

    def _node_generate(self, state: SyntheticDataState) -> SyntheticDataState:
        logger.info(f"Generating {state['num_rows']} synthetic rows …")

        fields_desc = json.dumps(state["schema"].get("fields", []), indent=2)
        constraints = state["schema"].get("constraints", "None specified.")

        prompt = (
            f"Problem context:\n{state['problem_statement']}\n\n"
            f"Generate exactly {state['num_rows']} realistic, diverse, anonymised "
            f"synthetic data records matching these fields:\n{fields_desc}\n\n"
            f"Constraints: {constraints}\n\n"
            "Return ONLY a valid JSON array of objects. No markdown, no explanation.\n"
            "Example: [{\"field1\": \"value1\", ...}, ...]\n"
            "Ensure:\n"
            "  • No real PII (names, emails, phones must be clearly fictional)\n"
            "  • Realistic value distributions\n"
            "  • All required fields present in every record\n"
        )

        try:
            raw = self._llm_call(
                system_prompt=self._system_prompt(
                    "synthetic data generation expert",
                    "Always return valid JSON arrays. Never use real PII."
                ),
                user_prompt=prompt,
            )
            # Strip markdown fences if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("LLM did not return a JSON array.")
            state["generated_data"] = data
        except Exception as exc:
            state["validation_errors"].append(str(exc))
            state["generated_data"] = None
        return state

    def _node_validate(self, state: SyntheticDataState) -> SyntheticDataState:
        data   = state.get("generated_data")
        errors = state["validation_errors"]

        if not data:
            errors.append("No data generated.")
            return state

        required_fields = [f["name"] for f in state["schema"].get("fields", [])]
        missing_all = []
        for i, row in enumerate(data):
            missing = [f for f in required_fields if f not in row]
            if missing:
                missing_all.append(f"Row {i}: missing {missing}")

        if missing_all:
            errors.extend(missing_all[:5])           # cap error list
            state["validation_errors"] = errors
        return state

    def _node_export(self, state: SyntheticDataState) -> SyntheticDataState:
        data   = state.get("generated_data", [])
        fmt    = state.get("output_format", "json")
        os.makedirs(CONFIG.EXPORT_DIR, exist_ok=True)

        base = os.path.join(CONFIG.EXPORT_DIR, "synthetic_data")
        try:
            if fmt == "csv" and data:
                path = base + ".csv"
                keys = list(data[0].keys())
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(data)
            else:
                path = base + ".json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            state["export_path"] = path
            logger.info(f"Synthetic data exported → {path}")
        except Exception as exc:
            state["error"] = f"Export failed: {exc}"
        return state

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route_after_parse(self, state: SyntheticDataState) -> str:
        return "error_end" if state.get("error") else "security_check"

    def _route_after_security(self, state: SyntheticDataState) -> str:
        return "error_end" if state.get("error") else "generate_data"

    def _route_after_validate(self, state: SyntheticDataState) -> str:
        if state.get("validation_errors") and state["retry_count"] < self.MAX_RETRIES:
            state["retry_count"] += 1
            logger.warning(
                f"Validation failed, retry {state['retry_count']}/{self.MAX_RETRIES}"
            )
            return "generate_data"
        if state.get("validation_errors"):
            return "error_end"
        return "export_data"

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(SyntheticDataState)

        g.add_node("parse_request",  self._node_parse)
        g.add_node("security_check", self._node_security)
        g.add_node("generate_data",  self._node_generate)
        g.add_node("validate_data",  self._node_validate)
        g.add_node("export_data",    self._node_export)
        g.add_node("error_end",      lambda s: s)   # terminal error node

        g.set_entry_point("parse_request")

        g.add_conditional_edges("parse_request",  self._route_after_parse,
                                {"security_check": "security_check",
                                 "error_end":      "error_end"})
        g.add_conditional_edges("security_check", self._route_after_security,
                                {"generate_data": "generate_data",
                                 "error_end":     "error_end"})
        g.add_edge("generate_data", "validate_data")
        g.add_conditional_edges("validate_data", self._route_after_validate,
                                {"generate_data": "generate_data",
                                 "export_data":   "export_data",
                                 "error_end":     "error_end"})
        g.add_edge("export_data", END)
        g.add_edge("error_end",   END)

        return g.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(self, inputs: Dict[str, Any]) -> AgentResponse:
        """
        inputs = {
            "problem_statement": str,
            "schema": {
                "fields": [{"name": str, "type": str, "description": str}, ...],
                "num_rows": int,
                "constraints": str   (optional)
            },
            "output_format": "json" | "csv"
        }
        """
        resp = AgentResponse(agent_name=self.name)
        try:
            initial_state: SyntheticDataState = {
                "problem_statement": inputs.get("problem_statement", ""),
                "schema":            inputs.get("schema", {}),
                "output_format":     inputs.get("output_format", "json"),
                "num_rows":          inputs.get("schema", {}).get("num_rows", 20),
                "generated_data":    None,
                "validation_errors": [],
                "retry_count":       0,
                "export_path":       None,
                "error":             None,
            }
            final = self._graph.invoke(initial_state)

            if final.get("error") or final.get("validation_errors"):
                errs = [final["error"]] if final.get("error") else []
                errs += final.get("validation_errors", [])
                return resp.fail(" | ".join(errs))

            resp.output      = final.get("generated_data", [])
            resp.raw_text    = json.dumps(resp.output, indent=2)
            resp.export_paths = [final["export_path"]] if final.get("export_path") else []
            resp.metadata = {
                "num_rows":     len(resp.output),
                "output_format": final.get("output_format"),
                "export_path":  final.get("export_path"),
            }
        except Exception as exc:
            return resp.fail(str(exc))
        return resp

"""
agents/visualization_agent.py — AI-Driven Visualization Agent
=============================================================
LangGraph StateGraph nodes:
    parse_data → choose_chart_type → render_chart → export_chart

Features:
  • LLM selects the best chart type for the data
  • Supports: bar, line, pie, scatter, histogram, heatmap, box
  • Exports PNG + interactive HTML (Plotly)
  • Works with dict, list-of-dicts, CSV string, or raw JSON

Usage:
    from agents.visualization_agent import VisualizationAgent
    agent = VisualizationAgent()
    result = agent.invoke({
        "data":       [...],             # list of dicts or raw JSON string
        "title":      "Incident Trends",
        "description": "Monthly incident counts by severity",
        "export_fmt": "png",             # "png" | "html" | "both"
    })
"""

import os
import json
from typing import Any, Dict, Optional, List, TypedDict

from langgraph.graph import StateGraph, END

from agents.base_agent import BaseAgent, AgentResponse
from core.logger import get_logger
from config import CONFIG

logger = get_logger("agent.visualization")

SUPPORTED_CHARTS = ["bar", "line", "pie", "scatter",
                    "histogram", "heatmap", "box", "area"]


# ── LangGraph State ───────────────────────────────────────────────────────────

class VizState(TypedDict):
    raw_data:    Any
    data_df:     Optional[Any]          # pandas DataFrame
    title:       str
    description: str
    chart_type:  str
    x_col:       Optional[str]
    y_col:       Optional[str]
    color_col:   Optional[str]
    export_fmt:  str
    export_paths: List[str]
    error:        Optional[str]


# ── Agent ─────────────────────────────────────────────────────────────────────

class VisualizationAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="visualization", enable_guardrails=False)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_parse(self, state: VizState) -> VizState:
        import pandas as pd

        raw = state["raw_data"]
        try:
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, list) and raw:
                df = pd.DataFrame(raw)
            elif isinstance(raw, dict):
                df = pd.DataFrame([raw])
            else:
                state["error"] = "Unsupported data format. Provide list-of-dicts or JSON string."
                return state

            state["data_df"] = df
            logger.info(f"Parsed data: {df.shape[0]} rows × {df.shape[1]} cols")
        except Exception as exc:
            state["error"] = f"Data parsing failed: {exc}"
        return state

    def _node_choose_chart(self, state: VizState) -> VizState:
        if state.get("chart_type") and state["chart_type"] in SUPPORTED_CHARTS:
            logger.info(f"Using user-specified chart type: {state['chart_type']}")
            return state

        df   = state["data_df"]
        cols = list(df.columns)
        dtypes = {c: str(df[c].dtype) for c in cols}
        nuniq  = {c: int(df[c].nunique()) for c in cols}

        prompt = (
            f"Data summary:\n"
            f"  Columns: {cols}\n"
            f"  Dtypes:  {dtypes}\n"
            f"  Unique values: {nuniq}\n"
            f"  Rows: {len(df)}\n\n"
            f"Title: {state['title']}\n"
            f"Description: {state['description']}\n\n"
            f"Choose the BEST chart type from: {SUPPORTED_CHARTS}\n"
            "Also specify the best x_column and y_column (if applicable), "
            "and optionally a color_column for grouping.\n"
            "Respond ONLY as JSON:\n"
            '{"chart_type": "bar", "x_col": "month", '
            '"y_col": "count", "color_col": "severity"}'
        )

        try:
            raw = self._llm_call(
                system_prompt=self._system_prompt("data visualisation expert"),
                user_prompt=prompt,
                check_input=False,
            )
            raw   = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            choice = json.loads(raw)

            state["chart_type"] = choice.get("chart_type", "bar")
            state["x_col"]      = choice.get("x_col")
            state["y_col"]      = choice.get("y_col")
            state["color_col"]  = choice.get("color_col")
            logger.info(
                f"AI selected chart: {state['chart_type']} "
                f"x={state['x_col']} y={state['y_col']}"
            )
        except Exception as exc:
            logger.warning(f"Chart selection LLM failed, defaulting to bar: {exc}")
            state["chart_type"] = "bar"
            num_cols = [c for c in df.columns if df[c].dtype in ["int64", "float64"]]
            cat_cols = [c for c in df.columns if df[c].dtype == "object"]
            state["x_col"] = cat_cols[0] if cat_cols else cols[0]
            state["y_col"] = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else cols[0])
        return state

    def _node_render(self, state: VizState) -> VizState:
        import plotly.express as px
        import plotly.graph_objects as go

        df         = state["data_df"]
        chart_type = state["chart_type"]
        x          = state.get("x_col")
        y          = state.get("y_col")
        color      = state.get("color_col") if state.get("color_col") in df.columns else None
        title      = state["title"]
        os.makedirs(CONFIG.EXPORT_DIR, exist_ok=True)

        try:
            chart_map = {
                "bar":       lambda: px.bar(df, x=x, y=y, color=color, title=title,
                                            template="plotly_white"),
                "line":      lambda: px.line(df, x=x, y=y, color=color, title=title,
                                             template="plotly_white"),
                "pie":       lambda: px.pie(df, names=x, values=y, title=title),
                "scatter":   lambda: px.scatter(df, x=x, y=y, color=color, title=title,
                                                template="plotly_white"),
                "histogram": lambda: px.histogram(df, x=x, color=color, title=title,
                                                  template="plotly_white"),
                "heatmap":   lambda: px.imshow(df.select_dtypes("number").corr(),
                                               title=title, text_auto=True,
                                               color_continuous_scale="RdBu"),
                "box":       lambda: px.box(df, x=color, y=y, title=title,
                                            template="plotly_white"),
                "area":      lambda: px.area(df, x=x, y=y, color=color, title=title,
                                             template="plotly_white"),
            }
            fig_fn = chart_map.get(chart_type, chart_map["bar"])
            fig    = fig_fn()

            fig.update_layout(
                font_family="Arial",
                title_font_size=18,
                legend_title_text=color or "",
                margin=dict(l=50, r=50, t=60, b=50),
            )

            state["_fig"] = fig       # pass figure forward
            logger.info(f"Chart rendered: {chart_type}")
        except Exception as exc:
            state["error"] = f"Chart rendering failed: {exc}"
        return state

    def _node_export(self, state: VizState) -> VizState:
        fig      = state.get("_fig")
        fmt      = state.get("export_fmt", "png").lower()
        base     = os.path.join(CONFIG.EXPORT_DIR, "chart")
        paths    = []

        if fig is None:
            state["error"] = "No figure to export."
            return state

        try:
            if fmt in ("png", "both"):
                png_path = base + ".png"
                fig.write_image(png_path, width=1200, height=700, scale=2)
                paths.append(png_path)

            if fmt in ("html", "both"):
                html_path = base + ".html"
                fig.write_html(html_path)
                paths.append(html_path)

            if fmt == "png" and not paths:         # fallback
                html_path = base + ".html"
                fig.write_html(html_path)
                paths.append(html_path)

            state["export_paths"] = paths
            logger.info(f"Chart exported → {paths}")
        except Exception as exc:
            # Plotly image export requires kaleido; fallback to HTML
            logger.warning(f"PNG export failed (kaleido?), saving HTML: {exc}")
            html_path = base + ".html"
            fig.write_html(html_path)
            state["export_paths"] = [html_path]

        return state

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route(self, state: VizState) -> str:
        return "error_end" if state.get("error") else "next"

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(VizState)

        g.add_node("parse_data",    self._node_parse)
        g.add_node("choose_chart",  self._node_choose_chart)
        g.add_node("render_chart",  self._node_render)
        g.add_node("export_chart",  self._node_export)
        g.add_node("error_end",     lambda s: s)

        g.set_entry_point("parse_data")

        for src, nxt in [
            ("parse_data",   "choose_chart"),
            ("choose_chart", "render_chart"),
            ("render_chart", "export_chart"),
        ]:
            g.add_conditional_edges(
                src, self._route, {"next": nxt, "error_end": "error_end"}
            )

        g.add_edge("export_chart", END)
        g.add_edge("error_end",    END)
        return g.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(self, inputs: Dict[str, Any]) -> AgentResponse:
        """
        inputs = {
            "data":        list | dict | str (JSON),
            "title":       str,
            "description": str,
            "chart_type":  str  (optional — AI will choose if omitted),
            "export_fmt":  "png" | "html" | "both"
        }
        """
        resp = AgentResponse(agent_name=self.name)
        try:
            initial: VizState = {
                "raw_data":    inputs.get("data", []),
                "data_df":     None,
                "title":       inputs.get("title", "Data Visualization"),
                "description": inputs.get("description", ""),
                "chart_type":  inputs.get("chart_type", ""),
                "x_col":       inputs.get("x_col"),
                "y_col":       inputs.get("y_col"),
                "color_col":   inputs.get("color_col"),
                "export_fmt":  inputs.get("export_fmt", "html"),
                "export_paths": [],
                "error":        None,
            }
            final = self._graph.invoke(initial)

            if final.get("error"):
                return resp.fail(final["error"])

            resp.output       = final.get("export_paths", [])
            resp.export_paths = final.get("export_paths", [])
            resp.metadata     = {
                "chart_type": final.get("chart_type"),
                "x_col":      final.get("x_col"),
                "y_col":      final.get("y_col"),
                "color_col":  final.get("color_col"),
            }
        except Exception as exc:
            return resp.fail(str(exc))
        return resp

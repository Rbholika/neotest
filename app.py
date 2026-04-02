"""
ui/app.py — Streamlit Frontend for the Modular AI Agent Framework
==================================================================
Run:  streamlit run ui/app.py --server.port 8501
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import streamlit as st
from datetime import datetime

from config import CONFIG
from orchestrator.multi_agent import MultiAgentOrchestrator
from agents.synthetic_data_agent import SyntheticDataAgent
from agents.file_upload_agent    import FileUploadAgent
from agents.rag_agent            import RAGAgent
from agents.visualization_agent  import VisualizationAgent
from tools.export_tool           import ExportTool
from tools.voice_tool            import VoiceTool

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Agent Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563EB 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .agent-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .success-badge {
        background: #D1FAE5; color: #065F46;
        padding: 2px 8px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600;
    }
    .error-badge {
        background: #FEE2E2; color: #991B1B;
        padding: 2px 8px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600;
    }
    .warning-badge {
        background: #FEF3C7; color: #92400E;
        padding: 2px 8px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600;
    }
    div[data-testid="stTab"] button { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────

for key, default in {
    "rag_collection": "default",
    "chat_history":   [],
    "last_response":  "",
    "export_paths":   [],
    "problem_stmt":   "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Singletons ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_orchestrator():
    return MultiAgentOrchestrator()

@st.cache_resource
def get_voice():
    return VoiceTool()

@st.cache_resource
def get_export():
    return ExportTool()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.title("⚙️ Framework Config")

    st.subheader("🤖 Model")
    provider = st.radio("Provider", ["Local (Ollama)", "Azure"], horizontal=True)
    CONFIG.PROVIDER = "local" if "Local" in provider else "azure"

    if "Local" in provider:
        from config import LocalChatModel
        model = st.selectbox(
            "Chat Model",
            [m.value for m in LocalChatModel],
            index=0,
        )
        CONFIG.LOCAL_CHAT_MODEL = model
    else:
        from config import AzureChatModel
        model = st.selectbox(
            "Azure Deployment",
            [m.value for m in AzureChatModel],
            index=2,
        )
        CONFIG.AZURE_CHAT_DEPLOYMENT = model

    st.divider()
    st.subheader("🔒 Security")
    CONFIG.ENABLE_GUARDRAILS    = st.toggle("Guardrails",     value=True)
    CONFIG.ENABLE_PII_DETECTION = st.toggle("PII Detection",  value=True)

    st.divider()
    st.subheader("🗃️ Collection")
    st.session_state.rag_collection = st.text_input(
        "Active Collection", value=st.session_state.rag_collection
    )

    st.divider()
    st.subheader("📁 Exports")
    if st.session_state.export_paths:
        for p in st.session_state.export_paths:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    ext = os.path.splitext(p)[1].lstrip(".")
                    mime_map = {
                        "pdf": "application/pdf",
                        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "md":   "text/markdown",
                        "html": "text/html",
                        "png":  "image/png",
                        "json": "application/json",
                        "csv":  "text/csv",
                    }
                    st.download_button(
                        f"⬇️ {os.path.basename(p)}",
                        data=f,
                        file_name=os.path.basename(p),
                        mime=mime_map.get(ext, "application/octet-stream"),
                    )
    else:
        st.caption("Exported files will appear here.")


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h2 style="margin:0">🤖 Modular AI Agent Framework</h2>
    <p style="margin:4px 0 0; opacity:0.85">
        Synthetic Data • Document Ingestion • RAG Analysis • Visualization
    </p>
</div>
""", unsafe_allow_html=True)

# ── Problem Statement ─────────────────────────────────────────────────────────

with st.expander("📋 Problem Statement / Context (optional)", expanded=False):
    st.session_state.problem_stmt = st.text_area(
        "Describe the problem context — agents will use this as background.",
        value=st.session_state.problem_stmt,
        height=100,
        placeholder="E.g. DevOps Incident Response: analyse post-mortem reports …",
    )

# ── Main Tabs ─────────────────────────────────────────────────────────────────

tab_orch, tab_synth, tab_upload, tab_rag, tab_viz = st.tabs([
    "🎯 Orchestrator",
    "🧬 Synthetic Data",
    "📤 File Upload",
    "🔍 RAG Analysis",
    "📊 Visualization",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

with tab_orch:
    st.subheader("🎯 Multi-Agent Orchestrator")
    st.caption("Describe what you need — the orchestrator routes to the right agents automatically.")

    col_input, col_voice = st.columns([5, 1])
    with col_input:
        user_input = st.text_area(
            "Your request",
            placeholder=(
                "E.g. 'Generate 30 synthetic incident records and visualise them as a bar chart'"
                "\n  or 'Analyse the uploaded post-mortem and summarise root causes'"
            ),
            height=100,
        )
    with col_voice:
        st.write("")
        st.write("")
        if st.button("🎤 Voice Input", use_container_width=True):
            with st.spinner("Listening …"):
                voice  = get_voice()
                spoken = voice.listen(timeout=15, phrase_limit=30)
                if spoken:
                    st.success(f"Heard: {spoken}")
                    user_input = spoken
                else:
                    st.warning("No speech detected.")

    context = {}
    with st.expander("⚙️ Extra Parameters (optional)"):
        col1, col2 = st.columns(2)
        with col1:
            context["collection"]  = st.text_input("Collection", st.session_state.rag_collection)
            context["num_rows"]    = st.number_input("Rows (synthetic)", 5, 500, 20)
        with col2:
            context["export"]      = st.checkbox("Export report", value=True)
            context["export_fmt"]  = st.selectbox("Export format", ["pdf", "docx", "md"])

    if st.button("🚀 Run Orchestrator", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a request.")
        else:
            with st.spinner("Orchestrating agents …"):
                orch = get_orchestrator()
                orch.problem_statement = st.session_state.problem_stmt
                result = orch.run(user_input, context=context)

            st.divider()
            if result["success"]:
                st.success(f"✅ Completed — Agents: {result['route_plan']}")
            else:
                st.error(f"❌ {result['error']}")

            # Agent result cards
            for agent_name, res in result["agent_results"].items():
                with st.expander(
                    f"{'✅' if res.get('success') else '❌'} {agent_name.replace('_',' ').title()}",
                    expanded=True,
                ):
                    if res.get("raw_text"):
                        st.markdown(res["raw_text"][:2000])
                    if res.get("export_paths"):
                        st.session_state.export_paths.extend(res["export_paths"])
                        for p in res["export_paths"]:
                            st.info(f"📁 Exported: {p}")
                    if res.get("warnings"):
                        for w in res["warnings"]:
                            st.warning(w)
                    if res.get("errors"):
                        for e in res["errors"]:
                            st.error(e)

            st.session_state.last_response = result.get("synthesised_output", "")

    # TTS for last result
    if st.session_state.last_response:
        if st.button("🔊 Read Result Aloud"):
            with st.spinner("Speaking …"):
                get_voice().speak(st.session_state.last_response[:1500])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════════════════

with tab_synth:
    st.subheader("🧬 Synthetic Data Generator")

    col1, col2 = st.columns([2, 1])
    with col1:
        problem_ctx = st.text_area(
            "Problem context",
            value=st.session_state.problem_stmt,
            height=80,
            placeholder="Describe the domain for realistic data …",
        )
    with col2:
        num_rows   = st.number_input("Number of rows", 5, 500, 20)
        output_fmt = st.radio("Output format", ["json", "csv"], horizontal=True)

    st.write("**Define fields:**")
    raw_schema = st.text_area(
        "Schema JSON",
        height=200,
        value=json.dumps({
            "fields": [
                {"name": "incident_id",  "type": "integer",    "description": "Unique incident ID"},
                {"name": "timestamp",    "type": "datetime",   "description": "ISO datetime"},
                {"name": "severity",     "type": "string",     "description": "P1/P2/P3/P4"},
                {"name": "service",      "type": "string",     "description": "Affected service name"},
                {"name": "description",  "type": "text",       "description": "Incident summary"},
                {"name": "resolved",     "type": "boolean",    "description": "Resolution status"},
            ],
            "num_rows": 20,
            "constraints": "severity distribution: 10% P1, 20% P2, 40% P3, 30% P4",
        }, indent=2),
    )

    if st.button("⚡ Generate Synthetic Data", type="primary"):
        try:
            schema = json.loads(raw_schema)
            schema["num_rows"] = num_rows
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON schema: {e}")
            st.stop()

        with st.spinner("Generating data …"):
            agent  = SyntheticDataAgent()
            result = agent.invoke({
                "problem_statement": problem_ctx,
                "schema":            schema,
                "output_format":     output_fmt,
            })

        if result.success:
            st.success(f"✅ Generated {len(result.output)} rows")
            st.dataframe(result.output, use_container_width=True)
            if result.export_paths:
                st.session_state.export_paths.extend(result.export_paths)
                st.info(f"📁 Saved to: {result.export_paths[0]}")
        else:
            for e in result.errors:
                st.error(e)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.subheader("📤 Document Ingestion Agent")

    collection = st.text_input(
        "Collection name",
        value=st.session_state.rag_collection,
        help="Documents are stored in this Chroma collection.",
    )

    uploaded = st.file_uploader(
        "Upload document",
        type=CONFIG.ALLOWED_FILE_TYPES,
        help=f"Supported: {', '.join(CONFIG.ALLOWED_FILE_TYPES)}",
    )

    if uploaded and st.button("📥 Ingest Document", type="primary"):
        os.makedirs(CONFIG.UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(CONFIG.UPLOAD_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner(f"Processing {uploaded.name} …"):
            agent  = FileUploadAgent()
            result = agent.invoke({
                "file_path":  save_path,
                "collection": collection,
            })

        if result.success:
            st.success(
                f"✅ Ingested **{uploaded.name}** — "
                f"{result.metadata.get('chunks', 0)} chunks indexed"
            )
            if result.warnings:
                for w in result.warnings:
                    st.warning(f"⚠️ {w}")
            st.session_state.rag_collection = collection
            st.json(result.metadata)
        else:
            for e in result.errors:
                st.error(e)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAG ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_rag:
    st.subheader("🔍 RAG Analysis Agent")

    col_q, col_v = st.columns([5, 1])
    with col_q:
        query = st.text_area(
            "Your question",
            placeholder="E.g. What were the top 3 root causes of incidents last month?",
            height=80,
        )
    with col_v:
        st.write("")
        st.write("")
        if st.button("🎤 Voice Query", use_container_width=True, key="voice_rag"):
            with st.spinner("Listening …"):
                spoken = get_voice().listen()
                if spoken:
                    query = spoken
                    st.success(spoken)

    col1, col2, col3 = st.columns(3)
    with col1:
        rag_collection = st.text_input("Collection", st.session_state.rag_collection, key="rag_col")
    with col2:
        do_export  = st.checkbox("Export report", value=True)
        export_fmt = st.selectbox("Format", ["pdf", "docx", "md"], key="rag_fmt")
    with col3:
        st.write("")

    if st.button("🔍 Analyse", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and analysing …"):
                agent  = RAGAgent()
                result = agent.invoke({
                    "query":      query,
                    "collection": rag_collection,
                    "export":     do_export,
                    "export_fmt": export_fmt,
                })

            if result.success:
                st.success("✅ Analysis complete")
                st.markdown("### 📝 Response")
                st.markdown(result.output)

                if result.metadata.get("sources"):
                    with st.expander("📚 Sources"):
                        for src in result.metadata["sources"]:
                            st.markdown(f"- {src}")

                if result.export_paths:
                    st.session_state.export_paths.extend(result.export_paths)
                    st.info(f"📁 Report exported: {result.export_paths[0]}")

                st.session_state.last_response = result.output

                if st.button("🔊 Read Response", key="tts_rag"):
                    get_voice().speak(result.output[:1500])
            else:
                for e in result.errors:
                    st.error(e)

    # Chat history
    if st.session_state.chat_history:
        with st.expander("💬 Query History"):
            for item in reversed(st.session_state.chat_history[-10:]):
                st.markdown(f"**Q:** {item['q']}")
                st.markdown(f"**A:** {item['a'][:300]} …")
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

with tab_viz:
    st.subheader("📊 Visualization Agent")

    viz_title = st.text_input("Chart title", "Data Analysis")
    viz_desc  = st.text_area(
        "Description (helps AI choose the best chart)",
        placeholder="E.g. Show monthly incident counts grouped by severity",
        height=60,
    )

    col1, col2 = st.columns(2)
    with col1:
        chart_type = st.selectbox(
            "Chart type (or let AI decide)",
            ["auto"] + ["bar", "line", "pie", "scatter",
                        "histogram", "heatmap", "box", "area"],
        )
    with col2:
        viz_fmt = st.selectbox("Export format", ["html", "png", "both"])

    data_src = st.radio("Data source", ["Upload JSON/CSV", "Paste JSON"], horizontal=True)

    data = None
    if data_src == "Upload JSON/CSV":
        viz_file = st.file_uploader("Upload data file", type=["json", "csv"])
        if viz_file:
            if viz_file.name.endswith(".json"):
                data = json.load(viz_file)
            else:
                import csv, io
                reader = csv.DictReader(io.StringIO(viz_file.read().decode()))
                data   = list(reader)
    else:
        raw_json = st.text_area(
            "Paste JSON array",
            height=150,
            placeholder='[{"month": "Jan", "incidents": 12, "severity": "P2"}, ...]',
        )
        if raw_json.strip():
            try:
                data = json.loads(raw_json)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    if data and st.button("📊 Generate Chart", type="primary"):
        with st.spinner("Rendering …"):
            agent  = VisualizationAgent()
            result = agent.invoke({
                "data":        data,
                "title":       viz_title,
                "description": viz_desc,
                "chart_type":  "" if chart_type == "auto" else chart_type,
                "export_fmt":  viz_fmt,
            })

        if result.success:
            st.success(
                f"✅ Chart generated — type: {result.metadata.get('chart_type')}"
            )
            for path in result.export_paths:
                st.session_state.export_paths.append(path)
                if path.endswith(".html"):
                    with open(path, "r") as f:
                        st.components.v1.html(f.read(), height=550, scrolling=True)
                elif path.endswith(".png"):
                    st.image(path)
        else:
            for e in result.errors:
                st.error(e)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"🤖 AI Agent Framework • {CONFIG.PROVIDER} • "
    f"Guardrails {'✅' if CONFIG.ENABLE_GUARDRAILS else '❌'} • "
    f"PII Detection {'✅' if CONFIG.ENABLE_PII_DETECTION else '❌'} • "
    f"© {datetime.now().year}"
)

# 🤖 Modular AI Agent Framework

A production-ready, modular multi-agent AI framework built on **LangGraph**, **LangChain**, and **Streamlit** — designed to plug into any problem statement.

---

## 📁 Project Structure

```
framework/
├── config.py                    # Central config (models, paths, security)
├── requirements.txt
├── core/
│   ├── logger.py                # Structured logging + webhook alerts
│   └── security.py              # PII detection, guardrails, prompt injection
├── agents/
│   ├── base_agent.py            # Abstract base with guarded LLM calls
│   ├── synthetic_data_agent.py  # Generate synthetic datasets
│   ├── file_upload_agent.py     # Ingest PDF/DOCX/CSV into Chroma
│   ├── rag_agent.py             # RAG Q&A with citations
│   └── visualization_agent.py  # AI-selected charts (Plotly)
├── tools/
│   ├── export_tool.py           # Export to PDF / DOCX / Markdown
│   └── voice_tool.py            # TTS (pyttsx3) + STT (SpeechRecognition)
├── orchestrator/
│   └── multi_agent.py           # LangGraph multi-agent router
└── ui/
    └── app.py                   # Streamlit frontend
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt --break-system-packages

# 2. Configure environment (copy and edit)
cp .env.example .env

# 3. Run the UI
streamlit run ui/app.py --server.port 8501
```

---

## ⚙️ Configuration

Edit `config.py` or set environment variables:

| Variable | Description | Default |
|---|---|---|
| `PROVIDER` | `local` or `azure` | `local` |
| `LOCAL_CHAT_MODEL` | Ollama model name | `llama3.2:3b` |
| `AZURE_API_KEY` | Azure API key | env var |
| `AZURE_ENDPOINT` | Azure endpoint | env var |
| `ENABLE_GUARDRAILS` | LLM security checks | `True` |
| `ENABLE_PII_DETECTION` | Mask PII in data | `True` |

---

## 🧩 Using Agents Independently

### Synthetic Data Agent
```python
from agents.synthetic_data_agent import SyntheticDataAgent

agent = SyntheticDataAgent()
result = agent.invoke({
    "problem_statement": "DevOps incident reporting system",
    "schema": {
        "fields": [
            {"name": "incident_id", "type": "integer"},
            {"name": "severity",    "type": "string"},
            {"name": "service",     "type": "string"},
            {"name": "resolved",    "type": "boolean"},
        ],
        "num_rows": 50,
        "constraints": "30% P1, 70% P2"
    },
    "output_format": "csv",
})
print(result.export_paths)
```

### File Upload Agent
```python
from agents.file_upload_agent import FileUploadAgent

agent = FileUploadAgent()
result = agent.invoke({
    "file_path":  "postmortem_report.pdf",
    "collection": "incidents",
})
```

### RAG Agent
```python
from agents.rag_agent import RAGAgent

agent = RAGAgent()
result = agent.invoke({
    "query":      "What were the top root causes?",
    "collection": "incidents",
    "export":     True,
    "export_fmt": "pdf",
})
print(result.output)          # Answer text
print(result.export_paths)    # PDF path
```

### Visualization Agent
```python
from agents.visualization_agent import VisualizationAgent

agent = VisualizationAgent()
result = agent.invoke({
    "data":        [{"month": "Jan", "incidents": 12}, ...],
    "title":       "Monthly Incidents",
    "description": "Show trend by month",
    "export_fmt":  "html",
})
```

### Multi-Agent Orchestrator
```python
from orchestrator.multi_agent import MultiAgentOrchestrator

orch = MultiAgentOrchestrator(
    problem_statement="DevOps Incident Response"
)
result = orch.run(
    "Generate 30 synthetic incidents and visualise by severity",
    context={"num_rows": 30, "export": True}
)
```

---

## 🛡️ Security Features

- **PII Detection & Masking** — emails, phones, SSN, credit cards, Aadhaar, PAN
- **Prompt Injection Detection** — blocks jailbreak attempts
- **Harmful Content Policy** — rejects requests for malware, weapons, etc.
- **Output Guardrails** — scans LLM responses for leaked PII
- **Audit Logging** — every LLM call logged with latency, token count

---

## 🎤 Voice Features

```python
from tools.voice_tool import voice_tool

# Text → Speech
voice_tool.speak("Analysis complete. 3 critical incidents detected.")

# Speech → Text
query = voice_tool.listen(timeout=10)

# Save to file
voice_tool.save_speech("Report summary.", "output.mp3")
```

---

## 📦 Exporting Results

```python
from tools.export_tool import ExportTool

tool = ExportTool()
tool.to_pdf({"title": "Report", "response": "..."})
tool.to_docx({"title": "Report", "response": "..."})
tool.to_markdown({"title": "Report", "response": "..."})
```

---

## 🔌 Adapting to a Problem Statement

Each agent reads `problem_statement` from its inputs and tailors prompts accordingly. To integrate a new use case:

1. Set `problem_statement` in the orchestrator or agent input
2. Define the appropriate `schema` for synthetic data
3. Choose the `collection` name for document storage
4. Run — agents adapt automatically

---

## 📋 Available Models

### Local (Ollama)
- `llama3.2:3b` — Fast general chat
- `gemma3:4b` — Balanced capability
- `qwen2.5-coder:latest` — Code-optimised
- `deepseek-r1:latest` — Reasoning tasks

### Azure
- `genailab-maas-gpt-40-mini` — Fast, cost-effective
- `genailab-maas-gpt-40` — Highest quality
- `genailab-maas-Llama-3.3-70B-Instruct` — Open source large
- `genailab-maas-DeepSeek-R1` — Advanced reasoning

---

*Built for Python 3.12.8 • Ollama + Azure AI • LangGraph + LangChain*

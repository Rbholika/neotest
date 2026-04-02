"""
config.py — Central Configuration for Modular AI Agent Framework
================================================================
All agents, tools, and orchestrators import from here.
Switch between LOCAL (Ollama) and AZURE providers via PROVIDER flag.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


# ── Provider & Model Enums ────────────────────────────────────────────────────

class ModelProvider(str, Enum):
    LOCAL = "local"   # Ollama
    AZURE = "azure"   # Azure AI / OpenAI


class LocalChatModel(str, Enum):
    LLAMA   = "llama3.2:3b"
    GEMMA   = "gemma3:4b"
    QWEN    = "qwen2.5-coder:latest"
    DEEPSEEK = "deepseek-r1:latest"


class AzureChatModel(str, Enum):
    GPT35       = "genailab-maas-gpt-35-turbo"
    GPT4O       = "genailab-maas-gpt-40"
    GPT4O_MINI  = "genailab-maas-gpt-40-mini"
    DEEPSEEK_R1 = "genailab-maas-DeepSeek-R1"
    LLAMA_70B   = "genailab-maas-Llama-3.3-70B-Instruct"
    LLAMA_4     = "genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8"


# ── Master Config ─────────────────────────────────────────────────────────────

@dataclass
class Config:

    # Provider selection — change this to switch between local and cloud
    PROVIDER: ModelProvider = ModelProvider.LOCAL

    # ── Ollama / Local ────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOCAL_CHAT_MODEL: str = LocalChatModel.LLAMA.value
    LOCAL_EMBEDDING_MODEL: str = "gte-large"

    # ── Azure ─────────────────────────────────────────────────────────────────
    AZURE_API_KEY: str      = field(default_factory=lambda: os.getenv("AZURE_API_KEY", ""))
    AZURE_ENDPOINT: str     = field(default_factory=lambda: os.getenv("AZURE_ENDPOINT", ""))
    AZURE_API_VERSION: str  = "2024-02-01"
    AZURE_CHAT_DEPLOYMENT: str      = AzureChatModel.GPT4O_MINI.value
    AZURE_EMBED_DEPLOYMENT: str     = "genailab-maas-text-embedding-3-large"

    # ── LLM Generation Params ─────────────────────────────────────────────────
    TEMPERATURE: float = 0.1    # Low temperature → less hallucination
    MAX_TOKENS: int    = 2048
    TOP_P: float       = 0.9

    # ── Paths ─────────────────────────────────────────────────────────────────
    CHROMA_DIR:  str = "./chroma_db"
    EXPORT_DIR:  str = "./exports"
    LOG_DIR:     str = "./logs"
    UPLOAD_DIR:  str = "./uploads"

    # ── Logging & Alerts ──────────────────────────────────────────────────────
    LOG_LEVEL:      str            = "INFO"
    ALERT_WEBHOOK:  Optional[str]  = None   # Slack / Teams webhook URL

    # ── Security ──────────────────────────────────────────────────────────────
    ENABLE_GUARDRAILS:   bool      = True
    ENABLE_PII_DETECTION: bool     = True
    MAX_FILE_SIZE_MB:    int       = 50
    ALLOWED_FILE_TYPES:  List[str] = field(default_factory=lambda: [
        "pdf", "docx", "txt", "csv", "json", "xlsx"
    ])

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_K: int = 5        # Top-k docs for RAG
    CHUNK_SIZE:  int = 512
    CHUNK_OVERLAP: int = 64

    # ── Voice ─────────────────────────────────────────────────────────────────
    TTS_RATE:   int = 175       # Words per minute
    TTS_VOLUME: float = 0.9


    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_llm(self, streaming: bool = False):
        """Return configured LangChain LLM instance."""
        if self.PROVIDER == ModelProvider.LOCAL:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                base_url=self.OLLAMA_BASE_URL,
                model=self.LOCAL_CHAT_MODEL,
                temperature=self.TEMPERATURE,
                streaming=streaming,
            )
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=self.AZURE_ENDPOINT,
            azure_deployment=self.AZURE_CHAT_DEPLOYMENT,
            api_version=self.AZURE_API_VERSION,
            api_key=self.AZURE_API_KEY,
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            streaming=streaming,
        )

    def get_embeddings(self):
        """Return configured LangChain Embeddings instance."""
        if self.PROVIDER == ModelProvider.LOCAL:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(
                base_url=self.OLLAMA_BASE_URL,
                model=self.LOCAL_EMBEDDING_MODEL,
            )
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_endpoint=self.AZURE_ENDPOINT,
            azure_deployment=self.AZURE_EMBED_DEPLOYMENT,
            api_version=self.AZURE_API_VERSION,
            api_key=self.AZURE_API_KEY,
        )

    def ensure_dirs(self):
        """Create all required directories if they don't exist."""
        for d in [self.CHROMA_DIR, self.EXPORT_DIR, self.LOG_DIR, self.UPLOAD_DIR]:
            os.makedirs(d, exist_ok=True)


# Singleton instance — import this everywhere
CONFIG = Config()
CONFIG.ensure_dirs()

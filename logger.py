"""
core/logger.py — Structured Logging + Alert System
====================================================
Features:
  • Rotating file handler (daily logs)
  • Colour-coded console output
  • Webhook alerts (Slack / Teams) for WARNING+
  • Structured JSON log records for machine parsing
  • Simple audit trail for LLM calls
"""

import logging
import logging.handlers
import json
import os
import httpx
from datetime import datetime
from typing import Optional
from pathlib import Path


# ── ANSI colours ──────────────────────────────────────────────────────────────
COLOURS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}


class ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = COLOURS.get(record.levelname, COLOURS["RESET"])
        reset  = COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname:8}{reset}"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """Machine-readable JSON records written to log files."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "module":    record.module,
            "funcName":  record.funcName,
            "lineno":    record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload["extra"] = record.extra
        return json.dumps(payload)


class WebhookHandler(logging.Handler):
    """Sends WARNING+ log records to a Slack / Teams webhook."""
    def __init__(self, webhook_url: str, level=logging.WARNING):
        super().__init__(level)
        self.webhook_url = webhook_url

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            payload = {"text": f"*[{record.levelname}]* `{record.name}`\n{msg}"}
            httpx.post(self.webhook_url, json=payload, timeout=5)
        except Exception:
            self.handleError(record)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_logger(
    name: str,
    log_dir: str = "./logs",
    level: str = "INFO",
    webhook_url: Optional[str] = None,
) -> logging.Logger:
    """
    Return a named logger with console + rotating-file + optional webhook.
    Call once per module:  logger = get_logger(__name__)
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:          # already configured
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColouredFormatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    # Rotating file handler (1 file per day, keep 30 days)
    log_path = Path(log_dir) / f"{name.replace('.', '_')}.log"
    fh = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)

    # Optional webhook
    if webhook_url:
        wh = WebhookHandler(webhook_url)
        wh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(wh)

    return logger


# ── Audit logger for LLM calls ────────────────────────────────────────────────

class LLMAuditLogger:
    """Logs every LLM prompt / response pair for traceability."""

    def __init__(self, log_dir: str = "./logs"):
        self.logger = get_logger("llm.audit", log_dir=log_dir, level="INFO")
        self._audit_path = Path(log_dir) / "llm_audit.jsonl"

    def log_call(
        self,
        agent: str,
        prompt: str,
        response: str,
        model: str,
        latency_ms: float,
        tokens: Optional[int] = None,
    ):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent":     agent,
            "model":     model,
            "latency_ms": round(latency_ms, 2),
            "tokens":    tokens,
            "prompt_len": len(prompt),
            "response_len": len(response),
        }
        with open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.logger.info(
            f"LLM call | agent={agent} model={model} "
            f"latency={latency_ms:.0f}ms tokens={tokens}"
        )

    def log_alert(self, agent: str, message: str, level: str = "WARNING"):
        getattr(self.logger, level.lower(), self.logger.warning)(
            f"[ALERT] agent={agent} | {message}"
        )


# Module-level singletons
framework_logger = get_logger("framework")
audit_logger      = LLMAuditLogger()

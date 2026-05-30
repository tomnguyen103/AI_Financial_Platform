"""Structured logging setup for the platform.

Replaces ad-hoc print() calls with the stdlib logging module configured for
structured (JSON) output, so logs are levelled, filterable, and ready to ship to
an aggregator (ELK/EFK, Grafana Loki, CloudWatch, etc.) without reparsing.

A `request_id` ContextVar is injected into every record, so logs emitted deep in
the call stack (services, detectors, the LLM client) can be correlated back to
the originating HTTP request without threading an id through every function.

Usage:
    from app.logging_config import configure_logging, get_logger, request_id_var
    configure_logging()              # once, at startup
    log = get_logger(__name__)
    log.info("message", extra={"key": "value"})
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar

# Correlation id for the current request (set by the gateway middleware).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Keys that the stdlib LogRecord already owns; everything else a caller passes
# via `extra=` is treated as a structured field and merged into the JSON output.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        # Merge any structured fields passed via `extra=`.
        for key, val in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class PlainFormatter(logging.Formatter):
    """Human-friendly formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<5} " \
               f"[{rid}] {record.name}: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str | None = None, *, json_output: bool | None = None) -> None:
    """Configure the root logger once.

    level:       LOG_LEVEL env var or "INFO".
    json_output: LOG_JSON env var ("1"/"true") or True by default; set False for
                 the readable dev formatter.
    """
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    if json_output is None:
        json_output = os.getenv("LOG_JSON", "true").lower() in ("1", "true", "yes")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else PlainFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn's own access log duplicates our request logging; quiet it.
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

"""
Structured logging configuration.

Emits JSON-formatted log records so they are natively parsed by Cloud
Logging on Cloud Run / GKE, while remaining human-readable in local dev.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.utils.config import get_settings


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Correlate with Cloud Trace / request IDs when present.
        for attr in ("request_id", "trace_id", "job_id"):
            value = getattr(record, attr, None)
            if value is not None:
                payload[attr] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure the root logger once at process startup."""
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    # Avoid duplicate handlers when reloaded (e.g. under uvicorn --reload).
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Quiet noisy third-party loggers unless we're debugging.
    for noisy in ("uvicorn.access", "google.auth", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)

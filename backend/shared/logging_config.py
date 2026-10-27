"""Logging setup.

One consistent format everywhere: a timestamp, the module that logged it, and
-- most usefully -- a short trace ID shared by every log line belonging to the
same request, so `grep <trace_id>` on the server log reconstructs one query's
entire journey from question to response.

Logs go to stdout AND to outputs/logs/backend.log (rotated daily, 14 days
kept). Structured per-question and per-LLM-call records are appended to
outputs/logs/questions.jsonl and outputs/logs/llm_calls.jsonl by
append_jsonl() -- the source for the /api/history endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from logging.handlers import TimedRotatingFileHandler

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_JSONL_LOCK = threading.Lock()


def _log_dir() -> str:
    from shared import config

    return config.LOG_DIR


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. uvicorn --reload re-imports)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    # A Windows console (cp1252) cannot print Hindi / Tamil questions: escape what it cannot
    # show instead of a logging traceback per line (backend.log below is UTF-8 and keeps them).
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    try:
        file_handler = TimedRotatingFileHandler(
            os.path.join(_log_dir(), "backend.log"), when="midnight", backupCount=14, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not open the backend log file -- logging to stdout only.")
    root.setLevel(LOG_LEVEL)
    # Quiet the noisiest third-party loggers unless LOG_LEVEL is DEBUG.
    if LOG_LEVEL != "DEBUG":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def append_jsonl(name: str, record: dict) -> None:
    """Append one JSON record to outputs/logs/<name>.jsonl. Never raises."""
    try:
        line = json.dumps(record, default=str, ensure_ascii=False)
        with _JSONL_LOCK, open(os.path.join(_log_dir(), f"{name}.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        logging.getLogger(__name__).debug("could not append to %s.jsonl", name, exc_info=True)


def new_trace_id() -> str:
    import uuid

    return uuid.uuid4().hex[:8]

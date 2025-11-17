"""Logging configuration for document review workflow."""

from __future__ import annotations

import logging
from pathlib import Path

from pythonjsonlogger import jsonlogger

_LOG_DIR = Path("external/logs")
_LOG_FILE = _LOG_DIR / "doc_review.log"
_HANDLER_MARKER = "_doc_review_log_handler"


class _DocReviewFilter(logging.Filter):
    """Filter to capture only document review related loggers."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple predicate
        name = record.name or ""
        return name.startswith("external.doc_review") or name.startswith("external.agent.doc_review")


def configure_doc_review_logging() -> None:
    """Attach a JSON file handler for document review logs."""
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(_DocReviewFilter())
    file_handler.setFormatter(
        jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    setattr(file_handler, _HANDLER_MARKER, True)
    root_logger.addHandler(file_handler)


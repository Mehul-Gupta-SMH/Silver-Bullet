"""Custom ASGI middleware for SilverBullet API.

Provides:
- RequestIDMiddleware  — attaches/generates X-Request-ID on every request
- LoggingMiddleware    — emits a structured JSON log line after each response
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context variable so that other parts of the app (e.g. endpoint handlers)
# can read the current request-ID without threading concerns.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Standard LogRecord attributes that should not be forwarded into the JSON
# payload as extra fields.  Defined once at module level to avoid rebuilding
# this set on every log call.
_LOG_RECORD_BUILTINS: frozenset[str] = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "taskName",
})


class _JsonFormatter(logging.Formatter):
    """Formats a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
        }
        # Merge any extra fields attached to the record.
        for key, value in record.__dict__.items():
            if key not in _LOG_RECORD_BUILTINS and not key.startswith("_"):
                payload[key] = value
        if record.getMessage():
            payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("silverbullet.api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_logger = _build_logger()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Read X-Request-ID from the incoming request or generate a UUID4.

    The value is:
    - Stored in the ``request_id_var`` context variable for downstream use.
    - Echoed back in the ``X-Request-ID`` response header.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(req_id)
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = req_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Emit a structured JSON log line after every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)

        _logger.info(
            "",
            extra={
                "request_id": request_id_var.get(),
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

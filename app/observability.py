import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


_CTX: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})

_SENSITIVE_KEY_EXACT = {
    "authorization",
    "authorization_code",
    "archive_api_key",
    "cookie",
    "email",
    "secret",
    "secret_key",
    "token",
    "token_hash",
    "token_encrypted",
    "x-archive-api-key",
}

_SENSITIVE_KEY_SUBSTR = (
    "api_key",
    "authorization",
    "cookie",
    "email",
    "password",
    "secret",
    "token",
)


def bind_context(**fields: Any) -> None:
    ctx = dict(_CTX.get())
    for key, value in fields.items():
        if value is None:
            ctx.pop(key, None)
        else:
            ctx[key] = value
    _CTX.set(ctx)


def clear_context(*keys: str) -> None:
    if not keys:
        _CTX.set({})
        return
    ctx = dict(_CTX.get())
    for key in keys:
        ctx.pop(key, None)
    _CTX.set(ctx)


def get_context() -> Dict[str, Any]:
    return dict(_CTX.get())


def _is_sensitive_key(key: str) -> bool:
    k = key.lower()
    if k in _SENSITIVE_KEY_EXACT:
        return True
    return any(substr in k for substr in _SENSITIVE_KEY_SUBSTR)


def _redact_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if _is_sensitive_key(str(k)):
                out[str(k)] = "[REDACTED]"
            else:
                out[str(k)] = _redact_obj(v)
        return out
    if isinstance(obj, list):
        return [_redact_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [_redact_obj(v) for v in obj]
    return obj


def _truncate_str(value: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars <= 0:
        return value, False
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars] + "…[TRUNCATED]", True


def sanitize_json_text(text: str, *, max_chars: Optional[int] = None) -> Tuple[str, bool]:
    max_chars_val = max_chars if max_chars is not None else int(os.getenv("LOG_BODY_MAX_CHARS", "8000"))
    try:
        parsed = json.loads(text)
        redacted = _redact_obj(parsed)
        rendered = json.dumps(redacted, ensure_ascii=False, separators=(",", ":"), default=str)
        return _truncate_str(rendered, max_chars_val)
    except Exception:
        return _truncate_str(text, max_chars_val)


def sanitize_json_bytes(data: bytes, *, max_chars: Optional[int] = None) -> Tuple[str, bool]:
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = repr(data)
    return sanitize_json_text(text, max_chars=max_chars)


def sanitize_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in headers.items():
        if _is_sensitive_key(k):
            out[k] = "[REDACTED]"
        else:
            out[k] = v
    return out


class JsonFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        ctx = get_context()
        if ctx:
            payload.update(ctx)

        # Include any explicit structured fields passed via `extra=`.
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        for k, v in record.__dict__.items():
            if k in reserved or k.startswith("_"):
                continue
            payload.setdefault(k, v)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


_CONFIGURED = False


def setup_logging(*, level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_service(service: str) -> None:
    bind_context(service=service)


def current_request_id() -> Optional[str]:
    ctx = get_context()
    rid = ctx.get("request_id")
    return str(rid) if rid else None


def set_request_id(request_id: str) -> None:
    bind_context(request_id=request_id)


def safe_dict(items: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    return {str(k): v for k, v in items.items()}


def safe_params(params: Optional[Iterable[Tuple[str, Any]]]) -> Dict[str, Any]:
    if not params:
        return {}
    out: Dict[str, Any] = {}
    for k, v in params:
        if _is_sensitive_key(k):
            out[str(k)] = "[REDACTED]"
        else:
            out[str(k)] = v
    return out

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorEnvelope(BaseModel):
    error: str
    message: str | None = None


def error_response(error: str, message: str | None = None, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorEnvelope(error=error, message=message).model_dump())


async def http_exception_handler(request: Request, exc):  # type: ignore[override]
    # Fallback handler to normalize FastAPI HTTPException.
    detail = getattr(exc, "detail", None)
    error = "error"
    message = None
    if isinstance(detail, dict):
        error = detail.get("error", error)
        message = detail.get("message")
    elif isinstance(detail, str):
        # Most endpoints raise string codes like "job_not_found"; treat as the error code.
        error = detail
    return error_response(error=error, message=message, status_code=getattr(exc, "status_code", 400))

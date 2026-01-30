import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from uuid import uuid4
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from sqlalchemy import func, or_, text
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import URLSafeSerializer

from . import accessories
from .crypto import decrypt
from .errors import error_response, http_exception_handler
from .schemas import (
    CodeResponse,
    ErrorResponse,
    FinalizeRequest,
    FinalizeResponse,
    AccessorySet,
    LinkStartResponse,
    RedirectResponse,
    RedirectClickRequest,
    RegisterEmailRequest,
    ReferralImpressionRequest,
    ReferralRequest,
    ReferralResponse,
    WrappedPayload,
    WrappedStatusResponse,
    WaitlistRequest,
    VerifyRegionResponse,
    AdminTestRunRequest,
    AdminTestRunResponse,
    AdminTestRunStatusResponse,
    AdminTestEnqueueResponse,
    AdminVerifyRegionEnqueueResponse,
    AdminVerifyRegionBatchRequest,
    AdminVerifyRegionBatchItem,
    AdminVerifyRegionBatchResponse,
    AdminVerifyRegionBatchStatusItem,
    AdminVerifyRegionBatchStatusResponse,
    AdminWrappedRetryRequest,
    AdminRetryFailedRunsRequest,
    AdminRetryFailedRunsResponse,
    AdminRetryFailedRunsItem,
    AdminRetryZeroVideosRequest,
    AdminRetryZeroVideosItem,
    AdminRetryZeroVideosResponse,
    AdminUserStageResponse,
    AdminUserStageVerify,
    AdminUserRestartRequest,
    AdminUserRestartResponse,
    AdminUserStageBatchRequest,
    AdminUserStageBatchResponse,
    AdminUserRestartBatchRequest,
    AdminUserRestartBatchItem,
    AdminUserRestartBatchResponse,
    AdminPromptPipelineRequest,
    AdminPromptPipelineResponse,
    AdminPromptCallResult,
    AdminBrainrotScoreRequest,
    AdminBrainrotScoreResponse,
    AdminJobStatusResponse,
)
from .db import SessionLocal
from .models import AppAuthJob, AppJob, AppSession, AppUser, AppUserEmail, AppWrappedRun, DeviceEmail
from .models import Referral, ReferralEvent
from .services.archive_client import ArchiveClient
from .services.job_queue import DBJobQueue
from .services.session_service import SessionService, parse_bearer
from .settings import Settings, cors_origins_list, get_settings
from .admin_metrics import render_admin_metrics
from .observability import (
    bind_context,
    clear_context,
    sanitize_json_bytes,
    set_request_id,
    set_service,
    setup_logging,
    get_logger,
)
from .prompts import (
    BRAINROT_EXPLANATION_PROMPT,
    BRAINROT_SCORE_PROMPT,
    KEYWORD_2026_PROMPT,
    NICHE_JOURNEY_PROMPT,
    PERSONALITY_EXPLANATION_PROMPT,
    PERSONALITY_PROMPT,
    ROAST_THUMB_PROMPT,
    TOP_NICHES_PROMPT,
    accessory_fallback_reason,
)
from .routers import upload, weekly_report

settings = get_settings()
setup_logging(level=settings.log_level)
logger = get_logger(__name__)
session_service = SessionService(ttl_days=settings.session_ttl_days, secret_key=settings.secret_key.get_secret_value())
archive_client = ArchiveClient(settings)
job_queue = DBJobQueue()
token_signer = URLSafeSerializer(settings.secret_key.get_secret_value(), salt="wrapped-token")
WATCH_HISTORY_VERIFY_FINALIZE_MAX_ATTEMPTS = max(
    1, int(os.getenv("WATCH_HISTORY_VERIFY_FINALIZE_MAX_ATTEMPTS", "60"))
)
AUTO_CANCEL_QUEUE_ON_MAX_POSITION = os.getenv("AUTO_CANCEL_QUEUE_ON_MAX_POSITION", "").lower() in ("1", "true", "yes", "on")
MAX_QUEUE_POSITION = max(0, int(os.getenv("MAX_QUEUE_POSITION", "0")))

app = FastAPI(title="TikTok Wrapped Backend")
app.add_exception_handler(HTTPException, http_exception_handler)

# Register routers
app.include_router(weekly_report.router)
app.include_router(upload.router)


async def request_validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response("validation_error", message=str(exc), status_code=422)


app.add_exception_handler(RequestValidationError, request_validation_exception_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list(settings),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    request_id = request.headers.get("X-Request-Id") or str(uuid4())
    set_service("web")
    set_request_id(request_id)
    client_ip, client_ip_source, client_ip_raw, x_real_ip, x_forwarded_for = _resolve_client_ip(request)

    logger.info(
        "http.request",
        extra={
            "event": "http.request",
            "http_method": request.method,
            "path": request.url.path,
            "query": str(request.url.query) if request.url.query else None,
            "client_ip": client_ip,
            "client_ip_source": client_ip_source,
            "client_ip_raw": client_ip_raw,
            "x_forwarded_for": x_forwarded_for,
            "x_real_ip": x_real_ip,
            "user_agent": request.headers.get("User-Agent"),
            "device_id": request.headers.get("X-Device-Id"),
            "platform": request.headers.get("X-Platform"),
            "app_version": request.headers.get("X-App-Version"),
            "os_version": request.headers.get("X-OS-Version"),
        },
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "http.exception",
            extra={
                "event": "http.exception",
                "http_method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            },
        )
        clear_context()
        raise

    duration_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Request-Id"] = request_id
    logger.info(
        "http.response",
        extra={
            "event": "http.response",
            "http_method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    clear_context()
    return response


# --- Helpers ---------------------------------------------------------------

def _resolve_client_ip(request: Request) -> Tuple[Optional[str], str, Optional[str], Optional[str], Optional[str]]:
    client_ip_raw = request.client.host if request.client else None
    x_real_ip = request.headers.get("X-Real-IP")
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = client_ip_raw
    client_ip_source = "socket"
    if x_real_ip:
        client_ip = x_real_ip.split(",")[0].strip()
        client_ip_source = "x_real_ip"
    elif x_forwarded_for:
        client_ip = x_forwarded_for.split(",")[0].strip()
        client_ip_source = "x_forwarded_for"
    if client_ip is not None:
        client_ip = client_ip.strip()
        if not client_ip:
            client_ip = None
    return client_ip, client_ip_source, client_ip_raw, x_real_ip, x_forwarded_for


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_session(
    authorization: str = Header(..., alias="Authorization"),
    device_id: str = Header(..., alias="X-Device-Id"),
    platform: str = Header(..., alias="X-Platform"),
    app_version: str = Header(..., alias="X-App-Version"),
    os_version: str = Header(..., alias="X-OS-Version"),
    db=Depends(get_db),
):
    token = parse_bearer(authorization)
    rec = session_service.validate(db, token, device_id=device_id)
    bind_context(
        app_user_id=rec.app_user_id,
        device_id=device_id,
        platform=platform,
        app_version=app_version,
        os_version=os_version,
        session_id=rec.id,
    )
    return rec


def require_device(
    device_id: str = Header(..., alias="X-Device-Id"),
    platform: str = Header(..., alias="X-Platform"),
    app_version: str = Header(..., alias="X-App-Version"),
    os_version: str = Header(..., alias="X-OS-Version"),
):
    bind_context(
        device_id=device_id,
        platform=platform,
        app_version=app_version,
        os_version=os_version,
    )
    return {
        "device_id": device_id,
        "platform": platform,
        "app_version": app_version,
        "os_version": os_version,
    }


def require_admin(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
):
    expected = settings.admin_api_key.get_secret_value() if settings.admin_api_key else None
    if not expected:
        raise HTTPException(status_code=404, detail="not_found")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_admin_key")
    return True


def _format_http_error(err: Exception) -> str:
    if isinstance(err, httpx.HTTPStatusError) and err.response is not None:
        return f"{err.response.status_code} {err.response.text}"
    return str(err)


def _clamp_0_100(value: Any) -> int:
    with suppress(Exception):
        return max(0, min(100, int(round(float(value)))))
    return 0


def _sigmoid(z: float) -> float:
    if z >= 60:
        return 1.0
    if z <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _compute_brainrot_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    brainrot = summary.get("brainrot") if isinstance(summary, dict) else None
    coverage = summary.get("coverage") if isinstance(summary, dict) else None

    raw = None
    confidence = None
    watch_seconds_enriched = None
    with suppress(Exception):
        if isinstance(brainrot, dict):
            raw = float(brainrot.get("raw"))
            confidence = float(brainrot.get("confidence"))
    with suppress(Exception):
        if isinstance(coverage, dict):
            watch_seconds_enriched = float(coverage.get("watch_seconds_enriched"))
    if raw is None:
        raise ValueError("missing_brainrot_raw")

    raw = max(0.0, min(1.0, float(raw)))
    if confidence is None:
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))
    if watch_seconds_enriched is None:
        watch_seconds_enriched = 0.0
    watch_seconds_enriched = max(0.0, float(watch_seconds_enriched))

    enriched_hours = watch_seconds_enriched / 3600.0
    intensity_linear_0_100 = max(0.0, min(100.0, raw * 100.0))

    eps = max(1e-12, float(os.getenv("BRAINROT_S_CURVE_EPS", "1e-9")))
    base_raw = max(eps, float(os.getenv("BRAINROT_S_CURVE_BASELINE_RAW", "0.0019632306430910507")))
    base_score_frac = float(os.getenv("BRAINROT_S_CURVE_BASELINE_SCORE", "60")) / 100.0
    base_score_frac = max(0.01, min(0.99, base_score_frac))
    k = max(0.05, float(os.getenv("BRAINROT_S_CURVE_SLOPE", "1.5")))
    zero_raw_floor_divisor = float(os.getenv("BRAINROT_ZERO_RAW_FLOOR_DIVISOR", "20"))

    raw_for_norm = raw
    if raw_for_norm <= 0.0:
        divisor = zero_raw_floor_divisor or 0.0
        if divisor <= 1.0:
            divisor = 20.0
        raw_for_norm = max(eps, base_raw / divisor)
    x = math.log10(max(raw_for_norm, eps))
    x0 = math.log10(base_raw)
    logit_p0 = math.log(base_score_frac / (1.0 - base_score_frac))
    x_mid = x0 - (logit_p0 / k)
    intensity_norm_0_100_f = 100.0 * _sigmoid(k * (x - x_mid))
    intensity_norm_0_100 = _clamp_0_100(intensity_norm_0_100_f)

    volume_hours = max(0.0, enriched_hours * raw)

    min_conf = max(0.0001, float(os.getenv("BRAINROT_MIN_CONFIDENCE", "0.1")))
    min_hours = max(0.0001, float(os.getenv("BRAINROT_MIN_ENRICHED_HOURS", "0.5")))
    w_conf = min(1.0, confidence / min_conf)
    w_hours = min(1.0, enriched_hours / min_hours)
    quality_weight = min(1.0, (w_conf * w_hours) ** 0.5)
    final_score = _clamp_0_100(intensity_norm_0_100_f * quality_weight)

    return {
        "brain_rot_score": final_score,
        "brainrot_intensity": intensity_norm_0_100,
        "brainrot_intensity_raw": round(raw, 8),
        "brainrot_intensity_raw_effective": round(raw_for_norm, 8),
        "brainrot_intensity_linear": round(intensity_linear_0_100, 4),
        "brainrot_confidence": round(confidence, 6),
        "brainrot_enriched_watch_pct": round(confidence * 100.0, 4),
        "brainrot_enriched_hours": round(enriched_hours, 4),
        "brainrot_volume_hours": round(volume_hours, 4),
        "brainrot_quality_weight": round(quality_weight, 6),
        "brainrot_normalization": {
            "baseline_raw": base_raw,
            "baseline_score": base_score_frac * 100.0,
            "slope": k,
            "zero_raw_floor_divisor": zero_raw_floor_divisor,
        },
    }


@app.get("/admin/metrics", dependencies=[Depends(require_admin)], include_in_schema=False)
def admin_metrics(db=Depends(get_db)) -> Response:
    body = render_admin_metrics(db)
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get(
    "/admin/jobs/{job_id}",
    response_model=AdminJobStatusResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    dependencies=[Depends(require_admin)],
    include_in_schema=False,
)
def admin_job_status(job_id: str, include_payload: bool = False, db=Depends(get_db)) -> AdminJobStatusResponse:
    job = db.get(AppJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    payload = job.payload if isinstance(job.payload, dict) else None
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        result = None
    def _dt(val: Optional[datetime]) -> Optional[str]:
        return val.isoformat() if val else None
    return AdminJobStatusResponse(
        job_id=job.id,
        task_name=job.task_name,
        status=job.status,
        attempts=int(job.attempts or 0),
        max_attempts=int(job.max_attempts or 0),
        not_before=_dt(job.not_before),
        locked_by=job.locked_by,
        locked_at=_dt(job.locked_at),
        created_at=_dt(job.created_at),
        updated_at=_dt(job.updated_at),
        idempotency_key=job.idempotency_key,
        result=result,
        payload=payload if include_payload else None,
    )


@app.post(
    "/admin/brainrot/score",
    dependencies=[Depends(require_admin)],
    response_model=AdminBrainrotScoreResponse,
    responses={401: {"model": ErrorResponse}},
    include_in_schema=False,
)
async def admin_brainrot_score(payload: AdminBrainrotScoreRequest) -> AdminBrainrotScoreResponse:
    range_spec = payload.range or {"type": "last_n_months", "months": 12}
    try:
        summary = await archive_client.interests_summary(
            sec_user_id=payload.sec_user_id,
            range=range_spec,
            time_zone=payload.time_zone or "UTC",
            group_by="none",
            limit=10,
            include_unknown=True,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "archive_unavailable", "message": _format_http_error(exc)},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"error": "archive_unavailable", "message": _format_http_error(exc)})

    try:
        computed = _compute_brainrot_from_summary(summary)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    updated_wrapped_run_id = None
    with SessionLocal() as db:
        run = (
            db.query(AppWrappedRun)
            .filter(AppWrappedRun.sec_user_id == payload.sec_user_id)
            .order_by(AppWrappedRun.updated_at.desc(), AppWrappedRun.created_at.desc())
            .first()
        )
        if not run:
            raise HTTPException(status_code=404, detail="wrapped_run_not_found")
        run_payload = run.payload if isinstance(run.payload, dict) else {}
        run_payload.update(computed)
        run.payload = run_payload
        db.add(run)
        db.commit()
        updated_wrapped_run_id = run.id

    return AdminBrainrotScoreResponse(
        sec_user_id=payload.sec_user_id,
        range=range_spec,
        time_zone=payload.time_zone or "UTC",
        updated_wrapped_run_id=updated_wrapped_run_id,
        **computed,
    )


_PROMPT_MAP: Dict[str, str] = {
    "personality": PERSONALITY_PROMPT,
    "personality_explanation": PERSONALITY_EXPLANATION_PROMPT,
    "niche_journey": NICHE_JOURNEY_PROMPT,
    "top_niches": TOP_NICHES_PROMPT,
    "brainrot_score": BRAINROT_SCORE_PROMPT,
    "brainrot_explanation": BRAINROT_EXPLANATION_PROMPT,
    "keyword_2026": KEYWORD_2026_PROMPT,
    "thumb_roast": ROAST_THUMB_PROMPT,
}


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _watch_history_since_ms() -> int:
    raw_ms = os.getenv("WATCH_HISTORY_SINCE_MS")
    if raw_ms:
        try:
            return int(raw_ms)
        except Exception:
            pass
    raw_date = (os.getenv("WATCH_HISTORY_SINCE_DATE") or "2025-01-01").strip()
    try:
        dt = datetime.fromisoformat(raw_date)
    except Exception:
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _months_between(start_utc: datetime, end_utc: datetime) -> int:
    if end_utc < start_utc:
        return 1
    return max(1, (end_utc.year - start_utc.year) * 12 + (end_utc.month - start_utc.month) + 1)


CAT_NAMES = [
    "cat_white",
    "cat_brown",
    "cat_leopard",
    "cat_orange",
    "cat_tabby",
    "cat_calico",
]


def _default_cat_name(seed: str) -> str:
    if not seed:
        return CAT_NAMES[0]
    try:
        h = hashlib.sha256(seed.encode("utf-8")).digest()[0]
        return CAT_NAMES[h % len(CAT_NAMES)]
    except Exception:
        return CAT_NAMES[0]


def _default_analogy_line(total_videos: int) -> str:
    total_videos = max(0, int(total_videos))
    videos_per_mile_env = os.getenv("THUMB_VIDEOS_PER_MILE", "180")
    try:
        videos_per_mile = float(videos_per_mile_env)
    except Exception:
        videos_per_mile = 180.0
    videos_per_mile = max(10.0, min(videos_per_mile, 10000.0))
    miles = total_videos / videos_per_mile

    def pick(bucket: str, options: List[str]) -> str:
        try:
            h = hashlib.sha256(f"{total_videos}:{bucket}".encode("utf-8")).digest()
            idx = int.from_bytes(h[:2], "big") % len(options)
            return options[idx]
        except Exception:
            return options[0]

    miles_label = f"{miles:.1f}" if miles < 2 else f"{int(round(miles))}"
    if miles < 2:
        suffix = pick("0_2", ["barely a warmup!", "barely broke a sweat!", "just a little stroll!"])
    elif miles < 8:
        suffix = pick("2_8", ["that’s a respectable cardio!", "your thumb’s got stamina!"])
    elif miles < 16:
        suffix = pick("8_16", ["longer than a half-marathon!"])
    elif miles < 26:
        suffix = pick("16_26", ["that’s literally a full marathon!"])
    elif miles < 50:
        suffix = pick("26_50", ["that’s two full marathons!", "take care, your thumb needs a spa day!"])
    else:
        suffix = pick("50_plus", ["your thumb is now a bodybuilder!", "purely unhinged behavior!"])
    return f"Your thumb ran **{miles_label} miles** on the screen — {suffix}"


def _format_hour_ampm(hour: int) -> str:
    hour = int(hour) % 24
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12
    return f"{hour12}{suffix}"


def _format_hour_window(start_hour: int, window_hours: int) -> str:
    start = _format_hour_ampm(start_hour)
    end = _format_hour_ampm(start_hour + window_hours)
    return f"{start}–{end}"


def _scroll_window_label(start_hour: int) -> str:
    hour = int(start_hour) % 24
    if 0 <= hour <= 3:
        return "a 3AM Goblin"
    if 22 <= hour <= 23:
        return "a Night Owl"
    if 18 <= hour <= 21:
        return "an Early Bird"
    if 4 <= hour <= 9:
        return "an Early Bird"
    return "a Day Scroller"


def _default_scroll_time(*, peak_hour: Optional[int], pct: Optional[float]) -> dict:
    window_hours_env = os.getenv("SCROLL_TIME_WINDOW_HOURS", "2")
    try:
        window_hours = int(window_hours_env)
    except Exception:
        window_hours = 2
    window_hours = max(1, min(window_hours, 6))

    start_hour = int(peak_hour) - 1 if peak_hour is not None else 20
    pct_val = float(pct) if pct is not None else 0.0
    return {
        "title": _scroll_window_label(start_hour),
        "rate": f"{round(pct_val):.0f}%",
        "between_time": _format_hour_window(start_hour, window_hours),
    }


def _ensure_wrapped_defaults(payload: dict, app_user_id: str) -> dict:
    """Backfill newly added presentation fields for legacy payloads."""
    if payload is None:
        payload = {}
    total_videos = payload.get("total_videos") or 0
    if "cat_name" not in payload:
        payload["cat_name"] = _default_cat_name(app_user_id)
    if "analogy_line" not in payload:
        payload["analogy_line"] = _default_analogy_line(total_videos)
    if "scroll_time" not in payload:
        payload["scroll_time"] = _default_scroll_time(peak_hour=payload.get("peak_hour"), pct=payload.get("night_pct"))

    accessory_set = payload.get("accessory_set")
    if not isinstance(accessory_set, dict):
        payload["accessory_set"] = accessories.select_accessory_set()
        return payload

    hints = {
        "scroll_time": payload.get("scroll_time"),
        "night_pct": payload.get("night_pct"),
        "peak_hour": payload.get("peak_hour"),
        "top_music": payload.get("top_music"),
        "top_creators": payload.get("top_creators"),
        "top_niches": payload.get("top_niches"),
    }

    normalized: Dict[str, Dict[str, str]] = {}
    changed = False
    for slot in ("head", "body", "other"):
        item = accessory_set.get(slot)
        internal_name = None
        reason = None
        if isinstance(item, dict):
            internal_name = item.get("internal_name")
            reason = item.get("reason")
            if not internal_name:
                item_id = item.get("item_id")
                if isinstance(item_id, str) and item_id:
                    internal_name = accessories.internal_name_for_item_id(item_id)
        elif isinstance(item, str) and item:
            internal_name = item
        if not internal_name:
            internal_name = "unknown"
            changed = True
        if not isinstance(reason, str) or not reason.strip():
            reason = accessory_fallback_reason(seed=str(app_user_id), slot=slot, internal_name=str(internal_name), hints=hints)
            changed = True
        normalized[slot] = {"internal_name": str(internal_name), "reason": str(reason).strip()}
        if (
            not isinstance(item, dict)
            or item.get("internal_name") != internal_name
            or str(item.get("reason") or "").strip() != str(reason).strip()
        ):
            changed = True

    if changed:
        payload["accessory_set"] = normalized
    return payload


def _generate_referral_code(app_user_id: str, db) -> str:
    """Generate a short, deterministic-ish referral code for a referrer identity; ensure uniqueness."""
    base = hashlib.sha256(app_user_id.encode("utf-8")).hexdigest()[:8]
    code = base
    suffix = 0
    while db.query(Referral).filter(Referral.code == code).first():
        suffix += 1
        code = f"{base}{suffix}"
    return code


def _referral_url(code: str) -> Optional[str]:
    frontend = os.getenv("FRONTEND_URL", "").rstrip("/")
    if not frontend:
        return None
    return f"{frontend}?ref={code}"


async def ensure_watch_history_available(
    user: AppUser,
    db,
    max_attempts: int = 3,
    auto_enqueue: bool = False,
    resend_email_on_ready: bool = False,
) -> Tuple[str, int, Optional[str]]:
    if not user.latest_sec_user_id:
        raise HTTPException(status_code=400, detail="sec_user_id_required")
    if user.is_watch_history_available == "yes":
        if auto_enqueue:
            queue_wrapped_run(user, db, resend_email_on_ready=resend_email_on_ready)
        return user.is_watch_history_available, 0, None

    attempts = 0
    last_error: Optional[str] = None
    for _ in range(max_attempts):
        attempts += 1
        try:
            start_resp = await archive_client.start_watch_history(
                sec_user_id=user.latest_sec_user_id,
                limit=1,
                max_pages=1,
                cursor=None,
            )
            status_code = getattr(start_resp, "status_code", None)
            if status_code == 429:
                await asyncio.sleep(1)
                attempts -= 1
                continue
            if status_code and status_code >= 400:
                last_error = f"{status_code} {getattr(start_resp, 'text', '')}"
                continue
            start_data = start_resp.json()
            data_job_id = start_data.get("data_job_id")
            if not data_job_id:
                last_error = "missing_data_job_id"
                continue
            finalize_attempts = 0
            while finalize_attempts < WATCH_HISTORY_VERIFY_FINALIZE_MAX_ATTEMPTS:
                finalize_attempts += 1
                fin = await archive_client.finalize_watch_history(
                    data_job_id=data_job_id, include_rows=False
                )
                fin_status = getattr(fin, "status_code", None)
                if fin_status == 202:
                    await asyncio.sleep(1)
                    continue
                if fin_status == 429:
                    await asyncio.sleep(1)
                    finalize_attempts -= 1
                    continue
                if fin_status == 200:
                    result = fin.json()
                    status_value = result.get("status")
                    if isinstance(status_value, str) and status_value.lower() in {"pending", "processing", "running"}:
                        await asyncio.sleep(1)
                        continue
                    videos_fetched = result.get("videos_fetched") or 0
                    if videos_fetched > 0:
                        user.is_watch_history_available = "yes"
                        last_error = None
                        break
                    last_error = "no_rows"
                    break
                if fin_status == 424:
                    last_error = "provider_failed"
                    break
                if fin_status == 404:
                    last_error = "not_found"
                    break
                last_error = f"{fin_status} {getattr(fin, 'text', '')}"
                break
            if user.is_watch_history_available != "yes" and last_error is None:
                last_error = "pending_timeout"
                break
            if user.is_watch_history_available == "yes":
                break
        except Exception as exc:
            last_error = _format_http_error(exc)

    if user.is_watch_history_available != "yes":
        if last_error in {"pending_timeout"}:
            user.is_watch_history_available = "unknown"
        else:
            user.is_watch_history_available = "no"
    db.add(user)
    db.commit()
    if user.is_watch_history_available in {"yes", "unknown"} and auto_enqueue:
        queue_wrapped_run(user, db, resend_email_on_ready=resend_email_on_ready)
    return user.is_watch_history_available, attempts, last_error


def _enqueue_reauth_notify(db, *, email: Optional[str], archive_job_id: str, reason: str) -> Optional[AppJob]:
    if os.getenv("REAUTH_NOTIFY_ENABLED", "true").lower() not in ("1", "true", "yes", "on"):
        return None
    email_value = (email or "").strip()
    if not email_value:
        return None
    now = datetime.utcnow()
    key = f"notify:reauth:{email_value}"
    existing = (
        db.query(AppJob)
        .filter(AppJob.task_name == "reauth_notify", AppJob.idempotency_key == key)
        .order_by(AppJob.created_at.desc())
        .first()
    )
    max_lifetime = int(os.getenv("REAUTH_NOTIFY_MAX_LIFETIME", "2"))
    if max_lifetime > 0:
        total_sent = (
            db.query(func.count(AppJob.id))
            .filter(AppJob.task_name == "reauth_notify", AppJob.idempotency_key == key)
            .scalar()
            or 0
        )
        if total_sent >= max_lifetime:
            return existing
    recent_cutoff = now - timedelta(hours=1)
    if existing and (
        (existing.not_before is not None and existing.not_before >= recent_cutoff)
        or (existing.created_at is not None and existing.created_at >= recent_cutoff)
    ):
        return existing

    return job_queue.enqueue(
        db,
        task_name="reauth_notify",
        payload={
            "email": email_value,
            "archive_job_id": archive_job_id,
            "reason": reason,
            "failed_at": now.replace(microsecond=0).isoformat() + "Z",
        },
        idempotency_key=key,
        force_new=True,
        not_before=now + timedelta(hours=1),
    )


def _enqueue_email_send_for_ready_run(db, run: AppWrappedRun, *, force_new: bool = True) -> Optional[AppJob]:
    app_user_id = run.app_user_id
    if not app_user_id:
        return None
    email_value = run.email
    if not email_value:
        latest_email = (
            db.query(AppUserEmail)
            .filter(AppUserEmail.app_user_id == app_user_id)
            .order_by(AppUserEmail.created_at.desc())
            .first()
        )
        if latest_email:
            email_value = latest_email.email
    if not email_value:
        return None

    job = job_queue.enqueue(
        db,
        task_name="email_send",
        payload={"wrapped_run_id": run.id, "app_user_id": app_user_id},
        idempotency_key=f"email:{run.id}",
        force_new=bool(force_new),
    )
    payload = run.payload if isinstance(run.payload, dict) else {}
    data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
    data_jobs["email_send"] = {"id": job.id, "status": job.status}
    payload["data_jobs"] = data_jobs
    payload.setdefault("email", email_value)
    run.payload = payload
    if not run.email:
        run.email = email_value
    db.add(run)
    db.commit()
    return job


def queue_wrapped_run(
    user: AppUser,
    db,
    *,
    resend_email_on_ready: bool = False,
    scrape_max_videos: Optional[int] = None,
) -> AppWrappedRun:
    if not user.latest_sec_user_id:
        raise HTTPException(status_code=400, detail="sec_user_id_required")
    existing = (
        db.query(AppWrappedRun)
        .filter(
            AppWrappedRun.app_user_id == user.app_user_id,
            AppWrappedRun.sec_user_id == user.latest_sec_user_id,
        )
        .order_by(AppWrappedRun.created_at.desc())
        .first()
    )
    if existing:
        if existing.status == "ready":
            if resend_email_on_ready:
                _enqueue_email_send_for_ready_run(db, existing)
            return existing

        payload = existing.payload if isinstance(existing.payload, dict) else {}

        watch_job = db.get(AppJob, existing.watch_history_job_id) if existing.watch_history_job_id else None
        if watch_job and watch_job.status in ("pending", "running"):
            return existing

        watch_done = False
        if watch_job and watch_job.status == "succeeded":
            watch_done = True
        elif isinstance(payload.get("_sample_texts"), list) and payload.get("_sample_texts"):
            watch_done = True
        elif payload.get("total_videos") is not None and payload.get("total_hours") is not None:
            watch_done = True

        if watch_done:
            analysis_job = (
                db.query(AppJob)
                .filter(AppJob.idempotency_key == f"analysis:{existing.id}")
                .order_by(AppJob.created_at.desc())
                .first()
            )
            if analysis_job and analysis_job.status in ("pending", "running"):
                return existing
            job = job_queue.enqueue(
                db,
                task_name="wrapped_analysis",
                payload={"wrapped_run_id": existing.id},
                idempotency_key=f"analysis:{existing.id}",
            )
            data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
            data_jobs["wrapped_analysis"] = {"id": job.id, "status": job.status}
            payload["data_jobs"] = data_jobs
            existing.payload = payload
            if existing.status == "failed":
                existing.status = "pending"
            db.add(existing)
            db.commit()
            return existing

        job = job_queue.enqueue(
            db,
            task_name="watch_history_fetch_2025",
            payload={
                "wrapped_run_id": existing.id,
                "app_user_id": user.app_user_id,
                "sec_user_id": user.latest_sec_user_id,
                "time_zone": user.time_zone,
                "platform_username": user.platform_username,
                "scrape_max_videos": scrape_max_videos,
            },
            idempotency_key=f"wrapped:{existing.id}",
        )
        existing.watch_history_job_id = job.id
        data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
        data_jobs["watch_history"] = {"id": job.id, "status": job.status}
        payload["data_jobs"] = data_jobs
        existing.payload = payload
        if existing.status == "failed":
            existing.status = "pending"
        db.add(existing)
        db.commit()
        return existing

    run = AppWrappedRun(
        id=str(uuid4()),
        app_user_id=user.app_user_id,
        sec_user_id=user.latest_sec_user_id,
        archive_user_id=user.archive_user_id,
        status="pending",
        email=None,
    )
    db.add(run)
    db.commit()

    job = job_queue.enqueue(
        db,
        task_name="watch_history_fetch_2025",
        payload={
            "wrapped_run_id": run.id,
            "app_user_id": user.app_user_id,
            "sec_user_id": user.latest_sec_user_id,
            "time_zone": user.time_zone,
            "platform_username": user.platform_username,
            "scrape_max_videos": scrape_max_videos,
        },
        idempotency_key=f"wrapped:{run.id}",
    )
    run.watch_history_job_id = job.id
    db.add(run)
    db.commit()
    return run


def _job_stuck_meta(rec: AppJob) -> Tuple[bool, Optional[str], Optional[float]]:
    now = datetime.utcnow()
    locked_for = (now - rec.locked_at).total_seconds() if rec.locked_at else None
    lease_seconds = int(rec.lease_seconds or 0)
    is_stuck = bool(
        rec.status == "running"
        and rec.locked_at
        and lease_seconds > 0
        and rec.locked_at + timedelta(seconds=lease_seconds) < now
    )
    stuck_reason = "lease_expired" if is_stuck else None
    return is_stuck, stuck_reason, locked_for


def _job_summary(rec: AppJob) -> Dict[str, Any]:
    is_stuck, stuck_reason, locked_for = _job_stuck_meta(rec)
    return {
        "id": rec.id,
        "task_name": rec.task_name,
        "status": rec.status,
        "attempts": rec.attempts,
        "max_attempts": rec.max_attempts,
        "not_before": rec.not_before.isoformat() if rec.not_before else None,
        "lease_seconds": rec.lease_seconds,
        "locked_by": rec.locked_by,
        "locked_at": rec.locked_at.isoformat() if rec.locked_at else None,
        "locked_for_seconds": locked_for,
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
        "updated_at": rec.updated_at.isoformat() if rec.updated_at else None,
        "idempotency_key": rec.idempotency_key,
        "is_stuck": is_stuck,
        "stuck_reason": stuck_reason,
    }


def _collect_run_status(
    db, run: AppWrappedRun
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Optional[List[str]], Optional[Dict[str, Any]]]:
    payload = run.payload if isinstance(run.payload, dict) else {}
    data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
    progress = payload.get("_watch_history_progress") if isinstance(payload.get("_watch_history_progress"), dict) else None
    analysis_warnings = payload.get("_analysis_warnings") if isinstance(payload.get("_analysis_warnings"), list) else None
    analysis_debug = payload.get("_analysis_debug") if isinstance(payload.get("_analysis_debug"), dict) else None

    job_ids: Dict[str, str] = {}
    if run.watch_history_job_id:
        job_ids["watch_history_fetch_2025"] = run.watch_history_job_id
    for name, job_ref in data_jobs.items():
        if isinstance(job_ref, dict):
            job_id = job_ref.get("id")
            if isinstance(job_id, str) and job_id:
                job_ids[name] = job_id

    analysis_job = (
        db.query(AppJob)
        .filter(AppJob.idempotency_key == f"analysis:{run.id}")
        .order_by(AppJob.created_at.desc())
        .first()
    )
    if analysis_job:
        job_ids.setdefault("wrapped_analysis", analysis_job.id)

    email_job = (
        db.query(AppJob)
        .filter(AppJob.idempotency_key == f"email:{run.id}")
        .order_by(AppJob.created_at.desc())
        .first()
    )
    if email_job:
        job_ids.setdefault("email_send", email_job.id)

    jobs: Dict[str, Any] = {}
    for name, job_id in job_ids.items():
        rec = db.get(AppJob, job_id)
        if not rec:
            jobs[name] = {"id": job_id, "status": "unknown"}
            continue
        jobs[name] = _job_summary(rec)

    effective_data_jobs: Dict[str, Any] = dict(data_jobs)
    watch_job = jobs.get("watch_history") or jobs.get("watch_history_fetch_2025")
    if isinstance(watch_job, dict) and isinstance(watch_job.get("id"), str):
        effective_data_jobs["watch_history"] = {"id": watch_job["id"], "status": watch_job.get("status", "unknown")}
    analysis_job_meta = jobs.get("wrapped_analysis")
    if isinstance(analysis_job_meta, dict) and isinstance(analysis_job_meta.get("id"), str):
        effective_data_jobs["wrapped_analysis"] = {
            "id": analysis_job_meta["id"],
            "status": analysis_job_meta.get("status", "unknown"),
        }
    email_job_meta = jobs.get("email_send")
    if isinstance(email_job_meta, dict) and isinstance(email_job_meta.get("id"), str):
        effective_data_jobs["email_send"] = {"id": email_job_meta["id"], "status": email_job_meta.get("status", "unknown")}

    return effective_data_jobs, jobs, progress, analysis_warnings, analysis_debug


def _normalize_status(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() or None
    return str(value).strip().lower() or None


def _latest_user_email(db, app_user_id: str) -> Optional[str]:
    latest_email = (
        db.query(AppUserEmail)
        .filter(AppUserEmail.app_user_id == app_user_id)
        .order_by(AppUserEmail.created_at.desc())
        .first()
    )
    if latest_email and isinstance(latest_email.email, str):
        return latest_email.email
    return None


def _latest_wrapped_run(db, app_user_id: str) -> Optional[AppWrappedRun]:
    return (
        db.query(AppWrappedRun)
        .filter(AppWrappedRun.app_user_id == app_user_id)
        .order_by(AppWrappedRun.created_at.desc(), AppWrappedRun.updated_at.desc())
        .first()
    )


def _latest_verify_job(db, app_user_id: str) -> Optional[AppJob]:
    return (
        db.query(AppJob)
        .filter(AppJob.idempotency_key == f"watch_history_verify:{app_user_id}")
        .order_by(AppJob.created_at.desc())
        .first()
    )


def _validate_app_user_id(app_user_id: str) -> None:
    if "@" in app_user_id:
        raise HTTPException(status_code=400, detail="invalid_app_user_id")


def _batch_user_query(
    db,
    *,
    limit: int,
    offset: int,
    include_admin_test: bool,
    watch_history_status: Optional[List[str]],
) -> Tuple[List[AppUser], int, int]:
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    base_query = db.query(AppUser)
    if not include_admin_test:
        base_query = base_query.filter(~AppUser.app_user_id.like("admin-test-%"))
    if watch_history_status:
        base_query = base_query.filter(AppUser.is_watch_history_available.in_(watch_history_status))
    users = (
        base_query.order_by(AppUser.updated_at.desc(), AppUser.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return users, limit, offset


def _run_status_value(run: Optional[AppWrappedRun]) -> str:
    return run.status if run else "no_run"


def _build_user_stage_response(
    *,
    app_user_id: str,
    user: AppUser,
    run: Optional[AppWrappedRun],
    verify_job: Optional[AppJob],
    effective_data_jobs: Dict[str, Any],
    jobs: Dict[str, Any],
    progress: Optional[Dict[str, Any]],
    stage: str,
    next_stage: str,
) -> AdminUserStageResponse:
    return _build_user_stage_response(
        app_user_id=app_user_id,
        user=user,
        run=run,
        verify_job=verify_job,
        effective_data_jobs=effective_data_jobs,
        jobs=jobs,
        progress=progress,
        stage=stage,
        next_stage=next_stage,
    )


def _analysis_reset(payload: Dict[str, Any]) -> None:
    for key in (
        "cat_name",
        "analogy_line",
        "scroll_time",
        "personality_type",
        "personality_explanation",
        "niche_journey",
        "top_niches",
        "top_niche_percentile",
        "brain_rot_score",
        "brainrot_intensity",
        "brainrot_volume_hours",
        "brainrot_confidence",
        "brainrot_enriched_hours",
        "brainrot_enriched_watch_pct",
        "brainrot_intensity_raw",
        "brainrot_intensity_raw_effective",
        "brainrot_intensity_linear",
        "brainrot_quality_weight",
        "brainrot_normalization",
        "brain_rot_explanation",
        "keyword_2026",
        "thumb_roast",
        "accessory_set",
        "_analysis_warnings",
        "_analysis_debug",
    ):
        payload.pop(key, None)
    for key in tuple(payload.keys()):
        if isinstance(key, str) and key.startswith("_brainrot_"):
            payload.pop(key, None)


def _full_payload_reset(payload: Dict[str, Any]) -> None:
    for key in (
        "_watch_history_progress",
        "_sample_texts",
        "_bucket_summaries",
        "_months_done",
        "_acc_state",
        "_analysis_warnings",
        "_analysis_debug",
        "total_hours",
        "total_videos",
        "night_pct",
        "peak_hour",
        "top_music",
        "top_creators",
        "cat_name",
        "analogy_line",
        "scroll_time",
        "personality_type",
        "personality_explanation",
        "niche_journey",
        "top_niches",
        "top_niche_percentile",
        "brain_rot_score",
        "brainrot_intensity",
        "brainrot_volume_hours",
        "brainrot_confidence",
        "brainrot_enriched_hours",
        "brainrot_enriched_watch_pct",
        "brainrot_intensity_raw",
        "brainrot_intensity_raw_effective",
        "brainrot_intensity_linear",
        "brainrot_quality_weight",
        "brainrot_normalization",
        "brain_rot_explanation",
        "keyword_2026",
        "thumb_roast",
        "source_spans",
        "accessory_set",
    ):
        payload.pop(key, None)
    for key in tuple(payload.keys()):
        if isinstance(key, str) and key.startswith("_brainrot_"):
            payload.pop(key, None)


def _derive_stage(
    user: AppUser,
    run: Optional[AppWrappedRun],
    effective_data_jobs: Dict[str, Any],
    *,
    email_needed: bool,
) -> Tuple[str, str]:
    if not run:
        if user.is_watch_history_available != "yes":
            return "verify_region", "verify_region"
        return "no_run", "watch_history"

    watch_status = _normalize_status((effective_data_jobs.get("watch_history") or {}).get("status"))
    analysis_status = _normalize_status((effective_data_jobs.get("wrapped_analysis") or {}).get("status"))
    email_status = _normalize_status((effective_data_jobs.get("email_send") or {}).get("status"))

    if watch_status != "succeeded":
        return "watch_history", "watch_history"
    if analysis_status != "succeeded":
        return "analysis", "analysis"
    if email_needed and email_status != "succeeded":
        return "email", "email"
    return "ready", "ready"


def _load_user_stage(
    db, app_user_id: str, *, wrapped_run_id: Optional[str] = None
) -> Tuple[
    AppUser,
    Optional[AppWrappedRun],
    Optional[AppJob],
    Dict[str, Any],
    Dict[str, Any],
    Optional[Dict[str, Any]],
    str,
    str,
    bool,
]:
    user = db.get(AppUser, app_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")

    run: Optional[AppWrappedRun] = None
    if wrapped_run_id:
        run = db.get(AppWrappedRun, wrapped_run_id)
        if not run or run.app_user_id != app_user_id:
            raise HTTPException(status_code=404, detail="wrapped_run_not_found")
    else:
        run = _latest_wrapped_run(db, app_user_id)

    verify_job = _latest_verify_job(db, app_user_id)
    effective_data_jobs: Dict[str, Any] = {}
    jobs: Dict[str, Any] = {}
    progress: Optional[Dict[str, Any]] = None
    if run:
        effective_data_jobs, jobs, progress, _, _ = _collect_run_status(db, run)

    email_value = run.email if run else None
    if not email_value:
        email_value = _latest_user_email(db, app_user_id)
    email_needed = bool(email_value)

    stage, next_stage = _derive_stage(
        user,
        run,
        effective_data_jobs,
        email_needed=email_needed,
    )
    return user, run, verify_job, effective_data_jobs, jobs, progress, stage, next_stage, email_needed


def _restart_user_stage(
    db, app_user_id: str, payload: AdminUserRestartRequest, *, dry_run: bool = False
) -> AdminUserRestartResponse:
    user, run, verify_job, effective_data_jobs, _jobs, _, _stage, next_stage, _ = _load_user_stage(
        db, app_user_id, wrapped_run_id=payload.wrapped_run_id
    )

    stage_value = (payload.stage or "next").strip().lower()
    if not stage_value or stage_value == "next":
        stage_value = next_stage

    if stage_value == "ready":
        return AdminUserRestartResponse(
            app_user_id=app_user_id,
            wrapped_run_id=run.id if run else None,
            selected_stage=stage_value,
            skipped_reason="already_ready",
        )

    if stage_value not in {"verify_region", "watch_history", "analysis", "email"}:
        raise HTTPException(status_code=400, detail="invalid_stage")

    if stage_value in {"verify_region", "watch_history"} and not user.latest_sec_user_id:
        raise HTTPException(status_code=400, detail="sec_user_id_required")

    def _in_progress(status: Optional[str]) -> bool:
        return status in {"pending", "running"}

    if stage_value == "verify_region":
        verify_status = _normalize_status(verify_job.status) if verify_job else None
        if _in_progress(verify_status):
            return AdminUserRestartResponse(
                app_user_id=app_user_id,
                wrapped_run_id=run.id if run else None,
                selected_stage=stage_value,
                enqueued_task="watch_history_verify",
                job_id=verify_job.id if verify_job else None,
                status=verify_job.status if verify_job else None,
                idempotency_key=verify_job.idempotency_key if verify_job else None,
                skipped_reason="verify_in_progress",
            )

        if dry_run:
            return AdminUserRestartResponse(
                app_user_id=app_user_id,
                wrapped_run_id=run.id if run else None,
                selected_stage=stage_value,
                enqueued_task="watch_history_verify",
                skipped_reason="dry_run",
            )

        job = job_queue.enqueue(
            db,
            task_name="watch_history_verify",
            payload={"app_user_id": user.app_user_id, "auto_enqueue": True},
            idempotency_key=f"watch_history_verify:{user.app_user_id}",
            force_new=payload.force_new_jobs,
        )
        return AdminUserRestartResponse(
            app_user_id=app_user_id,
            wrapped_run_id=run.id if run else None,
            selected_stage=stage_value,
            enqueued_task=job.task_name,
            job_id=job.id,
            status=job.status,
            idempotency_key=job.idempotency_key,
        )

    if stage_value == "watch_history":
        watch_status = _normalize_status((effective_data_jobs.get("watch_history") or {}).get("status"))
        if _in_progress(watch_status):
            return AdminUserRestartResponse(
                app_user_id=app_user_id,
                wrapped_run_id=run.id if run else None,
                selected_stage=stage_value,
                enqueued_task="watch_history_fetch_2025",
                job_id=(effective_data_jobs.get("watch_history") or {}).get("id"),
                status=watch_status,
                skipped_reason="watch_history_in_progress",
            )

        if not run and dry_run:
            return AdminUserRestartResponse(
                app_user_id=app_user_id,
                wrapped_run_id=None,
                selected_stage=stage_value,
                enqueued_task="watch_history_fetch_2025",
                skipped_reason="dry_run",
            )

        if not run:
            run = AppWrappedRun(
                id=str(uuid4()),
                app_user_id=user.app_user_id,
                sec_user_id=user.latest_sec_user_id,
                archive_user_id=user.archive_user_id,
                status="pending",
                email=None,
            )
            db.add(run)
            db.commit()

        run_payload = run.payload if isinstance(run.payload, dict) else {}
        if payload.reset_payload:
            _full_payload_reset(run_payload)
            data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
            data_jobs.pop("watch_history", None)
            data_jobs.pop("wrapped_analysis", None)
            data_jobs.pop("email_send", None)
            run_payload["data_jobs"] = data_jobs
            run.payload = run_payload

        if dry_run:
            return AdminUserRestartResponse(
                app_user_id=app_user_id,
                wrapped_run_id=run.id,
                selected_stage=stage_value,
                enqueued_task="watch_history_fetch_2025",
                skipped_reason="dry_run",
            )

        job = job_queue.enqueue(
            db,
            task_name="watch_history_fetch_2025",
            payload={
                "wrapped_run_id": run.id,
                "app_user_id": run.app_user_id,
                "sec_user_id": user.latest_sec_user_id,
                "time_zone": user.time_zone,
                "platform_username": user.platform_username,
                "force_new_analysis": True,
                "force_new_email": True,
            },
            idempotency_key=f"wrapped:{run.id}",
            force_new=payload.force_new_jobs,
        )
        run.watch_history_job_id = job.id
        data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
        data_jobs["watch_history"] = {"id": job.id, "status": job.status}
        run_payload["data_jobs"] = data_jobs
        run.payload = run_payload
        if run.status != "pending":
            run.status = "pending"
        db.add(run)
        db.commit()

        return AdminUserRestartResponse(
            app_user_id=app_user_id,
            wrapped_run_id=run.id,
            selected_stage=stage_value,
            enqueued_task=job.task_name,
            job_id=job.id,
            status=job.status,
            idempotency_key=job.idempotency_key,
        )

    if not run:
        raise HTTPException(status_code=400, detail="wrapped_run_required")

    if stage_value == "analysis":
        analysis_status = _normalize_status((effective_data_jobs.get("wrapped_analysis") or {}).get("status"))
        if _in_progress(analysis_status):
            return AdminUserRestartResponse(
                app_user_id=app_user_id,
                wrapped_run_id=run.id,
                selected_stage=stage_value,
                enqueued_task="wrapped_analysis",
                job_id=(effective_data_jobs.get("wrapped_analysis") or {}).get("id"),
                status=analysis_status,
                skipped_reason="analysis_in_progress",
            )

        run_payload = run.payload if isinstance(run.payload, dict) else {}
        if payload.reset_payload:
            _analysis_reset(run_payload)
            data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
            data_jobs.pop("wrapped_analysis", None)
            data_jobs.pop("email_send", None)
            run_payload["data_jobs"] = data_jobs
            run.payload = run_payload

        if dry_run:
            return AdminUserRestartResponse(
                app_user_id=app_user_id,
                wrapped_run_id=run.id,
                selected_stage=stage_value,
                enqueued_task="wrapped_analysis",
                skipped_reason="dry_run",
            )

        job = job_queue.enqueue(
            db,
            task_name="wrapped_analysis",
            payload={"wrapped_run_id": run.id, "force_new_email": True},
            idempotency_key=f"analysis:{run.id}",
            force_new=payload.force_new_jobs,
        )
        data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
        data_jobs["wrapped_analysis"] = {"id": job.id, "status": job.status}
        run_payload["data_jobs"] = data_jobs
        run.payload = run_payload
        if run.status != "pending":
            run.status = "pending"
        db.add(run)
        db.commit()

        return AdminUserRestartResponse(
            app_user_id=app_user_id,
            wrapped_run_id=run.id,
            selected_stage=stage_value,
            enqueued_task=job.task_name,
            job_id=job.id,
            status=job.status,
            idempotency_key=job.idempotency_key,
        )

    email_status = _normalize_status((effective_data_jobs.get("email_send") or {}).get("status"))
    if _in_progress(email_status):
        return AdminUserRestartResponse(
            app_user_id=app_user_id,
            wrapped_run_id=run.id,
            selected_stage=stage_value,
            enqueued_task="email_send",
            job_id=(effective_data_jobs.get("email_send") or {}).get("id"),
            status=email_status,
            skipped_reason="email_in_progress",
        )

    if dry_run:
        return AdminUserRestartResponse(
            app_user_id=app_user_id,
            wrapped_run_id=run.id,
            selected_stage=stage_value,
            enqueued_task="email_send",
            skipped_reason="dry_run",
        )

    email_job = _enqueue_email_send_for_ready_run(db, run, force_new=payload.force_new_jobs)
    if not email_job:
        return AdminUserRestartResponse(
            app_user_id=app_user_id,
            wrapped_run_id=run.id,
            selected_stage=stage_value,
            skipped_reason="missing_email",
        )

    return AdminUserRestartResponse(
        app_user_id=app_user_id,
        wrapped_run_id=run.id,
        selected_stage=stage_value,
        enqueued_task=email_job.task_name,
        job_id=email_job.id,
        status=email_job.status,
        idempotency_key=email_job.idempotency_key,
    )


def demo_wrapped_payload() -> WrappedPayload:
    accessory = accessories.select_accessory_set()
    return WrappedPayload(
        total_hours=0.0,
        total_videos=0,
        night_pct=0.0,
        peak_hour=None,
        top_music={"name": "", "count": 0},
        top_creators=[],
        personality_type="",
        personality_explanation=None,
        niche_journey=[],
        top_niches=[],
        top_niche_percentile=None,
        brain_rot_score=0,
        brain_rot_explanation=None,
        keyword_2026="",
        thumb_roast=None,
        platform_username=None,
        email=None,
        source_spans=[{"video_id": "demo", "reason": "placeholder"}],
        data_jobs={},
        accessory_set=AccessorySet.model_validate(accessory),
    )


# --- TikTok link via Xordi (Archive proxies) -------------------------------


@app.post(
    "/admin/test/run",
    response_model=AdminTestRunResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_test_run(payload: AdminTestRunRequest, _admin=Depends(require_admin), db=Depends(get_db)) -> AdminTestRunResponse:
    app_user_id = payload.app_user_id or f"admin-test-{uuid4()}"
    user = db.get(AppUser, app_user_id)
    if not user:
        user = AppUser(app_user_id=app_user_id)
    user.latest_sec_user_id = payload.sec_user_id
    user.platform_username = payload.platform_username
    if payload.time_zone:
        user.time_zone = payload.time_zone
    db.add(user)

    now = datetime.utcnow()
    existing_email = (
        db.query(AppUserEmail)
        .filter(AppUserEmail.app_user_id == app_user_id)
        .order_by(AppUserEmail.created_at.desc())
        .first()
    )
    if existing_email:
        existing_email.email = payload.email
        existing_email.created_at = now
        if existing_email.verified_at is None:
            existing_email.verified_at = now
        db.add(existing_email)
    else:
        db.add(
            AppUserEmail(
                id=str(uuid4()),
                app_user_id=app_user_id,
                email=payload.email,
                created_at=now,
                verified_at=now,
            )
        )
    db.commit()

    if payload.force_new_run:
        run = AppWrappedRun(
            id=str(uuid4()),
            app_user_id=app_user_id,
            sec_user_id=payload.sec_user_id,
            archive_user_id=user.archive_user_id,
            status="pending",
            email=None,
        )
        db.add(run)
        db.commit()

        job = job_queue.enqueue(
            db,
            task_name="watch_history_fetch_2025",
            payload={
                "wrapped_run_id": run.id,
                "app_user_id": app_user_id,
                "sec_user_id": payload.sec_user_id,
                "time_zone": user.time_zone,
                "platform_username": user.platform_username,
                "scrape_max_videos": payload.scrape_max_videos,
            },
            idempotency_key=f"wrapped:{run.id}",
        )
        run.watch_history_job_id = job.id
        run.payload = {"data_jobs": {"watch_history": {"id": job.id, "status": "pending"}}}
        db.add(run)
        db.commit()
    else:
        run = queue_wrapped_run(user, db, scrape_max_videos=payload.scrape_max_videos)
        if not run.watch_history_job_id:
            raise HTTPException(status_code=500, detail="missing_watch_history_job")

    frontend = os.getenv("FRONTEND_URL", "").rstrip("/")
    wrapped_link = (
        f"{frontend}/wrapped?app_user_id={quote(app_user_id, safe='')}"
        if frontend
        else f"/wrapped?app_user_id={quote(app_user_id, safe='')}"
    )
    return AdminTestRunResponse(
        app_user_id=app_user_id,
        wrapped_run_id=run.id,
        watch_history_job_id=run.watch_history_job_id or "",
        wrapped_link=wrapped_link,
        wrapped_status_endpoint=f"/wrapped/{app_user_id}",
        admin_status_endpoint=f"/admin/test/runs/{run.id}",
    )


@app.get(
    "/admin/test/runs/{wrapped_run_id}",
    response_model=AdminTestRunStatusResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_test_run_status(
    wrapped_run_id: str, _admin=Depends(require_admin), db=Depends(get_db)
) -> AdminTestRunStatusResponse:
    run = db.get(AppWrappedRun, wrapped_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="not_found")
    user = db.get(AppUser, run.app_user_id) if run.app_user_id else None
    email_value = run.email
    if not email_value and run.app_user_id:
        latest_email = (
            db.query(AppUserEmail)
            .filter(AppUserEmail.app_user_id == run.app_user_id)
            .order_by(AppUserEmail.created_at.desc())
            .first()
        )
        if latest_email:
            email_value = latest_email.email

    effective_data_jobs, jobs, progress, analysis_warnings, analysis_debug = _collect_run_status(db, run)

    watch_job = jobs.get("watch_history") or jobs.get("watch_history_fetch_2025")
    analysis_job_meta = jobs.get("wrapped_analysis")
    email_job_meta = jobs.get("email_send")

    effective_run_status = run.status
    if effective_run_status != "ready":
        if isinstance(analysis_job_meta, dict) and analysis_job_meta.get("status") == "failed":
            effective_run_status = "failed"
        elif isinstance(watch_job, dict) and watch_job.get("status") == "failed":
            effective_run_status = "failed"
        elif isinstance(email_job_meta, dict) and email_job_meta.get("status") == "failed":
            effective_run_status = "failed"

    frontend = os.getenv("FRONTEND_URL", "").rstrip("/")
    wrapped_link = (
        f"{frontend}/wrapped?app_user_id={quote(run.app_user_id, safe='')}"
        if frontend
        else f"/wrapped?app_user_id={quote(run.app_user_id, safe='')}"
    )
    return AdminTestRunStatusResponse(
        wrapped_run_id=run.id,
        app_user_id=run.app_user_id,
        sec_user_id=run.sec_user_id,
        run_status=effective_run_status,
        watch_history_job_id=run.watch_history_job_id,
        platform_username=user.platform_username if user else None,
        email=email_value,
        wrapped_link=wrapped_link,
        wrapped_status_endpoint=f"/wrapped/{run.app_user_id}",
        data_jobs=effective_data_jobs,
        watch_history_progress=progress,
        analysis_warnings=analysis_warnings,
        analysis_debug=analysis_debug,
        jobs=jobs,
    )


@app.post(
    "/admin/test/runs/{wrapped_run_id}/retry-analysis",
    response_model=AdminTestEnqueueResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_test_retry_analysis(
    wrapped_run_id: str, _admin=Depends(require_admin), db=Depends(get_db)
) -> AdminTestEnqueueResponse:
    run = db.get(AppWrappedRun, wrapped_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="not_found")
    if isinstance(run.payload, dict):
        payload = run.payload
        for key in (
            "cat_name",
            "analogy_line",
            "scroll_time",
            "personality_type",
            "personality_explanation",
            "niche_journey",
            "top_niches",
            "top_niche_percentile",
            "brain_rot_score",
            "brainrot_intensity",
            "brainrot_volume_hours",
            "brainrot_confidence",
            "brainrot_enriched_hours",
            "brainrot_enriched_watch_pct",
            "brainrot_intensity_raw",
            "brainrot_intensity_raw_effective",
            "brainrot_intensity_linear",
            "brainrot_quality_weight",
            "brainrot_normalization",
            "brain_rot_explanation",
            "keyword_2026",
            "thumb_roast",
            "accessory_set",
            "_analysis_warnings",
            "_analysis_debug",
        ):
            payload.pop(key, None)
        for key in tuple(payload.keys()):
            if isinstance(key, str) and key.startswith("_brainrot_"):
                payload.pop(key, None)
        run.payload = payload
    job = job_queue.enqueue(
        db,
        task_name="wrapped_analysis",
        payload={"wrapped_run_id": run.id, "force_new_email": True},
        idempotency_key=f"analysis:{run.id}",
        force_new=True,
    )
    if run.status != "pending":
        run.status = "pending"
    if isinstance(run.payload, dict):
        payload = run.payload
        data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
        data_jobs["wrapped_analysis"] = {"id": job.id, "status": job.status}
        payload["data_jobs"] = data_jobs
        run.payload = payload
    db.add(run)
    db.commit()
    return AdminTestEnqueueResponse(
        wrapped_run_id=run.id,
        task_name=job.task_name,
        job_id=job.id,
        status=job.status,
        idempotency_key=job.idempotency_key,
    )


@app.post(
    "/admin/test/runs/{wrapped_run_id}/retry-email",
    response_model=AdminTestEnqueueResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_test_retry_email(
    wrapped_run_id: str, _admin=Depends(require_admin), db=Depends(get_db)
) -> AdminTestEnqueueResponse:
    run = db.get(AppWrappedRun, wrapped_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="not_found")
    if not run.app_user_id:
        raise HTTPException(status_code=400, detail="missing_app_user_id")
    job = job_queue.enqueue(
        db,
        task_name="email_send",
        payload={"wrapped_run_id": run.id, "app_user_id": run.app_user_id},
        idempotency_key=f"email:{run.id}",
        force_new=True,
    )
    return AdminTestEnqueueResponse(
        wrapped_run_id=run.id,
        task_name=job.task_name,
        job_id=job.id,
        status=job.status,
        idempotency_key=job.idempotency_key,
    )


@app.post(
    "/admin/test/runs/{wrapped_run_id}/retry-watch-history",
    response_model=AdminTestEnqueueResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_test_retry_watch_history(
    wrapped_run_id: str, _admin=Depends(require_admin), db=Depends(get_db)
) -> AdminTestEnqueueResponse:
    run = db.get(AppWrappedRun, wrapped_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="not_found")
    if not run.app_user_id:
        raise HTTPException(status_code=400, detail="missing_app_user_id")

    user = db.get(AppUser, run.app_user_id)
    sec_user_id = run.sec_user_id or (user.latest_sec_user_id if user else None)
    if not sec_user_id:
        raise HTTPException(status_code=400, detail="sec_user_id_required")

    job = job_queue.enqueue(
        db,
        task_name="watch_history_fetch_2025",
        payload={
            "wrapped_run_id": run.id,
            "app_user_id": run.app_user_id,
            "sec_user_id": sec_user_id,
            "time_zone": (user.time_zone if user else None),
            "platform_username": (user.platform_username if user else None),
            "force_new_analysis": True,
            "force_new_email": True,
        },
        idempotency_key=f"wrapped:{run.id}",
        force_new=True,
    )

    run.watch_history_job_id = job.id
    payload = run.payload if isinstance(run.payload, dict) else {}
    data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
    # Force new analysis/email; clear previous computed analysis fields so the status payload reflects a true rerun.
    for key in (
        "_analysis_warnings",
        "_analysis_debug",
        "personality_type",
        "personality_explanation",
        "niche_journey",
        "top_niches",
        "top_niche_percentile",
        "brain_rot_score",
        "brainrot_intensity",
        "brainrot_volume_hours",
        "brainrot_confidence",
        "brainrot_enriched_hours",
        "brainrot_enriched_watch_pct",
        "brainrot_intensity_raw",
        "brainrot_intensity_raw_effective",
        "brainrot_intensity_linear",
        "brainrot_quality_weight",
        "brainrot_normalization",
        "brain_rot_explanation",
        "keyword_2026",
        "thumb_roast",
        "accessory_set",
        "cat_name",
        "analogy_line",
        "scroll_time",
    ):
        payload.pop(key, None)
    for key in tuple(payload.keys()):
        if isinstance(key, str) and key.startswith("_brainrot_"):
            payload.pop(key, None)
    data_jobs.pop("wrapped_analysis", None)
    data_jobs.pop("email_send", None)
    data_jobs["watch_history"] = {"id": job.id, "status": job.status}
    payload["data_jobs"] = data_jobs
    run.payload = payload
    if run.status != "pending":
        run.status = "pending"
    db.add(run)
    db.commit()

    return AdminTestEnqueueResponse(
        wrapped_run_id=run.id,
        task_name=job.task_name,
        job_id=job.id,
        status=job.status,
        idempotency_key=job.idempotency_key,
    )


@app.post(
    "/admin/test/users/{app_user_id}/fetch-watch-history",
    response_model=AdminTestEnqueueResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_test_fetch_watch_history(
    app_user_id: str,
    force_new_run: bool = False,
    scrape_max_videos: Optional[int] = None,
    since_days: Optional[int] = None,
    _admin=Depends(require_admin),
    db=Depends(get_db),
) -> AdminTestEnqueueResponse:
    user = db.get(AppUser, app_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")
    if not user.latest_sec_user_id:
        raise HTTPException(status_code=400, detail="sec_user_id_required")

    run = None
    if not force_new_run:
        run = (
            db.query(AppWrappedRun)
            .filter(AppWrappedRun.app_user_id == app_user_id)
            .order_by(AppWrappedRun.created_at.desc(), AppWrappedRun.updated_at.desc())
            .first()
        )
    if not run:
        run = AppWrappedRun(
            id=str(uuid4()),
            app_user_id=app_user_id,
            sec_user_id=user.latest_sec_user_id,
            archive_user_id=user.archive_user_id,
            status="pending",
            email=None,
        )
        db.add(run)
        db.commit()

    job = job_queue.enqueue(
        db,
        task_name="watch_history_fetch_2025",
        payload={
            "wrapped_run_id": run.id,
            "app_user_id": app_user_id,
            "sec_user_id": user.latest_sec_user_id,
            "time_zone": user.time_zone,
            "platform_username": user.platform_username,
            "scrape_max_videos": scrape_max_videos,
            "since_days": since_days,
            "force_new_analysis": True,
            "force_new_email": True,
        },
        idempotency_key=f"wrapped:{run.id}",
        force_new=True,
    )

    run.watch_history_job_id = job.id
    payload = run.payload if isinstance(run.payload, dict) else {}
    data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
    data_jobs["watch_history"] = {"id": job.id, "status": job.status}
    payload["data_jobs"] = data_jobs
    payload.pop("_watch_history_progress", None)
    run.payload = payload
    if run.status != "pending":
        run.status = "pending"
    db.add(run)
    db.commit()

    return AdminTestEnqueueResponse(
        wrapped_run_id=run.id,
        task_name=job.task_name,
        job_id=job.id,
        status=job.status,
        idempotency_key=job.idempotency_key,
    )


@app.post(
    "/admin/users/{app_user_id}/verify-region",
    response_model=AdminVerifyRegionEnqueueResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_verify_region(
    app_user_id: str, _admin=Depends(require_admin), db=Depends(get_db)
) -> AdminVerifyRegionEnqueueResponse:
    user = db.get(AppUser, app_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")
    if not user.latest_sec_user_id:
        raise HTTPException(status_code=400, detail="sec_user_id_required")
    job = job_queue.enqueue(
        db,
        task_name="watch_history_verify",
        payload={"app_user_id": user.app_user_id, "auto_enqueue": True},
        idempotency_key=f"watch_history_verify:{user.app_user_id}",
    )
    return AdminVerifyRegionEnqueueResponse(
        app_user_id=user.app_user_id,
        task_name=job.task_name,
        job_id=job.id,
        status=job.status,
        idempotency_key=job.idempotency_key,
    )


@app.post(
    "/admin/users/verify-region/batch",
    response_model=AdminVerifyRegionBatchResponse,
    responses={401: {"model": ErrorResponse}},
)
async def admin_verify_region_batch(
    payload: AdminVerifyRegionBatchRequest, _admin=Depends(require_admin), db=Depends(get_db)
) -> AdminVerifyRegionBatchResponse:
    statuses = ["no"]
    if payload.include_unknown:
        statuses.append("unknown")
    base_query = db.query(AppUser).filter(
        AppUser.is_watch_history_available.in_(statuses),
        AppUser.latest_sec_user_id.isnot(None),
        AppUser.latest_sec_user_id != "",
    )
    matched = int(base_query.count())
    users = base_query.order_by(AppUser.updated_at.asc()).all()

    results: List[AdminVerifyRegionBatchItem] = []
    job_ids: List[str] = []
    enqueued = 0
    for user in users:
        job = job_queue.enqueue(
            db,
            task_name="watch_history_verify",
            payload={"app_user_id": user.app_user_id, "auto_enqueue": payload.auto_enqueue},
            idempotency_key=f"watch_history_verify:{user.app_user_id}",
            force_new=payload.force_new,
        )
        enqueued += 1
        job_ids.append(job.id)
        results.append(
            AdminVerifyRegionBatchItem(
                app_user_id=user.app_user_id,
                sec_user_id=user.latest_sec_user_id,
                job_id=job.id,
                status=job.status,
                idempotency_key=job.idempotency_key,
            )
        )

    batch_id = str(uuid4())
    batch_payload = {
        "type": "watch_history_verify",
        "created_at": _iso_utc(datetime.utcnow()),
        "params": payload.model_dump(),
        "job_ids": job_ids,
        "matched": matched,
        "processed": len(results),
        "enqueued": enqueued,
    }
    batch_job = AppJob(
        id=batch_id,
        task_name="watch_history_verify_batch",
        payload=batch_payload,
        status="succeeded",
        attempts=0,
        max_attempts=0,
        not_before=None,
        idempotency_key=f"watch_history_verify_batch:{batch_id}",
        locked_by=None,
        locked_at=None,
        lease_seconds=0,
    )
    db.add(batch_job)
    db.commit()

    return AdminVerifyRegionBatchResponse(
        batch_id=batch_id,
        matched=matched,
        processed=len(results),
        enqueued=enqueued,
        results=results,
    )


@app.get(
    "/admin/users/verify-region/batch/{batch_id}",
    response_model=AdminVerifyRegionBatchStatusResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_verify_region_batch_status(
    batch_id: str, _admin=Depends(require_admin), db=Depends(get_db)
) -> AdminVerifyRegionBatchStatusResponse:
    batch = db.get(AppJob, batch_id)
    if not batch or batch.task_name != "watch_history_verify_batch":
        raise HTTPException(status_code=404, detail="batch_not_found")
    payload = batch.payload if isinstance(batch.payload, dict) else {}
    job_ids = payload.get("job_ids") if isinstance(payload.get("job_ids"), list) else []
    job_ids = [str(jid) for jid in job_ids if jid]
    jobs = db.query(AppJob).filter(AppJob.id.in_(job_ids)).all() if job_ids else []
    job_map = {job.id: job for job in jobs}

    results: List[AdminVerifyRegionBatchStatusItem] = []
    yes = no = unknown = error = completed = 0
    for job_id in job_ids:
        job = job_map.get(job_id)
        if not job:
            results.append(
                AdminVerifyRegionBatchStatusItem(
                    job_id=job_id,
                    job_status="missing",
                    error="job_not_found",
                )
            )
            continue
        job_payload = job.payload if isinstance(job.payload, dict) else {}
        result = job_payload.get("result") if isinstance(job_payload.get("result"), dict) else {}
        verify_status = result.get("status") if isinstance(result.get("status"), str) else None
        if verify_status:
            completed += 1
            if verify_status == "yes":
                yes += 1
            elif verify_status == "no":
                no += 1
            elif verify_status == "unknown":
                unknown += 1
            else:
                error += 1
        results.append(
            AdminVerifyRegionBatchStatusItem(
                app_user_id=result.get("app_user_id") or job_payload.get("app_user_id"),
                sec_user_id=result.get("sec_user_id"),
                job_id=job_id,
                job_status=job.status,
                verify_status=verify_status,
                attempts=result.get("attempts"),
                last_error=result.get("last_error"),
                checked_at=result.get("checked_at"),
            )
        )

    total = len(job_ids)
    pending = max(0, total - completed)
    created_at = payload.get("created_at") if isinstance(payload.get("created_at"), str) else None
    return AdminVerifyRegionBatchStatusResponse(
        batch_id=batch_id,
        created_at=created_at,
        total=total,
        completed=completed,
        pending=pending,
        yes=yes,
        no=no,
        unknown=unknown,
        error=error,
        results=results,
    )


@app.get(
    "/admin/users/{app_user_id}/stage",
    response_model=AdminUserStageResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_user_stage(
    app_user_id: str,
    wrapped_run_id: Optional[str] = None,
    _admin=Depends(require_admin),
    db=Depends(get_db),
) -> AdminUserStageResponse:
    _validate_app_user_id(app_user_id)
    user, run, verify_job, effective_data_jobs, jobs, progress, stage, next_stage, _ = _load_user_stage(
        db, app_user_id, wrapped_run_id=wrapped_run_id
    )

    verify_payload = verify_job.payload if verify_job and isinstance(verify_job.payload, dict) else {}
    verify_result = verify_payload.get("result") if isinstance(verify_payload.get("result"), dict) else None
    verify_stuck = False
    verify_stuck_reason = None
    verify_locked_for = None
    if verify_job:
        verify_stuck, verify_stuck_reason, verify_locked_for = _job_stuck_meta(verify_job)
    verify = AdminUserStageVerify(
        is_watch_history_available=user.is_watch_history_available,
        job_id=verify_job.id if verify_job else None,
        job_status=verify_job.status if verify_job else None,
        is_stuck=verify_stuck if verify_job else None,
        stuck_reason=verify_stuck_reason,
        locked_for_seconds=verify_locked_for,
        locked_by=verify_job.locked_by if verify_job else None,
        locked_at=verify_job.locked_at.isoformat() if verify_job and verify_job.locked_at else None,
        updated_at=verify_job.updated_at.isoformat() if verify_job and verify_job.updated_at else None,
        result=verify_result,
    )

    sec_user_id = None
    if run and run.sec_user_id:
        sec_user_id = run.sec_user_id
    elif user.latest_sec_user_id:
        sec_user_id = user.latest_sec_user_id

    return AdminUserStageResponse(
        app_user_id=app_user_id,
        sec_user_id=sec_user_id,
        wrapped_run_id=run.id if run else None,
        run_status=run.status if run else None,
        stage=stage,
        next_stage=next_stage,
        verify=verify,
        data_jobs=effective_data_jobs,
        watch_history_progress=progress,
        jobs=jobs,
    )


@app.post(
    "/admin/users/{app_user_id}/restart",
    response_model=AdminUserRestartResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_user_restart(
    app_user_id: str,
    payload: AdminUserRestartRequest,
    _admin=Depends(require_admin),
    db=Depends(get_db),
) -> AdminUserRestartResponse:
    _validate_app_user_id(app_user_id)
    return _restart_user_stage(db, app_user_id, payload)


@app.post(
    "/admin/users/stage/batch",
    response_model=AdminUserStageBatchResponse,
    responses={401: {"model": ErrorResponse}},
)
async def admin_user_stage_batch(
    payload: AdminUserStageBatchRequest,
    _admin=Depends(require_admin),
    db=Depends(get_db),
) -> AdminUserStageBatchResponse:
    users, limit, offset = _batch_user_query(
        db,
        limit=payload.limit,
        offset=payload.offset,
        include_admin_test=payload.include_admin_test,
        watch_history_status=payload.watch_history_status,
    )
    items: List[AdminUserStageResponse] = []
    requested = len(users)

    for user in users:
        if "@" in user.app_user_id:
            continue
        try:
            user_rec, run, verify_job, effective_data_jobs, jobs, progress, stage, next_stage, _ = _load_user_stage(
                db, user.app_user_id
            )
        except HTTPException:
            continue

        run_status_value = _run_status_value(run)
        if payload.run_status and run_status_value not in payload.run_status:
            continue
        if payload.stage and stage not in payload.stage:
            continue

        items.append(
            _build_user_stage_response(
                app_user_id=user_rec.app_user_id,
                user=user_rec,
                run=run,
                verify_job=verify_job,
                effective_data_jobs=effective_data_jobs,
                jobs=jobs,
                progress=progress,
                stage=stage,
                next_stage=next_stage,
            )
        )

    return AdminUserStageBatchResponse(
        limit=limit,
        offset=offset,
        requested=requested,
        returned=len(items),
        items=items,
    )


@app.post(
    "/admin/users/restart/batch",
    response_model=AdminUserRestartBatchResponse,
    responses={401: {"model": ErrorResponse}},
)
async def admin_user_restart_batch(
    payload: AdminUserRestartBatchRequest,
    _admin=Depends(require_admin),
    db=Depends(get_db),
) -> AdminUserRestartBatchResponse:
    users, limit, offset = _batch_user_query(
        db,
        limit=payload.limit,
        offset=payload.offset,
        include_admin_test=payload.include_admin_test,
        watch_history_status=payload.watch_history_status,
    )
    results: List[AdminUserRestartBatchItem] = []
    requested = len(users)
    processed = 0
    enqueued = 0
    skipped = 0

    for user in users:
        processed += 1
        if "@" in user.app_user_id:
            skipped += 1
            results.append(AdminUserRestartBatchItem(app_user_id=user.app_user_id, error="invalid_app_user_id"))
            continue

        try:
            user_rec, run, verify_job, effective_data_jobs, _jobs, _progress, stage, _next_stage, _ = _load_user_stage(
                db, user.app_user_id
            )
        except HTTPException as exc:
            skipped += 1
            results.append(AdminUserRestartBatchItem(app_user_id=user.app_user_id, error=str(exc.detail)))
            continue

        run_status_value = _run_status_value(run)
        if payload.run_status and run_status_value not in payload.run_status:
            skipped += 1
            results.append(
                AdminUserRestartBatchItem(
                    app_user_id=user_rec.app_user_id,
                    result=AdminUserRestartResponse(
                        app_user_id=user_rec.app_user_id,
                        wrapped_run_id=run.id if run else None,
                        selected_stage=stage,
                        skipped_reason="run_status_filtered",
                    ),
                )
            )
            continue

        if payload.stage_filter and stage not in payload.stage_filter:
            skipped += 1
            results.append(
                AdminUserRestartBatchItem(
                    app_user_id=user_rec.app_user_id,
                    result=AdminUserRestartResponse(
                        app_user_id=user_rec.app_user_id,
                        wrapped_run_id=run.id if run else None,
                        selected_stage=stage,
                        skipped_reason="stage_filtered",
                    ),
                )
            )
            continue

        restart_payload = AdminUserRestartRequest(
            stage=payload.restart_stage or "next",
            wrapped_run_id=None,
            force_new_jobs=payload.force_new_jobs,
            reset_payload=payload.reset_payload,
        )
        try:
            result = _restart_user_stage(db, user_rec.app_user_id, restart_payload, dry_run=payload.dry_run)
        except HTTPException as exc:
            skipped += 1
            results.append(AdminUserRestartBatchItem(app_user_id=user_rec.app_user_id, error=str(exc.detail)))
            continue

        if result.job_id and not result.skipped_reason and not payload.dry_run:
            enqueued += 1
        else:
            skipped += 1

        results.append(AdminUserRestartBatchItem(app_user_id=user_rec.app_user_id, result=result))

    return AdminUserRestartBatchResponse(
        limit=limit,
        offset=offset,
        requested=requested,
        processed=processed,
        enqueued=enqueued,
        skipped=skipped,
        dry_run=payload.dry_run,
        results=results,
    )


@app.delete(
    "/admin/users/{app_user_id}",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_delete_user(
    app_user_id: str,
    confirm: bool = False,
    _admin=Depends(require_admin),
    db=Depends(get_db),
) -> Dict[str, Any]:
    """
    Hard-delete an app user and related rows.

    Requires `?confirm=true` to reduce accidental deletions.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm_required")

    user = db.get(AppUser, app_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="not_found")

    runs = (
        db.query(AppWrappedRun)
        .filter(AppWrappedRun.app_user_id == app_user_id)
        .order_by(AppWrappedRun.created_at.desc())
        .all()
    )
    run_ids = [r.id for r in runs if r and r.id]

    job_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    for run in runs:
        if run.watch_history_job_id:
            job_ids.add(str(run.watch_history_job_id))
        idempotency_keys.add(f"wrapped:{run.id}")
        idempotency_keys.add(f"analysis:{run.id}")
        idempotency_keys.add(f"email:{run.id}")
        payload = run.payload if isinstance(run.payload, dict) else {}
        data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
        for meta in data_jobs.values():
            if isinstance(meta, dict):
                jid = meta.get("id")
                if isinstance(jid, str) and jid:
                    job_ids.add(jid)

    deleted: Dict[str, int] = {}

    deleted["app_sessions"] = db.query(AppSession).filter(AppSession.app_user_id == app_user_id).delete(
        synchronize_session=False
    )
    deleted["app_user_emails"] = db.query(AppUserEmail).filter(AppUserEmail.app_user_id == app_user_id).delete(
        synchronize_session=False
    )
    deleted["app_auth_jobs"] = db.query(AppAuthJob).filter(AppAuthJob.app_user_id == app_user_id).delete(
        synchronize_session=False
    )

    deleted["referral_events"] = (
        db.query(ReferralEvent)
        .filter((ReferralEvent.referrer_app_user_id == app_user_id) | (ReferralEvent.referred_app_user_id == app_user_id))
        .delete(synchronize_session=False)
    )
    deleted["referrals"] = db.query(Referral).filter(Referral.referrer_app_user_id == app_user_id).delete(
        synchronize_session=False
    )

    # Remove jobs associated to the user's runs.
    deleted["app_jobs_by_id"] = 0
    if job_ids:
        deleted["app_jobs_by_id"] = db.query(AppJob).filter(AppJob.id.in_(sorted(job_ids))).delete(
            synchronize_session=False
        )
    deleted["app_jobs_by_idempotency_key"] = 0
    if idempotency_keys:
        deleted["app_jobs_by_idempotency_key"] = db.query(AppJob).filter(
            AppJob.idempotency_key.in_(sorted(idempotency_keys))
        ).delete(synchronize_session=False)

    # Best-effort cleanup of any remaining jobs directly keyed to this user (postgres json filter).
    deleted["app_jobs_by_payload_app_user_id"] = 0
    dialect = ""
    with suppress(Exception):
        dialect = str(db.get_bind().dialect.name or "")
    if dialect == "postgresql":
        res = db.execute(
            text("DELETE FROM app_jobs WHERE payload->>'app_user_id' = :app_user_id"),
            {"app_user_id": app_user_id},
        )
        deleted["app_jobs_by_payload_app_user_id"] = int(getattr(res, "rowcount", 0) or 0)

    deleted["app_wrapped_runs"] = 0
    if run_ids:
        deleted["app_wrapped_runs"] = db.query(AppWrappedRun).filter(AppWrappedRun.id.in_(run_ids)).delete(
            synchronize_session=False
        )

    deleted["app_users"] = db.query(AppUser).filter(AppUser.app_user_id == app_user_id).delete(synchronize_session=False)
    db.commit()

    return {"deleted": deleted, "app_user_id": app_user_id, "wrapped_run_ids": run_ids}


@app.post(
    "/admin/test/prompt",
    response_model=AdminPromptPipelineResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def admin_test_prompt(payload: AdminPromptPipelineRequest, _admin=Depends(require_admin)) -> AdminPromptPipelineResponse:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    model = (payload.model or os.getenv("OPENROUTER_MODEL") or "").strip()
    if not api_key or not model:
        raise HTTPException(status_code=400, detail="openrouter_config_missing")

    since_ms = _watch_history_since_ms()
    now_dt = datetime.now(timezone.utc)
    start_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
    range_spec: Dict[str, Any] = {"type": "between", "start_at": _iso_utc(start_dt), "end_at": _iso_utc(now_dt)}

    sample_limit = 50
    months_span = _months_between(start_dt, now_dt)
    per_month = max(1, sample_limit // months_span)
    samples_limit = min(sample_limit, per_month * months_span)

    try:
        samples = await archive_client.watch_history_analytics_samples(
            sec_user_id=payload.sec_user_id,
            range=range_spec,
            time_zone=payload.time_zone,
            strategy={"type": "per_month", "per_month": per_month},
            limit=samples_limit,
            max_chars_per_item=300,
            fields=["title", "description", "hashtags", "music", "author"],
            include_video_id=True,
            include_watched_at=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"archive_samples_failed: {_format_http_error(exc)}") from exc

    try:
        sample_items = samples["items"]
        if not isinstance(sample_items, list):
            raise TypeError("samples.items must be a list")
    except Exception:
        raise HTTPException(status_code=502, detail="unexpected_samples_shape")

    sample_texts: List[str] = []
    for item in sample_items:
        if not isinstance(item, dict):
            continue
        text_val = str(item.get("text") or "").strip()
        if text_val:
            sample_texts.append(text_val)
    sample_texts = sample_texts[:sample_limit]
    if not sample_texts:
        raise HTTPException(status_code=422, detail="no_sample_texts")

    input_text = "\n".join(sample_texts[:20])
    prompt_order = [
        "personality",
        "personality_explanation",
        "niche_journey",
        "top_niches",
        "brainrot_score",
        "brainrot_explanation",
        "keyword_2026",
        "thumb_roast",
    ]

    async def call_openrouter(prompt: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": input_text},
                    ],
                    "temperature": float(payload.temperature),
                },
            )
        if resp.status_code != 200:
            raise RuntimeError(f"openrouter_{resp.status_code}: {resp.text}")
        data = resp.json()
        out = (
            (((data.get("choices") or [{}])[0] or {}).get("message") or {}).get("content") or ""
            if isinstance(data, dict)
            else ""
        )
        return str(out).strip()

    results: List[AdminPromptCallResult] = []
    personality_name: Optional[str] = None
    brainrot_score_val: Optional[int] = None
    for key in prompt_order:
        prompt_text = None
        if isinstance(payload.prompt_overrides, dict):
            override = payload.prompt_overrides.get(key)
            if isinstance(override, str) and override.strip():
                prompt_text = override.strip()
        if not prompt_text:
            prompt_text = _PROMPT_MAP.get(key)
        if not prompt_text:
            results.append(AdminPromptCallResult(prompt_key=key, prompt_used="", error="missing_prompt"))
            continue
        if key == "personality_explanation" and personality_name:
            prompt_text = f"{prompt_text}\n\nPersonality name: {personality_name}\nExplain why this exact name fits."
        if key == "brainrot_explanation" and brainrot_score_val is not None:
            prompt_text = f"{prompt_text}\n\nBrainrot score: {brainrot_score_val}"
        if payload.dry_run:
            results.append(AdminPromptCallResult(prompt_key=key, prompt_used=prompt_text, output_text=None, parsed=None))
            continue

        try:
            raw = await call_openrouter(prompt_text)
            parsed: Any = None
            if key == "personality":
                if not raw:
                    raise ValueError("empty_output")
                parsed = raw.strip().split()[0].lower().replace(" ", "_")
                personality_name = raw.strip()
            elif key == "niche_journey":
                parsed_json = json.loads(raw)
                if not isinstance(parsed_json, list):
                    raise TypeError("expected_json_list")
                parsed = parsed_json[:5]
            elif key == "top_niches":
                parsed_json = json.loads(raw)
                if not isinstance(parsed_json, dict):
                    raise TypeError("expected_json_object")
                top_niches = parsed_json.get("top_niches")
                pct = parsed_json.get("top_niche_percentile")
                if not isinstance(top_niches, list) or not pct:
                    raise ValueError("missing_keys")
                pct_val = str(pct).strip()
                match = re.search(r"\b(\d{1,2}(?:\.\d)?)%\b", pct_val)
                if match:
                    try:
                        num = float(match.group(1))
                        if num <= 0:
                            num = 0.1
                        if num > 3.0:
                            num = 3.0
                        pct_val = f"top {num:.1f}%" if num < 1.0 else f"top {int(num)}%"
                    except Exception:
                        pct_val = str(pct).strip()
                parsed = {
                    "top_niches": [str(x).strip() for x in top_niches if str(x).strip()][:2],
                    "top_niche_percentile": pct_val,
                }
            elif key == "brainrot_score":
                if not raw:
                    raise ValueError("empty_output")
                parsed = max(0, min(100, int(float(raw.strip().split()[0]))))
                brainrot_score_val = int(parsed)
            elif key == "keyword_2026":
                if not raw:
                    raise ValueError("empty_output")
                parsed = raw.strip().splitlines()[0]
            else:
                parsed = raw

            results.append(AdminPromptCallResult(prompt_key=key, prompt_used=prompt_text, output_text=raw, parsed=parsed))
        except Exception as exc:
            results.append(
                AdminPromptCallResult(prompt_key=key, prompt_used=prompt_text, output_text=None, parsed=None, error=str(exc))
            )

    return AdminPromptPipelineResponse(
        sec_user_id=payload.sec_user_id,
        model=model,
        range=range_spec,
        sample_texts_used=sample_texts[:20],
        results=results,
    )


@app.post(
    "/admin/wrapped/runs/{wrapped_run_id}/retry",
    response_model=AdminTestEnqueueResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def admin_wrapped_retry_run(
    wrapped_run_id: str,
    payload: AdminWrappedRetryRequest,
    _admin=Depends(require_admin),
    db=Depends(get_db),
) -> AdminTestEnqueueResponse:
    run = db.get(AppWrappedRun, wrapped_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="not_found")
    if not run.app_user_id:
        raise HTTPException(status_code=400, detail="missing_app_user_id")

    user = db.get(AppUser, run.app_user_id)
    if not user or not user.latest_sec_user_id:
        raise HTTPException(status_code=400, detail="sec_user_id_required")

    run_payload = run.payload if isinstance(run.payload, dict) else {}
    if payload.reset_payload:
        for key in (
            "_watch_history_progress",
            "_sample_texts",
            "_bucket_summaries",
            "_months_done",
            "_acc_state",
            "_analysis_warnings",
            "_analysis_debug",
            "total_hours",
            "total_videos",
            "night_pct",
            "peak_hour",
            "top_music",
            "top_creators",
            "cat_name",
            "analogy_line",
            "scroll_time",
            "personality_type",
            "personality_explanation",
            "niche_journey",
            "top_niches",
            "top_niche_percentile",
            "brain_rot_score",
            "brainrot_intensity",
            "brainrot_volume_hours",
            "brainrot_confidence",
            "brainrot_enriched_hours",
            "brainrot_enriched_watch_pct",
            "brainrot_intensity_raw",
            "brainrot_intensity_raw_effective",
            "brainrot_intensity_linear",
            "brainrot_quality_weight",
            "brainrot_normalization",
            "brain_rot_explanation",
            "keyword_2026",
            "thumb_roast",
            "source_spans",
            "accessory_set",
        ):
            run_payload.pop(key, None)
        for key in tuple(run_payload.keys()):
            if isinstance(key, str) and key.startswith("_brainrot_"):
                run_payload.pop(key, None)

    data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
    if payload.reset_payload:
        data_jobs.pop("watch_history", None)
        data_jobs.pop("wrapped_analysis", None)
        data_jobs.pop("email_send", None)
    run_payload["data_jobs"] = data_jobs
    run.payload = run_payload

    if run.status != "pending":
        run.status = "pending"

    if payload.include_watch_history:
        job = job_queue.enqueue(
            db,
            task_name="watch_history_fetch_2025",
            payload={
                "wrapped_run_id": run.id,
                "app_user_id": run.app_user_id,
                "sec_user_id": user.latest_sec_user_id,
                "time_zone": user.time_zone,
                "platform_username": user.platform_username,
                "force_new_analysis": bool(payload.include_analysis),
                "force_new_email": True,
            },
            idempotency_key=f"wrapped:{run.id}",
            force_new=True,
        )
        run.watch_history_job_id = job.id
        data_jobs["watch_history"] = {"id": job.id, "status": job.status}
        run.payload = run_payload
        db.add(run)
        db.commit()
        return AdminTestEnqueueResponse(
            wrapped_run_id=run.id,
            task_name=job.task_name,
            job_id=job.id,
            status=job.status,
            idempotency_key=job.idempotency_key,
        )

    if payload.include_analysis:
        job = job_queue.enqueue(
            db,
            task_name="wrapped_analysis",
            payload={"wrapped_run_id": run.id, "force_new_email": True},
            idempotency_key=f"analysis:{run.id}",
            force_new=True,
        )
        data_jobs["wrapped_analysis"] = {"id": job.id, "status": job.status}
        run.payload = run_payload
        db.add(run)
        db.commit()
        return AdminTestEnqueueResponse(
            wrapped_run_id=run.id,
            task_name=job.task_name,
            job_id=job.id,
            status=job.status,
            idempotency_key=job.idempotency_key,
        )

    raise HTTPException(status_code=400, detail="nothing_to_retry")


@app.post(
    "/admin/wrapped/runs/retry-failed",
    response_model=AdminRetryFailedRunsResponse,
    responses={401: {"model": ErrorResponse}},
)
async def admin_retry_failed_runs(
    payload: AdminRetryFailedRunsRequest,
    _admin=Depends(require_admin),
    db=Depends(get_db),
) -> AdminRetryFailedRunsResponse:
    """Find failed wrapped runs and enqueue retries (watch history / analysis / email) as appropriate."""
    limit = max(1, min(int(payload.limit or 50), 200))
    dry_run = bool(payload.dry_run)

    runs = (
        db.query(AppWrappedRun)
        .filter(AppWrappedRun.status == "failed")
        .filter(or_(AppWrappedRun.app_user_id.is_(None), ~AppWrappedRun.app_user_id.like("admin-test-%")))
        .order_by(AppWrappedRun.updated_at.desc(), AppWrappedRun.created_at.desc())
        .limit(limit)
        .all()
    )
    matched = len(runs)
    results: List[AdminRetryFailedRunsItem] = []
    enqueued = 0
    processed = 0

    def _pop_analysis_keys(run_payload: Dict[str, Any]) -> None:
        for key in (
            "_analysis_warnings",
            "_analysis_debug",
            "personality_type",
            "personality_explanation",
            "niche_journey",
            "top_niches",
            "top_niche_percentile",
            "brain_rot_score",
            "brainrot_intensity",
            "brainrot_volume_hours",
            "brainrot_confidence",
            "brainrot_enriched_hours",
            "brainrot_enriched_watch_pct",
            "brainrot_intensity_raw",
            "brainrot_intensity_raw_effective",
            "brainrot_intensity_linear",
            "brainrot_quality_weight",
            "brainrot_normalization",
            "brain_rot_explanation",
            "keyword_2026",
            "thumb_roast",
            "accessory_set",
            "cat_name",
            "analogy_line",
            "scroll_time",
        ):
            run_payload.pop(key, None)
        for key in tuple(run_payload.keys()):
            if isinstance(key, str) and key.startswith("_brainrot_"):
                run_payload.pop(key, None)

    for run in runs:
        processed += 1
        run_payload = run.payload if isinstance(run.payload, dict) else {}
        data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
        watch_meta = data_jobs.get("watch_history") if isinstance(data_jobs.get("watch_history"), dict) else {}
        analysis_meta = data_jobs.get("wrapped_analysis") if isinstance(data_jobs.get("wrapped_analysis"), dict) else {}
        email_meta = data_jobs.get("email_send") if isinstance(data_jobs.get("email_send"), dict) else {}

        watch_status = str(watch_meta.get("status") or "").lower() or None
        analysis_status = str(analysis_meta.get("status") or "").lower() or None
        email_status = str(email_meta.get("status") or "").lower() or None

        # Auto decide what to retry if caller didn't specify.
        include_watch = payload.include_watch_history
        include_analysis = payload.include_analysis
        include_email = payload.include_email

        if include_watch is None and include_analysis is None and include_email is None:
            # Prefer retrying the first failing stage.
            if watch_status in (None, "", "failed"):
                include_watch = True
            elif analysis_status in (None, "", "failed"):
                include_analysis = True
            elif email_status in (None, "", "failed"):
                include_email = True
            else:
                include_analysis = True

        user = db.get(AppUser, run.app_user_id) if run.app_user_id else None
        sec_user_id = run.sec_user_id or (user.latest_sec_user_id if user else None)

        if include_watch and watch_status in ("pending", "running"):
            results.append(
                AdminRetryFailedRunsItem(
                    wrapped_run_id=run.id,
                    app_user_id=run.app_user_id,
                    sec_user_id=sec_user_id,
                    action="skip",
                    skipped_reason="watch_history_in_progress",
                )
            )
            continue
        if include_analysis and analysis_status in ("pending", "running"):
            results.append(
                AdminRetryFailedRunsItem(
                    wrapped_run_id=run.id,
                    app_user_id=run.app_user_id,
                    sec_user_id=sec_user_id,
                    action="skip",
                    skipped_reason="analysis_in_progress",
                )
            )
            continue

        if payload.reset_payload and isinstance(run_payload, dict):
            if include_watch:
                # Full reset when re-running watch history.
                for key in (
                    "_watch_history_progress",
                    "_sample_texts",
                    "_bucket_summaries",
                    "_months_done",
                    "_acc_state",
                    "_analysis_warnings",
                    "_analysis_debug",
                    "total_hours",
                    "total_videos",
                    "night_pct",
                    "peak_hour",
                    "top_music",
                    "top_creators",
                    "cat_name",
                    "analogy_line",
                    "scroll_time",
                    "personality_type",
                    "personality_explanation",
                    "niche_journey",
                    "top_niches",
                    "top_niche_percentile",
                    "brain_rot_score",
                    "brain_rot_explanation",
                    "keyword_2026",
                    "thumb_roast",
                    "source_spans",
                    "accessory_set",
                ):
                    run_payload.pop(key, None)
                if isinstance(data_jobs, dict):
                    data_jobs.pop("watch_history", None)
                    data_jobs.pop("wrapped_analysis", None)
                    data_jobs.pop("email_send", None)
            elif include_analysis:
                _pop_analysis_keys(run_payload)
                if isinstance(data_jobs, dict):
                    data_jobs.pop("wrapped_analysis", None)
                    data_jobs.pop("email_send", None)
            run_payload["data_jobs"] = data_jobs

        if include_watch:
            if not run.app_user_id or not sec_user_id:
                results.append(
                    AdminRetryFailedRunsItem(
                        wrapped_run_id=run.id,
                        app_user_id=run.app_user_id,
                        sec_user_id=sec_user_id,
                        action="skip",
                        skipped_reason="missing_app_user_id_or_sec_user_id",
                    )
                )
                continue
            if dry_run:
                results.append(
                    AdminRetryFailedRunsItem(
                        wrapped_run_id=run.id,
                        app_user_id=run.app_user_id,
                        sec_user_id=sec_user_id,
                        action="dry_run",
                        enqueued_task="watch_history_fetch_2025",
                        idempotency_key=f"wrapped:{run.id}",
                    )
                )
                continue
            job = job_queue.enqueue(
                db,
                task_name="watch_history_fetch_2025",
                payload={
                    "wrapped_run_id": run.id,
                    "app_user_id": run.app_user_id,
                    "sec_user_id": sec_user_id,
                    "time_zone": (user.time_zone if user else None),
                    "platform_username": (user.platform_username if user else None),
                    "force_new_analysis": True,
                    "force_new_email": True,
                },
                idempotency_key=f"wrapped:{run.id}",
                force_new=True,
            )
            run.watch_history_job_id = job.id
            run.status = "pending"
            run.payload = run_payload
            data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
            data_jobs["watch_history"] = {"id": job.id, "status": job.status}
            run_payload["data_jobs"] = data_jobs
            db.add(run)
            db.commit()
            enqueued += 1
            results.append(
                AdminRetryFailedRunsItem(
                    wrapped_run_id=run.id,
                    app_user_id=run.app_user_id,
                    sec_user_id=sec_user_id,
                    action="enqueued",
                    enqueued_task=job.task_name,
                    job_id=job.id,
                    idempotency_key=job.idempotency_key,
                )
            )
            continue

        if include_analysis:
            if not run.id:
                results.append(AdminRetryFailedRunsItem(wrapped_run_id=run.id, action="skip", skipped_reason="missing_run_id"))
                continue
            if dry_run:
                results.append(
                    AdminRetryFailedRunsItem(
                        wrapped_run_id=run.id,
                        app_user_id=run.app_user_id,
                        sec_user_id=sec_user_id,
                        action="dry_run",
                        enqueued_task="wrapped_analysis",
                        idempotency_key=f"analysis:{run.id}",
                    )
                )
                continue
            job = job_queue.enqueue(
                db,
                task_name="wrapped_analysis",
                payload={"wrapped_run_id": run.id, "force_new_email": True},
                idempotency_key=f"analysis:{run.id}",
                force_new=True,
            )
            run.status = "pending"
            run.payload = run_payload
            data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
            data_jobs["wrapped_analysis"] = {"id": job.id, "status": job.status}
            run_payload["data_jobs"] = data_jobs
            db.add(run)
            db.commit()
            enqueued += 1
            results.append(
                AdminRetryFailedRunsItem(
                    wrapped_run_id=run.id,
                    app_user_id=run.app_user_id,
                    sec_user_id=sec_user_id,
                    action="enqueued",
                    enqueued_task=job.task_name,
                    job_id=job.id,
                    idempotency_key=job.idempotency_key,
                )
            )
            continue

        if include_email:
            # Only meaningful if the run is ready (some installs may mark failed due to email failures).
            if run.status != "ready":
                results.append(
                    AdminRetryFailedRunsItem(
                        wrapped_run_id=run.id,
                        app_user_id=run.app_user_id,
                        sec_user_id=sec_user_id,
                        action="skip",
                        skipped_reason="run_not_ready_for_email",
                    )
                )
                continue
            if dry_run:
                results.append(
                    AdminRetryFailedRunsItem(
                        wrapped_run_id=run.id,
                        app_user_id=run.app_user_id,
                        sec_user_id=sec_user_id,
                        action="dry_run",
                        enqueued_task="email_send",
                        idempotency_key=f"email:{run.id}",
                    )
                )
                continue
            job = job_queue.enqueue(
                db,
                task_name="email_send",
                payload={"wrapped_run_id": run.id, "app_user_id": run.app_user_id},
                idempotency_key=f"email:{run.id}",
                force_new=True,
            )
            enqueued += 1
            results.append(
                AdminRetryFailedRunsItem(
                    wrapped_run_id=run.id,
                    app_user_id=run.app_user_id,
                    sec_user_id=sec_user_id,
                    action="enqueued",
                    enqueued_task=job.task_name,
                    job_id=job.id,
                    idempotency_key=job.idempotency_key,
                )
            )
            continue

        results.append(
            AdminRetryFailedRunsItem(
                wrapped_run_id=run.id,
                app_user_id=run.app_user_id,
                sec_user_id=sec_user_id,
                action="skip",
                skipped_reason="no_action_selected",
            )
        )

    return AdminRetryFailedRunsResponse(
        matched=matched,
        processed=processed,
        enqueued=enqueued,
        dry_run=dry_run,
        results=results,
    )


@app.post(
    "/admin/wrapped/runs/retry-zero-videos",
    response_model=AdminRetryZeroVideosResponse,
    responses={401: {"model": ErrorResponse}},
)
async def admin_retry_zero_video_runs(
    payload: AdminRetryZeroVideosRequest,
    _admin=Depends(require_admin),
    db=Depends(get_db),
) -> AdminRetryZeroVideosResponse:
    if payload.limit is None:
        limit = None
    else:
        limit_value = int(payload.limit)
        limit = None if limit_value <= 0 else max(1, limit_value)
    dry_run = bool(payload.dry_run)
    run_statuses = payload.run_status or ["ready"]

    runs_query = db.query(AppWrappedRun).filter(AppWrappedRun.status.in_(run_statuses))
    if not payload.include_admin_test:
        runs_query = runs_query.filter(or_(AppWrappedRun.app_user_id.is_(None), ~AppWrappedRun.app_user_id.like("admin-test-%")))
    ordered_query = runs_query.order_by(AppWrappedRun.updated_at.desc(), AppWrappedRun.created_at.desc())
    runs = ordered_query.limit(limit).all() if limit is not None else ordered_query.all()
    matched = len(runs)
    results: List[AdminRetryZeroVideosItem] = []
    enqueued = 0
    processed = 0

    for run in runs:
        processed += 1
        total_videos = None
        payload_obj = run.payload if isinstance(run.payload, dict) else {}
        watch_history_progress = (
            payload_obj.get("_watch_history_progress")
            if isinstance(payload_obj.get("_watch_history_progress"), dict)
            else None
        )
        total_videos_raw = payload_obj.get("total_videos")
        if isinstance(total_videos_raw, (int, float)):
            total_videos = int(total_videos_raw)
        elif isinstance(total_videos_raw, str) and total_videos_raw.strip().isdigit():
            total_videos = int(total_videos_raw.strip())

        if total_videos != 0:
            results.append(
                AdminRetryZeroVideosItem(
                    wrapped_run_id=run.id,
                    app_user_id=run.app_user_id,
                    sec_user_id=run.sec_user_id,
                    total_videos=total_videos,
                    watch_history_progress=watch_history_progress,
                    action="skip",
                    skipped_reason="not_zero_videos",
                )
            )
            continue

        if not run.app_user_id:
            results.append(
                AdminRetryZeroVideosItem(
                    wrapped_run_id=run.id,
                    app_user_id=None,
                    sec_user_id=run.sec_user_id,
                    total_videos=total_videos,
                    watch_history_progress=watch_history_progress,
                    action="skip",
                    skipped_reason="missing_app_user_id",
                )
            )
            continue

        user = db.get(AppUser, run.app_user_id)
        sec_user_id = user.latest_sec_user_id if user else None
        if not sec_user_id:
            results.append(
                AdminRetryZeroVideosItem(
                    wrapped_run_id=run.id,
                    app_user_id=run.app_user_id,
                    sec_user_id=None,
                    total_videos=total_videos,
                    watch_history_progress=watch_history_progress,
                    action="skip",
                    skipped_reason="sec_user_id_required",
                )
            )
            continue

        watch_job = db.get(AppJob, run.watch_history_job_id) if run.watch_history_job_id else None
        if watch_job and watch_job.status in ("pending", "running"):
            results.append(
                AdminRetryZeroVideosItem(
                    wrapped_run_id=run.id,
                    app_user_id=run.app_user_id,
                    sec_user_id=sec_user_id,
                    total_videos=total_videos,
                    watch_history_progress=watch_history_progress,
                    action="skip",
                    skipped_reason="watch_history_in_progress",
                )
            )
            continue

        if dry_run:
            results.append(
                AdminRetryZeroVideosItem(
                    wrapped_run_id=run.id,
                    app_user_id=run.app_user_id,
                    sec_user_id=sec_user_id,
                    total_videos=total_videos,
                    watch_history_progress=watch_history_progress,
                    action="dry_run",
                    enqueued_task="watch_history_fetch_2025",
                )
            )
            continue

        run_payload = payload_obj
        if payload.reset_payload:
            _full_payload_reset(run_payload)
            data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
            data_jobs.pop("watch_history", None)
            data_jobs.pop("wrapped_analysis", None)
            data_jobs.pop("email_send", None)
            run_payload["data_jobs"] = data_jobs
            run.payload = run_payload

        job = job_queue.enqueue(
            db,
            task_name="watch_history_fetch_2025",
            payload={
                "wrapped_run_id": run.id,
                "app_user_id": run.app_user_id,
                "sec_user_id": sec_user_id,
                "time_zone": (user.time_zone if user else None),
                "platform_username": (user.platform_username if user else None),
                "force_new_analysis": True,
                "force_new_email": True,
            },
            idempotency_key=f"wrapped:{run.id}",
            force_new=True,
        )
        run.watch_history_job_id = job.id
        data_jobs = run_payload.get("data_jobs") if isinstance(run_payload.get("data_jobs"), dict) else {}
        data_jobs["watch_history"] = {"id": job.id, "status": job.status}
        run_payload["data_jobs"] = data_jobs
        run.payload = run_payload
        if run.status != "pending":
            run.status = "pending"
        db.add(run)
        db.commit()

        enqueued += 1
        results.append(
            AdminRetryZeroVideosItem(
            wrapped_run_id=run.id,
            app_user_id=run.app_user_id,
            sec_user_id=sec_user_id,
            total_videos=total_videos,
            watch_history_progress=watch_history_progress,
            action="enqueued",
            enqueued_task=job.task_name,
            job_id=job.id,
            idempotency_key=job.idempotency_key,
        )
        )

    return AdminRetryZeroVideosResponse(
        matched=matched,
        processed=processed,
        enqueued=enqueued,
        dry_run=dry_run,
        results=results,
    )


@app.post(
    "/link/tiktok/start",
    status_code=status.HTTP_201_CREATED,
    response_model=LinkStartResponse,
    responses={401: {"model": ErrorResponse}},
)
async def link_tiktok_start(
    request: Request,
    device=Depends(require_device),
    db=Depends(get_db),
) -> LinkStartResponse:
    try:
        res = await archive_client.start_xordi_auth(anchor_token=None)
    except httpx.HTTPStatusError as exc:
        detail = _format_http_error(exc)
        status_code = exc.response.status_code if exc.response else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(
            status_code=status_code,
            detail={"error": "archive_unavailable", "message": detail},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "archive_unavailable", "message": _format_http_error(exc)},
        )

    device_id = device["device_id"]
    device_email = db.get(DeviceEmail, device_id)
    email_snapshot = device_email.email if device_email else None

    # Only show failed_count for truly new users (no prior successful auth).
    succeeded_q = db.query(AppAuthJob).filter(AppAuthJob.status == "finalized")
    if email_snapshot:
        succeeded_q = succeeded_q.filter(or_(AppAuthJob.device_id == device_id, AppAuthJob.email == email_snapshot))
    else:
        succeeded_q = succeeded_q.filter(AppAuthJob.device_id == device_id)
    has_succeeded_before = succeeded_q.first() is not None

    failed_count = 0
    if not has_succeeded_before:
        failed_q = db.query(AppAuthJob).filter(AppAuthJob.status.in_(("failed", "expired", "error")))
        if email_snapshot:
            failed_q = failed_q.filter(or_(AppAuthJob.device_id == device_id, AppAuthJob.email == email_snapshot))
        else:
            failed_q = failed_q.filter(AppAuthJob.device_id == device_id)
        failed_count = int(failed_q.count())

    client_ip, _, _, _, _ = _resolve_client_ip(request)

    # Record auth job with device binding (app_user_id is unknown pre-finalize)
    job = AppAuthJob(
        archive_job_id=res.get("archive_job_id", ""),
        app_user_id=None,
        provider="xordi",
        device_id=device_id,
        client_ip=client_ip,
        email=email_snapshot,
        platform=device.get("platform"),
        app_version=device.get("app_version"),
        os_version=device.get("os_version"),
        status="pending",
    )
    db.add(job)
    db.commit()

    return LinkStartResponse(
        archive_job_id=res.get("archive_job_id", ""),
        expires_at=res.get("expires_at"),
        queue_position=res.get("queue_position"),
        failed_count=0 if has_succeeded_before else failed_count,
    )


@app.get(
    "/link/tiktok/redirect",
    response_model=RedirectResponse,
    responses={401: {"model": ErrorResponse}},
)
async def link_tiktok_redirect(
    job_id: str,
    time_zone: Optional[str] = None,
    device=Depends(require_device),
    db=Depends(get_db),
) -> RedirectResponse:
    job = db.get(AppAuthJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    if job.device_id and job.device_id != device["device_id"]:
        raise HTTPException(status_code=401, detail="invalid_device")

    # If a session exists, return it (covers normal finalize completion and rare races/crashes
    # between session minting and persisting `AppAuthJob.status`).
    device_id = device["device_id"]
    session: Optional[AppSession] = None
    user: Optional[AppUser] = None

    if job.app_user_id:
        session = (
            db.query(AppSession)
            .filter(
                AppSession.app_user_id == job.app_user_id,
                AppSession.device_id == device_id,
                AppSession.revoked_at.is_(None),
            )
            .order_by(AppSession.issued_at.desc())
            .first()
        )
        if session:
            user = db.get(AppUser, job.app_user_id)

    # If we don't yet know app_user_id, but a new session was minted after this auth job was created,
    # bind the auth job to that session so the frontend can proceed without extra polling state.
    if not session and job.created_at:
        recent = (
            db.query(AppSession)
            .filter(
                AppSession.device_id == device_id,
                AppSession.revoked_at.is_(None),
                AppSession.issued_at >= job.created_at,
            )
            .order_by(AppSession.issued_at.desc())
            .first()
        )
        if recent:
            session = recent
            user = db.get(AppUser, session.app_user_id)
            job.app_user_id = session.app_user_id
            job.status = "finalized"
            if job.finalized_at is None:
                job.finalized_at = datetime.utcnow()
            job.session_id = session.id
            db.add(job)

    if session:
        if user and time_zone:
            user.time_zone = time_zone
            db.add(user)
        if job.status != "finalized":
            job.status = "finalized"
            if job.finalized_at is None:
                job.finalized_at = datetime.utcnow()
            job.session_id = session.id
            db.add(job)
        db.commit()

        token_value = None
        with suppress(Exception):
            token_value = decrypt(session.token_encrypted, settings.secret_key.get_secret_value())
        return RedirectResponse(
            status="completed",
            app_user_id=job.app_user_id or session.app_user_id,
            token=token_value,
            expires_at=session.expires_at,
            platform_username=user.platform_username if user else None,
        )

    if job.status in ("failed", "expired", "error"):
        email_value = job.email
        if not email_value and job.device_id:
            device_email = db.get(DeviceEmail, job.device_id)
            if device_email and device_email.email:
                email_value = device_email.email
        if email_value:
            _enqueue_reauth_notify(db, email=email_value, archive_job_id=job.archive_job_id, reason="auth_failed")
        return RedirectResponse(status="reauth_needed", message=job.last_error)

    try:
        resp = await archive_client.get_redirect(job_id)
    except httpx.HTTPStatusError as exc:
        detail = _format_http_error(exc)
        status_code = exc.response.status_code if exc.response else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=status_code, detail={"error": "archive_unavailable", "message": detail})
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "archive_unavailable", "message": _format_http_error(exc)},
        )
    data = {}
    try:
        data = resp.json()
    except Exception:
        pass

    # Pending / processing
    if resp.status_code == 202 or (resp.status_code == 200 and data.get("status") in {"pending", "processing"}):
        queue_position_raw = data.get("queue_position")
        queue_request_id = data.get("queue_request_id")
        queue_status = data.get("queue_status")
        if AUTO_CANCEL_QUEUE_ON_MAX_POSITION and MAX_QUEUE_POSITION > 0:
            queue_position = None
            with suppress(Exception):
                queue_position = int(queue_position_raw)
            if queue_position is not None and queue_position > MAX_QUEUE_POSITION:
                with suppress(Exception):
                    await archive_client.delete_queue_request(job.archive_job_id)
                job.status = "expired"
                job.last_error = "queue_position_exceeded"
                db.add(job)
                db.commit()
                email_value = job.email
                if not email_value and job.device_id:
                    device_email = db.get(DeviceEmail, job.device_id)
                    if device_email and device_email.email:
                        email_value = device_email.email
                if email_value:
                    _enqueue_reauth_notify(
                        db, email=email_value, archive_job_id=job.archive_job_id, reason="queue_position_exceeded"
                    )
                payload = RedirectResponse(
                    status="expired",
                    message="queue_position_exceeded",
                    queue_status="cancelled",
                    queue_position=queue_position,
                    queue_request_id=queue_request_id,
                ).model_dump(exclude_none=True)
                return JSONResponse(status_code=410, content=payload)
        payload = RedirectResponse(
            status="pending",
            queue_status=queue_status,
            queue_position=queue_position_raw,
            queue_request_id=queue_request_id,
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=202, content=payload)
    # Ready with redirect
    if resp.status_code == 200 and data.get("redirect_url"):
        if job.status not in ("finalizing", "finalized"):
            queued = job_queue.enqueue(
                db,
                task_name="xordi_finalize",
                payload={"archive_job_id": job.archive_job_id, "time_zone": time_zone},
                idempotency_key=f"xordi_finalize:{job.archive_job_id}",
            )
            # If we already deduped an existing job, ensure it gets the latest time_zone.
            if time_zone:
                with suppress(Exception):
                    payload_existing = queued.payload if isinstance(queued.payload, dict) else {}
                    if not payload_existing.get("time_zone"):
                        payload_existing["time_zone"] = time_zone
                        queued.payload = payload_existing
                        db.add(queued)
                        db.commit()
            job.status = "finalizing"
            db.add(job)
            db.commit()
        payload = RedirectResponse(
            status="ready",
            redirect_url=data.get("redirect_url"),
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=resp.status_code, content=payload)
    # Completed / other 200 with status
    if resp.status_code == 200:
        status_value = data.get("status") or "completed"
        if isinstance(status_value, str) and status_value.lower().startswith("authentication already"):
            status_value = "finalizing"

            # Best-effort: if the worker already finalized (e.g. we saw redirect_url earlier),
            # return the session now. Do not enqueue finalize here.
            if not job.created_at:
                # Should never happen (DB default), but avoid breaking the request path.
                job_created_at = datetime.utcnow()
            else:
                job_created_at = job.created_at
            for delay in (0.2, 0.5, 0.8):
                await asyncio.sleep(delay)
                session = (
                    db.query(AppSession)
                    .filter(
                        AppSession.device_id == device_id,
                        AppSession.revoked_at.is_(None),
                        AppSession.issued_at >= job_created_at,
                    )
                    .order_by(AppSession.issued_at.desc())
                    .first()
                )
                if not session:
                    continue
                user = db.get(AppUser, session.app_user_id)
                if user and time_zone:
                    user.time_zone = time_zone
                    db.add(user)
                job.app_user_id = session.app_user_id
                job.status = "finalized"
                if job.finalized_at is None:
                    job.finalized_at = datetime.utcnow()
                job.session_id = session.id
                db.add(job)
                db.commit()
                token_value = None
                with suppress(Exception):
                    token_value = decrypt(session.token_encrypted, settings.secret_key.get_secret_value())
                payload = RedirectResponse(
                    status="completed",
                    app_user_id=session.app_user_id,
                    token=token_value,
                    expires_at=session.expires_at,
                    platform_username=user.platform_username if user else None,
                ).model_dump(exclude_none=True)
                return JSONResponse(status_code=200, content=jsonable_encoder(payload))
        payload = RedirectResponse(
            status="finalizing" if job.status == "finalizing" else status_value,
            message=data.get("message") or (data.get("status") if isinstance(data.get("status"), str) else None),
            queue_status=data.get("queue_status"),
            queue_position=data.get("queue_position"),
            queue_request_id=data.get("queue_request_id"),
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=resp.status_code, content=payload)
    # Expired
    if resp.status_code == 410:
        job.status = "expired"
        db.add(job)
        db.commit()
        email_value = job.email
        if not email_value and job.device_id:
            device_email = db.get(DeviceEmail, job.device_id)
            if device_email and device_email.email:
                email_value = device_email.email
        if email_value:
            _enqueue_reauth_notify(db, email=email_value, archive_job_id=job.archive_job_id, reason="expired")
        payload = RedirectResponse(
            status="expired",
            message=data.get("message"),
            queue_status=data.get("queue_status"),
            queue_position=data.get("queue_position"),
            queue_request_id=data.get("queue_request_id"),
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=resp.status_code, content=payload)
    raise HTTPException(status_code=resp.status_code, detail=data or resp.text)


@app.post("/link/tiktok/redirect/click", status_code=204, responses={401: {"model": ErrorResponse}})
async def link_tiktok_redirect_click(
    payload: RedirectClickRequest,
    device=Depends(require_device),
    db=Depends(get_db),
) -> Response:
    job = db.get(AppAuthJob, payload.archive_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    if job.device_id and job.device_id != device["device_id"]:
        raise HTTPException(status_code=401, detail="invalid_device")

    now = datetime.utcnow()
    if job.redirect_clicked_at is None:
        job.redirect_clicked_at = now
    job.redirect_clicks = int(job.redirect_clicks or 0) + 1
    db.add(job)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/link/tiktok/queue-status", responses={401: {"model": ErrorResponse}})
async def link_tiktok_queue_status(_device=Depends(require_device)) -> Dict[str, Any]:
    try:
        res = await archive_client.get_queue_status()
    except httpx.HTTPStatusError as exc:
        detail = _format_http_error(exc)
        status_code = exc.response.status_code if exc.response else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=status_code, detail={"error": "archive_unavailable", "message": detail})
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "archive_unavailable", "message": _format_http_error(exc)},
        )
    pending = res.get("pending")
    return {"pending": pending if isinstance(pending, int) else 0}


@app.delete("/link/tiktok/cancel-queue", responses={401: {"model": ErrorResponse}})
async def link_tiktok_cancel_queue(
    archive_job_id: str,
    device=Depends(require_device),
    db=Depends(get_db),
) -> Response:
    job = db.get(AppAuthJob, archive_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    if job.device_id and job.device_id != device["device_id"]:
        raise HTTPException(status_code=401, detail="invalid_device")

    resp = await archive_client.delete_queue_request(archive_job_id)
    content_type = resp.headers.get("content-type")
    try:
        payload = resp.json()
    except Exception:
        payload = resp.text

    if isinstance(payload, (dict, list)):
        return JSONResponse(status_code=resp.status_code, content=payload)
    return Response(status_code=resp.status_code, content=str(payload), media_type=content_type)


@app.get(
    "/link/tiktok/code",
    response_model=CodeResponse,
    responses={401: {"model": ErrorResponse}},
)
async def link_tiktok_code(job_id: str, device=Depends(require_device), db=Depends(get_db)) -> CodeResponse:
    job = db.get(AppAuthJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    if job.device_id and job.device_id != device["device_id"]:
        raise HTTPException(status_code=401, detail="invalid_device")
    resp = await archive_client.get_authorization_code(job_id)
    data = {}
    try:
        data = resp.json()
    except Exception:
        pass
    # Pending / processing
    if resp.status_code == 202 or (resp.status_code == 200 and data.get("status") in {"pending", "processing"}):
        payload = CodeResponse(
            status="pending",
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=202, content=payload)
    # Ready with code
    if resp.status_code == 200 and data.get("authorization_code"):
        payload = CodeResponse(
            status="ready",
            authorization_code=data.get("authorization_code"),
            expires_at=data.get("expires_at"),
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=resp.status_code, content=payload)
    # Completed/other 200 without code
    if resp.status_code == 200:
        payload = CodeResponse(
            status=data.get("status") or "completed",
            message=data.get("message"),
            expires_at=data.get("expires_at"),
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=resp.status_code, content=payload)
    # Expired
    if resp.status_code == 410:
        payload = CodeResponse(
            status="expired",
            queue_position=data.get("queue_position"),
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=resp.status_code, content=payload)
    raise HTTPException(status_code=resp.status_code, detail=data or resp.text)


@app.post(
    "/link/tiktok/finalize",
    response_model=FinalizeResponse,
    responses={401: {"model": ErrorResponse}},
)
async def link_tiktok_finalize(
    payload: FinalizeRequest,
    request: Request,
    device=Depends(require_device),
    db=Depends(get_db),
) -> FinalizeResponse:
    bind_context(archive_job_id=payload.archive_job_id)
    job = db.get(AppAuthJob, payload.archive_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    if job.device_id and job.device_id != device["device_id"]:
        raise HTTPException(status_code=401, detail="invalid_device")
    client_ip, _, _, _, _ = _resolve_client_ip(request)
    if client_ip and not job.client_ip:
        job.client_ip = client_ip
        db.add(job)
    anchor_token = None
    if job.app_user_id:
        existing_user = db.get(AppUser, job.app_user_id)
        anchor_token = existing_user.latest_anchor_token if existing_user else None
    try:
        data = await archive_client.finalize_xordi(
            archive_job_id=payload.archive_job_id,
            authorization_code=payload.authorization_code,
            anchor_token=anchor_token,
        )
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        body_text = None
        body_truncated = None
        with suppress(Exception):
            body_text, body_truncated = sanitize_json_bytes(exc.response.content if exc.response is not None else b"", max_chars=2000)

        # Record the error on the auth job for admin debugging.
        job = db.get(AppAuthJob, payload.archive_job_id)
        if job:
            job.last_error = f"{status_code} {body_text or ''}".strip()
            # Treat 4xx (except rate-limit) as terminal; 5xx are transient.
            if 400 <= status_code < 500 and status_code != 429:
                job.status = "error"
            db.add(job)
            db.commit()

        if status_code == 429 or status_code >= 500:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "archive_unavailable",
                    "message": f"Archive finalize failed ({status_code}){': ' + body_text if body_text else ''}",
                },
            )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "archive_finalize_failed",
                "message": f"Archive finalize rejected ({status_code}){': ' + body_text if body_text else ''}",
            },
        )
    except httpx.RequestError as exc:
        job = db.get(AppAuthJob, payload.archive_job_id)
        if job:
            job.last_error = str(exc)
            db.add(job)
            db.commit()
        raise HTTPException(
            status_code=502,
            detail={"error": "archive_unavailable", "message": "Archive finalize request failed"},
        )
    bind_context(
        archive_user_id=data.get("archive_user_id"),
        sec_user_id=data.get("provider_unique_id"),
    )
    # Bind to canonical app_user_id derived from archive_user_id
    final_app_user_id = data.get("archive_user_id") or (job.app_user_id or str(uuid4()))
    bind_context(app_user_id=final_app_user_id)
    canonical_user = db.get(AppUser, final_app_user_id)
    if not canonical_user:
        canonical_user = AppUser(app_user_id=final_app_user_id)
    previous_sec_user_id = canonical_user.latest_sec_user_id
    canonical_user.archive_user_id = data.get("archive_user_id")
    canonical_user.latest_sec_user_id = data.get("provider_unique_id")
    platform_username = data.get("platform_username")
    if platform_username:
        canonical_user.platform_username = platform_username
    canonical_user.time_zone = payload.time_zone or canonical_user.time_zone
    new_anchor = data.get("anchor_token")
    if new_anchor or anchor_token:
        canonical_user.latest_anchor_token = new_anchor or anchor_token
    if previous_sec_user_id != canonical_user.latest_sec_user_id:
        canonical_user.is_watch_history_available = "unknown"
    # Apply referral if provided (or captured on device) and not already set.
    ref = None
    referrer_identity = None
    if payload.referral_code:
        ref = db.get(Referral, payload.referral_code)
        if ref:
            referrer_identity = ref.referrer_app_user_id
    if not referrer_identity:
        device_email = db.get(DeviceEmail, device["device_id"])
        if device_email and device_email.referred_by:
            referrer_identity = device_email.referred_by
            ref = (
                db.query(Referral)
                .filter(Referral.referrer_app_user_id == referrer_identity)
                .order_by(Referral.created_at.desc())
                .first()
            )
    if referrer_identity and not canonical_user.referred_by:
        canonical_user.referred_by = referrer_identity
        if ref:
            ref.conversions = (ref.conversions or 0) + 1
            evt = ReferralEvent(
                id=str(uuid4()),
                referrer_app_user_id=ref.referrer_app_user_id,
                referred_app_user_id=canonical_user.app_user_id,
                event_type="conversion",
                archive_job_id=payload.archive_job_id,
                wrapped_run_id=None,
                ip=None,
                user_agent=None,
            )
            db.add(ref)
            db.add(evt)
    db.add(canonical_user)
    # Rebind auth job to canonical user if it exists
    job = db.get(AppAuthJob, payload.archive_job_id)
    if job:
        job.app_user_id = canonical_user.app_user_id
        job.status = "finalized"
        db.add(job)
    db.commit()
    token, expires_at = session_service.create_or_rotate(
        db=db,
        app_user_id=canonical_user.app_user_id,
        device_id=device["device_id"],
        platform=device["platform"],
        app_version=device["app_version"],
        os_version=device["os_version"],
    )
    # Auto-run availability check and enqueue wrapped pipeline on success
    await ensure_watch_history_available(canonical_user, db, auto_enqueue=True, resend_email_on_ready=True)
    return FinalizeResponse(
        archive_user_id=data.get("archive_user_id", ""),
        sec_user_id=data.get("provider_unique_id", ""),
        anchor_token=canonical_user.latest_anchor_token,
        app_user_id=canonical_user.app_user_id,
        token=token,
        expires_at=expires_at,
        platform_username=canonical_user.platform_username,
    )


@app.post(
    "/link/tiktok/verify-region",
    response_model=VerifyRegionResponse,
    responses={401: {"model": ErrorResponse}},
)
async def link_tiktok_verify_region(session=Depends(require_session), db=Depends(get_db)) -> VerifyRegionResponse:
    user = db.get(AppUser, session.app_user_id)
    if not user:
        raise HTTPException(status_code=400, detail="user_not_found")
    status_value, attempts, last_error = await ensure_watch_history_available(user, db, auto_enqueue=True)
    return VerifyRegionResponse(is_watch_history_available=status_value, attempts=attempts, last_error=last_error)


@app.post("/register-email", status_code=204, responses={401: {"model": ErrorResponse}})
async def register_email(
    payload: RegisterEmailRequest,
    request: Request,
    device=Depends(require_device),
    db=Depends(get_db),
) -> Response:
    device_id = device["device_id"]
    now = datetime.utcnow()

    # Persist email tied to device so the background finalize job can attach it to the user.
    device_email = db.get(DeviceEmail, device_id)
    ref = None
    referrer_identity = None
    if payload.referral_code:
        ref = db.get(Referral, payload.referral_code)
        if ref:
            referrer_identity = ref.referrer_app_user_id
    if device_email:
        device_email.email = payload.email
        device_email.updated_at = now
        new_referral = False
        if referrer_identity and not device_email.referred_by:
            device_email.referred_by = referrer_identity
            new_referral = True
        db.add(device_email)
    else:
        device_email = DeviceEmail(
            device_id=device_id,
            email=payload.email,
            referred_by=referrer_identity,
            created_at=now,
            updated_at=now,
        )
        db.add(device_email)
        new_referral = bool(referrer_identity)
    if new_referral and ref:
        evt = ReferralEvent(
            id=str(uuid4()),
            referrer_app_user_id=ref.referrer_app_user_id,
            referred_app_user_id=None,
            event_type="email_capture",
            archive_job_id=None,
            wrapped_run_id=None,
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )
        db.add(evt)
    db.commit()

    # Best-effort: if this device already has a finalized auth, attach email to that user.
    auth = (
        db.query(AppAuthJob)
        .filter(AppAuthJob.device_id == device_id, AppAuthJob.status == "finalized", AppAuthJob.app_user_id.isnot(None))
        .order_by(AppAuthJob.finalized_at.desc().nullslast(), AppAuthJob.created_at.desc())
        .first()
    )
    if not auth or not auth.app_user_id:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    user = db.get(AppUser, auth.app_user_id)
    if not user:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    existing = (
        db.query(AppUserEmail)
        .filter(AppUserEmail.app_user_id == user.app_user_id)
        .order_by(AppUserEmail.created_at.desc())
        .first()
    )
    if existing and existing.email == payload.email:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    if existing:
        existing.email = payload.email
        existing.created_at = now
        existing.verified_at = None
        db.add(existing)
    else:
        db.add(AppUserEmail(id=str(uuid4()), app_user_id=user.app_user_id, email=payload.email))
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Referrals -------------------------------------------------------------

@app.post("/referral", response_model=ReferralResponse)
async def create_or_get_referral(payload: ReferralRequest, db=Depends(get_db)) -> ReferralResponse:
    identity: Optional[str] = None
    if payload.app_user_id:
        identity = payload.app_user_id
    elif payload.email:
        identity = f"email:{str(payload.email).strip().lower()}"
    if not identity:
        raise HTTPException(status_code=400, detail="app_user_id_or_email_required")

    bind_context(app_user_id=payload.app_user_id, email=str(payload.email) if payload.email else None)

    # If the referrer is a real user ID, ensure the user exists.
    if payload.app_user_id:
        user = db.get(AppUser, payload.app_user_id)
        if not user:
            user = AppUser(app_user_id=payload.app_user_id)
            db.add(user)
            db.commit()
    existing = (
        db.query(Referral)
        .filter(Referral.referrer_app_user_id == identity)
        .order_by(Referral.created_at.desc())
        .first()
    )
    if existing:
        return ReferralResponse(code=existing.code, referral_url=_referral_url(existing.code))

    code = _generate_referral_code(identity, db)
    ref = Referral(
        code=code,
        referrer_app_user_id=identity,
        impressions=0,
        conversions=0,
        completions=0,
    )
    db.add(ref)
    db.commit()
    return ReferralResponse(code=code, referral_url=_referral_url(code))


@app.post("/referral/impression", status_code=204)
async def record_referral_impression(payload: ReferralImpressionRequest, request: Request, db=Depends(get_db)) -> Response:
    ref = db.get(Referral, payload.code)
    if not ref:
        raise HTTPException(status_code=404, detail="referral_not_found")
    ref.impressions = (ref.impressions or 0) + 1
    evt = ReferralEvent(
        id=str(uuid4()),
        referrer_app_user_id=ref.referrer_app_user_id,
        referred_app_user_id=None,
        event_type="impression",
        archive_job_id=None,
        wrapped_run_id=None,
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("User-Agent"),
    )
    db.add(ref)
    db.add(evt)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Wrapped orchestration -------------------------------------------------

@app.get(
    "/wrapped/{app_user_id}",
    response_model=WrappedStatusResponse,
)
async def wrapped_status(
    app_user_id: str, db=Depends(get_db)
) -> WrappedStatusResponse:
    run = (
        db.query(AppWrappedRun)
        .filter(AppWrappedRun.app_user_id == app_user_id)
        .order_by(AppWrappedRun.created_at.desc())
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="not_found")
    progress = None
    if isinstance(run.payload, dict):
        raw = run.payload.get("_watch_history_progress")
        if isinstance(raw, dict):
            progress = raw
    if not isinstance(run.payload, dict) or not run.payload:
        return WrappedStatusResponse(
            status="pending",
            wrapped_run_id=run.id,
            wrapped=None,
            watch_history_progress=progress,
            queue_position=None,
            queue_eta_seconds=None,
            queue_status="pending",
        )

    payload = _ensure_wrapped_defaults(run.payload or {}, app_user_id)
    if payload != run.payload:
        run.payload = payload
        db.add(run)
        db.commit()
    try:
        wrapped = WrappedPayload.model_validate(payload)
    except ValidationError as exc:
        logger.exception(
            "wrapped.payload_validation_error",
            extra={"event": "wrapped.payload_validation_error", "wrapped_run_id": run.id, "app_user_id": app_user_id},
        )
        with suppress(Exception):
            if isinstance(payload, dict):
                payload["_wrapped_payload_validation_error"] = str(exc)[:1000]
            run.payload = payload
            if run.status == "ready":
                run.status = "pending"
            db.add(run)
            db.commit()
        progress = None
        if isinstance(payload, dict):
            raw = payload.get("_watch_history_progress")
            if isinstance(raw, dict):
                progress = raw
        return WrappedStatusResponse(
            status="pending",
            wrapped_run_id=run.id,
            wrapped=None,
            watch_history_progress=progress,
            queue_position=None,
            queue_eta_seconds=None,
            queue_status="pending",
        )
    if run.status != "ready":
        with suppress(Exception):
            run.status = "ready"
            db.add(run)
            db.commit()
    return WrappedStatusResponse(status="ready", wrapped_run_id=run.id, wrapped=wrapped)


@app.post("/waitlist", status_code=204, responses={401: {"model": ErrorResponse}})
async def join_waitlist(payload: WaitlistRequest, device=Depends(require_device), db=Depends(get_db)) -> Response:
    device_id = device["device_id"]
    now = datetime.utcnow()

    rec = db.get(DeviceEmail, device_id)
    if rec:
        rec.email = payload.email
        rec.updated_at = now
    else:
        rec = DeviceEmail(device_id=device_id, email=payload.email, created_at=now, updated_at=now)

    rec.waitlist_opt_in = True
    if rec.waitlist_opt_in_at is None:
        rec.waitlist_opt_in_at = now
    db.add(rec)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Health ---------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/readyz")
async def readyz() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)

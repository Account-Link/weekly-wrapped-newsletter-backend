import asyncio
import math
import json
import os
import re
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from sqlalchemy import func

from app import accessories
from app.db import SessionLocal
from app.emailer import Emailer
from app.models import AppAuthJob, AppJob, AppSession, AppUser, AppUserEmail, AppWrappedRun, DeviceEmail, Referral, ReferralEvent, WeeklyReport
from app.observability import bind_context, clear_context, get_logger, sanitize_json_text, set_service, setup_logging
from app.prompts import (
    ACCESSORY_SET_PROMPT,
    ACCESSORY_REASONS_PROMPT,
    BRAINROT_EXPLANATION_PROMPT,
    BRAINROT_SCORE_PROMPT,
    KEYWORD_2026_PROMPT,
    NICHE_JOURNEY_PROMPT,
    PERSONALITY_EXPLANATION_PROMPT,
    PERSONALITY_PROMPT,
    ROAST_THUMB_PROMPT,
    TOP_NICHES_PROMPT,
    WEEKLY_FEEDING_STATE_PROMPT,
    WEEKLY_TOPICS_PROMPT,
    WEEKLY_RABBIT_HOLE_PROMPT,
    WEEKLY_NUDGE_PROMPT,
    accessory_fallback_reason,
)
from app.services.archive_client import ArchiveClient
from app.services.job_queue import DBJobQueue
from app.services.session_service import SessionService
from app.settings import get_settings

job_queue = DBJobQueue()
settings = get_settings()
setup_logging(level=settings.log_level)
logger = get_logger(__name__)
archive_client = ArchiveClient(settings)
session_service = SessionService(ttl_days=settings.session_ttl_days, secret_key=settings.secret_key.get_secret_value())
_emailer: Optional[Emailer] = None

WATCH_HISTORY_PAGE_LIMIT = max(1, int(os.getenv("WATCH_HISTORY_PAGE_LIMIT", "200")))
WATCH_HISTORY_MAX_PAGES = max(1, int(os.getenv("WATCH_HISTORY_MAX_PAGES", "3")))
WATCH_HISTORY_SINCE_DATE = os.getenv("WATCH_HISTORY_SINCE_DATE", "2025-01-01")
WATCH_HISTORY_SINCE_MS = os.getenv("WATCH_HISTORY_SINCE_MS")
WATCH_HISTORY_FINALIZE_PROGRESS_PERSIST_SECONDS = max(
    1.0, float(os.getenv("WATCH_HISTORY_FINALIZE_PROGRESS_PERSIST_SECONDS", "15"))
)
WATCH_HISTORY_FINALIZE_MAX_SECONDS = max(30.0, float(os.getenv("WATCH_HISTORY_FINALIZE_MAX_SECONDS", "900")))
WATCH_HISTORY_FINALIZE_PROVIDER_FAILED_MAX_SECONDS = max(
    5.0, float(os.getenv("WATCH_HISTORY_FINALIZE_PROVIDER_FAILED_MAX_SECONDS", "60"))
)
WATCH_HISTORY_VERIFY_FINALIZE_MAX_ATTEMPTS = max(
    1, int(os.getenv("WATCH_HISTORY_VERIFY_FINALIZE_MAX_ATTEMPTS", "60"))
)
LLM_CONCURRENCY = max(1, int(os.getenv("LLM_CONCURRENCY", "2")))

llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

WORKER_JOB_CONCURRENCY = max(1, int(os.getenv("WORKER_JOB_CONCURRENCY", "1")))
WORKER_JOB_LEASE_SECONDS = max(30, int(os.getenv("WORKER_JOB_LEASE_SECONDS", "60")))
_DEFAULT_HEARTBEAT_SECONDS = max(5.0, WORKER_JOB_LEASE_SECONDS / 2)
WORKER_JOB_HEARTBEAT_SECONDS = max(1.0, float(os.getenv("WORKER_JOB_HEARTBEAT_SECONDS", str(_DEFAULT_HEARTBEAT_SECONDS))))
WORKER_POLL_INTERVAL = max(0.1, float(os.getenv("WORKER_POLL_INTERVAL", "1.0")))
WORKER_TASK_ALLOW = [t.strip() for t in os.getenv("WORKER_TASK_ALLOW", "").split(",") if t.strip()]

XORDI_AUTH_CODE_POLL_MAX_SECONDS = max(30, int(os.getenv("XORDI_AUTH_CODE_POLL_MAX_SECONDS", "300")))

BRAINROT_ENRICH_POLL_MAX_SECONDS = max(0, int(os.getenv("BRAINROT_ENRICH_POLL_MAX_SECONDS", "300")))
BRAINROT_ENRICH_MAX_TOTAL_MINUTES = max(0, int(os.getenv("BRAINROT_ENRICH_MAX_TOTAL_MINUTES", "10")))
BRAINROT_ENRICH_MAX_ATTEMPTS = max(0, int(os.getenv("BRAINROT_ENRICH_MAX_ATTEMPTS", "10")))
BRAINROT_MIN_CONFIDENCE = float(os.getenv("BRAINROT_MIN_CONFIDENCE", "0.1"))
BRAINROT_MIN_ENRICHED_HOURS = float(os.getenv("BRAINROT_MIN_ENRICHED_HOURS", "0.5"))
BRAINROT_S_CURVE_BASELINE_RAW = float(os.getenv("BRAINROT_S_CURVE_BASELINE_RAW", "0.0019632306430910507"))
BRAINROT_S_CURVE_BASELINE_SCORE = float(os.getenv("BRAINROT_S_CURVE_BASELINE_SCORE", "60"))
BRAINROT_S_CURVE_SLOPE = float(os.getenv("BRAINROT_S_CURVE_SLOPE", "1.5"))
BRAINROT_S_CURVE_EPS = float(os.getenv("BRAINROT_S_CURVE_EPS", "1e-9"))
BRAINROT_ZERO_RAW_FLOOR_DIVISOR = float(os.getenv("BRAINROT_ZERO_RAW_FLOOR_DIVISOR", "20"))

# Disable automatic wrapped (yearly report) trigger on user authorization
DISABLE_AUTO_WRAPPED = os.getenv("DISABLE_AUTO_WRAPPED", "").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class LeasedJob:
    id: str
    task_name: str
    payload: Dict[str, Any]

CAT_NAMES = [
    "cat_white",
    "cat_brown",
    "cat_leopard",
    "cat_orange",
    "cat_tabby",
    "cat_calico",
]


def _get_emailer() -> Emailer:
    global _emailer
    if _emailer is None:
        _emailer = Emailer()
    return _emailer


def _safe_zone(tz_name: Optional[str]) -> ZoneInfo:
    if not tz_name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    with suppress(Exception):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _enqueue_reauth_notify(
    db,
    *,
    email: str,
    archive_job_id: str,
    reason: str,
    failed_at: Optional[datetime] = None,
) -> Optional[AppJob]:
    if os.getenv("REAUTH_NOTIFY_ENABLED", "true").lower() not in ("1", "true", "yes", "on"):
        return None
    email_value = (email or "").strip()
    if not email_value:
        return None
    now = failed_at or datetime.utcnow()
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


async def handle_reauth_notify(job: LeasedJob) -> bool:
    if os.getenv("REAUTH_NOTIFY_ENABLED", "true").lower() not in ("1", "true", "yes", "on"):
        return True
    email = job.payload.get("email")
    failed_at_raw = job.payload.get("failed_at")
    if not isinstance(email, str) or not email.strip():
        return True
    failed_at = _parse_iso_datetime(failed_at_raw if isinstance(failed_at_raw, str) else None)

    platform_username: Optional[str] = None
    wrapped_link: Optional[str] = None
    with SessionLocal() as db:
        # If the user successfully authenticated since the failure time, don't send.
        success_q = db.query(AppAuthJob).filter(AppAuthJob.status == "finalized", AppAuthJob.email == email)
        if failed_at is not None:
            success_q = success_q.filter(AppAuthJob.finalized_at.isnot(None), AppAuthJob.finalized_at >= failed_at)
        if success_q.first() is not None:
            return True

        # If the user already has a ready wrapped, prefer linking directly to it.
        app_user_ids = [
            row[0]
            for row in db.query(AppUserEmail.app_user_id).filter(AppUserEmail.email == email).all()
            if row and row[0]
        ]
        ready_run = (
            db.query(AppWrappedRun)
            .filter(
                AppWrappedRun.status == "ready",
                (AppWrappedRun.email == email) | (AppWrappedRun.app_user_id.in_(app_user_ids) if app_user_ids else False),
            )
            .order_by(AppWrappedRun.updated_at.desc().nullslast(), AppWrappedRun.created_at.desc())
            .first()
        )
        if ready_run and ready_run.app_user_id:
            payload = ready_run.payload if isinstance(ready_run.payload, dict) else {}
            data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
            email_job = data_jobs.get("email_send") if isinstance(data_jobs.get("email_send"), dict) else {}
            if email_job.get("status") == "succeeded":
                return True

            frontend = os.getenv("FRONTEND_URL", "").rstrip("/") or None
            if frontend:
                wrapped_link = f"{frontend}/wrapped?app_user_id={quote(ready_run.app_user_id, safe='')}"
            else:
                wrapped_link = f"/wrapped?app_user_id={quote(ready_run.app_user_id, safe='')}"

        latest_auth = (
            db.query(AppAuthJob)
            .filter(AppAuthJob.status == "finalized", AppAuthJob.email == email, AppAuthJob.app_user_id.isnot(None))
            .order_by(AppAuthJob.finalized_at.desc().nullslast(), AppAuthJob.created_at.desc())
            .first()
        )
        if latest_auth and latest_auth.app_user_id:
            user = db.get(AppUser, latest_auth.app_user_id)
            if user and user.platform_username:
                platform_username = user.platform_username

    fallback_frontend = os.getenv("FRONTEND_URL", "").rstrip("/") or None
    subject, text_body, html_body = _get_emailer().format_reauth_email(
        platform_username,
        wrapped_link or fallback_frontend,
    )
    resp = _get_emailer().send_email(email, subject, text_body, html_body)
    return resp is not None


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

    queued = job_queue.enqueue(
        db,
        task_name="email_send",
        payload={"wrapped_run_id": run.id, "app_user_id": app_user_id},
        idempotency_key=f"email:{run.id}",
        force_new=bool(force_new),
    )
    payload = run.payload if isinstance(run.payload, dict) else {}
    data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
    data_jobs["email_send"] = {"id": queued.id, "status": queued.status}
    payload["data_jobs"] = data_jobs
    payload.setdefault("email", email_value)
    run.payload = payload
    if not run.email:
        run.email = email_value
    db.add(run)
    db.commit()
    return queued


def _queue_wrapped_run_for_user(
    db,
    user: AppUser,
    *,
    resend_email_on_ready: bool = False,
    scrape_max_videos: Optional[int] = None,
    bypass_disable: bool = False,
) -> Optional[AppWrappedRun]:
    """Queue a wrapped (yearly report) run for a user.
    
    Args:
        bypass_disable: If True, ignore DISABLE_AUTO_WRAPPED setting (for admin manual triggers)
    """
    if DISABLE_AUTO_WRAPPED and not bypass_disable:
        return None
    if not user.app_user_id or not user.latest_sec_user_id:
        return None
    existing = (
        db.query(AppWrappedRun)
        .filter(AppWrappedRun.app_user_id == user.app_user_id, AppWrappedRun.sec_user_id == user.latest_sec_user_id)
        .order_by(AppWrappedRun.created_at.desc())
        .first()
    )
    if existing:
        if existing.status == "ready":
            if resend_email_on_ready:
                _enqueue_email_send_for_ready_run(db, existing, force_new=True)
            return existing

        watch_job = db.get(AppJob, existing.watch_history_job_id) if existing.watch_history_job_id else None
        if watch_job and watch_job.status in ("pending", "running"):
            return existing

        payload = existing.payload if isinstance(existing.payload, dict) else {}
        queued = job_queue.enqueue(
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
            force_new=True,
        )
        existing.watch_history_job_id = queued.id
        data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
        data_jobs["watch_history"] = {"id": queued.id, "status": queued.status}
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

    queued = job_queue.enqueue(
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
    run.watch_history_job_id = queued.id
    db.add(run)
    db.commit()
    return run


async def _ensure_watch_history_available(
    user: AppUser,
    db,
    *,
    max_attempts: int = 3,
    auto_enqueue: bool = False,
    resend_email_on_ready: bool = False,
) -> tuple[str, int, Optional[str]]:
    if not user.latest_sec_user_id:
        return "no", 0, "sec_user_id_required"
    if user.is_watch_history_available == "yes":
        if auto_enqueue:
            _queue_wrapped_run_for_user(db, user, resend_email_on_ready=resend_email_on_ready)
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
                fin = await archive_client.finalize_watch_history(data_job_id=data_job_id, include_rows=False)
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
            last_error = str(exc)

    if user.is_watch_history_available != "yes":
        if last_error in {"pending_timeout"}:
            user.is_watch_history_available = "unknown"
        else:
            user.is_watch_history_available = "no"
    db.add(user)
    db.commit()
    if user.is_watch_history_available in {"yes", "unknown"} and auto_enqueue:
        _queue_wrapped_run_for_user(db, user, resend_email_on_ready=resend_email_on_ready)
    return user.is_watch_history_available, attempts, last_error


def _record_watch_history_verify_result(
    db,
    job_id: str,
    *,
    app_user_id: str,
    sec_user_id: Optional[str],
    status: str,
    attempts: int,
    last_error: Optional[str],
) -> None:
    rec = db.get(AppJob, job_id)
    if not rec:
        return
    payload = rec.payload if isinstance(rec.payload, dict) else {}
    payload["result"] = {
        "app_user_id": app_user_id,
        "sec_user_id": sec_user_id,
        "status": status,
        "attempts": attempts,
        "last_error": last_error,
        "checked_at": _iso_utc(datetime.now(timezone.utc)),
    }
    rec.payload = payload
    db.add(rec)
    db.commit()


async def handle_watch_history_verify(job: LeasedJob) -> bool:
    app_user_id = job.payload.get("app_user_id")
    if not isinstance(app_user_id, str) or not app_user_id:
        logger.warning(
            "watch_history_verify.missing_app_user_id",
            extra={"event": "watch_history_verify.missing_app_user_id", "job_id": job.id},
        )
        return True

    auto_enqueue = bool(job.payload.get("auto_enqueue", False))
    with SessionLocal() as db:
        user = db.get(AppUser, app_user_id)
        if not user:
            _record_watch_history_verify_result(
                db,
                job.id,
                app_user_id=app_user_id,
                sec_user_id=None,
                status="error",
                attempts=0,
                last_error="user_not_found",
            )
            return True
        if not user.latest_sec_user_id:
            _record_watch_history_verify_result(
                db,
                job.id,
                app_user_id=app_user_id,
                sec_user_id=None,
                status="error",
                attempts=0,
                last_error="sec_user_id_required",
            )
            return True

        status_value, attempts, last_error = await _ensure_watch_history_available(
            user,
            db,
            auto_enqueue=auto_enqueue,
        )
        _record_watch_history_verify_result(
            db,
            job.id,
            app_user_id=app_user_id,
            sec_user_id=user.latest_sec_user_id,
            status=status_value,
            attempts=attempts,
            last_error=last_error,
        )
        return True


async def handle_xordi_finalize(job: LeasedJob) -> bool:
    """Poll Xordi auth code, finalize, mint session, then enqueue region/wrapped pipeline."""
    archive_job_id = job.payload.get("archive_job_id")
    if not isinstance(archive_job_id, str) or not archive_job_id:
        logger.warning("xordi_finalize.missing_archive_job_id", extra={"event": "xordi_finalize.missing_archive_job_id"})
        return True

    with SessionLocal() as db:
        auth = db.get(AppAuthJob, archive_job_id)
        if not auth:
            return True
        bind_context(archive_job_id=archive_job_id, device_id=auth.device_id)
        if auth.status == "finalized":
            return True
        auth.status = "finalizing"
        db.add(auth)
        db.commit()

    deadline = time.monotonic() + float(XORDI_AUTH_CODE_POLL_MAX_SECONDS)
    backoff = 0.5
    authorization_code: Optional[str] = None

    while time.monotonic() < deadline:
        resp = await archive_client.get_authorization_code(archive_job_id)
        data: Dict[str, Any] = {}
        with suppress(Exception):
            data = resp.json()

        if resp.status_code == 202:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)
            continue
        if resp.status_code == 200 and isinstance(data.get("authorization_code"), str) and data.get("authorization_code"):
            authorization_code = data.get("authorization_code")
            break
        if resp.status_code == 410:
            with SessionLocal() as db:
                auth = db.get(AppAuthJob, archive_job_id)
                if auth:
                    auth.status = "expired"
                    auth.last_error = "authorization_code_expired"
                    notify_email = auth.email
                    if not notify_email and auth.device_id:
                        device_email = db.get(DeviceEmail, auth.device_id)
                        notify_email = device_email.email if device_email else None
                    if notify_email:
                        _enqueue_reauth_notify(
                            db,
                            email=notify_email,
                            archive_job_id=archive_job_id,
                            reason="authorization_code_expired",
                        )
                    db.add(auth)
                    db.commit()
            return True
        if resp.status_code == 429 or resp.status_code >= 500:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)
            continue

        with SessionLocal() as db:
            auth = db.get(AppAuthJob, archive_job_id)
            if auth:
                auth.status = "failed"
                auth.last_error = f"authorization_code_failed:{resp.status_code}"
                notify_email = auth.email
                if not notify_email and auth.device_id:
                    device_email = db.get(DeviceEmail, auth.device_id)
                    notify_email = device_email.email if device_email else None
                if notify_email:
                    _enqueue_reauth_notify(
                        db,
                        email=notify_email,
                        archive_job_id=archive_job_id,
                        reason="authorization_code_failed",
                    )
                db.add(auth)
                db.commit()
        return True

    if not authorization_code:
        with SessionLocal() as db:
            auth = db.get(AppAuthJob, archive_job_id)
            if auth:
                auth.status = "failed"
                auth.last_error = "authorization_code_timeout"
                notify_email = auth.email
                if not notify_email and auth.device_id:
                    device_email = db.get(DeviceEmail, auth.device_id)
                    notify_email = device_email.email if device_email else None
                if notify_email:
                    _enqueue_reauth_notify(
                        db,
                        email=notify_email,
                        archive_job_id=archive_job_id,
                        reason="authorization_code_timeout",
                    )
                db.add(auth)
                db.commit()
        return True

    with SessionLocal() as db:
        auth = db.get(AppAuthJob, archive_job_id)
        if not auth:
            return True

        anchor_token = None
        if auth.app_user_id:
            existing_user = db.get(AppUser, auth.app_user_id)
            anchor_token = existing_user.latest_anchor_token if existing_user else None

        try:
            data = await archive_client.finalize_xordi(
                archive_job_id=archive_job_id,
                authorization_code=authorization_code,
                anchor_token=anchor_token,
            )
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else 502
            body = (exc.response.text if exc.response is not None else "").strip()
            auth.last_error = f"{status_code} {body}".strip()
            # IMPORTANT: finalize must be at-most-once per auth job.
            # Treat any finalize error as terminal so we never retry and call finalize again.
            auth.status = "failed"
            notify_email = auth.email
            if not notify_email and auth.device_id:
                device_email = db.get(DeviceEmail, auth.device_id)
                notify_email = device_email.email if device_email else None
            if notify_email:
                _enqueue_reauth_notify(
                    db,
                    email=notify_email,
                    archive_job_id=archive_job_id,
                    reason="finalize_failed",
                )
            db.add(auth)
            db.commit()
            return True
        except Exception as exc:
            auth.last_error = str(exc)
            auth.status = "failed"
            notify_email = auth.email
            if not notify_email and auth.device_id:
                device_email = db.get(DeviceEmail, auth.device_id)
                notify_email = device_email.email if device_email else None
            if notify_email:
                _enqueue_reauth_notify(
                    db,
                    email=notify_email,
                    archive_job_id=archive_job_id,
                    reason="finalize_exception",
                )
            db.add(auth)
            db.commit()
            return True

        final_app_user_id = data.get("archive_user_id") or auth.app_user_id or str(uuid4())
        user = db.get(AppUser, final_app_user_id)
        if not user:
            user = AppUser(app_user_id=final_app_user_id)

        previous_sec_user_id = user.latest_sec_user_id
        user.archive_user_id = data.get("archive_user_id")
        user.latest_sec_user_id = data.get("provider_unique_id")
        platform_username = data.get("platform_username")
        if platform_username:
            user.platform_username = platform_username
        tz_hint = job.payload.get("time_zone")
        if isinstance(tz_hint, str) and tz_hint.strip():
            user.time_zone = tz_hint.strip()
        new_anchor = data.get("anchor_token")
        if new_anchor or anchor_token:
            user.latest_anchor_token = new_anchor or anchor_token
        if previous_sec_user_id != user.latest_sec_user_id:
            user.is_watch_history_available = "unknown"
        db.add(user)
        db.commit()

        if not auth.device_id or not auth.platform or not auth.app_version or not auth.os_version:
            auth.status = "failed"
            auth.last_error = "missing_device_metadata"
            notify_email = auth.email
            if not notify_email and auth.device_id:
                device_email = db.get(DeviceEmail, auth.device_id)
                notify_email = device_email.email if device_email else None
            if notify_email:
                _enqueue_reauth_notify(
                    db,
                    email=notify_email,
                    archive_job_id=archive_job_id,
                    reason="missing_device_metadata",
                )
            db.add(auth)
            db.commit()
            return True

        token, session_expires_at = session_service.create_or_rotate(
            db=db,
            app_user_id=user.app_user_id,
            device_id=auth.device_id,
            platform=auth.platform,
            app_version=auth.app_version,
            os_version=auth.os_version,
        )
        _ = token
        _ = session_expires_at

        session = (
            db.query(AppSession)
            .filter(
                AppSession.app_user_id == user.app_user_id,
                AppSession.device_id == auth.device_id,
                AppSession.revoked_at.is_(None),
            )
            .order_by(AppSession.issued_at.desc())
            .first()
        )

        auth.app_user_id = user.app_user_id
        auth.status = "finalized"
        auth.finalized_at = datetime.utcnow()
        auth.session_id = session.id if session else None
        auth.last_error = None

        device_email = db.get(DeviceEmail, auth.device_id)
        if device_email and device_email.email:
            auth.email = device_email.email
            latest_email = (
                db.query(AppUserEmail)
                .filter(AppUserEmail.app_user_id == user.app_user_id)
                .order_by(AppUserEmail.created_at.desc())
                .first()
            )
            if not latest_email or latest_email.email != device_email.email:
                db.add(
                    AppUserEmail(
                        id=str(uuid4()),
                        app_user_id=user.app_user_id,
                        email=device_email.email,
                        created_at=datetime.utcnow(),
                        verified_at=datetime.utcnow(),
                    )
                )

        db.add(auth)
        db.commit()

        await _ensure_watch_history_available(user, db, auto_enqueue=True, resend_email_on_ready=True)
        return True


def _watch_history_since_ms() -> int:
    if WATCH_HISTORY_SINCE_MS:
        with suppress(Exception):
            return int(WATCH_HISTORY_SINCE_MS)

    raw_date = (WATCH_HISTORY_SINCE_DATE or "2025-01-01").strip()
    with suppress(Exception):
        dt = datetime.fromisoformat(raw_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    return int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _months_between(start_utc: datetime, end_utc: datetime) -> int:
    if end_utc < start_utc:
        return 1
    return max(1, (end_utc.year - start_utc.year) * 12 + (end_utc.month - start_utc.month) + 1)


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


async def _compute_brainrot_score(
    *,
    sec_user_id: str,
    range_spec: Dict[str, Any],
    time_zone: Optional[str],
    payload: Dict[str, Any],
    debug_enabled: bool,
) -> Dict[str, Any]:
    """
    Compute brainrot intensity/volume via Archive cached interests enrichment.

    Returns a dict with:
    - ok: bool (computed and stable this attempt)
    - pending: bool (enrichment still running; retry later)
    - used_fallback_llm: bool
    - fields: Dict[str, Any] (payload updates)
    """
    fields: Dict[str, Any] = {}
    now = datetime.now(timezone.utc)

    max_creators = 500
    with suppress(Exception):
        raw = os.getenv("BRAINROT_ENRICH_MAX_CREATORS")
        if raw:
            max_creators = max(1, min(5000, int(raw)))

    # Loop driver (best-effort): reconcile → poll status (until ready/failed) → reconcile … until done
    # or until we run out of wall-clock time for this analysis attempt.
    deadline = time.perf_counter() + float(BRAINROT_ENRICH_POLL_MAX_SECONDS or 0)
    cycles = 0
    done = False
    while True:
        cycles += 1
        fields["_brainrot_interests_reconcile_cycles"] = cycles

        reconcile_resp = None
        reconcile_data: Dict[str, Any] = {}
        try:
            reconcile_resp = await archive_client.interests_enrichment_reconcile(
                sec_user_id=sec_user_id,
                range=range_spec,
                time_zone=time_zone,
                mode="missing_only",
                max_creators=max_creators,
                enqueue=True,
            )
            with suppress(Exception):
                reconcile_data = reconcile_resp.json()
        except Exception as exc:
            if debug_enabled:
                fields["_brainrot_interests_error"] = {"stage": "reconcile", "error": str(exc)}
            return {"ok": False, "pending": False, "used_fallback_llm": True, "fields": fields}

        if isinstance(reconcile_data, dict):
            fields["_brainrot_interests_reconcile"] = {
                "updated_at": _iso_utc(datetime.now(timezone.utc)),
                "done": reconcile_data.get("done"),
                "job_id": reconcile_data.get("job_id"),
                "interest_set_version": reconcile_data.get("interest_set_version"),
                "creators_enrichable": reconcile_data.get("creators_enrichable"),
                "creators_cached_any": reconcile_data.get("creators_cached_any"),
                "creators_cached_ok": reconcile_data.get("creators_cached_ok"),
                "missing_creators": reconcile_data.get("missing_creators"),
                "coverage_any_pct": reconcile_data.get("coverage_any_pct"),
                "coverage_ok_pct": reconcile_data.get("coverage_ok_pct"),
                "enqueued": reconcile_data.get("enqueued"),
                "deduped": reconcile_data.get("deduped"),
            }

        done = bool(reconcile_data.get("done")) if isinstance(reconcile_data, dict) else False
        job_id = reconcile_data.get("job_id") if isinstance(reconcile_data, dict) else None
        if done:
            break
        if not job_id:
            break
        if not BRAINROT_ENRICH_POLL_MAX_SECONDS:
            break
        if time.perf_counter() >= deadline:
            break

        sleep_s = 0.5
        last_state = None
        while time.perf_counter() < deadline:
            try:
                status_resp = await archive_client.interests_enrichment_status(job_id=str(job_id))
                last_status_body = None
                with suppress(Exception):
                    last_status_body = status_resp.json()
                if isinstance(last_status_body, dict):
                    last_state = str(last_status_body.get("state") or last_status_body.get("status") or "").lower()
                    fields["_brainrot_interests_status"] = {"updated_at": _iso_utc(datetime.now(timezone.utc)), **last_status_body}
                if last_state in {"ready"}:
                    break
                if last_state in {"failed", "expired"}:
                    break
            except Exception as exc:
                if debug_enabled:
                    fields["_brainrot_interests_error"] = {"stage": "status", "error": str(exc)}
                break
            await asyncio.sleep(sleep_s)
            sleep_s = min(sleep_s * 1.6, 8.0)

        if last_state in {"failed", "expired"}:
            break
        if time.perf_counter() >= deadline:
            break

    # Even if reconcile isn't fully done, compute from whatever cache exists; downweight with confidence/hours.
    try:
        summary = await archive_client.interests_summary(
            sec_user_id=sec_user_id,
            range=range_spec,
            time_zone=time_zone,
            group_by="none",
            limit=10,
            include_unknown=True,
        )
    except Exception as exc:
        if debug_enabled:
            fields["_brainrot_interests_error"] = {"stage": "summary", "error": str(exc)}
        return {"ok": False, "pending": False, "used_fallback_llm": True, "fields": fields}

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
        return {"ok": False, "pending": False, "used_fallback_llm": True, "fields": fields}

    raw = max(0.0, min(1.0, float(raw)))
    if confidence is None:
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))
    if watch_seconds_enriched is None:
        watch_seconds_enriched = 0.0
    watch_seconds_enriched = max(0.0, float(watch_seconds_enriched))

    enriched_hours = watch_seconds_enriched / 3600.0
    intensity_linear_0_100 = max(0.0, min(100.0, raw * 100.0))

    eps = max(1e-12, float(BRAINROT_S_CURVE_EPS))
    base_raw = max(eps, float(BRAINROT_S_CURVE_BASELINE_RAW))
    base_score_frac = float(BRAINROT_S_CURVE_BASELINE_SCORE) / 100.0
    base_score_frac = max(0.01, min(0.99, base_score_frac))
    k = max(0.05, float(BRAINROT_S_CURVE_SLOPE))

    raw_for_norm = raw
    if raw_for_norm <= 0.0:
        divisor = float(BRAINROT_ZERO_RAW_FLOOR_DIVISOR or 0.0)
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

    # Downweight intensity if quality is low.
    min_conf = max(0.0001, float(BRAINROT_MIN_CONFIDENCE))
    min_hours = max(0.0001, float(BRAINROT_MIN_ENRICHED_HOURS))
    w_conf = min(1.0, confidence / min_conf)
    w_hours = min(1.0, enriched_hours / min_hours)
    quality_weight = min(1.0, (w_conf * w_hours) ** 0.5)
    final_score = _clamp_0_100(intensity_norm_0_100_f * quality_weight)

    fields.update(
        {
            "_brainrot_source": "tokyodata_interests",
            "brainrot_intensity_raw": round(raw, 8),
            "brainrot_intensity_raw_effective": round(raw_for_norm, 8),
            "brainrot_intensity_linear": round(intensity_linear_0_100, 4),
            "brainrot_intensity": intensity_norm_0_100,
            "brainrot_confidence": round(confidence, 6),
            "brainrot_enriched_watch_pct": round(confidence * 100.0, 4),
            "brainrot_enriched_hours": round(enriched_hours, 4),
            "brainrot_volume_hours": round(volume_hours, 4),
            "brainrot_quality_weight": round(quality_weight, 6),
            "brainrot_normalization": {
                "baseline_raw": base_raw,
                "baseline_score": float(BRAINROT_S_CURVE_BASELINE_SCORE),
                "slope": k,
                "zero_raw_floor_divisor": float(BRAINROT_ZERO_RAW_FLOOR_DIVISOR),
            },
            "brain_rot_score": final_score,
        }
    )
    return {"ok": True, "pending": False, "used_fallback_llm": False, "fields": fields}


async def _call_llm(
    prompt: str,
    sample_texts: List[str],
    debug_meta: Optional[Dict[str, Any]] = None,
    *,
    temperature: float = 0.7,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    if not api_key or not model:
        if debug_meta is not None:
            debug_meta.update({"error": "missing_openrouter_config", "model": model})
        return ""
    async with llm_semaphore:
        async with httpx.AsyncClient(timeout=20.0) as client:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "\n".join(sample_texts[:20])},
            ]
            backoff = 1.0
            for attempt in range(1, 4):
                try:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": messages, "temperature": float(temperature)},
                    )
                    request_id = resp.headers.get("x-request-id") or resp.headers.get("x-openrouter-request-id")
                    if debug_meta is not None:
                        debug_meta.update(
                            {
                                "attempt": attempt,
                                "status_code": resp.status_code,
                                "request_id": request_id,
                                "model": model,
                            }
                        )
                    if resp.status_code == 200:
                        data = resp.json()
                        if debug_meta is not None:
                            debug_meta.update(
                                {
                                    "provider": data.get("provider"),
                                    "response_model": data.get("model"),
                                    "usage": data.get("usage"),
                                }
                            )
                        content_val = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        content = (content_val or "").strip()
                        if debug_meta is not None:
                            debug_meta.update({"content_len": len(content)})
                        return content
                    body_text, body_truncated = sanitize_json_text(resp.text, max_chars=2000)
                    if debug_meta is not None:
                        debug_meta.update({"error_body": body_text, "error_body_truncated": body_truncated})
                    logger.warning(
                        "openrouter.error",
                        extra={
                            "event": "openrouter.error",
                            "openrouter_status": resp.status_code,
                            "openrouter_body": body_text,
                            "openrouter_body_truncated": body_truncated,
                            "openrouter_request_id": request_id,
                            "openrouter_model": model,
                            "attempt": attempt,
                        },
                    )
                    if resp.status_code < 500 and resp.status_code != 429:
                        break
                except Exception:
                    if debug_meta is not None:
                        debug_meta.update({"attempt": attempt, "error": "exception"})
                    logger.exception(
                        "openrouter.exception",
                        extra={"event": "openrouter.exception", "openrouter_model": model, "attempt": attempt},
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
    return ""


def _data_job_key_for_task(task_name: str) -> Optional[str]:
    if task_name == "watch_history_fetch_2025":
        return "watch_history"
    if task_name in ("wrapped_analysis", "email_send"):
        return task_name
    return None


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if "```" not in cleaned:
        return cleaned
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return (match.group(1) or "").strip()
    return cleaned


def _extract_json_value(text: str) -> Optional[Any]:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(cleaned[idx:])
            return value
        except Exception:
            continue
    return None


def _extract_first_int(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    with suppress(Exception):
        return int(match.group(0))
    return None


def _extract_percentile(text: str) -> Optional[str]:
    if not text:
        return None
    # IMPORTANT: don't use `\b` after `%` (percent sign is a non-word char, so `%\b` never matches).
    range_match = re.search(
        r"\btop\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:-|to)\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:%|(?:percent|pct)\b)",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        raw = range_match.group(2)
    else:
        match = re.search(
            r"\btop\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:%|(?:percent|pct)\b)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"\b(\d{1,2}(?:\.\d{1,2})?)\s*(?:%|(?:percent|pct)\b)",
                text,
                flags=re.IGNORECASE,
            )
        if not match:
            return None
        raw = match.group(1)
    try:
        value = float(raw)
    except Exception:
        return None
    if value <= 0:
        value = 0.1
    if value > 3.0:
        value = 3.0
    if value < 1.0:
        value = max(0.1, round(value, 1))
        if value >= 1.0:
            return "top 1%"
        return f"top {value:.1f}%"
    if float(value).is_integer():
        return f"top {int(value)}%"
    return f"top {value:g}%"


def _normalize_title_phrase(text: str, *, max_words: int) -> str:
    if not text:
        return ""
    cleaned = _strip_code_fences(text).strip()
    if not cleaned:
        return ""
    first_line = ""
    for line in cleaned.splitlines():
        candidate = line.strip()
        if candidate:
            first_line = candidate
            break
    if not first_line:
        return ""
    first_line = first_line.strip().strip("{}").strip()
    first_line = first_line.replace("_", " ")
    first_line = re.sub(r"^[*•-]+\s*", "", first_line)
    first_line = re.sub(r"[^A-Za-z0-9 ]+", "", first_line)
    first_line = re.sub(r"\s+", " ", first_line).strip()
    if not first_line:
        return ""
    words = first_line.split()
    if max_words > 0:
        words = words[:max_words]
    return " ".join(w.title() for w in words)

def _mark_wrapped_run_failed(db, wrapped_run_id: str, *, task_name: str, job_id: str) -> None:
    run = db.get(AppWrappedRun, wrapped_run_id)
    if not run or run.status == "ready":
        return

    payload = run.payload if isinstance(run.payload, dict) else {}
    data_jobs = payload.get("data_jobs")
    data_jobs = data_jobs if isinstance(data_jobs, dict) else {}

    key = _data_job_key_for_task(task_name)
    if key:
        data_jobs[key] = {"id": job_id, "status": "failed"}
        payload["data_jobs"] = data_jobs

    if key == "watch_history":
        progress = payload.get("_watch_history_progress")
        if isinstance(progress, dict):
            progress["status"] = "failed"
            progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
            if not progress.get("error"):
                progress["error"] = "job_failed"
                progress["error_detail"] = {"task_name": task_name, "job_id": job_id}
            payload["_watch_history_progress"] = progress

    run.payload = payload
    run.status = "failed"
    db.add(run)
    db.commit()


def _pick_cat_name(seed: str) -> str:
    try:
        import hashlib

        h = hashlib.sha256(seed.encode("utf-8")).digest()[0]
        return CAT_NAMES[h % len(CAT_NAMES)]
    except Exception:
        return CAT_NAMES[0]


def _pick_from_list(seed: str, options: List[str], default: str) -> str:
    if not options:
        return default
    try:
        import hashlib

        h = hashlib.sha256(seed.encode("utf-8")).digest()
        idx = int.from_bytes(h[:2], "big") % len(options)
        return options[idx]
    except Exception:
        return options[0]


def _build_analogy_line(total_videos: int) -> str:
    total_videos = max(0, int(total_videos))
    videos_per_mile_env = os.getenv("THUMB_VIDEOS_PER_MILE", "180")
    try:
        videos_per_mile = float(videos_per_mile_env)
    except Exception:
        videos_per_mile = 180.0
    videos_per_mile = max(10.0, min(videos_per_mile, 10000.0))
    miles = total_videos / videos_per_mile
    miles_label = f"{miles:.1f}" if miles < 2 else f"{int(round(miles))}"
    seed = f"analogy:{total_videos}"
    if miles < 2:
        suffix = _pick_from_list(seed + ":0_2", ["barely a warmup!", "barely broke a sweat!", "just a little stroll!"], "barely a warmup!")
    elif miles < 8:
        suffix = _pick_from_list(seed + ":2_8", ["that’s a respectable cardio!", "your thumb’s got stamina!"], "your thumb’s got stamina!")
    elif miles < 16:
        suffix = "longer than a half-marathon!"
    elif miles < 26:
        suffix = "that’s literally a full marathon!"
    elif miles < 50:
        suffix = _pick_from_list(seed + ":26_50", ["that’s two full marathons!", "take care, your thumb needs a spa day!"], "that’s two full marathons!")
    else:
        suffix = _pick_from_list(seed + ":50_plus", ["your thumb is now a bodybuilder!", "purely unhinged behavior!"], "purely unhinged behavior!")
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


def _best_hour_window(hour_histogram_seconds: Dict[str, Any], window_hours: int) -> Optional[Dict[str, Any]]:
    if not isinstance(hour_histogram_seconds, dict):
        return None
    if window_hours <= 0 or window_hours > 24:
        return None

    per_hour: List[float] = []
    for hour in range(24):
        raw = hour_histogram_seconds.get(str(hour))
        try:
            per_hour.append(float(raw) if raw is not None else 0.0)
        except Exception:
            per_hour.append(0.0)

    total = sum(per_hour)
    if total <= 0:
        return None

    best_start = 0
    best_sum = -1.0
    for start in range(24):
        window_sum = 0.0
        for offset in range(window_hours):
            window_sum += per_hour[(start + offset) % 24]
        if window_sum > best_sum:
            best_sum = window_sum
            best_start = start

    pct = 100.0 * best_sum / total if total > 0 else 0.0
    return {"start_hour": best_start, "window_seconds": best_sum, "pct": pct}


def _build_scroll_time(
    *,
    hour_histogram_seconds: Optional[Dict[str, Any]] = None,
    peak_hour: Optional[int] = None,
) -> Dict[str, str]:
    window_hours_env = os.getenv("SCROLL_TIME_WINDOW_HOURS", "2")
    try:
        window_hours = int(window_hours_env)
    except Exception:
        window_hours = 2
    window_hours = max(1, min(window_hours, 6))

    best = _best_hour_window(hour_histogram_seconds or {}, window_hours)
    if best:
        start_hour = int(best["start_hour"])
        pct = float(best["pct"])
    elif peak_hour is not None:
        start_hour = int(peak_hour) - 1
        pct = 0.0
    else:
        start_hour = 20
        pct = 0.0

    return {
        "title": _scroll_window_label(start_hour),
        "rate": f"{round(pct):.0f}%",
        "between_time": _format_hour_window(start_hour, window_hours),
    }


async def handle_watch_history_fetch(job: LeasedJob) -> bool:
    wrapped_run_id = job.payload.get("wrapped_run_id")
    app_user_id = job.payload.get("app_user_id")
    if not wrapped_run_id or not app_user_id:
        return True

    def _exc_details(exc: Exception, *, max_body_chars: int = 2000) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": exc.__class__.__name__, "message": str(exc)}
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            out["status_code"] = exc.response.status_code
            with suppress(Exception):
                body_text, body_truncated = sanitize_json_text(exc.response.text, max_chars=max_body_chars)
                out["body"] = body_text
                out["body_truncated"] = body_truncated
        return out

    with SessionLocal() as db:
        run = db.get(AppWrappedRun, wrapped_run_id)
        if not run:
            return True
        user = db.get(AppUser, app_user_id)
        run_id = run.id
        run_app_user_id = run.app_user_id
        sec_user_id = (
            (run.sec_user_id or "").strip()
            or str(job.payload.get("sec_user_id") or "").strip()
            or (user.latest_sec_user_id if user and user.latest_sec_user_id else None)
        )
        if not sec_user_id:
            return True
        tz_hint = job.payload.get("time_zone")
        tz_name = str(_safe_zone(str(tz_hint) if tz_hint else (user.time_zone if user else None)))
        existing_progress = run.payload.get("_watch_history_progress") if isinstance(run.payload, dict) else {}
        existing_progress = existing_progress if isinstance(existing_progress, dict) else {}

    since_ms = _watch_history_since_ms()
    with suppress(Exception):
        payload_since_ms = job.payload.get("since_ms")
        if payload_since_ms is not None:
            since_ms = int(payload_since_ms)
    with suppress(Exception):
        payload_since_days = job.payload.get("since_days")
        if payload_since_days is not None:
            days = float(payload_since_days)
            if days > 0:
                since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_dt = datetime.now(timezone.utc)
    now_ms = int(now_dt.timestamp() * 1000)
    start_at = _iso_utc(datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc))
    end_at = _iso_utc(now_dt)

    scrape_max_videos = None
    with suppress(Exception):
        raw = job.payload.get("scrape_max_videos")
        if raw is not None:
            scrape_max_videos = int(raw)
    if scrape_max_videos is None:
        with suppress(Exception):
            scrape_max_videos = int(os.getenv("WATCH_HISTORY_SCRAPE_MAX_VIDEOS") or 0) or None
    if scrape_max_videos is not None and scrape_max_videos <= 0:
        scrape_max_videos = None

    scrape_max_data_jobs = None
    with suppress(Exception):
        raw = job.payload.get("scrape_max_data_jobs")
        if raw is not None:
            scrape_max_data_jobs = int(raw)
    if scrape_max_data_jobs is None:
        with suppress(Exception):
            scrape_max_data_jobs = int(os.getenv("WATCH_HISTORY_SCRAPE_MAX_DATA_JOBS") or 0) or None
    if scrape_max_data_jobs is not None and scrape_max_data_jobs <= 0:
        scrape_max_data_jobs = None

    completed_data_jobs = 0
    with suppress(Exception):
        completed_data_jobs = int(existing_progress.get("completed_data_jobs") or 0)
    videos_fetched_total = 0
    with suppress(Exception):
        videos_fetched_total = int(existing_progress.get("videos_fetched_total") or 0)

    resume_cursor_ms = None
    with suppress(Exception):
        candidate = int(existing_progress.get("next_cursor_ms"))
        if candidate > 0:
            resume_cursor_ms = candidate
    with suppress(Exception):
        if resume_cursor_ms is None and completed_data_jobs > 0:
            candidate = int(existing_progress.get("cursor_ms"))
            if candidate > 0:
                resume_cursor_ms = candidate
    cursor_ms = resume_cursor_ms if resume_cursor_ms and resume_cursor_ms >= since_ms else now_ms

    logger.info("watch_history_fetch_2025 start run=%s since_ms=%s cursor_ms=%s", run_id, since_ms, cursor_ms)

    progress: Dict[str, Any] = {
        "status": "running",
        "since_ms": since_ms,
        "scrape_max_videos": scrape_max_videos,
        "scrape_max_data_jobs": scrape_max_data_jobs,
        "scrape_truncated": False,
        "scrape_stop_reason": None,
        "range": {"type": "between", "start_at": start_at, "end_at": end_at},
        "cursor_ms": cursor_ms,
        "next_cursor_ms": None,
        "has_more": None,
        "completed_data_jobs": completed_data_jobs,
        "videos_fetched_total": videos_fetched_total,
        "last_data_job_id": None,
        "sec_user_id": sec_user_id,
        "updated_at": _iso_utc(now_dt),
    }

    with SessionLocal() as db:
        run = db.get(AppWrappedRun, run_id)
        if not run:
            return True
        partial_payload = run.payload if isinstance(run.payload, dict) else {}
        if job.payload.get("force_new_analysis"):
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
                partial_payload.pop(key, None)
            for key in tuple(partial_payload.keys()):
                if isinstance(key, str) and key.startswith("_brainrot_"):
                    partial_payload.pop(key, None)
            data_jobs_existing = partial_payload.get("data_jobs") if isinstance(partial_payload.get("data_jobs"), dict) else {}
            data_jobs_existing.pop("wrapped_analysis", None)
            data_jobs_existing.pop("email_send", None)
            partial_payload["data_jobs"] = data_jobs_existing
        data_jobs = partial_payload.get("data_jobs") or {}
        data_jobs["watch_history"] = {"id": job.id, "status": "running"}
        partial_payload["data_jobs"] = data_jobs
        partial_payload["_watch_history_progress"] = progress
        partial_payload.pop("_months_done", None)
        run.payload = partial_payload
        db.add(run)
        db.commit()

    def _should_skip_scrape(progress_dict: Dict[str, Any]) -> bool:
        try:
            status_val = str(progress_dict.get("status") or "")
            # If the cursor-walk already completed, don't enqueue more scrape data jobs on retry.
            if status_val not in {"scrape_done", "failed"}:
                return False
            if progress_dict.get("scrape_truncated") is True and int(progress_dict.get("completed_data_jobs") or 0) > 0:
                return True
            has_more_val = progress_dict.get("has_more")
            next_cursor_val = progress_dict.get("next_cursor_ms")
            completed = int(progress_dict.get("completed_data_jobs") or 0)
            if has_more_val is False and not next_cursor_val and completed > 0:
                return True
        except Exception:
            return False
        return False

    if _should_skip_scrape(existing_progress):
        logger.info("watch_history_fetch_2025 skipping scrape (already scrape_done) run=%s", run_id)
    else:
        last_cursor_ms = None
        use_null_cursor_first = resume_cursor_ms is None and completed_data_jobs == 0
        while True:
            if cursor_ms < since_ms:
                break
            if last_cursor_ms is not None and cursor_ms >= last_cursor_ms:
                logger.warning(
                    "watch_history.cursor_not_decreasing run=%s cursor_ms=%s last_cursor_ms=%s",
                    run_id,
                    cursor_ms,
                    last_cursor_ms,
                )
                break
            last_cursor_ms = cursor_ms

            bind_context(cursor_ms=cursor_ms, since_ms=since_ms)
            try:
                backoff = 1.0
                start_data = None
                last_start_status: Optional[int] = None
                last_start_body: Optional[str] = None
                last_start_body_truncated: Optional[bool] = None
                cursor_arg = None if use_null_cursor_first else str(cursor_ms)
                for _ in range(5):
                    start_resp = await archive_client.start_watch_history(
                        sec_user_id=sec_user_id,
                        limit=WATCH_HISTORY_PAGE_LIMIT,
                        max_pages=WATCH_HISTORY_MAX_PAGES,
                        cursor=cursor_arg,
                    )
                    last_start_status = start_resp.status_code
                    with suppress(Exception):
                        last_start_body, last_start_body_truncated = sanitize_json_text(start_resp.text, max_chars=1200)
                    if start_resp.status_code == 202:
                        start_data = start_resp.json()
                        break
                    if start_resp.status_code == 429:
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 8.0)
                        continue
                    if start_resp.status_code >= 500:
                        logger.warning(
                            "watch_history.start_5xx",
                            extra={
                                "event": "watch_history.start_5xx",
                                "sec_user_id": sec_user_id,
                                "status_code": start_resp.status_code,
                                "cursor_ms": cursor_ms,
                            },
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 8.0)
                        continue
                    if start_resp.status_code == 404:
                        logger.warning(
                            "watch_history.start_not_found",
                            extra={"event": "watch_history.start_not_found", "sec_user_id": sec_user_id},
                        )
                        break
                    logger.warning(
                        "watch_history.start_unexpected_status",
                        extra={
                            "event": "watch_history.start_unexpected_status",
                            "sec_user_id": sec_user_id,
                            "status_code": start_resp.status_code,
                            "cursor_ms": cursor_ms,
                        },
                    )
                    break
                use_null_cursor_first = False
                if not start_data:
                    progress["status"] = "failed"
                    progress["error"] = "archive_start_failed"
                    progress["error_detail"] = {
                        "cursor": cursor_arg,
                        "status_code": last_start_status,
                        "body": last_start_body,
                        "body_truncated": last_start_body_truncated,
                    }
                    progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
                    with SessionLocal() as db:
                        run = db.get(AppWrappedRun, run_id)
                        if run:
                            payload = run.payload if isinstance(run.payload, dict) else {}
                            data_jobs = payload.get("data_jobs") if isinstance(payload.get("data_jobs"), dict) else {}
                            if last_start_status in (400, 401, 403, 404):
                                data_jobs["watch_history"] = {
                                    "id": job.id,
                                    "status": "failed",
                                    "error": "archive_not_found" if last_start_status == 404 else "archive_start_failed",
                                }
                                payload["data_jobs"] = data_jobs
                                run.status = "failed"
                            payload["_watch_history_progress"] = progress
                            run.payload = payload
                            db.add(run)
                            db.commit()
                    if last_start_status in (400, 401, 403, 404):
                        return True
                    return False

                data_job_id = start_data.get("data_job_id")
                if not data_job_id:
                    progress["status"] = "failed"
                    progress["error"] = "missing_data_job_id"
                    progress["error_detail"] = {"cursor": cursor_arg}
                    progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
                    with SessionLocal() as db:
                        run = db.get(AppWrappedRun, run_id)
                        if run:
                            payload = run.payload if isinstance(run.payload, dict) else {}
                            payload["_watch_history_progress"] = progress
                            run.payload = payload
                            db.add(run)
                            db.commit()
                    return False

                progress["last_data_job_id"] = data_job_id
                progress["cursor_ms"] = cursor_ms
                progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))

                backoff_fin = 1.0
                success_fin = False
                fin_data: Dict[str, Any] = {}
                last_fin_status: Optional[int] = None
                last_fin_body: Optional[str] = None
                last_fin_body_truncated: Optional[bool] = None
                finalize_started_perf = time.perf_counter()
                finalize_started_at = datetime.now(timezone.utc)
                last_progress_persist_perf = finalize_started_perf
                provider_failed_started_perf: Optional[float] = None
                max_cursor_skips = max(0, int(os.getenv("WATCH_HISTORY_SCRAPE_MAX_CURSOR_SKIPS", "8")))
                cursor_skip_week_ms = 7 * 24 * 60 * 60 * 1000
                for attempt in range(1, 301):
                    if (time.perf_counter() - finalize_started_perf) > WATCH_HISTORY_FINALIZE_MAX_SECONDS:
                        break
                    fin = await archive_client.finalize_watch_history(
                        data_job_id=data_job_id,
                        include_rows=False,
                    )
                    last_fin_status = fin.status_code
                    with suppress(Exception):
                        last_fin_body, last_fin_body_truncated = sanitize_json_text(fin.text, max_chars=1200)
                    if fin.status_code == 202:
                        with suppress(Exception):
                            fin_data = fin.json()
                        queue_pos = fin_data.get("queue_position")
                        if (time.perf_counter() - last_progress_persist_perf) >= WATCH_HISTORY_FINALIZE_PROGRESS_PERSIST_SECONDS:
                            progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
                            progress["finalize"] = {
                                "data_job_id": data_job_id,
                                "attempt": attempt,
                                "status_code": fin.status_code,
                                "queue_position": queue_pos,
                                "started_at": _iso_utc(finalize_started_at),
                            }
                            with SessionLocal() as db:
                                run = db.get(AppWrappedRun, run_id)
                                if run:
                                    payload = run.payload if isinstance(run.payload, dict) else {}
                                    payload["_watch_history_progress"] = progress
                                    run.payload = payload
                                    db.add(run)
                                    db.commit()
                            last_progress_persist_perf = time.perf_counter()
                        wait = 2.0
                        with suppress(Exception):
                            if queue_pos is not None:
                                pos = int(queue_pos)
                                if pos <= 1:
                                    wait = 0.1
                                else:
                                    # Archive queue advances at ~2s per position; wait close to expected turn.
                                    wait = min(60.0, max(0.5, 2.0 * float(pos - 1)))
                        await asyncio.sleep(wait)
                        continue
                    if fin.status_code == 200:
                        with suppress(Exception):
                            fin_data = fin.json()
                        success_fin = True
                        break
                    if fin.status_code == 424:
                        if provider_failed_started_perf is None:
                            provider_failed_started_perf = time.perf_counter()
                        logger.warning(
                            "watch_history.finalize_provider_failed",
                            extra={
                                "event": "watch_history.finalize_provider_failed",
                                "run_id": run_id,
                                "sec_user_id": sec_user_id,
                                "data_job_id": data_job_id,
                                "status_code": fin.status_code,
                            },
                        )
                        if (time.perf_counter() - provider_failed_started_perf) >= WATCH_HISTORY_FINALIZE_PROVIDER_FAILED_MAX_SECONDS:
                            break
                        await asyncio.sleep(backoff_fin)
                        backoff_fin = min(backoff_fin * 2, 8.0)
                        continue
                    if fin.status_code >= 500:
                        logger.warning(
                            "watch_history.finalize_5xx",
                            extra={
                                "event": "watch_history.finalize_5xx",
                                "sec_user_id": sec_user_id,
                                "data_job_id": data_job_id,
                                "status_code": fin.status_code,
                            },
                        )
                        await asyncio.sleep(backoff_fin)
                        backoff_fin = min(backoff_fin * 2, 8.0)
                        continue
                    if fin.status_code == 410:
                        break
                    if fin.status_code == 429:
                        await asyncio.sleep(backoff_fin)
                        backoff_fin = min(backoff_fin * 2, 8.0)
                        continue
                    await asyncio.sleep(backoff_fin)
                    backoff_fin = min(backoff_fin * 2, 8.0)
                if not success_fin:
                    error_detail = {
                        "status_code": last_fin_status,
                        "body": last_fin_body,
                        "body_truncated": last_fin_body_truncated,
                    }
                    progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
                    progress["finalize"] = {
                        "data_job_id": data_job_id,
                        "attempt": attempt if "attempt" in locals() else None,
                        "status_code": last_fin_status,
                        "started_at": _iso_utc(finalize_started_at),
                    }

                    # If we've already ingested some pages, attempt to "nudge" the cursor and continue
                    # scraping rather than failing the whole run. After enough skips, fall back to using
                    # the partial data we already have (analytics + extrapolation).
                    can_continue_partial = completed_data_jobs > 0 or videos_fetched_total > 0
                    if can_continue_partial:
                        skip_count = 0
                        with suppress(Exception):
                            skip_count = int(progress.get("cursor_skip_count") or 0)
                        skip_count += 1
                        progress["cursor_skip_count"] = skip_count
                        progress["scrape_error"] = "archive_finalize_failed"
                        progress["scrape_error_detail"] = error_detail

                        next_cursor_candidate = cursor_ms - cursor_skip_week_ms
                        if next_cursor_candidate <= since_ms:
                            next_cursor_candidate = since_ms

                        logger.warning(
                            "watch_history.finalize_failed_cursor_skip",
                            extra={
                                "event": "watch_history.finalize_failed_cursor_skip",
                                "run_id": run_id,
                                "sec_user_id": sec_user_id,
                                "data_job_id": data_job_id,
                                "cursor_ms": cursor_ms,
                                "next_cursor_ms": next_cursor_candidate,
                                "skip_count": skip_count,
                                "max_cursor_skips": max_cursor_skips,
                                "completed_data_jobs": completed_data_jobs,
                                "videos_fetched_total": videos_fetched_total,
                                **error_detail,
                            },
                        )

                        progress["next_cursor_ms"] = next_cursor_candidate
                        progress["cursor_ms"] = cursor_ms
                        progress["has_more"] = True

                        stop_reason = None
                        if max_cursor_skips and skip_count >= max_cursor_skips:
                            progress["scrape_truncated"] = True
                            stop_reason = "cursor_skips_exhausted"
                        elif next_cursor_candidate <= since_ms or next_cursor_candidate == cursor_ms:
                            progress["scrape_truncated"] = True
                            stop_reason = "cursor_skip_no_progress"
                        if stop_reason:
                            progress["scrape_stop_reason"] = stop_reason

                        with SessionLocal() as db:
                            run = db.get(AppWrappedRun, run_id)
                            if run:
                                payload = run.payload if isinstance(run.payload, dict) else {}
                                payload["_watch_history_progress"] = progress
                                run.payload = payload
                                db.add(run)
                                db.commit()

                        if stop_reason:
                            break

                        cursor_ms = next_cursor_candidate
                        continue

                    progress["status"] = "failed"
                    progress["error"] = "archive_finalize_failed"
                    progress["error_detail"] = error_detail
                    with SessionLocal() as db:
                        run = db.get(AppWrappedRun, run_id)
                        if run:
                            payload = run.payload if isinstance(run.payload, dict) else {}
                            payload["_watch_history_progress"] = progress
                            run.payload = payload
                            db.add(run)
                            db.commit()
                    return False

                completed_data_jobs += 1
                progress["completed_data_jobs"] = completed_data_jobs

                with suppress(Exception):
                    videos_fetched_total += int(fin_data.get("videos_fetched") or 0)
                    progress["videos_fetched_total"] = videos_fetched_total

                pagination = fin_data.get("pagination", {}) if fin_data else {}
                next_cursor = pagination.get("next_cursor")
                has_more = pagination.get("has_more")
                has_more_bool = None if has_more is None else bool(has_more)
                progress["has_more"] = has_more_bool

                next_cursor_ms = None
                with suppress(Exception):
                    next_cursor_ms = int(next_cursor) if next_cursor is not None else None
                if next_cursor_ms is not None and next_cursor_ms <= 0:
                    next_cursor_ms = None
                # Prefer has_more when provided: Archive may return a next_cursor even on the terminal page.
                should_continue = bool(next_cursor_ms) and has_more_bool is not False
                if has_more_bool is False and next_cursor_ms:
                    logger.info(
                        "watch_history.pagination_has_more_false_ignoring_next_cursor",
                        extra={
                            "event": "watch_history.pagination_has_more_false_ignoring_next_cursor",
                            "run_id": run_id,
                            "sec_user_id": sec_user_id,
                            "cursor_ms": cursor_ms,
                            "next_cursor_ms": next_cursor_ms,
                        },
                    )
                progress["next_cursor_ms"] = next_cursor_ms if should_continue else None
                progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))

                with SessionLocal() as db:
                    run = db.get(AppWrappedRun, run_id)
                    if not run:
                        return True
                    partial_payload = run.payload if isinstance(run.payload, dict) else {}
                    data_jobs = partial_payload.get("data_jobs") or {}
                    data_jobs["watch_history"] = {"id": job.id, "status": "running"}
                    partial_payload["data_jobs"] = data_jobs
                    partial_payload["_watch_history_progress"] = progress
                    run.payload = partial_payload
                    db.add(run)
                    db.commit()

                if scrape_max_data_jobs is not None and completed_data_jobs >= scrape_max_data_jobs:
                    progress["scrape_truncated"] = True
                    progress["scrape_stop_reason"] = "max_data_jobs"
                    break
                if scrape_max_videos is not None and videos_fetched_total >= scrape_max_videos:
                    progress["scrape_truncated"] = True
                    progress["scrape_stop_reason"] = "max_videos"
                    break

                if not should_continue:
                    break
                if next_cursor_ms < since_ms:
                    break
                cursor_ms = min(next_cursor_ms, now_ms)
            finally:
                clear_context("cursor_ms", "since_ms")

    progress["status"] = "scrape_done"
    progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
    with SessionLocal() as db:
        run = db.get(AppWrappedRun, run_id)
        if run:
            partial_payload = run.payload if isinstance(run.payload, dict) else {}
            partial_payload["_watch_history_progress"] = progress
            run.payload = partial_payload
            db.add(run)
            db.commit()

    def _is_retryable_analytics_error(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            # Analytics is eventually consistent; treat common "not ready" and transient errors as retryable.
            code = exc.response.status_code
            return code in (404, 408, 409, 422, 424, 429, 500, 502, 503, 504)
        if isinstance(exc, httpx.RequestError):
            return True
        return True

    async def _retry_analytics(label: str, fn, *, attempts: int = 6, base_sleep: float = 1.0):
        backoff = float(base_sleep)
        for attempt in range(1, attempts + 1):
            try:
                return await fn()
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_analytics_error(exc):
                    raise
                logger.warning(
                    "watch_history.analytics_retry",
                    extra={
                        "event": "watch_history.analytics_retry",
                        "run_id": run_id,
                        "sec_user_id": sec_user_id,
                        "label": label,
                        "attempt": attempt,
                        "sleep_seconds": backoff,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    coverage = None
    watched_at_min_dt = None
    watched_at_max_dt = None
    try:
        coverage = await _retry_analytics(
            "coverage",
            lambda: archive_client.watch_history_analytics_coverage(sec_user_id=sec_user_id),
            attempts=5,
        )
        if isinstance(coverage, dict):
            watched_at_min_dt = _parse_iso_datetime(coverage.get("watched_at_min"))
            watched_at_max_dt = _parse_iso_datetime(coverage.get("watched_at_max"))
    except Exception as exc:
        logger.warning("Watch history analytics coverage failed run=%s sec_user_id=%s err=%s", run_id, sec_user_id, exc)

    target_start_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
    actual_start_dt = watched_at_min_dt if watched_at_min_dt and watched_at_min_dt > target_start_dt else target_start_dt
    target_months = _months_between(target_start_dt, now_dt)
    coverage_months = _months_between(actual_start_dt, now_dt)

    range_spec: Dict[str, Any] = {"type": "between", "start_at": _iso_utc(actual_start_dt), "end_at": end_at}
    try:
        summary = await _retry_analytics(
            "summary",
            lambda: archive_client.watch_history_analytics_summary(
                sec_user_id=sec_user_id,
                range=range_spec,
                time_zone=tz_name,
                include_hour_histogram=True,
                top_creators_limit=5,
                top_music_limit=1,
            ),
            attempts=8,
        )
    except Exception as exc:
        logger.warning("Watch history analytics summary failed run=%s sec_user_id=%s err=%s", run_id, sec_user_id, exc)
        progress["status"] = "failed"
        progress["error"] = "analytics_summary_failed"
        progress["error_detail"] = _exc_details(exc)
        progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
        with SessionLocal() as db:
            run = db.get(AppWrappedRun, run_id)
            if run:
                payload = run.payload if isinstance(run.payload, dict) else {}
                payload["_watch_history_progress"] = progress
                run.payload = payload
                db.add(run)
                db.commit()
        return False

    sample_limit_env = int(os.getenv("WATCH_HISTORY_SAMPLES_LIMIT", "200"))
    sample_limit = max(1, min(sample_limit_env, 200))
    per_month_raw = max(1, sample_limit // coverage_months)
    if per_month_raw > 50:
        samples_strategy: Dict[str, Any] = {"type": "recent"}
        samples_limit = sample_limit
    else:
        samples_strategy = {"type": "per_month", "per_month": per_month_raw}
        samples_limit = min(sample_limit, per_month_raw * coverage_months)
    progress["samples_strategy"] = dict(samples_strategy)
    progress["samples_limit"] = samples_limit
    try:
        samples = await _retry_analytics(
            "samples",
            lambda: archive_client.watch_history_analytics_samples(
                sec_user_id=sec_user_id,
                range=range_spec,
                time_zone=tz_name,
                strategy=samples_strategy,
                limit=samples_limit,
                max_chars_per_item=300,
                fields=["title", "description", "hashtags", "music", "author"],
                include_video_id=True,
                include_watched_at=True,
            ),
            attempts=8,
        )
    except Exception as exc:
        logger.warning("Watch history analytics samples failed run=%s sec_user_id=%s err=%s", run_id, sec_user_id, exc)
        progress["status"] = "failed"
        progress["error"] = "analytics_samples_failed"
        progress["error_detail"] = _exc_details(exc)
        progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
        with SessionLocal() as db:
            run = db.get(AppWrappedRun, run_id)
            if run:
                payload = run.payload if isinstance(run.payload, dict) else {}
                payload["_watch_history_progress"] = progress
                run.payload = payload
                db.add(run)
                db.commit()
        return False

    try:
        totals = summary["totals"]
        night = summary["night"]
        total_videos = int(totals["videos"])
        total_hours = float(totals["watch_hours"])
        night_pct = float(night["watch_pct"])
        peak_hour_val = summary.get("peak_hour")
        peak_hour = int(peak_hour_val) if peak_hour_val is not None else None
        hour_histogram_seconds = summary.get("hour_histogram_seconds")

        top_music: Dict[str, Any] = {"name": "", "count": 0}
        top_music_items = summary.get("top_music") or []
        if top_music_items:
            first = top_music_items[0]
            top_music = {"name": str(first["music"] or ""), "count": int(first["video_count"] or 0)}

        top_creators: List[str] = []
        top_creator_items = summary.get("top_creators") or []
        for item in top_creator_items:
            if not isinstance(item, dict):
                continue
            author = item["author"] or item["author_id"]
            if author:
                top_creators.append(str(author))
        top_creators = top_creators[:5]
    except Exception as exc:
        logger.warning("Unexpected analytics summary shape run=%s sec_user_id=%s err=%s", run_id, sec_user_id, exc)
        progress["status"] = "failed"
        progress["error"] = "analytics_summary_shape"
        progress["error_detail"] = _exc_details(exc)
        progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
        with SessionLocal() as db:
            run = db.get(AppWrappedRun, run_id)
            if run:
                payload = run.payload if isinstance(run.payload, dict) else {}
                payload["_watch_history_progress"] = progress
                run.payload = payload
                db.add(run)
                db.commit()
        return False

    extrapolate_enabled = os.getenv("WRAPPED_EXTRAPOLATE_PARTIAL", "1") == "1"
    extrapolation_factor = float(target_months) / float(coverage_months) if coverage_months else 1.0
    extrapolated = bool(extrapolate_enabled and extrapolation_factor > 1.01)
    if extrapolated:
        observed_videos = total_videos
        observed_hours = total_hours
        total_videos = int(round(total_videos * extrapolation_factor))
        total_hours = float(total_hours * extrapolation_factor)
        if isinstance(top_music.get("count"), int):
            top_music["count"] = int(round(top_music["count"] * extrapolation_factor))
        logger.info(
            "watch_history.extrapolated",
            extra={
                "event": "watch_history.extrapolated",
                "wrapped_run_id": run_id,
                "sec_user_id": sec_user_id,
                "target_months": target_months,
                "coverage_months": coverage_months,
                "extrapolation_factor": round(extrapolation_factor, 4),
                "observed_videos": observed_videos,
                "projected_videos": total_videos,
                "observed_hours": observed_hours,
                "projected_hours": total_hours,
            },
        )

    total_hours = round(total_hours, 2)
    night_pct = round(night_pct, 2)

    try:
        sample_items = samples["items"]
        if not isinstance(sample_items, list):
            raise TypeError("samples.items must be a list")
    except Exception as exc:
        logger.warning("Unexpected analytics samples shape run=%s sec_user_id=%s err=%s", run_id, sec_user_id, exc)
        progress["status"] = "failed"
        progress["error"] = "analytics_samples_shape"
        progress["error_detail"] = _exc_details(exc)
        progress["updated_at"] = _iso_utc(datetime.now(timezone.utc))
        with SessionLocal() as db:
            run = db.get(AppWrappedRun, run_id)
            if run:
                payload = run.payload if isinstance(run.payload, dict) else {}
                payload["_watch_history_progress"] = progress
                run.payload = payload
                db.add(run)
                db.commit()
        return False

    sample_texts: List[str] = []
    source_spans: List[Dict[str, Any]] = []
    for item in sample_items:
        if not isinstance(item, dict):
            continue
        text_val = str(item["text"] or "").strip()
        if text_val:
            sample_texts.append(text_val)
        vid = item.get("video_id")
        if vid:
            source_spans.append({"video_id": str(vid), "reason": "sample"})
    sample_texts = sample_texts[:sample_limit]
    source_spans = source_spans[:200]

    if total_videos == 0:
        logger.warning("No rows available in analytics for run %s", run_id)
    cat_name = _pick_cat_name(run_id or run_app_user_id or sec_user_id or "cat")
    scroll_time = _build_scroll_time(hour_histogram_seconds=hour_histogram_seconds, peak_hour=peak_hour)
    analogy_line = _build_analogy_line(total_videos)

    with SessionLocal() as db:
        run = db.get(AppWrappedRun, run_id)
        if not run:
            return True
        user = db.get(AppUser, run.app_user_id) if run.app_user_id else None
        payload = run.payload if isinstance(run.payload, dict) else {}
        payload.update(
            {
                "total_hours": total_hours,
                "total_videos": total_videos,
                "night_pct": night_pct,
                "peak_hour": peak_hour,
                "top_music": top_music,
                "top_creators": top_creators,
                "cat_name": cat_name,
                "analogy_line": analogy_line,
                "scroll_time": scroll_time,
                "platform_username": user.platform_username if user else None,
                "email": run.email,
                "source_spans": source_spans,
                "_sample_texts": sample_texts,
                "accessory_set": accessories.select_accessory_set(),
            }
        )
        if isinstance(payload.get("_watch_history_progress"), dict):
            payload["_watch_history_progress"]["status"] = "succeeded"
            payload["_watch_history_progress"]["updated_at"] = _iso_utc(datetime.now(timezone.utc))
            payload["_watch_history_progress"]["coverage"] = {
                "watched_at_min": _iso_utc(watched_at_min_dt) if watched_at_min_dt else None,
                "watched_at_max": _iso_utc(watched_at_max_dt) if watched_at_max_dt else None,
                "target_months": target_months,
                "coverage_months": coverage_months,
                "extrapolated": extrapolated,
                "extrapolation_factor": round(extrapolation_factor, 4) if extrapolated else 1.0,
            }
        data_jobs = payload.get("data_jobs") or {}
        data_jobs["watch_history"] = {"id": job.id, "status": "succeeded"}
        payload["data_jobs"] = data_jobs
        payload.pop("_acc_state", None)
        payload.pop("_bucket_summaries", None)
        payload.pop("_months_done", None)
        run.payload = payload
        run.status = "pending"
        db.add(run)
        db.commit()
        logger.info("watch_history_fetch_2025 done run=%s total_videos=%s", run_id, total_videos)

        job_queue.enqueue(
            db,
            task_name="wrapped_analysis",
            payload={
                "wrapped_run_id": run.id,
                "force_new_email": bool(job.payload.get("force_new_email")),
            },
            idempotency_key=f"analysis:{run.id}",
            force_new=bool(job.payload.get("force_new_analysis")),
        )
    return True


async def handle_wrapped_analysis(job: LeasedJob) -> bool:
    wrapped_run_id = job.payload.get("wrapped_run_id")
    if not wrapped_run_id:
        return True

    with SessionLocal() as db:
        run = db.get(AppWrappedRun, wrapped_run_id)
        if not run or not isinstance(run.payload, dict):
            return True
        if run.status == "ready":
            return True
        user = db.get(AppUser, run.app_user_id) if run.app_user_id else None
        run_id = run.id
        run_app_user_id = run.app_user_id or ""
        watch_history_job_id = run.watch_history_job_id or ""
        sec_user_id = run.sec_user_id or (user.latest_sec_user_id if user else None)
        tz_name = str(_safe_zone(user.time_zone)) if user else None
        payload: Dict[str, Any] = dict(run.payload or {})
        raw_samples = payload.get("_sample_texts") or []
        sample_texts = [str(x) for x in raw_samples if str(x).strip()] if isinstance(raw_samples, list) else []

    required_watch_fields = ("total_hours", "total_videos", "night_pct", "top_music", "top_creators")
    missing_watch_fields = [k for k in required_watch_fields if payload.get(k) is None]
    if missing_watch_fields:
        logger.warning(
            "wrapped_analysis.missing_watch_history_metrics",
            extra={
                "event": "wrapped_analysis.missing_watch_history_metrics",
                "wrapped_run_id": wrapped_run_id,
                "sec_user_id": sec_user_id,
                "missing": missing_watch_fields,
            },
        )
        with SessionLocal() as db:
            run = db.get(AppWrappedRun, wrapped_run_id)
            if run and isinstance(run.payload, dict):
                payload_db = run.payload
                warnings = payload_db.get("_analysis_warnings")
                warning_list = [str(x) for x in warnings] if isinstance(warnings, list) else []
                warning_list.append("missing_watch_history_metrics")
                payload_db["_analysis_warnings"] = sorted(set(warning_list))
                data_jobs = payload_db.get("data_jobs") or {}
                data_jobs["wrapped_analysis"] = {
                    "id": job.id,
                    "status": "failed",
                    "error": "missing_watch_history_metrics",
                    "missing": missing_watch_fields,
                }
                payload_db["data_jobs"] = data_jobs
                run.payload = payload_db
                if run.status == "ready":
                    run.status = "pending"
                db.add(run)
                db.commit()
        return False

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    if not api_key or not model:
        logger.error("OpenRouter config missing; cannot run wrapped_analysis for run %s", run_id)
        with SessionLocal() as db:
            run = db.get(AppWrappedRun, wrapped_run_id)
            if not run:
                return True
            payload_db = run.payload if isinstance(run.payload, dict) else {}
            data_jobs = payload_db.get("data_jobs") or {}
            data_jobs["wrapped_analysis"] = {"id": job.id, "status": "failed", "error": "missing_openrouter_config"}
            payload_db["data_jobs"] = data_jobs
            run.payload = payload_db
            run.status = "failed"
            db.add(run)
            db.commit()
        return True

    range_spec: Dict[str, Any] = {"type": "last_n_months", "months": max(1, int(os.getenv("WATCH_HISTORY_MONTH_LIMIT", "12")))}
    progress = payload.get("_watch_history_progress")
    if isinstance(progress, dict):
        raw_range = progress.get("range")
        if isinstance(raw_range, dict):
            if (
                raw_range.get("type") == "between"
                and isinstance(raw_range.get("start_at"), str)
                and isinstance(raw_range.get("end_at"), str)
            ):
                range_spec = {"type": "between", "start_at": raw_range["start_at"], "end_at": raw_range["end_at"]}
            elif raw_range.get("type") == "last_n_months":
                with suppress(Exception):
                    months_val = int(raw_range.get("months"))
                    if months_val > 0:
                        range_spec = {"type": "last_n_months", "months": months_val}

    sample_source = "payload"
    if not sample_texts and sec_user_id:
        try:
            months_hint = 12
            if range_spec.get("type") == "last_n_months":
                with suppress(Exception):
                    months_hint = max(1, int(range_spec.get("months") or 12))
            elif range_spec.get("type") == "between":
                start_dt = _parse_iso_datetime(range_spec.get("start_at"))
                end_dt = _parse_iso_datetime(range_spec.get("end_at"))
                if start_dt and end_dt:
                    months_hint = _months_between(start_dt, end_dt)

            sample_limit_env = int(os.getenv("WATCH_HISTORY_SAMPLES_LIMIT", "200"))
            sample_limit = max(1, min(sample_limit_env, 200))
            per_month_raw = max(1, sample_limit // months_hint)
            if per_month_raw > 50:
                strategy: Dict[str, Any] = {"type": "recent"}
                limit = sample_limit
            else:
                strategy = {"type": "per_month", "per_month": per_month_raw}
                limit = min(sample_limit, per_month_raw * months_hint, 200)

            samples = await archive_client.watch_history_analytics_samples(
                sec_user_id=sec_user_id,
                range=range_spec,
                time_zone=tz_name,
                strategy=strategy,
                limit=limit,
                max_chars_per_item=300,
                fields=["title", "description", "hashtags", "music", "author"],
                include_video_id=False,
                include_watched_at=False,
            )
            items = samples.get("items") if isinstance(samples, dict) else None
            if isinstance(items, list):
                fetched = [str(it.get("text") or "").strip() for it in items if isinstance(it, dict)]
                sample_texts = [t for t in fetched if t]
                sample_source = "archive"
        except Exception as exc:
            logger.warning("wrapped_analysis.fetch_samples_failed run=%s err=%s", run_id, exc)

    debug_enabled = os.getenv("WRAPPED_ANALYSIS_DEBUG") == "1" or run_app_user_id.startswith("admin-test-")
    analysis_debug: Optional[Dict[str, Any]] = None
    if debug_enabled:
        analysis_debug = {
            "started_at": _iso_utc(datetime.now(timezone.utc)),
            "model": model,
            "sample_source": sample_source,
            "sample_count": len(sample_texts),
            "sample_total_chars": sum(len(s) for s in sample_texts),
            "steps": {},
        }
        with SessionLocal() as db:
            run = db.get(AppWrappedRun, wrapped_run_id)
            if run and isinstance(run.payload, dict):
                payload_db = run.payload
                payload_db["_analysis_debug"] = analysis_debug
                data_jobs = payload_db.get("data_jobs") or {}
                data_jobs["wrapped_analysis"] = {"id": job.id, "status": "running"}
                payload_db["data_jobs"] = data_jobs
                run.payload = payload_db
                db.add(run)
                db.commit()

    warnings: List[str] = []
    retry_needed = False
    retry_reasons: List[str] = []

    # Brainrot score (0..100): computed via TokyoData interests summary (cached). LLM fallback if unavailable.
    existing_watch_id = payload.get("_brainrot_watch_history_job_id")
    if "brain_rot_score" in payload and isinstance(existing_watch_id, str) and existing_watch_id == watch_history_job_id:
        payload["brain_rot_score"] = _clamp_0_100(payload.get("brain_rot_score"))
    elif sec_user_id:
        try:
            brainrot_result = await _compute_brainrot_score(
                sec_user_id=str(sec_user_id),
                range_spec=range_spec,
                time_zone=tz_name,
                payload=payload,
                debug_enabled=debug_enabled,
            )
        except Exception as exc:
            if debug_enabled:
                payload["_brainrot_interests_error"] = {"stage": "compute", "error": str(exc)}
            warnings.append("brain_rot_score_compute")
            brainrot_result = {"ok": False, "pending": False, "used_fallback_llm": True, "fields": {}}
        payload.update(brainrot_result.get("fields") or {})
        if watch_history_job_id:
            payload["_brainrot_watch_history_job_id"] = watch_history_job_id
        if brainrot_result.get("used_fallback_llm"):
            llm_score_raw = await _call_llm(BRAINROT_SCORE_PROMPT, sample_texts, temperature=0.2)
            llm_score = _extract_first_int(llm_score_raw)
            if llm_score is None:
                warnings.append("brain_rot_score_parse")
                retry_reasons.append("brain_rot_score_parse")
                llm_score = 0
            payload["brain_rot_score"] = _clamp_0_100(llm_score)
            payload["_brainrot_source"] = "llm_fallback"
            if watch_history_job_id:
                payload["_brainrot_watch_history_job_id"] = watch_history_job_id
    if "brain_rot_score" not in payload:
        warnings.append("brain_rot_score_missing")
        payload["brain_rot_score"] = 0

    prompts = [
        ("personality_type", PERSONALITY_PROMPT, "llm_personality"),
        ("personality_explanation", PERSONALITY_EXPLANATION_PROMPT, "llm_personality_explanation"),
        ("niche_journey", NICHE_JOURNEY_PROMPT, "llm_niche_journey"),
        ("top_niche_percentile", TOP_NICHES_PROMPT, "llm_top_niche_percentile"),
        ("brain_rot_explanation", BRAINROT_EXPLANATION_PROMPT, "llm_brainrot_explanation"),
        ("keyword_2026", KEYWORD_2026_PROMPT, "llm_keyword_2026"),
        ("thumb_roast", ROAST_THUMB_PROMPT, "llm_thumb_roast"),
    ]
    try:
        for field, prompt, task_name in prompts:
            step_meta: Dict[str, Any] = {}
            effective_prompt = prompt
            if task_name == "llm_personality_explanation":
                personality = payload.get("personality_type")
                if isinstance(personality, str) and personality.strip():
                    effective_prompt = f"{prompt}\n\nPersonality name: {personality.strip()}\nExplain why this exact name fits."
            if task_name == "llm_brainrot_explanation":
                score_val = payload.get("brain_rot_score")
                if isinstance(score_val, int):
                    effective_prompt = f"{prompt}\n\nBrainrot score: {score_val}"
            result = await _call_llm(effective_prompt, sample_texts, debug_meta=step_meta if debug_enabled else None)
            step_warnings: List[str] = []
            if analysis_debug is not None:
                preview = (result or "")[:500]
                analysis_debug["steps"][task_name] = {
                    "field": field,
                    "meta": step_meta,
                    "result_preview": preview,
                    "result_truncated": len(result or "") > len(preview),
                }
            if task_name == "llm_niche_journey":
                def _parse_niche_journey(value: str) -> List[str]:
                    parsed_val = _extract_json_value(value)
                    if isinstance(parsed_val, dict):
                        if isinstance(parsed_val.get("niche_journey"), list):
                            parsed_val = parsed_val.get("niche_journey")
                        elif isinstance(parsed_val.get("themes"), list):
                            parsed_val = parsed_val.get("themes")
                    if isinstance(parsed_val, list):
                        return [str(x).strip() for x in parsed_val if str(x).strip()][:5]
                    # Fallback: accept plain text lists (bullets/lines/commas) to avoid throwing away usable output.
                    cleaned = _strip_code_fences(value or "").strip()
                    if not cleaned:
                        return []
                    parts: List[str] = []
                    for line in cleaned.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        line = re.sub(r"^[\s\-•*\d)+.]+", "", line).strip()
                        if not line:
                            continue
                        parts.extend([p.strip() for p in line.split(",") if p.strip()])
                    return parts[:5]

                out = _parse_niche_journey(result)
                if len(out) < 5:
                    strict_prompt = (
                        f"{prompt}\n\n"
                        "Return ONLY valid JSON (no markdown, no commentary): "
                        "a JSON array of exactly 5 short niche phrases (strings)."
                    )
                    for temp in (0.0, 0.0):
                        retry_meta: Dict[str, Any] = {}
                        retry_result = await _call_llm(
                            strict_prompt,
                            sample_texts,
                            debug_meta=retry_meta if debug_enabled else None,
                            temperature=temp,
                        )
                        out = _parse_niche_journey(retry_result)
                        if len(out) >= 5:
                            break
                if len(out) < 5:
                    warnings.append("niche_journey_parse")
                    step_warnings.append("niche_journey_parse")
                    retry_needed = True
                    retry_reasons.append("niche_journey_parse")
                payload[field] = out[:5]
                if analysis_debug is not None:
                    analysis_debug["steps"][task_name]["parsed_count"] = len(out)
            elif task_name == "llm_top_niche_percentile":
                def _parse_top_niches_and_percentile(value: str) -> tuple[List[str], Optional[str]]:
                    data_val = _extract_json_value(value)
                    top_niches_val: List[str] = []
                    top_pct_val: Optional[str] = None
                    if isinstance(data_val, dict):
                        tn = data_val.get("top_niches")
                        if isinstance(tn, list):
                            top_niches_val = [str(x).strip() for x in tn if str(x).strip()]
                        pct = (
                            data_val.get("top_niche_percentile")
                            or data_val.get("percentile")
                            or data_val.get("top_percentile")
                            or data_val.get("niche_percentile")
                        )
                        if pct:
                            if isinstance(pct, (int, float)):
                                top_pct_val = _extract_percentile(f"top {pct}%")
                            else:
                                top_pct_val = _extract_percentile(str(pct).strip()) or _extract_percentile(value or "")
                    if not top_pct_val:
                        top_pct_val = _extract_percentile(value or "")
                    return top_niches_val[:5], top_pct_val

                top_niches, top_pct = _parse_top_niches_and_percentile(result)
                if not top_niches or not top_pct:
                    strict_prompt = prompt
                    last_llm_raw = result or ""
                    for temp in (0.0, 0.2):
                        retry_meta: Dict[str, Any] = {}
                        retry_result = await _call_llm(
                            strict_prompt,
                            sample_texts,
                            debug_meta=retry_meta if debug_enabled else None,
                            temperature=temp,
                        )
                        last_llm_raw = retry_result or last_llm_raw
                        top_niches, top_pct = _parse_top_niches_and_percentile(retry_result)
                        if top_niches and top_pct:
                            break
                if not top_niches:
                    warnings.append("top_niches_parse")
                    step_warnings.append("top_niches_parse")
                    retry_needed = True
                    retry_reasons.append("top_niches_parse")
                if not top_pct:
                    raw_source = last_llm_raw if "last_llm_raw" in locals() else (result or "")
                    raw_text, raw_truncated = sanitize_json_text(raw_source, max_chars=4000)
                    log_fields = {
                        "wrapped_run_id": wrapped_run_id,
                        "job_id": job.id,
                        "openrouter_model": model,
                        "llm_raw": raw_text,
                        "llm_raw_truncated": raw_truncated,
                        "llm_parsed_preview": _extract_json_value(raw_source),
                    }
                    logger.error(
                        "wrapped_analysis.top_niche_percentile_parse",
                        extra={
                            "event": "wrapped_analysis.top_niche_percentile_parse",
                            **log_fields,
                        },
                    )
                    warnings.append("top_niche_percentile_parse")
                    step_warnings.append("top_niche_percentile_parse")
                    retry_needed = True
                    retry_reasons.append("top_niche_percentile_parse")

                payload["top_niches"] = top_niches[:5]
                payload["top_niche_percentile"] = top_pct
                if analysis_debug is not None:
                    analysis_debug["steps"][task_name]["parsed"] = {
                        "top_niches": payload.get("top_niches"),
                        "top_niche_percentile": payload.get("top_niche_percentile"),
                    }
            elif task_name == "llm_personality":
                label = _normalize_title_phrase(result or "", max_words=4)
                if not label:
                    warnings.append("personality_type_parse")
                    step_warnings.append("personality_type_parse")
                    label = "Unknown"
                payload[field] = label
                if analysis_debug is not None:
                    analysis_debug["steps"][task_name]["parsed"] = label
            elif task_name == "llm_keyword_2026":
                keyword = _normalize_title_phrase(result or "", max_words=4)
                if not keyword:
                    warnings.append("keyword_2026_parse")
                    step_warnings.append("keyword_2026_parse")
                payload[field] = keyword
                if analysis_debug is not None:
                    analysis_debug["steps"][task_name]["parsed"] = keyword
            else:
                if not result:
                    warnings.append(f"{task_name}_empty")
                    step_warnings.append(f"{task_name}_empty")
                payload[field] = result or ""

            if analysis_debug is not None and step_warnings:
                analysis_debug["steps"][task_name]["warnings"] = step_warnings
        items = accessories.load_items()
        grouped = accessories.build_accessory_lookup(items)
        head_options = sorted({row.get("internal_name", "") for row in grouped.get("Head", []) if row.get("internal_name")})
        body_options = sorted({row.get("internal_name", "") for row in grouped.get("Body", []) if row.get("internal_name")})
        other_options = sorted({row.get("internal_name", "") for row in grouped.get("Other", []) if row.get("internal_name")})

        allowed = {
            "cat_names": CAT_NAMES,
            "head": head_options,
            "body": body_options,
            "other": other_options,
        }
        accessory_prompt = f"{ACCESSORY_SET_PROMPT}\nAllowed JSON:\n{json.dumps(allowed, ensure_ascii=False)}"
        acc_step_warnings: List[str] = []
        used_random_fallback = False
        accessory_attempts: List[Dict[str, Any]] = []

        fallback_cat = payload.get("cat_name")
        if fallback_cat not in CAT_NAMES:
            fallback_cat = _pick_cat_name(run_id or run_app_user_id or "cat")

        def _norm(val: Any) -> Optional[str]:
            if not isinstance(val, str):
                return None
            v = val.strip()
            return v if v else None

        def _parse_accessory_choice(acc_data: Any) -> tuple[
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
        ]:
            cat_choice = None
            head_choice = None
            body_choice = None
            other_choice = None
            head_reason = None
            body_reason = None
            other_reason = None
            if isinstance(acc_data, dict):
                cat_choice = acc_data.get("cat_name") or acc_data.get("cat")
                head_choice = acc_data.get("head")
                body_choice = acc_data.get("body")
                other_choice = acc_data.get("other")
                if isinstance(head_choice, dict):
                    head_reason = head_choice.get("reason")
                    head_choice = head_choice.get("internal_name")
                if isinstance(body_choice, dict):
                    body_reason = body_choice.get("reason")
                    body_choice = body_choice.get("internal_name")
                if isinstance(other_choice, dict):
                    other_reason = other_choice.get("reason")
                    other_choice = other_choice.get("internal_name")
                if head_reason is None:
                    head_reason = acc_data.get("head_reason")
                if body_reason is None:
                    body_reason = acc_data.get("body_reason")
                if other_reason is None:
                    other_reason = acc_data.get("other_reason")
            return (
                _norm(cat_choice),
                _norm(head_choice),
                _norm(body_choice),
                _norm(other_choice),
                _norm(head_reason),
                _norm(body_reason),
                _norm(other_reason),
            )

        def _validate_accessory_choice(
            cat_choice: Optional[str],
            head_choice: Optional[str],
            body_choice: Optional[str],
            other_choice: Optional[str],
        ) -> List[str]:
            invalid: List[str] = []
            if cat_choice not in CAT_NAMES:
                invalid.append("cat_name_invalid")
            if head_choice not in head_options:
                invalid.append("head_invalid")
            if body_choice not in body_options:
                invalid.append("body_invalid")
            if other_choice not in other_options:
                invalid.append("other_invalid")
            return invalid

        # Regenerate once if the first attempt doesn't validate.
        # If we still can't validate, fall back to random selection from items.csv.
        acc_result = None
        acc_meta: Dict[str, Any] = {}
        cat_choice = None
        head_choice = None
        body_choice = None
        other_choice = None
        head_reason = None
        body_reason = None
        other_reason = None
        for attempt_idx, temp in enumerate((0.2, 0.0), start=1):
            acc_meta = {}
            acc_result = await _call_llm(
                accessory_prompt,
                sample_texts,
                debug_meta=acc_meta if debug_enabled else None,
                temperature=temp,
            )
            acc_data = _extract_json_value(acc_result)
            (
                candidate_cat,
                candidate_head,
                candidate_body,
                candidate_other,
                candidate_head_reason,
                candidate_body_reason,
                candidate_other_reason,
            ) = _parse_accessory_choice(acc_data)
            invalid = _validate_accessory_choice(candidate_cat, candidate_head, candidate_body, candidate_other)
            accessory_attempts.append(
                {
                    "attempt": attempt_idx,
                    "temperature": temp,
                    "invalid": invalid,
                    "parsed": {
                        "cat_name": candidate_cat,
                        "head": candidate_head,
                        "body": candidate_body,
                        "other": candidate_other,
                    },
                }
            )
            if not invalid:
                cat_choice = candidate_cat
                head_choice = candidate_head
                body_choice = candidate_body
                other_choice = candidate_other
                head_reason = candidate_head_reason
                body_reason = candidate_body_reason
                other_reason = candidate_other_reason
                break

        if acc_result is None:
            acc_result = ""

        if analysis_debug is not None:
            preview = (acc_result or "")[:500]
            analysis_debug["steps"]["llm_accessory_set"] = {
                "field": "accessory_set",
                "meta": acc_meta,
                "result_preview": preview,
                "result_truncated": len(acc_result or "") > len(preview),
                "attempts": accessory_attempts,
            }

        invalid_final = _validate_accessory_choice(cat_choice, head_choice, body_choice, other_choice)
        if invalid_final:
            used_random_fallback = True
            acc_step_warnings.append("accessory_set_fallback")
            fallback_accessories = accessories.select_accessory_set()
            cat_choice = fallback_cat
            head_choice = fallback_accessories.get("head", {}).get("internal_name") or "unknown"
            body_choice = fallback_accessories.get("body", {}).get("internal_name") or "unknown"
            other_choice = fallback_accessories.get("other", {}).get("internal_name") or "unknown"
            head_reason = None
            body_reason = None
            other_reason = None

        def _clean_reason(reason: Optional[str]) -> Optional[str]:
            if not reason:
                return None
            cleaned = _strip_code_fences(str(reason)).strip()
            if not cleaned:
                return None
            cleaned = cleaned.splitlines()[0].strip()
            cleaned = cleaned.replace("_", " ")
            cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            words = cleaned.split()
            if len(words) < 3:
                return None
            words = words[:3]
            if any(len(w) > 9 for w in words):
                return None
            return " ".join(w.title() for w in words)

        head_reason = _clean_reason(head_reason)
        body_reason = _clean_reason(body_reason)
        other_reason = _clean_reason(other_reason)

        force_reasons = False

        if not head_reason:
            head_reason = None
            force_reasons = True
        if not body_reason:
            body_reason = None
            force_reasons = True
        if not other_reason:
            other_reason = None
            force_reasons = True

        def _first_short_words(text: Any, max_words: int) -> List[str]:
            phrase = _normalize_title_phrase(str(text or ""), max_words=max_words)
            out: List[str] = []
            for w in phrase.split():
                if 1 <= len(w) <= 9:
                    out.append(w)
            return out[:max_words]

        def _fallback_reason(slot: str, internal_name: str) -> str:
            hints = {
                "scroll_time": payload.get("scroll_time"),
                "night_pct": payload.get("night_pct"),
                "peak_hour": payload.get("peak_hour"),
                "top_music": payload.get("top_music"),
                "top_creators": payload.get("top_creators"),
                "top_niches": payload.get("top_niches"),
            }
            return accessory_fallback_reason(seed=str(run_id), slot=slot, internal_name=internal_name, hints=hints)

        needs_reason_llm = sample_texts and (force_reasons or not (head_reason and body_reason and other_reason))
        if needs_reason_llm:
            reasons_meta: Dict[str, Any] = {}
            chosen = {"head": str(head_choice), "body": str(body_choice), "other": str(other_choice)}
            personality_hint = payload.get("personality_type")
            top_niches_hint = payload.get("top_niches")
            hints = {
                "personality_type": personality_hint,
                "top_niches": top_niches_hint,
                "scroll_time": payload.get("scroll_time"),
                "night_pct": payload.get("night_pct"),
                "peak_hour": payload.get("peak_hour"),
            }
            reasons_prompt = (
                f"{ACCESSORY_REASONS_PROMPT}\nChosen internal_names:\n{json.dumps(chosen, ensure_ascii=False)}\n"
                f"Hints:\n{json.dumps(hints, ensure_ascii=False)}"
            )
            reasons_result = await _call_llm(
                reasons_prompt,
                sample_texts,
                debug_meta=reasons_meta if debug_enabled else None,
                temperature=0.4,
            )
            if analysis_debug is not None:
                preview = (reasons_result or "")[:500]
                analysis_debug["steps"]["llm_accessory_reasons"] = {
                    "field": "accessory_set.reasons",
                    "meta": reasons_meta,
                    "result_preview": preview,
                    "result_truncated": len(reasons_result or "") > len(preview),
                }
            reasons_data = _extract_json_value(reasons_result)
            if isinstance(reasons_data, dict):
                new_head = _clean_reason(_norm(reasons_data.get("head_reason")))
                new_body = _clean_reason(_norm(reasons_data.get("body_reason")))
                new_other = _clean_reason(_norm(reasons_data.get("other_reason")))
                if new_head:
                    head_reason = new_head
                if new_body:
                    body_reason = new_body
                if new_other:
                    other_reason = new_other
            if not head_reason:
                head_reason = _fallback_reason("head", str(head_choice))
            if not body_reason:
                body_reason = _fallback_reason("body", str(body_choice))
            if not other_reason:
                other_reason = _fallback_reason("other", str(other_choice))
        else:
            head_reason = head_reason or _fallback_reason("head", str(head_choice))
            body_reason = body_reason or _fallback_reason("body", str(body_choice))
            other_reason = other_reason or _fallback_reason("other", str(other_choice))

        payload["cat_name"] = cat_choice or fallback_cat
        payload["accessory_set"] = {
            "head": {"internal_name": str(head_choice), "reason": str(head_reason)},
            "body": {"internal_name": str(body_choice), "reason": str(body_reason)},
            "other": {"internal_name": str(other_choice), "reason": str(other_reason)},
        }
        if used_random_fallback:
            warnings.append("accessory_set_validation")
            if analysis_debug is not None:
                analysis_debug["steps"]["llm_accessory_set"]["warnings"] = acc_step_warnings
        if analysis_debug is not None:
            analysis_debug["steps"]["llm_accessory_set"]["parsed"] = {
                "cat_name": payload.get("cat_name"),
                "head": payload.get("accessory_set", {}).get("head"),
                "body": payload.get("accessory_set", {}).get("body"),
                "other": payload.get("accessory_set", {}).get("other"),
            }
    except Exception as exc:
        if analysis_debug is not None:
            analysis_debug["completed_at"] = _iso_utc(datetime.now(timezone.utc))
            tb, tb_truncated = sanitize_json_text(traceback.format_exc(), max_chars=4000)
            analysis_debug["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb,
                "traceback_truncated": tb_truncated,
            }
            if warnings:
                analysis_debug["warnings"] = warnings
        with SessionLocal() as db:
            run = db.get(AppWrappedRun, wrapped_run_id)
            if run and isinstance(run.payload, dict):
                payload_db = run.payload
                if warnings:
                    payload_db["_analysis_warnings"] = warnings
                if analysis_debug is not None:
                    payload_db["_analysis_debug"] = analysis_debug
                data_jobs = payload_db.get("data_jobs") or {}
                data_jobs["wrapped_analysis"] = {"id": job.id, "status": "failed", "error": type(exc).__name__}
                payload_db["data_jobs"] = data_jobs
                run.payload = payload_db
                db.add(run)
                db.commit()
        raise

    payload.pop("_sample_texts", None)
    payload.pop("_bucket_summaries", None)
    if warnings:
        logger.warning(
            "wrapped_analysis.fallback",
            extra={"event": "wrapped_analysis.fallback", "wrapped_run_id": run_id, "job_id": job.id, "warnings": warnings},
        )
        payload["_analysis_warnings"] = warnings
    if analysis_debug is not None:
        analysis_debug["completed_at"] = _iso_utc(datetime.now(timezone.utc))
        if warnings:
            analysis_debug["warnings"] = warnings
        payload["_analysis_debug"] = analysis_debug
    data_jobs = payload.get("data_jobs") or {}
    if retry_needed:
        data_jobs["wrapped_analysis"] = {
            "id": job.id,
            "status": "failed",
            "error": "llm_parse_retry",
            "retry_reasons": sorted(set(retry_reasons)),
        }
    else:
        data_jobs["wrapped_analysis"] = {"id": job.id, "status": "succeeded"}
    payload["data_jobs"] = data_jobs

    with SessionLocal() as db:
        run = db.get(AppWrappedRun, wrapped_run_id)
        if not run:
            return True

        if not run.email:
            latest_email = (
                db.query(AppUserEmail)
                .filter(AppUserEmail.app_user_id == run.app_user_id)
                .order_by(AppUserEmail.created_at.desc())
                .first()
            )
            if latest_email:
                run.email = latest_email.email
        if run.email:
            payload["email"] = run.email

        user = db.get(AppUser, run.app_user_id) if run.app_user_id else None
        if user and user.referred_by:
            existing_evt = (
                db.query(ReferralEvent)
                .filter(ReferralEvent.event_type == "completion", ReferralEvent.wrapped_run_id == run.id)
                .first()
            )
            if not existing_evt:
                ref = (
                    db.query(Referral)
                    .filter(Referral.referrer_app_user_id == user.referred_by)
                    .order_by(Referral.created_at.desc())
                    .first()
                )
                if ref:
                    ref.completions = (ref.completions or 0) + 1
                    evt = ReferralEvent(
                        id=str(uuid4()),
                        referrer_app_user_id=ref.referrer_app_user_id,
                        referred_app_user_id=user.app_user_id,
                        event_type="completion",
                        archive_job_id=None,
                        wrapped_run_id=run.id,
                        ip=None,
                        user_agent=None,
                    )
                    db.add(ref)
                    db.add(evt)

        run.payload = payload
        if not retry_needed:
            run.status = "ready"
        else:
            # Don't regress an already-ready run; just keep retrying analysis to fill missing fields.
            if run.status != "ready":
                run.status = "pending"
        db.add(run)
        db.commit()

        if retry_needed:
            return False

        job_queue.enqueue(
            db,
            task_name="email_send",
            payload={"wrapped_run_id": run.id, "app_user_id": run.app_user_id},
            idempotency_key=f"email:{run.id}",
            force_new=bool(job.payload.get("force_new_email")),
        )
    return True


async def handle_llm_task(job: LeasedJob, prompt: str, field: str, task_name: str) -> bool:
    wrapped_run_id = job.payload.get("wrapped_run_id")
    if not wrapped_run_id:
        return True

    with SessionLocal() as db:
        run = db.get(AppWrappedRun, wrapped_run_id)
        if not run or not isinstance(run.payload, dict):
            return True
        payload = run.payload or {}
        if field in payload and not (task_name == "llm_top_niche_percentile" and "top_niches" not in payload):
            return True
        raw_samples = payload.get("_sample_texts") or []
        sample_texts = [str(x) for x in raw_samples if str(x).strip()] if isinstance(raw_samples, list) else []

    effective_prompt = prompt
    if task_name == "llm_brainrot_explanation":
        score_val = payload.get("brain_rot_score")
        if isinstance(score_val, int):
            effective_prompt = f"{prompt}\n\nBrainrot score: {score_val}"
    result = await _call_llm(effective_prompt, sample_texts)

    updates: Dict[str, Any] = {}
    if task_name == "llm_brainrot":
        try:
            updates[field] = max(0, min(100, int(float(result.strip().split()[0]))))
        except Exception:
            return False
    elif task_name == "llm_niche_journey":
        try:
            parsed = json.loads(result)
            if not isinstance(parsed, list):
                return False
        except Exception:
            return False
        updates[field] = parsed[:5]
    elif task_name == "llm_top_niche_percentile":
        try:
            data = json.loads(result)
            if not isinstance(data, dict):
                return False
            tn = data.get("top_niches")
            if not isinstance(tn, list):
                return False
            updates["top_niches"] = [str(x).strip() for x in tn if str(x).strip()]
            pct = data.get("top_niche_percentile")
            if not pct:
                return False
            updates["top_niche_percentile"] = _extract_percentile(str(pct).strip()) or _extract_percentile(result or "")
            if not updates["top_niche_percentile"]:
                return False
        except Exception:
            return False
    elif task_name == "llm_personality":
        if not result:
            return False
        updates[field] = result.strip().split()[0].lower().replace(" ", "_")
    elif task_name == "llm_keyword_2026":
        if not result:
            return False
        updates[field] = result.strip().splitlines()[0]
    else:
        updates[field] = result

    with SessionLocal() as db:
        run = db.get(AppWrappedRun, wrapped_run_id)
        if not run:
            return True
        payload = run.payload if isinstance(run.payload, dict) else {}
        payload.update(updates)
        run.payload = payload
        db.add(run)
        db.commit()
    return True


async def handle_email_send(job: LeasedJob) -> bool:
    wrapped_run_id = job.payload.get("wrapped_run_id")
    if not wrapped_run_id:
        return True

    with SessionLocal() as db:
        run = db.get(AppWrappedRun, wrapped_run_id)
        if not run:
            return True
        if run.status != "ready":
            logger.info(
                "email_send.not_ready",
                extra={"event": "email_send.not_ready", "wrapped_run_id": wrapped_run_id, "run_status": run.status},
            )
            return False
        app_user_id = run.app_user_id
        platform_username = None
        if app_user_id:
            user = db.get(AppUser, app_user_id)
            platform_username = user.platform_username if user else None
        email = run.email
        if not email and app_user_id:
            latest_email = (
                db.query(AppUserEmail)
                .filter(AppUserEmail.app_user_id == app_user_id)
                .order_by(AppUserEmail.created_at.desc())
                .first()
            )
            if latest_email:
                email = latest_email.email
        if email and app_user_id:
            payload = run.payload if isinstance(run.payload, dict) else {}
            data_jobs = payload.get("data_jobs") or {}
            data_jobs["email_send"] = {"id": job.id, "status": "running"}
            payload["data_jobs"] = data_jobs
            run.payload = payload
            db.add(run)
            db.commit()
        else:
            return True

    frontend = os.getenv("FRONTEND_URL", "")
    subject, text_body, html_body = _get_emailer().format_wrapped_email(app_user_id, platform_username, frontend)
    resp = _get_emailer().send_email(email, subject, text_body, html_body)
    sent_ok = resp is not None

    # Delete watch history from archive after successful email send
    if sent_ok:
        with SessionLocal() as db:
            run = db.get(AppWrappedRun, wrapped_run_id)
            if run:
                user = db.get(AppUser, run.app_user_id) if run.app_user_id else None
                sec_user_id = run.sec_user_id or (user.latest_sec_user_id if user else None)
                if sec_user_id:
                    try:
                        delete_result = await archive_client.delete_watch_history(sec_user_id)
                        deleted_count = delete_result.get("deleted_count", 0)
                        logger.info(
                            "email_send.watch_history_deleted",
                            extra={
                                "event": "email_send.watch_history_deleted",
                                "wrapped_run_id": wrapped_run_id,
                                "sec_user_id": sec_user_id,
                                "deleted_count": deleted_count,
                            },
                        )
                    except httpx.HTTPStatusError as exc:
                        status_code = exc.response.status_code if exc.response is not None else None
                        if status_code == 404:
                            logger.warning(
                                "email_send.watch_history_delete_not_found",
                                extra={
                                    "event": "email_send.watch_history_delete_not_found",
                                    "wrapped_run_id": wrapped_run_id,
                                    "sec_user_id": sec_user_id,
                                },
                            )
                        else:
                            logger.warning(
                                "email_send.watch_history_delete_failed",
                                extra={
                                    "event": "email_send.watch_history_delete_failed",
                                    "wrapped_run_id": wrapped_run_id,
                                    "sec_user_id": sec_user_id,
                                    "status_code": status_code,
                                },
                            )
                    except Exception as exc:
                        logger.warning(
                            "email_send.watch_history_delete_exception",
                            extra={
                                "event": "email_send.watch_history_delete_exception",
                                "wrapped_run_id": wrapped_run_id,
                                "sec_user_id": sec_user_id,
                                "error": str(exc),
                            },
                        )

    with SessionLocal() as db:
        run = db.get(AppWrappedRun, wrapped_run_id)
        if not run:
            return True
        if sent_ok and not run.email:
            run.email = email
        payload = run.payload if isinstance(run.payload, dict) else {}
        data_jobs = payload.get("data_jobs") or {}
        if sent_ok:
            data_jobs["email_send"] = {"id": job.id, "status": "succeeded"}
        else:
            data_jobs["email_send"] = {"id": job.id, "status": "failed", "error": "send_failed"}
        payload["data_jobs"] = data_jobs
        run.payload = payload
        db.add(run)
        db.commit()
    return sent_ok


async def handle_weekly_report_analyze(job: LeasedJob) -> bool:
    """Analyze user's wrapped data and create/update weekly report."""
    app_user_id = job.payload.get("app_user_id")
    if not app_user_id:
        return True
    
    with SessionLocal() as db:
        user = db.get(AppUser, app_user_id)
        if not user:
            return True
        if not user.latest_sec_user_id:
            logger.warning(
                "weekly_report_analyze.missing_sec_user_id",
                extra={"event": "weekly_report_analyze.missing_sec_user_id", "app_user_id": app_user_id},
            )
            return True
        
        now = datetime.now(timezone.utc)
        week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)
        window_end = min(week_end, now)

        existing_report = (
            db.query(WeeklyReport)
            .filter(
                WeeklyReport.app_user_id == app_user_id,
                WeeklyReport.period_start.isnot(None),
                WeeklyReport.period_start >= week_start,
                WeeklyReport.period_start < week_end,
            )
            .order_by(WeeklyReport.updated_at.desc(), WeeklyReport.created_at.desc())
            .first()
        )
        if not existing_report:
            existing_report = (
                db.query(WeeklyReport)
                .filter(
                    WeeklyReport.app_user_id == app_user_id,
                    WeeklyReport.created_at >= week_start,
                )
                .order_by(WeeklyReport.updated_at.desc(), WeeklyReport.created_at.desc())
                .first()
            )
        
        if existing_report:
            # Update existing report
            report = existing_report
        else:
            # Create new report
            report = WeeklyReport(
                app_user_id=app_user_id,
                send_status="pending",
            )
        
        # 总结起止时间（周一 00:00 UTC 到下周一 00:00 UTC）
        report.period_start = week_start
        report.period_end = week_end

        async def _coverage_ok() -> bool:
            try:
                coverage = await archive_client.watch_history_analytics_coverage(sec_user_id=user.latest_sec_user_id)
            except Exception as exc:
                logger.warning(
                    "weekly_report_analyze.coverage_failed",
                    extra={"event": "weekly_report_analyze.coverage_failed", "app_user_id": app_user_id, "error": str(exc)},
                )
                return False
            watched_min = _parse_iso_datetime(coverage.get("watched_at_min")) if isinstance(coverage, dict) else None
            watched_max = _parse_iso_datetime(coverage.get("watched_at_max")) if isinstance(coverage, dict) else None
            if not watched_min or not watched_max:
                return False
            grace_hours = float(os.getenv("WEEKLY_REPORT_COVERAGE_GRACE_HOURS", "0") or 0)
            grace = timedelta(hours=max(0.0, grace_hours))
            return watched_min <= week_start and watched_max >= (window_end - grace)

        async def _fetch_until_week_start() -> None:
            max_jobs = max(1, int(os.getenv("WEEKLY_REPORT_FETCH_MAX_DATA_JOBS", "8")))
            cursor_ms = None
            target_ms = int(week_start.timestamp() * 1000)
            completed = 0
            while completed < max_jobs:
                start_resp = await archive_client.start_watch_history(
                    sec_user_id=user.latest_sec_user_id,
                    limit=WATCH_HISTORY_PAGE_LIMIT,
                    max_pages=WATCH_HISTORY_MAX_PAGES,
                    cursor=str(cursor_ms) if cursor_ms is not None else None,
                )
                if start_resp.status_code != 202:
                    if start_resp.status_code in (400, 401, 403, 404):
                        return
                    await asyncio.sleep(1)
                    continue
                with suppress(Exception):
                    start_data = start_resp.json()
                data_job_id = start_data.get("data_job_id") if isinstance(start_data, dict) else None
                if not data_job_id:
                    return

                fin_ok = False
                fin_data: Dict[str, Any] = {}
                backoff = 1.0
                for _ in range(60):
                    fin = await archive_client.finalize_watch_history(data_job_id=data_job_id, include_rows=False)
                    if fin.status_code == 202:
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 8.0)
                        continue
                    if fin.status_code == 200:
                        with suppress(Exception):
                            fin_data = fin.json()
                        fin_ok = True
                    break

                if not fin_ok:
                    return

                completed += 1
                pagination = fin_data.get("pagination", {}) if isinstance(fin_data, dict) else {}
                next_cursor = pagination.get("next_cursor")
                has_more = pagination.get("has_more")
                next_cursor_ms = None
                with suppress(Exception):
                    next_cursor_ms = int(next_cursor) if next_cursor is not None else None
                if not next_cursor_ms or has_more is False or next_cursor_ms <= target_ms:
                    return
                cursor_ms = next_cursor_ms

        if not await _coverage_ok():
            await _fetch_until_week_start()
            if not await _coverage_ok():
                logger.info(
                    "weekly_report_analyze.coverage_incomplete",
                    extra={"event": "weekly_report_analyze.coverage_incomplete", "app_user_id": app_user_id},
                )
                return True

        range_spec: Dict[str, Any] = {"type": "between", "start_at": _iso_utc(week_start), "end_at": _iso_utc(window_end)}
        summary: Optional[Dict[str, Any]] = None
        try:
            summary = await archive_client.watch_history_analytics_summary(
                sec_user_id=user.latest_sec_user_id,
                range=range_spec,
                time_zone=user.time_zone or "UTC",
                include_hour_histogram=False,
                top_creators_limit=5,
                top_music_limit=1,
            )
        except Exception as exc:
            logger.warning(
                "weekly_report_analyze.summary_failed",
                extra={"event": "weekly_report_analyze.summary_failed", "app_user_id": app_user_id, "error": str(exc)},
            )
            return True

        try:
            totals = summary.get("totals") if isinstance(summary, dict) else None
            total_videos = int(totals.get("videos") or 0) if isinstance(totals, dict) else 0
            total_hours = float(totals.get("watch_hours") or 0.0) if isinstance(totals, dict) else 0.0
        except Exception as exc:
            logger.warning(
                "weekly_report_analyze.summary_shape",
                extra={"event": "weekly_report_analyze.summary_shape", "app_user_id": app_user_id, "error": str(exc)},
            )
            return True

        report.total_videos = total_videos
        report.total_time = int(round(total_hours * 3600))
        report.timezone = user.time_zone or "UTC"

        # Calculate miles_scrolled (estimate: ~3 inches per video swipe, converted to miles)
        # 3 inches = 0.0000473485 miles, so videos * 0.0000473485
        report.miles_scrolled = int(round(total_videos * 0.0000473485 * 1000))  # in 1/1000 miles for precision

        # Get previous week's total_time for comparison
        prev_week_start = week_start - timedelta(days=7)
        prev_report = (
            db.query(WeeklyReport)
            .filter(
                WeeklyReport.app_user_id == app_user_id,
                WeeklyReport.period_start.isnot(None),
                WeeklyReport.period_start >= prev_week_start,
                WeeklyReport.period_start < week_start,
            )
            .order_by(WeeklyReport.created_at.desc())
            .first()
        )
        if prev_report and prev_report.total_time:
            report.pre_total_time = prev_report.total_time

        # Fetch sample data for LLM analysis
        sample_texts: List[str] = []
        try:
            samples = await archive_client.watch_history_analytics_samples(
                sec_user_id=user.latest_sec_user_id,
                range=range_spec,
                time_zone=user.time_zone or "UTC",
                strategy={"type": "recent"},
                limit=50,
                max_chars_per_item=300,
                fields=["title", "description", "hashtags", "music", "author"],
                include_video_id=False,
                include_watched_at=False,
            )
            items = samples.get("items") if isinstance(samples, dict) else None
            if isinstance(items, list):
                sample_texts = [str(it.get("text") or "").strip() for it in items if isinstance(it, dict)]
                sample_texts = [t for t in sample_texts if t]
        except Exception as exc:
            logger.warning(
                "weekly_report_analyze.samples_failed",
                extra={"event": "weekly_report_analyze.samples_failed", "app_user_id": app_user_id, "error": str(exc)},
            )

        # LLM analysis (only if we have sample data)
        if sample_texts:
            # 1. Feeding state analysis
            try:
                feeding_result = await _call_llm(WEEKLY_FEEDING_STATE_PROMPT, sample_texts, temperature=0.3)
                feeding_state = feeding_result.strip().lower()
                if feeding_state in ("curious", "excited", "cozy", "sleepy", "dizzy"):
                    report.feeding_state = feeding_state
            except Exception as exc:
                logger.warning(
                    "weekly_report_analyze.feeding_state_failed",
                    extra={"event": "weekly_report_analyze.feeding_state_failed", "app_user_id": app_user_id, "error": str(exc)},
                )

            # 2. Topics analysis
            try:
                topics_result = await _call_llm(WEEKLY_TOPICS_PROMPT, sample_texts, temperature=0.5)
                topics_data = _extract_json_value(topics_result)
                if isinstance(topics_data, list):
                    # Add placeholder pic_url for now (can be filled by frontend)
                    topics = []
                    for item in topics_data[:3]:
                        if isinstance(item, dict) and item.get("topic"):
                            topics.append({"topic": str(item["topic"]), "pic_url": ""})
                        elif isinstance(item, str):
                            topics.append({"topic": item, "pic_url": ""})
                    if topics:
                        report.topics = topics
            except Exception as exc:
                logger.warning(
                    "weekly_report_analyze.topics_failed",
                    extra={"event": "weekly_report_analyze.topics_failed", "app_user_id": app_user_id, "error": str(exc)},
                )

            # 3. Rabbit hole analysis
            try:
                rabbit_result = await _call_llm(WEEKLY_RABBIT_HOLE_PROMPT, sample_texts, temperature=0.3)
                rabbit_data = _extract_json_value(rabbit_result)
                if isinstance(rabbit_data, dict):
                    category = rabbit_data.get("category")
                    count = rabbit_data.get("count")
                    if category and isinstance(category, str):
                        report.rabbit_hole_category = category
                        if isinstance(count, (int, float)):
                            report.rabbit_hole_count = int(count)
            except Exception as exc:
                logger.warning(
                    "weekly_report_analyze.rabbit_hole_failed",
                    extra={"event": "weekly_report_analyze.rabbit_hole_failed", "app_user_id": app_user_id, "error": str(exc)},
                )

            # 4. Nudge text analysis
            try:
                nudge_result = await _call_llm(WEEKLY_NUDGE_PROMPT, sample_texts, temperature=0.7)
                nudge_text = nudge_result.strip()
                if nudge_text and len(nudge_text) <= 80:
                    report.nudge_text = nudge_text
            except Exception as exc:
                logger.warning(
                    "weekly_report_analyze.nudge_failed",
                    extra={"event": "weekly_report_analyze.nudge_failed", "app_user_id": app_user_id, "error": str(exc)},
                )

        # Call external Node.js service to generate email HTML content.
        node_url = (os.getenv("WEEKLY_REPORT_NODE_URL") or "").strip()
        if node_url:
            try:
                # Build params with all WeeklyReport fields
                node_params = {
                    "period_start": _iso_utc(week_start) if week_start else None,
                    "period_end": _iso_utc(week_end) if week_end else None,
                    "timezone": report.timezone,
                    "total_videos": report.total_videos,
                    "total_time": report.total_time,
                    "pre_total_time": report.pre_total_time,
                    "miles_scrolled": report.miles_scrolled,
                    "feeding_state": report.feeding_state,
                    "topics": report.topics,
                    "trend_name": report.trend_name,
                    "trend_type": report.trend_type,
                    "discovery_rank": report.discovery_rank,
                    "total_discoverers": report.total_discoverers,
                    "origin_niche_text": report.origin_niche_text,
                    "spread_end_text": report.spread_end_text,
                    "reach_start": report.reach_start,
                    "reach_end": report.reach_end,
                    "current_reach": report.current_reach,
                    "rabbit_hole_datetime": _iso_utc(report.rabbit_hole_datetime) if report.rabbit_hole_datetime else None,
                    "rabbit_hole_date": report.rabbit_hole_date,
                    "rabbit_hole_time": report.rabbit_hole_time,
                    "rabbit_hole_count": report.rabbit_hole_count,
                    "rabbit_hole_category": report.rabbit_hole_category,
                    "nudge_text": report.nudge_text,
                }
                # New API format: {"uid": app_user_id, "params": {...}}
                node_payload = {
                    "uid": app_user_id,
                    "params": node_params,
                }
                headers = {"Content-Type": "application/json"}
                node_token = (os.getenv("WEEKLY_REPORT_NODE_TOKEN") or "").strip()
                if node_token:
                    headers["Authorization"] = f"Bearer {node_token}"
                resp = httpx.post(node_url, json=node_payload, headers=headers, timeout=30.0)
                resp.raise_for_status()
                with suppress(Exception):
                    node_data = resp.json()
                    # Response format: {"data": {}, "html": "..."}
                    email_html = node_data.get("html") or node_data.get("email_content")
                    if isinstance(email_html, str) and email_html.strip():
                        report.email_content = email_html
            except Exception as exc:
                logger.warning(
                    "weekly_report_analyze.node_failed",
                    extra={"event": "weekly_report_analyze.node_failed", "app_user_id": app_user_id, "error": str(exc)},
                )

        db.add(report)
        db.commit()

        logger.info(
            "weekly_report_analyze.completed",
            extra={"event": "weekly_report_analyze.completed", "app_user_id": app_user_id, "report_id": report.id},
        )
    
    return True


async def handle_weekly_report_send(job: LeasedJob) -> bool:
    """Send weekly report email to user.
    
    Supports optional email_override in payload for testing (sends to that email instead of user's).
    """
    app_user_id = job.payload.get("app_user_id")
    if not app_user_id:
        return True
    
    # Check for email override (for testing purposes)
    email_override = job.payload.get("email_override")
    
    report_id = None
    email_address = None
    platform_username = None
    
    with SessionLocal() as db:
        user = db.get(AppUser, app_user_id)
        if not user:
            return True
        
        # Check if user is unsubscribed (skip this check if using email_override for testing)
        if not email_override and user.weekly_report_unsubscribed:
            logger.info(
                "weekly_report_send.unsubscribed",
                extra={"event": "weekly_report_send.unsubscribed", "app_user_id": app_user_id},
            )
            return True
        
        platform_username = user.platform_username
        
        # Get latest weekly report
        report = (
            db.query(WeeklyReport)
            .filter(WeeklyReport.app_user_id == app_user_id)
            .order_by(WeeklyReport.created_at.desc())
            .first()
        )
        if not report:
            logger.warning(
                "weekly_report_send.no_report",
                extra={"event": "weekly_report_send.no_report", "app_user_id": app_user_id},
            )
            return True
        
        report_id = report.id
        
        # Use email_override if provided, otherwise get user's email
        if email_override and isinstance(email_override, str) and email_override.strip():
            email_address = email_override.strip()
            logger.info(
                "weekly_report_send.using_override_email",
                extra={"event": "weekly_report_send.using_override_email", "app_user_id": app_user_id, "email": email_address},
            )
        else:
            # Get user email
            email = (
                db.query(AppUserEmail)
                .filter(AppUserEmail.app_user_id == app_user_id)
                .order_by(AppUserEmail.created_at.desc())
                .first()
            )
            if not email or not email.email:
                logger.warning(
                    "weekly_report_send.no_email",
                    extra={"event": "weekly_report_send.no_email", "app_user_id": app_user_id},
                )
                return True
            email_address = email.email
        
        # Update send status to pending
        report.send_status = "pending"
        db.add(report)
        db.commit()
    
    # Send email
    frontend = os.getenv("FRONTEND_URL", "")
    
    # Use email_content if available, otherwise create basic email
    with SessionLocal() as db:
        report = db.get(WeeklyReport, report_id)
        if not report:
            return False
        
        if report.email_content:
            html_body = report.email_content
            text_body = ""  # Could extract text from HTML if needed
            subject = "Your Weekly TikTok Report"
        else:
            # Fallback: create basic email
            subject, text_body, html_body = _get_emailer().format_wrapped_email(app_user_id, platform_username, frontend)
    
    resp = _get_emailer().send_email(email_address, subject, text_body, html_body)
    sent_ok = resp is not None
    
    # Update send status
    with SessionLocal() as db:
        report = db.get(WeeklyReport, report_id)
        if report:
            report.send_status = "sent" if sent_ok else "failed"
            db.add(report)
            db.commit()
    
    logger.info(
        "weekly_report_send.completed",
        extra={
            "event": "weekly_report_send.completed",
            "app_user_id": app_user_id,
            "report_id": report_id,
            "sent": sent_ok,
        },
    )
    
    return sent_ok


async def handle_job(job: LeasedJob) -> bool:
    """Dispatch jobs by task_name. Return True on success, False on retry."""
    set_service("worker")
    bind_context(
        job_id=job.id,
        task_name=job.task_name,
        wrapped_run_id=job.payload.get("wrapped_run_id"),
        app_user_id=job.payload.get("app_user_id"),
        sec_user_id=job.payload.get("sec_user_id"),
    )
    start = datetime.now(timezone.utc)
    logger.info(
        "job.start",
        extra={"event": "job.start", "job_id": job.id, "task_name": job.task_name},
    )
    try:
        if job.task_name == "watch_history_fetch_2025":
            return await handle_watch_history_fetch(job)
        if job.task_name == "xordi_finalize":
            return await handle_xordi_finalize(job)
        if job.task_name == "reauth_notify":
            return await handle_reauth_notify(job)
        if job.task_name == "wrapped_analysis":
            return await handle_wrapped_analysis(job)
        if job.task_name == "llm_personality":
            return await handle_llm_task(job, PERSONALITY_PROMPT, "personality_type", job.task_name)
        if job.task_name == "llm_personality_explanation":
            return await handle_llm_task(job, PERSONALITY_EXPLANATION_PROMPT, "personality_explanation", job.task_name)
        if job.task_name == "llm_niche_journey":
            return await handle_llm_task(job, NICHE_JOURNEY_PROMPT, "niche_journey", job.task_name)
        if job.task_name == "llm_top_niche_percentile":
            return await handle_llm_task(job, TOP_NICHES_PROMPT, "top_niche_percentile", job.task_name)
        if job.task_name == "llm_brainrot":
            return await handle_llm_task(job, BRAINROT_SCORE_PROMPT, "brain_rot_score", job.task_name)
        if job.task_name == "llm_brainrot_explanation":
            return await handle_llm_task(job, BRAINROT_EXPLANATION_PROMPT, "brain_rot_explanation", job.task_name)
        if job.task_name == "llm_keyword_2026":
            return await handle_llm_task(job, KEYWORD_2026_PROMPT, "keyword_2026", job.task_name)
        if job.task_name == "llm_thumb_roast":
            return await handle_llm_task(job, ROAST_THUMB_PROMPT, "thumb_roast", job.task_name)
        if job.task_name == "email_send":
            return await handle_email_send(job)
        if job.task_name == "watch_history_verify":
            return await handle_watch_history_verify(job)
        if job.task_name == "weekly_report_analyze":
            return await handle_weekly_report_analyze(job)
        if job.task_name == "weekly_report_send":
            return await handle_weekly_report_send(job)
        return True
    except Exception as exc:
        logger.exception(
            "job.exception",
            extra={"event": "job.exception", "job_id": job.id, "task_name": job.task_name},
        )
        return False
    finally:
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        logger.info(
            "job.end",
            extra={
                "event": "job.end",
                "job_id": job.id,
                "task_name": job.task_name,
                "duration_ms": duration_ms,
            },
        )
        clear_context("job_id", "task_name", "wrapped_run_id", "app_user_id", "sec_user_id")


def _lease_next_job(worker_id: str, lease_seconds: int) -> Optional[LeasedJob]:
    with SessionLocal() as db:
        leased = job_queue.lease(
            db,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            task_names=WORKER_TASK_ALLOW or None,
        )
        if not leased:
            return None
        payload = leased.payload if isinstance(leased.payload, dict) else {}
        return LeasedJob(id=leased.id, task_name=leased.task_name, payload=payload)


async def _lease_heartbeat(job_id: str, worker_id: str, interval_seconds: float, lease_seconds: int) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        with SessionLocal() as db:
            ok = job_queue.heartbeat(db, job_id=job_id, worker_id=worker_id, lease_seconds=lease_seconds)
        if not ok:
            logger.warning(
                "job.heartbeat_lost",
                extra={"event": "job.heartbeat_lost", "job_id": job_id, "worker_id": worker_id},
            )
            return


async def _run_leased_job(worker_id: str, job: LeasedJob, lease_seconds: int, heartbeat_seconds: float) -> None:
    heartbeat_task = asyncio.create_task(_lease_heartbeat(job.id, worker_id, heartbeat_seconds, lease_seconds))
    ok = False
    try:
        ok = await handle_job(job)
    except Exception:
        logger.exception(
            "job.runner.exception",
            extra={"event": "job.runner.exception", "job_id": job.id, "task_name": job.task_name},
        )
        ok = False
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task

    with SessionLocal() as db:
        if ok:
            job_queue.complete(db, job.id, worker_id=worker_id)
        else:
            job_queue.fail(db, job.id, retry_delay_seconds=lease_seconds, worker_id=worker_id)
            try:
                rec = db.get(AppJob, job.id)
                wrapped_run_id = job.payload.get("wrapped_run_id")
                if rec and rec.status == "failed" and isinstance(wrapped_run_id, str) and wrapped_run_id:
                    if _data_job_key_for_task(job.task_name):
                        _mark_wrapped_run_failed(db, wrapped_run_id, task_name=job.task_name, job_id=job.id)
            except Exception:
                logger.exception(
                    "job.terminal_fail.propagation_error",
                    extra={"event": "job.terminal_fail.propagation_error", "job_id": job.id, "task_name": job.task_name},
                )


async def run_worker(
    poll_interval: float = WORKER_POLL_INTERVAL,
    lease_seconds: int = WORKER_JOB_LEASE_SECONDS,
    job_concurrency: int = WORKER_JOB_CONCURRENCY,
    heartbeat_seconds: float = WORKER_JOB_HEARTBEAT_SECONDS,
) -> None:
    worker_id = f"worker-{os.getpid()}"
    logger.info(
        "worker.start",
        extra={
            "event": "worker.start",
            "worker_id": worker_id,
            "job_concurrency": job_concurrency,
            "lease_seconds": lease_seconds,
            "heartbeat_seconds": heartbeat_seconds,
        },
    )

    inflight: set[asyncio.Task] = set()
    while True:
        while len(inflight) < job_concurrency:
            leased = _lease_next_job(worker_id, lease_seconds)
            if not leased:
                break
            inflight.add(asyncio.create_task(_run_leased_job(worker_id, leased, lease_seconds, heartbeat_seconds)))

        if not inflight:
            await asyncio.sleep(poll_interval)
            continue

        done, inflight = await asyncio.wait(inflight, timeout=poll_interval, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                task.result()
            except Exception:
                logger.exception("worker.job_task_crash", extra={"event": "worker.job_task_crash", "worker_id": worker_id})


if __name__ == "__main__":
    asyncio.run(run_worker())

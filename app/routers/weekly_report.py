"""Weekly report routes."""
import os
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import AppUser, AppUserEmail, WeeklyReport
from app.schemas import (
    TopicItem,
    UnsubscribeRequest,
    UnsubscribeResponse,
    WeeklyReportResponse,
    WeeklyReportTestRequest,
    WeeklyReportTestResponse,
)
from app.services.job_queue import DBJobQueue
from app.settings import get_settings
from app.observability import get_logger

router = APIRouter()
job_queue = DBJobQueue()
settings = get_settings()
logger = get_logger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _latest_user_email(db: Session, app_user_id: str) -> Optional[str]:
    latest_email = (
        db.query(AppUserEmail)
        .filter(AppUserEmail.app_user_id == app_user_id)
        .order_by(AppUserEmail.created_at.desc())
        .first()
    )
    if latest_email and isinstance(latest_email.email, str):
        return latest_email.email
    return None


def require_weekly_token(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_weekly_token: Optional[str] = Header(None, alias="X-Weekly-Token"),
):
    expected = settings.weekly_token.get_secret_value() if settings.weekly_token else None
    if not expected:
        raise HTTPException(status_code=404, detail="not_found")
    
    # Check Authorization header (Bearer token) or X-Weekly-Token header
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_weekly_token:
        token = x_weekly_token.strip()
    
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_weekly_token")
    return True


@router.get(
    "/weekly-report/{app_user_id}",
    response_model=WeeklyReportResponse,
    tags=["weekly-report"],
)
async def get_weekly_report(
    app_user_id: str, db: Session = Depends(get_db)
) -> WeeklyReportResponse:
    report = (
        db.query(WeeklyReport)
        .filter(WeeklyReport.app_user_id == app_user_id)
        .order_by(WeeklyReport.created_at.desc())
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="not_found")
    
    # Convert topics JSON to list of TopicItem if present
    topics = None
    if report.topics:
        try:
            topics = [TopicItem(**item) if isinstance(item, dict) else item for item in report.topics]
        except Exception:
            topics = None
    
    return WeeklyReportResponse(
        id=report.id,
        app_user_id=report.app_user_id,
        email_content=report.email_content,
        period_start=report.period_start,
        period_end=report.period_end,
        created_at=report.created_at,
        updated_at=report.updated_at,
        send_status=report.send_status,
        feeding_state=report.feeding_state,
        trend_name=report.trend_name,
        trend_type=report.trend_type,
        discovery_rank=report.discovery_rank,
        total_discoverers=report.total_discoverers,
        origin_niche_text=report.origin_niche_text,
        spread_end_text=report.spread_end_text,
        reach_start=report.reach_start,
        reach_end=report.reach_end,
        current_reach=report.current_reach,
        total_videos=report.total_videos,
        total_time=report.total_time,
        pre_total_time=report.pre_total_time,
        miles_scrolled=report.miles_scrolled,
        topics=topics,
        timezone=report.timezone,
        rabbit_hole_datetime=report.rabbit_hole_datetime,
        rabbit_hole_date=report.rabbit_hole_date,
        rabbit_hole_time=report.rabbit_hole_time,
        rabbit_hole_count=report.rabbit_hole_count,
        rabbit_hole_category=report.rabbit_hole_category,
        nudge_text=report.nudge_text,
    )


@router.post(
    "/weekly-report/unsubscribe",
    response_model=UnsubscribeResponse,
    tags=["weekly-report"],
)
async def unsubscribe_weekly_report(
    payload: UnsubscribeRequest, db: Session = Depends(get_db)
) -> UnsubscribeResponse:
    user = db.get(AppUser, payload.app_user_id)
    if user:
        user.weekly_report_unsubscribed = True
        user.updated_at = datetime.utcnow()
        db.add(user)
        db.commit()
    # Return success even if user not found (idempotent)
    return UnsubscribeResponse(success=True, message="Unsubscribed")


@router.post(
    "/admin/cron/weekly-report-analyze",
    status_code=204,
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def cron_weekly_report_analyze(db: Session = Depends(get_db)) -> Response:
    """Cron endpoint to enqueue weekly report analysis jobs for eligible users.
    
    Eligible users: have sec_user_id and are not unsubscribed from weekly reports.
    """
    # Query users who have sec_user_id and are not unsubscribed
    users = (
        db.query(AppUser)
        .filter(
            AppUser.weekly_report_unsubscribed == False,
            AppUser.latest_sec_user_id.isnot(None),
            AppUser.latest_sec_user_id != "",
        )
        .all()
    )
    
    enqueued = 0
    for user in users:
        # Enqueue analysis job for each eligible user
        job_queue.enqueue(
            db,
            task_name="weekly_report_analyze",
            payload={"app_user_id": user.app_user_id},
            idempotency_key=f"weekly_report_analyze:{user.app_user_id}",
        )
        enqueued += 1
    
    logger.info(
        "cron.weekly_report_analyze.enqueued",
        extra={"event": "cron.weekly_report_analyze.enqueued", "enqueued": enqueued, "total_users": len(users)},
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/admin/cron/weekly-report-send",
    status_code=204,
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def cron_weekly_report_send(db: Session = Depends(get_db)) -> Response:
    """Cron endpoint to enqueue weekly report email send jobs for eligible users (runs on Monday)."""
    # Query users who have latest weekly report, verified email, and are not unsubscribed
    users = (
        db.query(AppUser)
        .filter(
            AppUser.weekly_report_unsubscribed == False,
        )
        .all()
    )
    
    enqueued = 0
    for user in users:
        # Check if user has latest weekly report
        latest_report = (
            db.query(WeeklyReport)
            .filter(WeeklyReport.app_user_id == user.app_user_id)
            .order_by(WeeklyReport.created_at.desc())
            .first()
        )
        if not latest_report:
            continue
        
        # Check if user has verified email
        email = _latest_user_email(db, user.app_user_id)
        if not email:
            continue
        
        # Enqueue send job
        job_queue.enqueue(
            db,
            task_name="weekly_report_send",
            payload={"app_user_id": user.app_user_id},
            idempotency_key=f"weekly_report_send:{user.app_user_id}",
        )
        enqueued += 1
    
    logger.info(
        "cron.weekly_report_send.enqueued",
        extra={"event": "cron.weekly_report_send.enqueued", "enqueued": enqueued, "total_users": len(users)},
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/admin/test/weekly-report/{app_user_id}",
    response_model=WeeklyReportTestResponse,
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
)
async def admin_test_weekly_report(
    app_user_id: str,
    payload: WeeklyReportTestRequest,
    db: Session = Depends(get_db),
) -> WeeklyReportTestResponse:
    """Test endpoint to run the complete weekly report flow for a single user.
    
    This endpoint:
    1. Validates the user exists and has sec_user_id
    2. Enqueues the weekly_report_analyze job
    3. If send_email is True, also enqueues the weekly_report_send job
    4. Returns the job information
    
    Args:
        payload.send_email: Whether to send the email (default True)
        payload.email: Override email address for testing (sends to this instead of user's email)
    
    Note: Jobs are processed asynchronously by the worker. Use GET /weekly-report/{app_user_id}
    to check the report status after the job completes.
    """
    # Validate user exists
    user = db.get(AppUser, app_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")
    
    # Validate user has sec_user_id
    if not user.latest_sec_user_id:
        raise HTTPException(status_code=400, detail="sec_user_id_required")
    
    # Determine email address for sending (if needed)
    email_address: Optional[str] = None
    if payload.send_email:
        if payload.email:
            # Use the provided test email
            email_address = payload.email
        else:
            # Fall back to user's registered email
            email_address = _latest_user_email(db, app_user_id)
            if not email_address:
                raise HTTPException(status_code=400, detail="email_required_for_sending")

    # Enqueue analysis job (force new to override any idempotency)
    analyze_job = job_queue.enqueue(
        db,
        task_name="weekly_report_analyze",
        payload={"app_user_id": app_user_id},
        idempotency_key=f"weekly_report_analyze_test:{app_user_id}:{datetime.utcnow().isoformat()}",
    )
    
    # Optionally enqueue send job (will run after analyze completes based on job ordering)
    send_job_id = None
    if payload.send_email:
        send_payload = {"app_user_id": app_user_id}
        # If a custom email is provided, pass it to the send job
        if payload.email:
            send_payload["email_override"] = payload.email
        send_job = job_queue.enqueue(
            db,
            task_name="weekly_report_send",
            payload=send_payload,
            idempotency_key=f"weekly_report_send_test:{app_user_id}:{datetime.utcnow().isoformat()}",
        )
        send_job_id = send_job.id
    
    logger.info(
        "admin.test.weekly_report.enqueued",
        extra={
            "event": "admin.test.weekly_report.enqueued",
            "app_user_id": app_user_id,
            "analyze_job_id": analyze_job.id,
            "send_job_id": send_job_id,
            "send_email": payload.send_email,
            "email_override": payload.email,
        },
    )
    
    # Get or create a placeholder report to return
    report = (
        db.query(WeeklyReport)
        .filter(WeeklyReport.app_user_id == app_user_id)
        .order_by(WeeklyReport.created_at.desc())
        .first()
    )
    
    # If no report exists yet, create a placeholder
    if not report:
        report = WeeklyReport(
            app_user_id=app_user_id,
            send_status="pending",
        )
        db.add(report)
        db.commit()
        db.refresh(report)
    
    # Convert topics JSON to list of TopicItem if present
    topics = None
    if report.topics:
        try:
            topics = [TopicItem(**item) if isinstance(item, dict) else item for item in report.topics]
        except Exception:
            topics = None
    
    report_response = WeeklyReportResponse(
        id=report.id,
        app_user_id=report.app_user_id,
        email_content=report.email_content,
        period_start=report.period_start,
        period_end=report.period_end,
        created_at=report.created_at,
        updated_at=report.updated_at,
        send_status=report.send_status,
        feeding_state=report.feeding_state,
        trend_name=report.trend_name,
        trend_type=report.trend_type,
        discovery_rank=report.discovery_rank,
        total_discoverers=report.total_discoverers,
        origin_niche_text=report.origin_niche_text,
        spread_end_text=report.spread_end_text,
        reach_start=report.reach_start,
        reach_end=report.reach_end,
        current_reach=report.current_reach,
        total_videos=report.total_videos,
        total_time=report.total_time,
        pre_total_time=report.pre_total_time,
        miles_scrolled=report.miles_scrolled,
        topics=topics,
        timezone=report.timezone,
        rabbit_hole_datetime=report.rabbit_hole_datetime,
        rabbit_hole_date=report.rabbit_hole_date,
        rabbit_hole_time=report.rabbit_hole_time,
        rabbit_hole_count=report.rabbit_hole_count,
        rabbit_hole_category=report.rabbit_hole_category,
        nudge_text=report.nudge_text,
    )
    
    return WeeklyReportTestResponse(
        app_user_id=app_user_id,
        report_id=report.id,
        analyze_job_id=analyze_job.id,
        send_job_id=send_job_id,
        email_sent=False,  # Jobs are asynchronous, email not sent yet
        email_address=email_address,
        report=report_response,
    )

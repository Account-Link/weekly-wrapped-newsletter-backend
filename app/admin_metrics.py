from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import AppAuthJob, AppJob, AppWrappedRun


def _esc_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\"", "\\\"")


def _render_labels(labels: Optional[Dict[str, str]]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_esc_label_value(str(v))}"' for k, v in sorted(labels.items()) if v is not None]
    return "{" + ",".join(parts) + "}" if parts else ""


def _metric_line(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> str:
    return f"{name}{_render_labels(labels)} {value}"


def _add_help_type(lines: List[str], name: str, help_text: str, metric_type: str) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")


def render_admin_metrics(db: Session) -> str:
    """
    Render low-cardinality, DB-backed operational metrics in Prometheus text format.

    This endpoint is intended for an internal/admin scrape and MUST avoid user/job-id labels.
    """
    lines: List[str] = []

    now = datetime.now(timezone.utc)
    _add_help_type(lines, "tk_wrapped_time_seconds", "Current server time (unix seconds).", "gauge")
    lines.append(_metric_line("tk_wrapped_time_seconds", now.timestamp()))

    # Queue depth by task/status.
    _add_help_type(lines, "tk_wrapped_jobs_total", "Count of internal jobs by task/status.", "gauge")
    rows: Iterable[Tuple[str, str, int]] = (
        db.query(AppJob.task_name, AppJob.status, func.count(AppJob.id))
        .group_by(AppJob.task_name, AppJob.status)
        .all()
    )
    for task_name, status, count in rows:
        lines.append(_metric_line("tk_wrapped_jobs_total", float(count), {"task_name": task_name, "status": status}))

    # Queue summary gauges.
    _add_help_type(lines, "tk_wrapped_queue_pending_total", "Number of pending jobs.", "gauge")
    pending_total = db.query(func.count(AppJob.id)).filter(AppJob.status == "pending").scalar() or 0
    lines.append(_metric_line("tk_wrapped_queue_pending_total", float(pending_total)))

    _add_help_type(lines, "tk_wrapped_queue_running_total", "Number of running jobs.", "gauge")
    running_total = db.query(func.count(AppJob.id)).filter(AppJob.status == "running").scalar() or 0
    lines.append(_metric_line("tk_wrapped_queue_running_total", float(running_total)))

    _add_help_type(lines, "tk_wrapped_queue_failed_total", "Number of failed jobs.", "gauge")
    failed_total = db.query(func.count(AppJob.id)).filter(AppJob.status == "failed").scalar() or 0
    lines.append(_metric_line("tk_wrapped_queue_failed_total", float(failed_total)))

    _add_help_type(
        lines,
        "tk_wrapped_queue_oldest_pending_age_seconds",
        "Age in seconds of the oldest pending job (0 if none).",
        "gauge",
    )
    oldest_pending_created_at = db.query(func.min(AppJob.created_at)).filter(AppJob.status == "pending").scalar()
    if oldest_pending_created_at:
        if oldest_pending_created_at.tzinfo is None:
            oldest_pending_created_at = oldest_pending_created_at.replace(tzinfo=timezone.utc)
        age = max(0.0, (now - oldest_pending_created_at).total_seconds())
    else:
        age = 0.0
    lines.append(_metric_line("tk_wrapped_queue_oldest_pending_age_seconds", age))

    # Auth jobs (Archive/Xordi) by provider/status.
    _add_help_type(lines, "tk_wrapped_auth_jobs_total", "Count of auth jobs by provider/status.", "gauge")
    auth_rows: Iterable[Tuple[str, str, int]] = (
        db.query(AppAuthJob.provider, AppAuthJob.status, func.count(AppAuthJob.archive_job_id))
        .group_by(AppAuthJob.provider, AppAuthJob.status)
        .all()
    )
    for provider, status, count in auth_rows:
        lines.append(_metric_line("tk_wrapped_auth_jobs_total", float(count), {"provider": provider or "", "status": status}))

    # Wrapped runs by status.
    _add_help_type(lines, "tk_wrapped_wrapped_runs_total", "Count of wrapped runs by status.", "gauge")
    run_rows: Iterable[Tuple[str, int]] = db.query(AppWrappedRun.status, func.count(AppWrappedRun.id)).group_by(AppWrappedRun.status).all()
    for status, count in run_rows:
        lines.append(_metric_line("tk_wrapped_wrapped_runs_total", float(count), {"status": status}))

    return "\n".join(lines) + "\n"


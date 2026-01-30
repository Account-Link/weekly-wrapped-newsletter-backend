import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import AppJob
from app.observability import get_logger, sanitize_json_text


class DBJobQueue:
    """DB-backed queue with leases, idempotency, and retry backoff."""

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    def _reclaim_expired_leases(self, db: Session, now: datetime) -> None:
        """Move expired running jobs back to pending so they can be re-leased."""
        dialect = ""
        try:
            dialect = str(db.get_bind().dialect.name or "")
        except Exception:
            dialect = ""

        if dialect == "postgresql":
            try:
                res = db.execute(
                    text(
                        """
UPDATE app_jobs
SET status = 'pending', locked_by = NULL, locked_at = NULL
WHERE status = 'running'
  AND locked_at IS NOT NULL
  AND locked_at + (lease_seconds || ' seconds')::interval < :now
RETURNING id, task_name, locked_by, lease_seconds
"""
                    ),
                    {"now": now},
                )
                reclaimed = res.fetchall()
            except Exception:
                reclaimed = None

            if reclaimed is not None:
                if reclaimed:
                    db.commit()
                    for row in reclaimed[:10]:
                        self._logger.warning(
                            "job.lease_expired",
                            extra={
                                "event": "job.lease_expired",
                                "job_id": row[0],
                                "task_name": row[1],
                                "locked_by": row[2],
                                "lease_seconds": row[3],
                            },
                        )
                    if len(reclaimed) > 10:
                        self._logger.warning(
                            "job.lease_expired.batch",
                            extra={"event": "job.lease_expired.batch", "count": len(reclaimed)},
                        )
                return

        running = db.query(AppJob).filter(AppJob.status == "running").all()
        expired = False
        for job in running:
            if job.locked_at and job.locked_at + timedelta(seconds=job.lease_seconds) < now:
                self._logger.warning(
                    "job.lease_expired",
                    extra={
                        "event": "job.lease_expired",
                        "job_id": job.id,
                        "task_name": job.task_name,
                        "locked_by": job.locked_by,
                        "lease_seconds": job.lease_seconds,
                    },
                )
                job.status = "pending"
                job.locked_by = None
                job.locked_at = None
                db.add(job)
                expired = True
        if expired:
            db.commit()

    def enqueue(
        self,
        db: Session,
        task_name: str,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None,
        force_new: bool = False,
        not_before: Optional[datetime] = None,
    ) -> AppJob:
        if idempotency_key:
            existing = (
                db.query(AppJob)
                .filter(AppJob.idempotency_key == idempotency_key)
                .order_by(AppJob.created_at.desc())
                .first()
            )
            if existing:
                # Always dedup pending/running; optionally allow re-running succeeded jobs.
                should_dedup = existing.status in ("pending", "running") or (existing.status == "succeeded" and not force_new)
                if should_dedup:
                    payload_text, payload_truncated = sanitize_json_text(json.dumps(payload, default=str))
                    self._logger.info(
                        "job.enqueue.dedup",
                        extra={
                            "event": "job.enqueue.dedup",
                            "job_id": existing.id,
                            "task_name": existing.task_name,
                            "status": existing.status,
                            "idempotency_key": idempotency_key,
                            "payload": payload_text,
                            "payload_truncated": payload_truncated,
                        },
                    )
                    return existing

        job = AppJob(
            id=str(uuid.uuid4()),
            task_name=task_name,
            payload=payload,
            status="pending",
            attempts=0,
            max_attempts=5,
            not_before=not_before,
            idempotency_key=idempotency_key,
            locked_by=None,
            locked_at=None,
            lease_seconds=60,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        payload_text, payload_truncated = sanitize_json_text(json.dumps(payload, default=str))
        self._logger.info(
            "job.enqueue",
            extra={
                "event": "job.enqueue",
                "job_id": job.id,
                "task_name": task_name,
                "status": job.status,
                "idempotency_key": idempotency_key,
                "payload": payload_text,
                "payload_truncated": payload_truncated,
            },
        )
        return job

    def lease(
        self,
        db: Session,
        worker_id: str,
        lease_seconds: int = 60,
        task_names: Optional[list[str]] = None,
    ) -> Optional[AppJob]:
        # Use naive UTC timestamps; DB columns are `timestamp without time zone`.
        now = datetime.utcnow()
        self._reclaim_expired_leases(db, now)

        query = db.query(AppJob).filter(
            AppJob.status == "pending",
            (AppJob.not_before.is_(None)) | (AppJob.not_before <= now),
        )
        if task_names:
            query = query.filter(AppJob.task_name.in_(task_names))
        job = query.order_by(AppJob.created_at.asc()).with_for_update(skip_locked=True).first()
        if not job:
            return None
        job.status = "running"
        job.locked_by = worker_id
        job.locked_at = now
        job.lease_seconds = lease_seconds
        db.commit()
        db.refresh(job)
        self._logger.info(
            "job.lease",
            extra={
                "event": "job.lease",
                "job_id": job.id,
                "task_name": job.task_name,
                "attempts": job.attempts,
                "max_attempts": job.max_attempts,
                "worker_id": worker_id,
                "lease_seconds": lease_seconds,
            },
        )
        return job

    def heartbeat(self, db: Session, job_id: str, worker_id: str, lease_seconds: Optional[int] = None) -> bool:
        """Extend a running job's lease. Returns False if the job is no longer owned by this worker."""
        now = datetime.utcnow()
        update_values: Dict[str, Any] = {"locked_at": now}
        if lease_seconds is not None:
            update_values["lease_seconds"] = int(lease_seconds)
        updated = (
            db.query(AppJob)
            .filter(AppJob.id == job_id, AppJob.status == "running", AppJob.locked_by == worker_id)
            .update(update_values)
        )
        if updated:
            db.commit()
            self._logger.debug(
                "job.heartbeat",
                extra={"event": "job.heartbeat", "job_id": job_id, "worker_id": worker_id, "lease_seconds": lease_seconds},
            )
            return True
        db.rollback()
        return False

    def complete(self, db: Session, job_id: str, worker_id: Optional[str] = None) -> None:
        if worker_id:
            updated = (
                db.query(AppJob)
                .filter(AppJob.id == job_id, AppJob.status == "running", AppJob.locked_by == worker_id)
                .update({"status": "succeeded", "locked_by": None, "locked_at": None})
            )
            if not updated:
                db.rollback()
                self._logger.warning(
                    "job.complete.not_owned",
                    extra={"event": "job.complete.not_owned", "job_id": job_id, "worker_id": worker_id},
                )
                return
            db.commit()
            self._logger.info("job.complete", extra={"event": "job.complete", "job_id": job_id, "worker_id": worker_id})
            return

        job = db.get(AppJob, job_id)
        if not job:
            return
        job.status = "succeeded"
        job.locked_by = None
        job.locked_at = None
        db.commit()
        self._logger.info(
            "job.complete",
            extra={
                "event": "job.complete",
                "job_id": job.id,
                "task_name": job.task_name,
                "attempts": job.attempts,
            },
        )

    def fail(self, db: Session, job_id: str, retry_delay_seconds: int = 60, worker_id: Optional[str] = None) -> None:
        if worker_id:
            job = db.get(AppJob, job_id)
            if not job or job.status != "running" or job.locked_by != worker_id:
                db.rollback()
                self._logger.warning(
                    "job.fail.not_owned",
                    extra={"event": "job.fail.not_owned", "job_id": job_id, "worker_id": worker_id},
                )
                return
            job.attempts += 1
            job.status = "pending" if job.attempts < job.max_attempts else "failed"
            job.not_before = datetime.utcnow() + timedelta(seconds=retry_delay_seconds)
            job.locked_by = None
            job.locked_at = None
            db.commit()
            self._logger.warning(
                "job.fail",
                extra={
                    "event": "job.fail",
                    "job_id": job.id,
                    "task_name": job.task_name,
                    "attempts": job.attempts,
                    "max_attempts": job.max_attempts,
                    "status": job.status,
                    "retry_delay_seconds": retry_delay_seconds,
                    "not_before": job.not_before,
                    "worker_id": worker_id,
                },
            )
            return

        job = db.get(AppJob, job_id)
        if not job:
            return
        job.attempts += 1
        job.status = "pending" if job.attempts < job.max_attempts else "failed"
        job.not_before = datetime.utcnow() + timedelta(seconds=retry_delay_seconds)
        job.locked_by = None
        job.locked_at = None
        db.commit()
        self._logger.warning(
            "job.fail",
            extra={
                "event": "job.fail",
                "job_id": job.id,
                "task_name": job.task_name,
                "attempts": job.attempts,
                "max_attempts": job.max_attempts,
                "status": job.status,
                "retry_delay_seconds": retry_delay_seconds,
                "not_before": job.not_before,
            },
        )

from uuid import uuid4
import os
from app.db import SessionLocal
from app.models import AppWrappedRun
from app.services.job_queue import DBJobQueue

# Set these before running or edit inline
SEC_USER_ID = os.environ.get("SEC_USER_ID", "")      # REQUIRED
APP_USER_ID = os.environ.get("APP_USER_ID", "ref-test-001")
ARCHIVE_USER_ID = os.environ.get("ARCHIVE_USER_ID", "manual-test")
TIME_ZONE = os.environ.get("TIME_ZONE", "UTC")

if not SEC_USER_ID:
    raise SystemExit("SEC_USER_ID is required (env SEC_USER_ID)")

sess = SessionLocal()
run_id = str(uuid4())
run = AppWrappedRun(
    id=run_id,
    app_user_id=APP_USER_ID,
    sec_user_id=SEC_USER_ID,
    archive_user_id=ARCHIVE_USER_ID,
    status="pending",
)
sess.add(run)
sess.commit()

DBJobQueue().enqueue(
    sess,
    task_name="watch_history_fetch_2025",
    payload={
        "wrapped_run_id": run_id,
        "app_user_id": APP_USER_ID,
        "sec_user_id": SEC_USER_ID,
        "time_zone": TIME_ZONE,
        "platform_username": None,
    },
    idempotency_key=f"wrapped:{APP_USER_ID}:{SEC_USER_ID}",
)

sess.close()
print(f"Enqueued run_id={run_id} for app_user_id={APP_USER_ID} sec_user_id={SEC_USER_ID}")

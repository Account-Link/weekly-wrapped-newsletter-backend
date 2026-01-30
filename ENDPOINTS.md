# API Endpoints

Source of truth: `app/main.py` and `app/schemas.py`. All timestamps are ISO8601. Unless stated otherwise, success responses are HTTP 200 and errors use the envelope:

```json
{ "error": "<code>", "message": "<optional human message>" }
```

## Conventions
- **Device headers** (required where noted): `X-Device-Id`, `X-Platform`, `X-App-Version`, `X-OS-Version`.
- **Session auth**: `Authorization: Bearer <token>` returned by `/link/tiktok/redirect` when it reaches `status="completed"`.
- **Content-Type**: `application/json` on bodies.

## Link & Session Flow (TikTok via Xordi)

### POST /link/tiktok/start
- Auth: device headers required; no bearer.
- Body: none.
- 201 response (`LinkStartResponse`):
  ```json
  {
    "archive_job_id": "string",
    "expires_at": "2025-02-04T12:00:00Z",
    "queue_position": 5,
    "failed_count": 0
  }
  ```

### GET /link/tiktok/redirect
- Auth: device headers required; no bearer.
- Query: `job_id` (string, from `archive_job_id` above), optional `time_zone`.
- Responses (`RedirectResponse`):
  - 202 pending (also used when Archive returns 200 with `status: pending/processing`):
    ```json
    {
      "status": "pending",
      "queue_status": "pending or processing",
      "queue_position": 3,
      "queue_request_id": "uuid"
    }
    ```
  - 200 redirect ready:
    ```json
    {
      "status": "ready",
      "redirect_url": "https://www.tiktok.com/..."
    }
    ```
  - 200 completed (session minted):
    ```json
    {
      "status": "completed",
      "app_user_id": "string",
      "token": "bearer-token",
      "expires_at": "2025-03-06T12:00:00Z",
      "platform_username": "string or null"
    }
    ```
  - 200 finalizing/other (Archive says “authentication already completed” but session not available yet):
    ```json
    {
      "status": "finalizing",
      "message": "authentication already completed",
      "queue_status": "processing",
      "queue_position": null,
      "queue_request_id": "uuid"
    }
    ```
  - 410 expired/failed:
    ```json
    {
      "status": "expired",
      "message": "Job expired",
      "queue_status": "failed",
      "queue_position": null,
      "queue_request_id": "uuid"
    }
    ```
- Optional auto-cancel: when `AUTO_CANCEL_QUEUE_ON_MAX_POSITION=true` and `queue_position > MAX_QUEUE_POSITION`, the server cancels the Archive queue request and returns 410 with `message="queue_position_exceeded"` and `queue_status="cancelled"`.
- Errors: 404 `job_not_found`; 401 `invalid_device`; other HTTP status bubbled.
> Notes: Queue fields may be omitted if Archive does not return them.

### POST /link/tiktok/redirect/click
- Auth: device headers required; no bearer.
- Body:
  ```json
  { "archive_job_id": "string" }
  ```
- 204 No Content on success.
- Behavior: sets `redirect_clicked_at` on first click and increments `redirect_clicks` on every call.
- Errors: 404 `job_not_found`; 401 `invalid_device`.

### GET /link/tiktok/queue-status
- Auth: device headers required; no bearer.
- 200 response: `{ "pending": 5 }`

### DELETE /link/tiktok/cancel-queue
- Auth: device headers required; no bearer.
- Query: `archive_job_id=<id>`
- 200 response:
  ```json
  { "request_id": "req_123", "status": "cancelled", "cancelled_at": "2025-12-27T15:11:29.302000Z" }
  ```
- Errors: 404 `job_not_found`; 401 `invalid_device`; 4xx/5xx bubbled from Archive.

### GET /link/tiktok/code
- Legacy: frontend no longer uses this endpoint (worker polls Archive authorization code internally).

### POST /link/tiktok/finalize
- Legacy: frontend no longer uses this endpoint (finalize runs in the worker after redirect).

### POST /link/tiktok/verify-region
- Auth: bearer token + device headers.
- Note: frontend no longer calls this endpoint; the worker runs the region probe automatically. Keep this for debugging/manual retries.
- Body: none.
- 200 response (`VerifyRegionResponse`):
  ```json
  { "is_watch_history_available": "unknown|yes|no", "attempts": 3, "last_error": "string or null" }
  ```
- Errors: 400 `user_not_found`; 401 `invalid_session` (bearer).

## Email & Waitlist

### POST /register-email
- Auth: device headers only.
- Body (`RegisterEmailRequest`): `{ "email": "user@example.com", "referral_code": "optional" }`
- 204 No Content on success.

### POST /waitlist
- Auth: device headers only.
- Body (`WaitlistRequest`): `{ "email": "user@example.com" }`
- 204 No Content on success (sets `device_emails.waitlist_opt_in=true` and stamps `waitlist_opt_in_at` once).

## Referrals

### POST /referral
- Auth: public.
- Body (`ReferralRequest`): `{ "app_user_id": "..." }` or `{ "email": "user@example.com" }`
- 200 response (`ReferralResponse`):
  ```json
  { "code": "abc12345", "referral_url": "https://wrapped.feedling.app?ref=abc12345" }
  ```
- Behavior: returns existing code for the referrer or creates one.

### POST /referral/impression
- Auth: public.
- Body (`ReferralImpressionRequest`): `{ "code": "abc12345" }`
- 204 No Content on success; increments impressions and logs an event. 404 if code is unknown.

## Wrapped Pipeline

### GET /wrapped/{app_user_id}
- Auth: public; typically reached via the emailed link containing `app_user_id` (no device or bearer needed).
- 200 response (`WrappedStatusResponse`):
  - Pending: `{ "status": "pending", "wrapped_run_id": "uuid", "wrapped": null, "queue_status": "pending", "queue_position": null, "queue_eta_seconds": null }`
  - Ready: `{ "status": "ready", "wrapped_run_id": "uuid", "wrapped": { /* WrappedPayload */ }, "queue_status": null, "queue_position": null, "queue_eta_seconds": null }`
    - `wrapped.email` is included if the user registered an email.
- Errors: 404 `not_found`.

## Admin Test Harness

These endpoints are intended for operators to test the full pipeline end-to-end (watch history fetch → analysis → email send).

**Auth**
- Header: `X-Admin-Key: <ADMIN_API_KEY>` (server-side env var `ADMIN_API_KEY` must be set, otherwise these endpoints 404).

### POST /admin/test/run
- Auth: `X-Admin-Key` only.
- Body (`AdminTestRunRequest`):
  ```json
  {
    "sec_user_id": "string",
    "platform_username": "optional",
    "email": "user@example.com",
    "time_zone": "America/New_York",
    "app_user_id": "optional (defaults to new admin-test-uuid)",
    "force_new_run": true
  }
  ```
- 200 response (`AdminTestRunResponse`):
  ```json
  {
    "app_user_id": "string",
    "wrapped_run_id": "uuid",
    "watch_history_job_id": "uuid",
    "wrapped_link": "https://wrapped.feedling.app/?app_user_id=<app_user_id>",
    "wrapped_status_endpoint": "/wrapped/<app_user_id>",
    "admin_status_endpoint": "/admin/test/runs/<wrapped_run_id>"
  }
  ```

**Example**
```bash
curl -sS -X POST "$BASE/admin/test/run" \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "sec_user_id":"7504768731169719339",
    "platform_username":"hell.technologies",
    "email":"tanxiaoting7@gmail.com",
    "time_zone":"America/New_York"
  }' | jq -S .
```

### GET /admin/test/runs/{wrapped_run_id}
- Auth: `X-Admin-Key` only.
- 200 response (`AdminTestRunStatusResponse`) includes:
  - `run_status` (derived from pipeline outcome),
  - `jobs.*` (DB job records),
  - `data_jobs.*` (per-stage status),
  - `watch_history_progress` (cursor / pagination info).
  - `analysis_warnings` / `analysis_debug` (when present; useful for LLM debugging).

**Example**
```bash
curl -sS "$BASE/admin/test/runs/$WRAPPED_RUN_ID" \
  -H "X-Admin-Key: $ADMIN_API_KEY" | jq -S .
```

### POST /admin/test/runs/{wrapped_run_id}/retry-analysis
- Enqueues (or dedups) a `wrapped_analysis` job for an existing run.

**Example**
```bash
curl -sS -X POST "$BASE/admin/test/runs/$WRAPPED_RUN_ID/retry-analysis" \
  -H "X-Admin-Key: $ADMIN_API_KEY" | jq -S .
```

### POST /admin/test/runs/{wrapped_run_id}/retry-email
- Enqueues (or dedups) an `email_send` job for an existing run.

**Example**
```bash
curl -sS -X POST "$BASE/admin/test/runs/$WRAPPED_RUN_ID/retry-email" \
  -H "X-Admin-Key: $ADMIN_API_KEY" | jq -S .
```

### POST /admin/users/{app_user_id}/verify-region
- Auth: `X-Admin-Key` only.
- Behavior: enqueues a background watch history verification job; auto-enqueues the wrapped pipeline on `yes` or `unknown`.
- 200 response (`AdminVerifyRegionEnqueueResponse`):
  ```json
  {
    "app_user_id": "string",
    "task_name": "watch_history_verify",
    "job_id": "uuid",
    "status": "pending",
    "idempotency_key": "watch_history_verify:<app_user_id>"
  }
  ```

### POST /admin/users/verify-region/batch
- Auth: `X-Admin-Key` only.
- Body:
  ```json
  { "include_unknown": false, "auto_enqueue": false, "force_new": false }
  ```
- Behavior: enqueues watch history verification jobs for all users with `is_watch_history_available="no"` (and optionally `unknown`) and a `sec_user_id`; auto-enqueues wrapped pipeline on `yes` or `unknown` when `auto_enqueue=true`.
- 200 response (`AdminVerifyRegionBatchResponse`):
  ```json
  {
    "batch_id": "uuid",
    "matched": 123,
    "processed": 123,
    "enqueued": 123,
    "results": [
      {
        "app_user_id": "user_1",
        "sec_user_id": "sec_123",
        "job_id": "uuid",
        "status": "pending",
        "idempotency_key": "watch_history_verify:user_1",
        "error": null
      }
    ]
  }
  ```

### GET /admin/users/verify-region/batch/{batch_id}
- Auth: `X-Admin-Key` only.
- 200 response (`AdminVerifyRegionBatchStatusResponse`):
  ```json
  {
    "batch_id": "uuid",
    "created_at": "2025-12-25T00:00:00Z",
    "total": 123,
    "completed": 100,
    "pending": 23,
    "yes": 80,
    "no": 15,
    "unknown": 4,
    "error": 1,
    "results": [
      {
        "app_user_id": "user_1",
        "sec_user_id": "sec_123",
        "job_id": "uuid",
        "job_status": "succeeded",
        "verify_status": "yes",
        "attempts": 2,
        "last_error": null,
        "checked_at": "2025-12-25T00:00:02Z",
        "error": null
      }
    ]
  }
  ```

### GET /admin/users/{app_user_id}/stage
- Auth: `X-Admin-Key` only.
- Query params: `wrapped_run_id` (optional; defaults to latest run).
- Behavior: returns the current pipeline stage + next stage for a user, plus verify/job/run details.
- `is_stuck` is true when a job is `running` and `locked_at + lease_seconds < now`.
- Note: `app_user_id` must be a real user id, not an email address.
- 200 response (`AdminUserStageResponse`):
  ```json
  {
    "app_user_id": "user_1",
    "sec_user_id": "sec_123",
    "wrapped_run_id": "uuid",
    "run_status": "pending",
    "stage": "verify_region|watch_history|analysis|email|ready|no_run",
    "next_stage": "verify_region|watch_history|analysis|email|ready",
    "verify": {
      "is_watch_history_available": "unknown",
      "job_id": "uuid",
      "job_status": "running",
      "is_stuck": false,
      "stuck_reason": null,
      "locked_for_seconds": 12.3,
      "locked_by": "worker_1",
      "locked_at": "2025-12-25T00:00:02Z",
      "updated_at": "2025-12-25T00:00:05Z",
      "result": { "status": "unknown" }
    },
    "data_jobs": { "watch_history": { "id": "uuid", "status": "running" } },
    "watch_history_progress": { "status": "running" },
    "jobs": { "watch_history_fetch_2025": { "id": "uuid", "status": "running", "is_stuck": false } }
  }
  ```

### POST /admin/users/{app_user_id}/restart
- Auth: `X-Admin-Key` only.
- Body (`AdminUserRestartRequest`):
  ```json
  {
    "stage": "next|verify_region|watch_history|analysis|email",
    "wrapped_run_id": "optional",
    "force_new_jobs": true,
    "reset_payload": true
  }
  ```
- Behavior: restarts the requested stage for the user; if `stage="next"` it restarts the next pending stage. If no run exists and next stage is `watch_history`, a new run is created. Downstream stages auto-enqueue on success.
- Note: `app_user_id` must be a real user id, not an email address.
- 200 response (`AdminUserRestartResponse`):
  ```json
  {
    "app_user_id": "user_1",
    "wrapped_run_id": "uuid",
    "selected_stage": "watch_history",
    "enqueued_task": "watch_history_fetch_2025",
    "job_id": "uuid",
    "status": "pending",
    "idempotency_key": "wrapped:<run_id>",
    "skipped_reason": null
  }
  ```

### POST /admin/users/stage/batch
- Auth: `X-Admin-Key` only.
- Body (`AdminUserStageBatchRequest`):
  ```json
  {
    "limit": 50,
    "offset": 0,
    "include_admin_test": false,
    "watch_history_status": ["unknown", "no"],
    "run_status": ["pending", "failed", "no_run"],
    "stage": ["verify_region", "watch_history", "analysis", "email", "ready", "no_run"]
  }
  ```
- Behavior: returns stage details for a batch of users. Filters are applied after stage derivation.
- 200 response (`AdminUserStageBatchResponse`):
  ```json
  {
    "limit": 50,
    "offset": 0,
    "requested": 50,
    "returned": 42,
    "items": [ /* AdminUserStageResponse */ ]
  }
  ```

### POST /admin/users/restart/batch
- Auth: `X-Admin-Key` only.
- Body (`AdminUserRestartBatchRequest`):
  ```json
  {
    "limit": 50,
    "offset": 0,
    "include_admin_test": false,
    "watch_history_status": ["unknown", "no"],
    "run_status": ["pending", "failed", "no_run"],
    "stage_filter": ["verify_region", "watch_history", "analysis", "email", "no_run"],
    "restart_stage": "next",
    "force_new_jobs": true,
    "reset_payload": true,
    "dry_run": false
  }
  ```
- Behavior: restarts the selected stage for each user in the batch; if `restart_stage="next"` it restarts the next pending stage. Skips users filtered out by `stage_filter` or `run_status`.
- 200 response (`AdminUserRestartBatchResponse`):
  ```json
  {
    "limit": 50,
    "offset": 0,
    "requested": 50,
    "processed": 50,
    "enqueued": 12,
    "skipped": 38,
    "dry_run": false,
    "results": [
      {
        "app_user_id": "user_1",
        "result": {
          "app_user_id": "user_1",
          "wrapped_run_id": "uuid",
          "selected_stage": "watch_history",
          "enqueued_task": "watch_history_fetch_2025",
          "job_id": "uuid",
          "status": "pending",
          "idempotency_key": "wrapped:<run_id>",
          "skipped_reason": null
        }
      }
    ]
  }
  ```

### GET /admin/jobs/{job_id}
- Auth: `X-Admin-Key` only.
- Query params: `include_payload` (bool, default false).
- 200 response (`AdminJobStatusResponse`):
  ```json
  {
    "job_id": "uuid",
    "task_name": "watch_history_verify",
    "status": "succeeded",
    "attempts": 1,
    "max_attempts": 5,
    "not_before": null,
    "locked_by": null,
    "locked_at": null,
    "created_at": "2025-12-25T00:00:00Z",
    "updated_at": "2025-12-25T00:00:10Z",
    "idempotency_key": "watch_history_verify:user_1",
    "result": {
      "app_user_id": "user_1",
      "sec_user_id": "sec_123",
      "status": "yes",
      "attempts": 1,
      "last_error": null,
      "checked_at": "2025-12-25T00:00:10Z"
    },
    "payload": null
  }
  ```

## Admin: Retry Real User Runs

Auth: `X-Admin-Key: <ADMIN_API_KEY>`

### POST /admin/wrapped/runs/{wrapped_run_id}/retry
- Restarts an existing wrapped run for a real user.

### POST /admin/wrapped/runs/retry-failed
- Bulk retries failed wrapped runs (up to `limit`).
- Auth: `X-Admin-Key` only.
- Body (`AdminRetryFailedRunsRequest`):
  ```json
  { "limit": 50, "dry_run": false, "reset_payload": true }
  ```
- Behavior: by default auto-detects which stage failed and re-enqueues watch history or analysis; can be forced via `include_watch_history/include_analysis/include_email`.
- Default behavior: re-run watch history, then force a fresh `wrapped_analysis` when watch history completes.
- Body (`AdminWrappedRetryRequest`):
  ```json
  {
    "include_watch_history": true,
    "include_analysis": true,
    "force_new_jobs": true,
    "reset_payload": true
  }
  ```
- 200 response (`AdminTestEnqueueResponse`): returns the enqueued job record for the first stage.

**Example**
```bash
curl -sS -X POST "$BASE/admin/wrapped/runs/$WRAPPED_RUN_ID/retry" \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"include_watch_history":true,"include_analysis":true,"force_new_jobs":true,"reset_payload":true}' | jq -S .
```

### POST /admin/wrapped/runs/retry-zero-videos
- Bulk retries wrapped runs where `payload.total_videos == 0`.
- Auth: `X-Admin-Key` only.
- Body (`AdminRetryZeroVideosRequest`):
  ```json
  { "limit": 50, "dry_run": false, "reset_payload": true, "include_admin_test": false, "run_status": ["ready"] }
  ```
- `limit`: set to `0` or `null` to scan all runs (no limit).
- Behavior: re-enqueues `watch_history_fetch_2025` with `force_new_analysis=true` and `force_new_email=true`, then analysis/email auto-enqueue on success.
- 200 response (`AdminRetryZeroVideosResponse`):
  ```json
  {
    "matched": 10,
    "processed": 10,
    "enqueued": 3,
    "dry_run": false,
    "results": [
      {
        "wrapped_run_id": "uuid",
        "app_user_id": "user_1",
        "sec_user_id": "sec_123",
        "total_videos": 0,
        "watch_history_progress": { "status": "running" },
        "action": "enqueued",
        "enqueued_task": "watch_history_fetch_2025",
        "job_id": "uuid",
        "idempotency_key": "wrapped:<run_id>",
        "skipped_reason": null
      }
    ]
  }
  ```

## Admin: Metrics

Auth: `X-Admin-Key: <ADMIN_API_KEY>`

### GET /admin/metrics
- Returns an HTML page with basic backend + DB health/queue stats (intended for operators).

### WrappedPayload shape
Returned inside wrapped responses:
```json
{
  "total_hours": 1.23,
  "total_videos": 120,
  "night_pct": 12.5,
  "peak_hour": 22,
  "top_music": { "name": "string", "count": 5 },
  "top_creators": ["creator1", "creator2"],
  "cat_name": "string",
  "analogy_line": "string or null",
  "scroll_time": { "title": "string", "rate": "string", "between_time": "string" },
  "personality_type": "string",
  "personality_explanation": "string or null",
  "niche_journey": ["string", "..."],
  "top_niches": ["string", "..."],
  "top_niche_percentile": "string or null",
  "brain_rot_score": 42,
  "brain_rot_explanation": "string or null",
  "keyword_2026": "string",
  "thumb_roast": "string or null",
  "platform_username": "string or null",
  "email": "string or null",
  "source_spans": [ { "video_id": "string", "reason": "aggregate" } ],
  "data_jobs": { "watch_history": { "id": "uuid", "status": "succeeded" }, "wrapped_analysis": { ... }, "email_send": { ... } },
  "accessory_set": {
    "head": { "internal_name": "string", "reason": "string" },
    "body": { "internal_name": "string", "reason": "string" },
    "other": { "internal_name": "string", "reason": "string" }
  }
}
```

## Health

### GET /healthz
- 204 No Content.

### GET /readyz
- 204 No Content.

# Admin Dashboard Data Model (Read SQL + Write via Admin API)

This repo’s “admin dashboard” can be implemented as a hybrid:
- **Reads**: query Postgres, Polling Postgres every ~5s can work fine if you do it
  carefully; the “big hit” comes from doing wide table
  scans or heavy JSON processing repeatedly.
- **Writes/actions**: call backend **admin HTTP endpoints** (keeps idempotency + business rules in one place).

For endpoint details and request/response shapes, also reference:
- `ENDPOINTS.md`
- `scripts/admin-curl.md`

## Core entities & where “stage” lives

### `app_wrapped_runs` (one run = one Wrapped generation attempt)
Primary row you’ll list/monitor.
- **PK**: `id` (aka `wrapped_run_id`)
- **Key FKs/refs**:
  - `app_user_id` → `app_users.id`
  - `watch_history_job_id` → `app_jobs.id` (the `watch_history_fetch_2025` job)
- **Important columns**:
  - `status`: high-level run state (`pending|ready|failed`)
  - `sec_user_id`: TikTok provider id (may also exist on `app_users.latest_sec_user_id`)
  - `email`: may be null; also mirrored into `payload["email"]` when present
  - `payload` (JSON): the detailed state machine for progress + analysis output
- **Important `payload` keys** (JSON):
  - `data_jobs`: map of stage → `{id,status}` (mirrors/derives from `app_jobs`)
    - `watch_history`, `wrapped_analysis`, `email_send`
  - `_watch_history_progress`: crawl/progress details, including:
    - `status`: `running|scrape_done|succeeded|failed`
    - `cursor_ms`, `next_cursor_ms`, `has_more`, `completed_data_jobs`, `videos_fetched_total`
    - `scrape_truncated`, `scrape_stop_reason`
    - `scrape_error`, `scrape_error_detail` (when we continue after a scrape error)
    - `coverage`: `watched_at_min/max`, `coverage_months`, `extrapolated`, `extrapolation_factor`
  - Analysis outputs (when available):
    - `personality_type`, `personality_explanation`
    - `niche_journey`, `top_niches`, `top_niche_percentile`
    - `brain_rot_score`, `brain_rot_explanation`
    - Brainrot metrics: `brainrot_intensity`, `brainrot_volume_hours`, `brainrot_confidence`
    - Brainrot telemetry: `brainrot_enriched_hours`, `brainrot_enriched_watch_pct`, `brainrot_intensity_raw`,
      `brainrot_intensity_raw_effective`, `brainrot_intensity_linear`, `brainrot_quality_weight`, `brainrot_normalization`
    - Presentation: `cat_name`, `analogy_line`, `scroll_time` (object with `title`, `rate`, `between_time`)
    - Presentation: `accessory_set` (object with `head`/`body`/`other`, each `{internal_name, reason}`)
    - `keyword_2026`, `thumb_roast`, `source_spans`
  - `_analysis_warnings`: non-fatal issues (parse warnings, fallbacks, etc.)
    - surfaced as `analysis_warnings` in admin endpoints
  - `_analysis_debug`: verbose LLM/debug metadata when enabled
  - `_brainrot_*`: internal compute metadata (source, reconciliation status, watch job ids)

**Dashboard stage recommendation** (derived):
- `run.status == "ready"` → “Ready” (email may or may not be sent; check `data_jobs.email_send`)
- else if `payload.data_jobs.watch_history.status != "succeeded"` → “Fetching watch history”
- else if `payload.data_jobs.wrapped_analysis.status != "succeeded"` → “Running analysis”
- else if `payload.data_jobs.email_send.status != "succeeded"` → “Email pending/failed”
- else → “Pending (unknown)” (should be rare)

### `app_jobs` (DB-backed queue jobs)
This is the worker queue and the source of truth for retries/attempts.
- **PK**: `id`
- **Key columns**:
  - `task_name`: e.g. `watch_history_fetch_2025`, `wrapped_analysis`, `email_send`, `xordi_finalize`, `reauth_notify`
  - `status`: `pending|running|succeeded|failed`
  - `attempts`, `max_attempts`, `not_before`
  - `idempotency_key`: used to dedupe/ensure “only once” where required
  - `payload` (JSON): contains identifiers like `wrapped_run_id`, `app_user_id`, `sec_user_id`, etc.
- **Join patterns**:
  - `app_wrapped_runs.watch_history_job_id = app_jobs.id` for the watch-history job record.
  - For analysis/email: use `app_wrapped_runs.payload->'data_jobs'->'wrapped_analysis'->>'id'` (and `email_send`) to join to `app_jobs.id`.

### `app_users` (user identity)
- **PK**: `id` (`app_user_id`)
- **Useful columns**:
  - `latest_sec_user_id`, `platform_username`, `time_zone`
  - any email field if present in your schema version (some flows also keep email in runs)

### `app_auth_jobs` (TikTok linking/auth state)
Use this for “auth failed/needs reauth” monitoring.
- **PK**: `id`
- **Common join keys**:
  - `device_id` (ties to `device_emails.device_id`)
  - `archive_job_id` (ties to Archive/xordi)
- **Dashboard uses**:
  - show current auth state per device/email
  - correlate auth failures with downstream wrapped runs

### `device_emails` (device → email, waitlist opt-in)
- **PK**: `device_id`
- **Important columns** (schema-dependent but expected):
  - `email` (lowercased)
  - `referred_by` (referrer identity captured at email entry, if present)
  - `waitlist_opt_in` (bool)
  - timestamps

### `app_sessions` (if the dashboard needs to impersonate/verify sessions)
Most dashboards won’t need this, but it exists if you’re debugging session issuance/expiry.

### `referrals` / `referral_events`
If you plan to include referral monitoring.
- Current write API accepts either `app_user_id` or `email` (stored as identity `email:<lowercased>`).

## Read-side “views” to implement in the dashboard

### 1) Runs list (main table)
Recommended columns:
- `wrapped_run_id`, `run.status`, `run.updated_at`, `app_user_id`, `sec_user_id`, `email`
- derived stage (see above)
- `watch_history_progress.completed_data_jobs`, `videos_fetched_total`, `has_more`, `cursor_ms`
- `analysis_warnings` count

### 2) Run details page
Show:
- Full `app_wrapped_runs.payload` (pretty JSON)
- Linked `app_jobs` rows for each stage (watch_history / analysis / email_send)
- Worker events (from logs; correlate via `wrapped_run_id` + `job_id`)

### 3) Auth jobs list (device/email centric)
Show:
- per `device_id`: `email`, latest auth job status, archive_job_id, last updated, “needs reauth” signals

### 4) Queue view (jobs)
From `app_jobs`:
- filter by `status in ('pending','running','failed')`
- group by `task_name`
- show `attempts/max_attempts`, `not_before`, `idempotency_key`

## Write-side actions (use admin HTTP endpoints)

Run-level actions:
- Retry stage-aware: `POST /admin/wrapped/runs/{wrapped_run_id}/retry`
- Bulk retry failed runs: `POST /admin/wrapped/runs/retry-failed`
- Retry specific stages:
  - `POST /admin/test/runs/{wrapped_run_id}/retry-watch-history`
  - `POST /admin/test/runs/{wrapped_run_id}/retry-analysis`
  - `POST /admin/test/runs/{wrapped_run_id}/retry-email`

All admin endpoints require `X-Admin-Key`.

## Example read queries (dashboard-friendly)

List recently-updated runs (cheap poll):

```sql
select
  id as wrapped_run_id,
  status,
  app_user_id,
  sec_user_id,
  email,
  updated_at
from app_wrapped_runs
where updated_at > :last_seen
order by updated_at desc
limit 200;
```

Join run → stage jobs (watch_history is a real FK; analysis/email ids live in JSON payload):

```sql
select
  r.id as wrapped_run_id,
  r.status as run_status,
  r.updated_at,
  wh.status as watch_history_job_status,
  a.status as analysis_job_status,
  e.status as email_job_status
from app_wrapped_runs r
left join app_jobs wh on wh.id = r.watch_history_job_id
left join app_jobs a on a.id = (r.payload->'data_jobs'->'wrapped_analysis'->>'id')::uuid
left join app_jobs e on e.id = (r.payload->'data_jobs'->'email_send'->>'id')::uuid
where r.updated_at > :last_seen
order by r.updated_at desc
limit 200;
```

Note: if your `app_jobs.id` type is not `uuid`, remove the cast and compare as text.

## Notes / footguns
- Many “stage” fields are stored in `app_wrapped_runs.payload` JSON; your dashboard should handle missing keys gracefully.
- “Run is pending but email received” is prevented by gating `email_send` on `run.status == 'ready'`, but a dashboard should still treat `data_jobs.email_send` as the authoritative “email sent?” indicator.
- Prefer **read-only** DB creds for the dashboard; do not mutate `app_jobs` or `app_wrapped_runs` directly from the dashboard (use HTTP actions).
- “User registered but no run visible” can be normal: users may exist only in `device_emails`/`app_auth_jobs` (pre-finalize) or only in `app_users`/`app_auth_jobs` (finalized but run not created yet). To avoid “invisible users”, keep two primary views: **Auth/Linking** (device/email-centric) and **Runs** (wrapped-run-centric), and support search by `email`, `device_id`, `app_user_id`, and `sec_user_id` (don’t rely on `app_wrapped_runs.email` alone).

## What to share with the dashboard implementer
- Schema: `app/models.py` and `alembic/versions/*` (authoritative column definitions)
- Read model: `docs/admin-dashboard-data-model.md` (this file)
- Write actions: `ENDPOINTS.md` + `scripts/admin-curl.md` (retry endpoints + example calls)

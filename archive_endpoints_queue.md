# Archive API — Xordi (TikTok) Queue Mode

## Auth (QR/OAuth, queue)
Flow: `start-auth` → poll `get-redirect` (QR decode / oauth redirect) → user completes → poll `get-authorization-code` → `finalize`.

### Start — `POST /archive/xordi/start-auth` (201)
- Submits `operation: qr_auth` to Xordi queue with `oauth_client_id` (or `callback_url` fallback).
- Response: `{ "archive_job_id": "...", "expires_at": "...", "queue_position": <int|null> }`.
- Submit errors bubble through as 4xx/5xx (e.g., `user_not_authorized`).

### Poll redirect — `GET /archive/xordi/get-redirect?archive_job_id=...`
- Pending: 202 `{ "status": "pending", "queue_status": "pending|processing", "queue_position": <int|null>, "queue_request_id": "<uuid>" }`.
- If Xordi sends a decoded URL while still pending: 200 `{ "redirect_url": "https://www.tiktok.com/..." }`.
- Completed but no redirect: 200 `{ "status": "authentication already completed" }`.
- Failed/cancelled: 410 `{ "error": "expired", "message": "Job expired", "queue_status": "failed|cancelled", "queue_position": null, "queue_request_id": "<uuid>" }`.

### Poll authorization code — `GET /archive/xordi/get-authorization-code?archive_job_id=...`
- Pending: 202 `{ "status": "pending" }`.
- Completed: 200 `{ "authorization_code": "...", "expires_at": "..." }` (expires_at only when provided by Xordi).
- Failed/cancelled: 410 `{ "status": "expired", "queue_position": <int|null> }`.

### Finalize — `POST /archive/xordi/finalize`
Body: `{ "archive_job_id": "...", "authorization_code": "...", "anchor_token": "<optional>" }`
- Requires `authorization_code`; validates `anchor_token` when provided.
- Exchanges code via Xordi OAuth, links the TikTok account, finalizes the auth job, and returns:
```json
{
  "archive_user_id": "<uuid>",
  "provider_unique_id": "<sec_user_id>",
  "platform_username": "<username>",
  "anchor_merge_performed": false,
  "anchor_token_status": "omitted",
  "anchor_token": "<optional new token>"
}
```
- Errors: 400 (missing/invalid anchor or code), 501 (XORDI_URL/client id/secret missing), 502 (exchange failed).

### Auth curl quickstart
- Start auth:
```bash
curl -s -X POST "$BASE/archive/xordi/start-auth" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" | jq .
```
- Poll redirect:
```bash
curl -s "$BASE/archive/xordi/get-redirect?archive_job_id=$JOB" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" | jq .
```
- Poll authorization code:
```bash
curl -s "$BASE/archive/xordi/get-authorization-code?archive_job_id=$JOB" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" | jq .
```
- Finalize with code:
```bash
curl -s -X POST "$BASE/archive/xordi/finalize" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "archive_job_id": "'$JOB'", "authorization_code": "<code>" }' | jq .
```

### Example happy path (queue)
1) `POST /archive/xordi/start-auth` → 201
```json
{ "archive_job_id": "aj_Ui7kHb7rjZjAS4caNRC5tw", "expires_at": "2025-12-13T06:01:44.423937+00:00", "queue_position": 1 }
```
2) `GET /archive/xordi/get-redirect` → 202 pending
```json
{ "status": "pending", "queue_status": "processing", "queue_position": null, "queue_request_id": "req_8c332911ae2a1f4ff653bbf0" }
```
3) Next poll → 200 redirect
```json
{ "redirect_url": "https://www.tiktok.com/t/ZTHwCjWPcsHwG-JRiFA/" }
```
4) `GET /archive/xordi/get-authorization-code` → 202 pending (may repeat)
```json
{ "status": "pending" }
```
5) Later poll → 200 with code
```json
{ "authorization_code": "SJhHcu0aM1hK0osHJvDn67D4R21jerbk9PgR1tpvLs4" }
```
6) `POST /archive/xordi/finalize` → 200
```json
{
  "archive_user_id": "4d4e3369-e78f-4101-ae2c-e8d4c545e149",
  "provider_unique_id": "7428349000778073130",
  "platform_username": "big.red.button.says",
  "anchor_merge_performed": false,
  "anchor_token_status": "omitted",
  "anchor_token": "at_wKt1uES5C2TrKPaJgmMYAkSWZpASsuuokLgjNiwzpyw"
}
```

## Watch History (queue)
Flow: `watch-history/start` (202) → poll `watch-history/finalize` until not pending.

### Start — `POST /archive/xordi/watch-history/start` (202)
Body:
```json
{ "sec_user_id": "...", "max_pages": 50, "limit": 200, "cursor": "1765604604073" }
```
- `sec_user_id` must already be linked; otherwise 404 `{ "error": "not_found", "message": "TikTok account not found" }`.
- `cursor`: Unix ms timestamp string for older history; `null` starts from newest.
- Success: `{ "data_job_id": "...", "provider_unique_id": "<sec_user_id>", "expires_at": "..." }`.
- Submit errors bubble up (e.g., `user_not_authorized`).
- Per-account queue behavior:
  - Archive will accept and queue up to 10 pending watch-history jobs per TikTok account; they are processed ~1/sec per account.
  - Different TikTok accounts are independent; one account’s queue does not block another.
  - If the per-account queue is exceeded, the start call returns 429 `{ "error": "account_queue_limit_reached", "message": "Xordi queue submit failed" }`.

Curl:
```bash
curl -s -X POST "$BASE/archive/xordi/watch-history/start" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "sec_user_id": "<sec_user_id>", "max_pages": 50, "cursor": null }' | jq .
```

### Finalize — `POST /archive/xordi/watch-history/finalize`
Body: `{ "data_job_id": "...", "return_limit": 200, "include_rows": true|false }`
- Pending: 202 `{ "status": "pending", "data_job_id": "...", "provider_unique_id": "...", "queue_position": <int|null> }` (may include `Retry-After` header).
- Failed/cancelled: 424 `{ "error": "<provider code|provider_failed>", "message": "...", "queue_position": <int|null> }`.
  - Example queue overload: 429 `{ "error": "queue_limit_reached", "message": "Xordi queue submit failed" }` (provider rate/queue limit).
- Success: 200
```json
{
  "provider_unique_id": "<sec_user_id>",
  "status": "success",
  "videos_fetched": 20,
  "videos_stored": 0,
  "videos_updated": 20,
  "approx_times_watched": { "processed": 825, "updated": 0, "skipped": 825, "errors": 0 },
  "pagination": { "has_more": true, "next_cursor": "1763832171206" },
  "rows": null,
  "truncated": null
}
```
- When `include_rows=true`, `rows` contains newest-first rows up to `return_limit` and `truncated` is set.
- Queue failure example surfaced by provider 500:
```json
{
  "request_id": "req_e9142f48705287393dc35369",
  "operation": "watch_history",
  "status": "failed",
  "position": null,
  "error": { "code": "unknown_error", "message": "Request failed with status code 500", "retryable": false }
}
```
Archive returns 424 `{ "error": "unknown_error", "message": "Request failed with status code 500", "queue_position": null }`.

Curl:
```bash
curl -s -X POST "$BASE/archive/xordi/watch-history/finalize" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "data_job_id": "<data_job_id>", "include_rows": false, "return_limit": 1 }' | jq .
```

## Timeline (queue)

### Start — `POST /archive/xordi/timeline/start` (202)
Body: `{ "sec_user_id": "...", "max_videos": 10 }`
- Success: `{ "data_job_id": "...", "provider_unique_id": "<sec_user_id>", "expires_at": "...", "queue_position": <int|null> }`.
- Errors: 404 if TikTok account missing; provider submit errors bubble up.

Curl:
```bash
curl -s -X POST "$BASE/archive/xordi/timeline/start" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "sec_user_id": "<sec_user_id>", "max_videos": 10 }' | jq .
```

### Finalize — `POST /archive/xordi/timeline/finalize`
Body: `{ "data_job_id": "...", "return_limit": 100, "include_rows": true|false }`
- Pending: 202 with queue hints.
- Failed/cancelled: 424 with provider error/message.
- Success: 200 `{ "status": "success", "videos_fetched": <int>, "videos_stored": <int>, "videos_updated": <int>, "summary": { ... }, "rows": [ ... ], "truncated": false }` (rows only when `include_rows=true`).

Curl:
```bash
curl -s -X POST "$BASE/archive/xordi/timeline/finalize" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "data_job_id": "<data_job_id>", "include_rows": false, "return_limit": 100 }' | jq .
```

## Stored Reads (DB)
- `GET /archive/xordi/watch-history?sec_user_id=...&limit=...` → stored watch history; 404 if account missing.

Pagination for stored watch history:
- Default `limit` is 200; max 1000.
- Results are newest-first. Use `next_before` from the response to page older rows:
  - First call: `?sec_user_id=...&limit=1000`
  - Next call: add `before=<next_before>` until `next_before` is null/empty.

- `GET /archive/xordi/timeline?sec_user_id=...&limit=...` → stored timeline; 404 if account missing.
Curl:
```bash
curl -s "$BASE/archive/xordi/watch-history?sec_user_id=<sec_user_id>&limit=50" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" | jq .
```
```bash
curl -s "$BASE/archive/xordi/timeline?sec_user_id=<sec_user_id>&limit=50" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" | jq .
```

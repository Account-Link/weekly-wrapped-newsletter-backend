# Xordi Watch History Behaviors (Queue Mode)

This note illustrates how `/archive/xordi/watch-history/*` behaves for:
- A linked TikTok account with available watch history
- A linked TikTok account where the provider fails to return data
- A bogus/unknown `sec_user_id`

Assumes queue mode is on (`XORDI_QUEUE_MODE=true`), requests are sent to Archive, and `X-Archive-API-Key` is provided.

## 1) Account with watch history available
Start:
```bash
curl -s -X POST "$BASE/archive/xordi/watch-history/start" \
  -H "X-Archive-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{ "sec_user_id": "7504768731169719339", "max_pages": 1, "cursor": null }' | jq .
```
Response (202):
```json
{ "data_job_id": "dj_v6t6MODKlVtW4auvwikM8g", "provider_unique_id": "7504768731169719339", "expires_at": "2025-12-13T22:30:37.154527Z" }
```
Finalize (rows included):
```bash
curl -s -X POST "$BASE/archive/xordi/watch-history/finalize" \
  -H "X-Archive-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{ "data_job_id": "dj_v6t6MODKlVtW4auvwikM8g", "include_rows": true, "return_limit": 1 }' | jq .
```
Response (200 success, rows + pagination):
```json
{
  "provider_unique_id": "7504768731169719339",
  "status": "success",
  "videos_fetched": 19,
  "videos_stored": 0,
  "videos_updated": 19,
  "approx_times_watched": { "processed": 2797, "updated": 0, "skipped": 2797, "errors": 0 },
  "pagination": { "has_more": true, "next_cursor": "1764876517427" },
  "rows": [
    {
      "video_id": "7582759928645864717",
      "url": "https://www.tiktok.com/@etymologynerd/video/7582759928645864717",
      "title": "nothing is objective #etymology #linguistics #slang #lowkenuinely #memes",
      "description": "nothing is objective #etymology #linguistics #slang #lowkenuinely #memes",
      "author": "etymologynerd",
      "author_id": "6903235979081368581",
      "likes": 92017,
      "comments": 816,
      "views": 558371,
      "shares": 6384,
      "duration_ms": 63600,
      "approx_times_watched": 7.547169811320755,
      "music": "original sound - etymologynerd",
      "hashtags": ["etymology", "linguistics", "slang", "lowkenuinely", "memes"],
      "watched_at": "2025-12-13T20:54:24.754000Z"
    }
  ],
  "truncated": false
}
```

## 2) Account linked, provider fails (no rows returned)
Start:
```bash
curl -s -X POST "$BASE/archive/xordi/watch-history/start" \
  -H "X-Archive-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{ "sec_user_id": "6772195950947943429", "max_pages": 1, "cursor": null }' | jq .
```
Response (202):
```json
{ "data_job_id": "dj_YkKvSggb6Yj553K5qXnljg", "provider_unique_id": "6772195950947943429", "expires_at": "2025-12-13T23:04:42.970407Z" }
```
Finalize:
```bash
curl -s -X POST "$BASE/archive/xordi/watch-history/finalize" \
  -H "X-Archive-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{ "data_job_id": "dj_YkKvSggb6Yj553K5qXnljg", "include_rows": true, "return_limit": 1 }' | jq .
```
Response (424 provider failed):
```json
{ "error": "provider_failed", "message": "Request failed with status code 500", "queue_position": null }
```
Notes:
- The job still exists; clients can retry later. The error comes directly from Xordi (e.g., upstream 500).
- Pending responses may precede the failure; once failed, finalize will keep returning 424 until retried with a fresh start.

## 3) Bogus/unknown `sec_user_id`
Start:
```bash
curl -s -X POST "$BASE/archive/xordi/watch-history/start" \
  -H "X-Archive-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{ "sec_user_id": "000", "max_pages": 1, "cursor": null }' | jq .
```
Response (404):
```json
{ "error": "not_found", "message": "TikTok account not found" }
```
Notes:
- Archive validates the TikTok account exists in its DB before queuing; nonexistent accounts return 404 immediately.
- Use `GET /archive/xordi/watch-history?sec_user_id=...` to confirm stored data presence, or re-link the TikTok account before starting.

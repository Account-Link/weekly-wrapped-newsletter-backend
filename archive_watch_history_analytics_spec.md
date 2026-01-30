# Xordi TikTok Watch History Analytics (Stored DB)

These endpoints compute **bounded analytics** from watch history rows already stored in Archive’s database. They **do not** trigger scraping.

Related endpoints:
- Fetch (scrape + store): `POST /archive/xordi/watch-history/start`, `POST /archive/xordi/watch-history/finalize`
- Raw stored rows: `GET /archive/xordi/watch-history`

## Auth
- Header: `X-Archive-API-Key: <key>`

## Errors
- If `sec_user_id` is unknown/unlinked:
  ```json
  { "error": "not_found", "message": "TikTok account not found" }
  ```

## Time range (`range`)

All analytics endpoints accept a `range` object.

**Between (half-open interval)** `[start_at, end_at)`:
```json
{ "type": "between", "start_at": "2025-01-01T00:00:00Z", "end_at": "2026-01-01T00:00:00Z" }
```

**Last N months** (calendar months, anchored by `end_at`, defaults to “now” in UTC):
```json
{ "type": "last_n_months", "months": 12, "end_at": "2025-12-15T00:00:00Z" }
```

## Time zones
- `time_zone` must be an IANA TZ name (e.g. `America/Los_Angeles`, `UTC`).
- Invalid values return `422`:
  ```json
  { "error": "invalid_time_zone", "message": "Unknown IANA time zone" }
  ```

## Metrics

Per-row watch time:
- `duration_ms` null ⇒ `0`
- `approx_times_watched` null ⇒ `1`
- `watch_seconds = (duration_ms / 1000.0) * approx_times_watched`

Night window (local time): hours `22, 23, 0, 1, 2, 3`.

## 1) Coverage

### `GET /archive/xordi/watch-history/analytics/coverage`

**Query params**
- `sec_user_id` (required)

**Example**
```bash
curl -s "$BASE/archive/xordi/watch-history/analytics/coverage?sec_user_id=$SEC_USER_ID" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" | jq .
```

**200 response example**
```json
{
  "sec_user_id": "7504768731169719339",
  "tiktok_account_id": "0a1b2c3d-1111-2222-3333-444455556666",
  "username": "optional_handle_or_null",
  "rows_total": 12345,
  "rows_with_watched_at": 12200,
  "watched_at_min": "2024-02-01T12:00:00Z",
  "watched_at_max": "2025-12-15T01:02:03Z",
  "scraped_at_max": "2025-12-15T01:02:10Z",
  "created_at_max": "2025-12-15T01:02:10Z"
}
```

## 2) Summary

### `POST /archive/xordi/watch-history/analytics/summary`

Computes totals, night %, peak hour, and top lists **within the requested time range** (applied to `watched_at`).

**Request body example**
```json
{
  "sec_user_id": "7504768731169719339",
  "range": { "type": "last_n_months", "months": 12 },
  "time_zone": "America/Los_Angeles",
  "include_hour_histogram": true,
  "top_creators_limit": 5,
  "top_music_limit": 1
}
```

**Example**
```bash
curl -s -X POST "$BASE/archive/xordi/watch-history/analytics/summary" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "sec_user_id": "'"$SEC_USER_ID"'",
    "range": { "type": "last_n_months", "months": 12 },
    "time_zone": "America/Los_Angeles",
    "include_hour_histogram": true,
    "top_creators_limit": 5,
    "top_music_limit": 1
  }' | jq .
```

**Notes**
- `range` is echoed back; for `last_n_months`, Archive includes a concrete `end_at` in the response (even if omitted in the request).
- `hour_histogram_seconds` is only returned when `include_hour_histogram=true` and always includes keys `"0"`..`"23"`.
- `nulls.watched_at` counts rows for the account with missing `watched_at` (these rows are excluded from range-bounded analytics).

**200 response example**
```json
{
  "sec_user_id": "7504768731169719339",
  "tiktok_account_id": "0a1b2c3d-1111-2222-3333-444455556666",
  "range": { "type": "last_n_months", "months": 12, "end_at": "2025-12-15T00:00:00Z" },
  "time_zone": "America/Los_Angeles",
  "totals": { "videos": 1200, "watch_seconds": 54321.0, "watch_hours": 15.09 },
  "night": { "watch_seconds": 4200.0, "watch_pct": 7.73 },
  "peak_hour": 22,
  "hour_histogram_seconds": {
    "0": 12.3,
    "1": 9.0,
    "2": 0.0,
    "3": 0.0,
    "4": 1.2,
    "5": 3.4,
    "6": 2.0,
    "7": 1.0,
    "8": 0.5,
    "9": 0.8,
    "10": 1.1,
    "11": 0.9,
    "12": 2.2,
    "13": 3.3,
    "14": 4.4,
    "15": 5.5,
    "16": 6.6,
    "17": 7.7,
    "18": 8.8,
    "19": 9.9,
    "20": 10.1,
    "21": 11.2,
    "22": 13.3,
    "23": 14.4
  },
  "top_creators": [
    { "author": "pigskin.takes", "author_id": "7572727838634574903", "video_count": 123 }
  ],
  "top_music": [
    { "music": "Fernando", "video_count": 55 }
  ],
  "nulls": { "watched_at": 0, "duration_ms": 10, "approx_times_watched": 200 }
}
```

## 3) Samples

### `POST /archive/xordi/watch-history/analytics/samples`

Returns a small list of “LLM-ready” sample strings without paging through the full dataset.

**Strategy**
- Recent:
  ```json
  { "type": "recent" }
  ```
- Per-month (local time buckets, up to `per_month` per month across the range):
  ```json
  { "type": "per_month", "per_month": 5 }
  ```

**Request body example**
```json
{
  "sec_user_id": "7504768731169719339",
  "range": { "type": "last_n_months", "months": 12 },
  "time_zone": "America/Los_Angeles",
  "strategy": { "type": "per_month", "per_month": 4 },
  "limit": 48,
  "max_chars_per_item": 300,
  "fields": ["title", "description", "hashtags", "music", "author"],
  "include_video_id": true,
  "include_watched_at": true
}
```

**Text formatting**
Archive builds `text` by concatenating, in order:
1) `title`
2) `description`
3) `hashtags` (if JSON array in storage, joined with spaces; otherwise included as-is)
4) `music`
5) `author` (or `author_id` if `author` is null)
Then trims to `max_chars_per_item`.

**Example**
```bash
curl -s -X POST "$BASE/archive/xordi/watch-history/analytics/samples" \
  -H "X-Archive-API-Key: $ARCHIVE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "sec_user_id": "'"$SEC_USER_ID"'",
    "range": { "type": "last_n_months", "months": 12 },
    "time_zone": "America/Los_Angeles",
    "strategy": { "type": "per_month", "per_month": 4 },
    "limit": 48,
    "max_chars_per_item": 300,
    "fields": ["title", "description", "hashtags", "music", "author"],
    "include_video_id": true,
    "include_watched_at": true
  }' | jq .
```

**200 response example**
```json
{
  "sec_user_id": "7504768731169719339",
  "range": { "type": "last_n_months", "months": 12, "end_at": "2025-12-15T00:00:00Z" },
  "time_zone": "America/Los_Angeles",
  "strategy": { "type": "per_month", "per_month": 4 },
  "items": [
    {
      "bucket": "2025-12",
      "watched_at": "2025-12-13T20:54:24.754000Z",
      "video_id": "7582759928645864717",
      "text": "nothing is objective ... #etymology #linguistics ... original sound - etymologynerd etymologynerd"
    }
  ],
  "truncated": false
}
```

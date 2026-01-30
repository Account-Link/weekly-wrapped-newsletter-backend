import time
from typing import Any, Dict, Optional

import httpx
import logging

from app.settings import Settings
from app.observability import get_logger, sanitize_json_bytes

JSONDict = Dict[str, Any]


class ArchiveClient:
    def __init__(self, settings: Settings) -> None:
        base_url = str(settings.archive_base_url).rstrip("/")
        self.base = base_url
        self.api_key = settings.archive_api_key.get_secret_value()
        self._logger = get_logger(__name__)
        self._client = httpx.AsyncClient(
            base_url=self.base,
            timeout=30.0,
            event_hooks={"request": [self._log_request], "response": [self._log_response]},
        )

    async def _headers(self) -> Dict[str, str]:
        return {"X-Archive-API-Key": self.api_key, "Content-Type": "application/json"}

    async def _log_request(self, request: httpx.Request) -> None:
        request.extensions["start_time"] = time.perf_counter()
        include_body = self._logger.isEnabledFor(logging.DEBUG)
        body_text = None
        body_truncated = None
        if include_body:
            try:
                if request.content:
                    body_text, body_truncated = sanitize_json_bytes(request.content)
            except Exception:
                body_text = None
                body_truncated = None
        self._logger.info(
            "archive.request",
            extra={
                "event": "archive.request",
                "archive_method": request.method,
                "archive_url": str(request.url),
                "archive_path": request.url.path,
                "archive_query": str(request.url.query) if request.url.query else None,
                "archive_request_body": body_text,
                "archive_request_body_truncated": body_truncated,
            },
        )

    async def _log_response(self, response: httpx.Response) -> None:
        start = response.request.extensions.get("start_time")
        duration_ms = None
        try:
            if start is not None:
                duration_ms = int((time.perf_counter() - float(start)) * 1000)
        except Exception:
            duration_ms = None

        include_body = response.status_code >= 400 or self._logger.isEnabledFor(logging.DEBUG)
        body_text = None
        body_truncated = None
        if include_body:
            try:
                content = await response.aread()
                # Preserve content for downstream `.json()` / `.text`.
                response._content = content  # type: ignore[attr-defined]
                if content:
                    body_text, body_truncated = sanitize_json_bytes(content)
            except Exception:
                body_text = None
                body_truncated = None

        self._logger.info(
            "archive.response",
            extra={
                "event": "archive.response",
                "archive_method": response.request.method,
                "archive_url": str(response.request.url),
                "archive_path": response.request.url.path,
                "archive_status": response.status_code,
                "archive_duration_ms": duration_ms,
                "archive_response_body": body_text,
                "archive_response_body_truncated": body_truncated,
            },
        )

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.request(method, url, **kwargs)
        except Exception as exc:
            self._logger.exception(
                "archive.exception",
                extra={
                    "event": "archive.exception",
                    "archive_method": method,
                    "archive_path": url,
                },
            )
            raise

    async def start_xordi_auth(self, anchor_token: Optional[str] = None) -> JSONDict:
        body: JSONDict = {}
        if anchor_token:
            body["anchor_token"] = anchor_token
        resp = await self._request("POST", "/archive/xordi/start-auth", json=body, headers=await self._headers())
        resp.raise_for_status()
        return resp.json()

    async def get_redirect(self, archive_job_id: str) -> httpx.Response:
        return await self._request(
            "GET",
            "/archive/xordi/get-redirect",
            params={"archive_job_id": archive_job_id},
            headers=await self._headers(),
        )

    async def get_authorization_code(self, archive_job_id: str) -> httpx.Response:
        return await self._request(
            "GET",
            "/archive/xordi/get-authorization-code",
            params={"archive_job_id": archive_job_id},
            headers=await self._headers(),
        )

    async def get_queue_status(self) -> JSONDict:
        resp = await self._request(
            "GET",
            "/archive/xordi/queue-status",
            headers=await self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def delete_queue_request(self, archive_job_id: str) -> httpx.Response:
        return await self._request(
            "DELETE",
            "/archive/xordi/queue-request",
            params={"archive_job_id": archive_job_id},
            headers=await self._headers(),
        )

    async def finalize_xordi(self, archive_job_id: str, authorization_code: str, anchor_token: Optional[str]) -> JSONDict:
        body: JSONDict = {"archive_job_id": archive_job_id, "authorization_code": authorization_code}
        if anchor_token:
            body["anchor_token"] = anchor_token
        resp = await self._request("POST", "/archive/xordi/finalize", json=body, headers=await self._headers())
        resp.raise_for_status()
        return resp.json()

    async def get_watch_history(self, sec_user_id: str, limit: int = 200, before: Optional[str] = None) -> JSONDict:
        params: Dict[str, Any] = {"sec_user_id": sec_user_id, "limit": limit}
        if before:
            params["before"] = before
        resp = await self._request("GET", "/archive/xordi/watch-history", params=params, headers=await self._headers())
        resp.raise_for_status()
        return resp.json()

    async def delete_watch_history(self, sec_user_id: str) -> JSONDict:
        resp = await self._request(
            "DELETE",
            "/archive/xordi/watch-history",
            params={"sec_user_id": sec_user_id},
            headers=await self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def start_watch_history(self, sec_user_id: str, limit: int = 200, max_pages: int = 1, cursor: Optional[str] = None) -> httpx.Response:
        body: JSONDict = {"sec_user_id": sec_user_id, "limit": limit, "max_pages": max_pages, "cursor": cursor}
        return await self._request("POST", "/archive/xordi/watch-history/start", json=body, headers=await self._headers())

    async def finalize_watch_history(
        self, data_job_id: str, include_rows: bool = True, return_limit: Optional[int] = None
    ) -> httpx.Response:
        body: JSONDict = {"data_job_id": data_job_id, "include_rows": include_rows}
        if return_limit is not None:
            body["return_limit"] = return_limit
        return await self._request("POST", "/archive/xordi/watch-history/finalize", json=body, headers=await self._headers())

    async def watch_history_analytics_coverage(self, sec_user_id: str) -> JSONDict:
        resp = await self._request(
            "GET",
            "/archive/xordi/watch-history/analytics/coverage",
            params={"sec_user_id": sec_user_id},
            headers=await self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def watch_history_analytics_summary(
        self,
        sec_user_id: str,
        range: JSONDict,
        time_zone: Optional[str] = None,
        include_hour_histogram: bool = False,
        top_creators_limit: int = 5,
        top_music_limit: int = 1,
    ) -> JSONDict:
        body: JSONDict = {
            "sec_user_id": sec_user_id,
            "range": range,
            "include_hour_histogram": include_hour_histogram,
            "top_creators_limit": top_creators_limit,
            "top_music_limit": top_music_limit,
        }
        if time_zone:
            body["time_zone"] = time_zone
        resp = await self._request(
            "POST",
            "/archive/xordi/watch-history/analytics/summary",
            json=body,
            headers=await self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def watch_history_analytics_samples(
        self,
        sec_user_id: str,
        range: JSONDict,
        time_zone: Optional[str] = None,
        strategy: Optional[JSONDict] = None,
        limit: int = 50,
        max_chars_per_item: int = 300,
        fields: Optional[list] = None,
        include_video_id: bool = True,
        include_watched_at: bool = True,
    ) -> JSONDict:
        body: JSONDict = {
            "sec_user_id": sec_user_id,
            "range": range,
            "strategy": strategy or {"type": "recent"},
            "limit": limit,
            "max_chars_per_item": max_chars_per_item,
            "include_video_id": include_video_id,
            "include_watched_at": include_watched_at,
        }
        if time_zone:
            body["time_zone"] = time_zone
        if fields:
            body["fields"] = fields
        resp = await self._request(
            "POST",
            "/archive/xordi/watch-history/analytics/samples",
            json=body,
            headers=await self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def interests_enrichment_reconcile(
        self,
        *,
        sec_user_id: str,
        range: JSONDict,
        time_zone: Optional[str] = None,
        mode: str = "missing_only",
        max_creators: int = 500,
        enqueue: bool = True,
    ) -> httpx.Response:
        body: JSONDict = {
            "sec_user_id": sec_user_id,
            "range": range,
            "time_zone": time_zone or "UTC",
            "mode": mode,
            "max_creators": int(max_creators),
            "enqueue": bool(enqueue),
        }
        return await self._request(
            "POST",
            "/archive/xordi/watch-history/enrichment/interests/reconcile",
            json=body,
            headers=await self._headers(),
        )

    async def interests_enrichment_status(self, *, job_id: str) -> httpx.Response:
        return await self._request(
            "GET",
            "/archive/xordi/watch-history/enrichment/interests/status",
            params={"job_id": job_id},
            headers=await self._headers(),
        )

    async def interests_summary(
        self,
        *,
        sec_user_id: str,
        range: JSONDict,
        time_zone: Optional[str] = None,
        group_by: str = "none",
        limit: int = 10,
        include_unknown: bool = True,
    ) -> JSONDict:
        body: JSONDict = {
            "sec_user_id": sec_user_id,
            "range": range,
            "time_zone": time_zone or "UTC",
            "group_by": group_by,
            "limit": int(limit),
            "include_unknown": bool(include_unknown),
        }
        resp = await self._request(
            "POST",
            "/archive/xordi/watch-history/analytics/interests/summary",
            json=body,
            headers=await self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

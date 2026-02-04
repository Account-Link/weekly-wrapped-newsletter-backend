"""TikTok Ads Creative Radar API client for popular trends (creator, sound, hashtag)."""

import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.observability import get_logger

MAX_TREND_ITEMS = 100
PAGE_SIZE = 20
MAX_PAGES = 10  # cap to avoid infinite loops


class TikTokCreativeRadarError(Exception):
    """API returned code != 0 or request failed."""

    def __init__(self, message: str, code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.response = response


class TikTokCreativeRadarClient:
    """Client for TikTok Ads Creative Radar popular trend APIs (paginated)."""

    BASE_URL = "https://ads.tiktok.com/creative_radar_api/v1/popular_trend"

    def __init__(
        self,
        *,
        cookie: Optional[str] = None,
        user_sign: Optional[str] = None,
        web_id: Optional[str] = None,
        creator_cookie: Optional[str] = None,
        creator_user_sign: Optional[str] = None,
        creator_web_id: Optional[str] = None,
        sound_cookie: Optional[str] = None,
        sound_user_sign: Optional[str] = None,
        sound_web_id: Optional[str] = None,
        hashtag_cookie: Optional[str] = None,
        hashtag_user_sign: Optional[str] = None,
        hashtag_web_id: Optional[str] = None,
        country_code: str = "US",
        timeout: float = 30.0,
    ) -> None:
        self._fallback_cookie = (cookie or "").strip()
        self._fallback_user_sign = (user_sign or "").strip()
        self._fallback_web_id = (web_id or "").strip()

        self._creator_cookie = (creator_cookie or "").strip()
        self._creator_user_sign = (creator_user_sign or "").strip()
        self._creator_web_id = (creator_web_id or "").strip()

        self._sound_cookie = (sound_cookie or "").strip()
        self._sound_user_sign = (sound_user_sign or "").strip()
        self._sound_web_id = (sound_web_id or "").strip()

        self._hashtag_cookie = (hashtag_cookie or "").strip()
        self._hashtag_user_sign = (hashtag_user_sign or "").strip()
        self._hashtag_web_id = (hashtag_web_id or "").strip()

        self._country_code = country_code.upper()
        self._timeout = timeout
        self._logger = get_logger(__name__)
        self._client = httpx.AsyncClient(timeout=timeout)

    def _creds_for(self, trend_type: str) -> Tuple[str, str, str]:
        if trend_type == "creator":
            c = self._creator_cookie or self._fallback_cookie
            u = self._creator_user_sign or self._fallback_user_sign
            w = self._creator_web_id or self._fallback_web_id
            return c, u, w
        if trend_type == "sound":
            c = self._sound_cookie or self._fallback_cookie
            u = self._sound_user_sign or self._fallback_user_sign
            w = self._sound_web_id or self._fallback_web_id
            return c, u, w
        c = self._hashtag_cookie or self._fallback_cookie
        u = self._hashtag_user_sign or self._fallback_user_sign
        w = self._hashtag_web_id or self._fallback_web_id
        return c, u, w

    def _headers(self, trend_type: str, referer: str) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        cookie, user_sign, web_id = self._creds_for(trend_type)
        return {
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,zh-HK;q=0.6,ko;q=0.5,fr;q=0.4,ru;q=0.3,tr;q=0.2,vi;q=0.1",
            "cache-control": "no-cache",
            "lang": "en",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": referer,
            "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "timestamp": timestamp,
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
            "user-sign": user_sign,
            "web-id": web_id,
            "Cookie": cookie,
        }

    def is_configured(self) -> bool:
        cookie, user_sign, web_id = self._creds_for("creator")
        return bool(cookie and user_sign and web_id)

    async def _request(
        self,
        method: str,
        url: str,
        params: Dict[str, Any],
        trend_type: str,
        referer: str,
    ) -> Dict[str, Any]:
        resp = await self._client.request(
            method,
            url,
            params=params,
            headers=self._headers(trend_type, referer),
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise TikTokCreativeRadarError("Invalid response: not a JSON object")
        code = data.get("code")
        if code != 0:
            raise TikTokCreativeRadarError(
                data.get("msg") or f"API error code {code}",
                code=code,
                response=data,
            )
        return data

    async def fetch_creator_page(self, page: int = 1, limit: int = PAGE_SIZE) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/creator/list"
        params = {
            "page": page,
            "limit": limit,
            "sort_by": "follower",
            "creator_country": self._country_code,
        }
        return await self._request(
            "GET",
            url,
            params,
            "creator",
            "https://ads.tiktok.com/business/creativecenter/inspiration/popular/creator/pc/en",
        )

    async def fetch_sound_page(self, page: int = 1, limit: int = PAGE_SIZE) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/sound/rank_list"
        params = {
            "period": 7,
            "page": page,
            "limit": limit,
            "rank_type": "popular",
            "new_on_board": "false",
            "commercial_music": "false",
            "country_code": self._country_code,
        }
        return await self._request(
            "GET",
            url,
            params,
            "sound",
            "https://ads.tiktok.com/business/creativecenter/inspiration/popular/song/pc/en",
        )

    async def fetch_hashtag_page(self, page: int = 1, limit: int = PAGE_SIZE) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/hashtag/list"
        params = {
            "page": page,
            "limit": limit,
            "period": 7,
            "country_code": self._country_code,
            "sort_by": "popular",
        }
        return await self._request(
            "GET",
            url,
            params,
            "hashtag",
            "https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en",
        )

    async def fetch_all_creators(self, max_items: int = MAX_TREND_ITEMS) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        page = 1
        for _ in range(MAX_PAGES):
            if len(results) >= max_items:
                break
            limit = min(PAGE_SIZE, max_items - len(results))
            raw = await self.fetch_creator_page(page=page, limit=limit)
            data = raw.get("data") or {}
            creators = data.get("creators") or []
            for c in creators:
                if isinstance(c, dict):
                    results.append(c)
                    if len(results) >= max_items:
                        break
            pagination = data.get("pagination") or {}
            if not pagination.get("has_more"):
                break
            page += 1
        return results[:max_items]

    async def fetch_all_sounds(self, max_items: int = MAX_TREND_ITEMS) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        page = 1
        for _ in range(MAX_PAGES):
            if len(results) >= max_items:
                break
            limit = min(PAGE_SIZE, max_items - len(results))
            raw = await self.fetch_sound_page(page=page, limit=limit)
            data = raw.get("data") or {}
            sound_list = data.get("sound_list") or []
            for s in sound_list:
                if isinstance(s, dict):
                    results.append(s)
                    if len(results) >= max_items:
                        break
            pagination = data.get("pagination") or {}
            if not pagination.get("has_more"):
                break
            page += 1
        return results[:max_items]

    async def fetch_all_hashtags(self, max_items: int = MAX_TREND_ITEMS) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        page = 1
        for _ in range(MAX_PAGES):
            if len(results) >= max_items:
                break
            limit = min(PAGE_SIZE, max_items - len(results))
            raw = await self.fetch_hashtag_page(page=page, limit=limit)
            data = raw.get("data") or {}
            lst = data.get("list") or []
            for h in lst:
                if isinstance(h, dict):
                    results.append(h)
                    if len(results) >= max_items:
                        break
            pagination = data.get("pagination") or {}
            if not pagination.get("has_more"):
                break
            page += 1
        return results[:max_items]

    async def close(self) -> None:
        await self._client.aclose()

"""TikTok Ads Creative Radar API client for popular trends (creator, sound, hashtag).

Supports two modes:
- Anonymous (default): when TIKTOK_ADS_COOKIE / USER_SIGN / WEB_ID are not set,
  uses built-in anonymous headers so the API can be called without logging in.
- Logged-in: set the three env vars (or pass to constructor) to use a real session;
  see docs/tiktok-creative-radar.md if you get "no permission".
"""

import time
from typing import Any, Dict, List, Optional

import httpx

from app.observability import get_logger

MAX_TREND_ITEMS = 100
PAGE_SIZE = 20
MAX_PAGES = 10  # cap to avoid infinite loops

# Default headers for unauthenticated (anonymous) access to Creative Radar.
# Override via TIKTOK_ADS_COOKIE / TIKTOK_ADS_USER_SIGN / TIKTOK_ADS_WEB_ID if needed.
DEFAULT_ANONYMOUS_USER_ID = "27d8c5bc-dcae-42dc-9cb0-d2d6143051a6"
DEFAULT_ANONYMOUS_USER_SIGN = "d19e733d3e3c5520"
DEFAULT_ANONYMOUS_WEB_ID = "7601202182150424071"
DEFAULT_ANONYMOUS_COOKIE = (
    "passport_csrf_token=a7231afff3f4d93ac9b79ff3c99ad7cd; passport_csrf_token_default=a7231afff3f4d93ac9b79ff3c99ad7cd; "
    "store-country-code=us; store-country-code-src=uid; tt-target-idc=useast5; multi_sids=7589159141932434446%3A5692b1702bfdde57ee60107e5f5b7310; "
    "cmpl_token=AgQQAPNSF-RO0rls8qBG-F038zOCSq4H_53ZYKLGiw; sid_guard=5692b1702bfdde57ee60107e5f5b7310%7C1766991334%7C15552000%7CSat%2C+27-Jun-2026+06%3A55%3A34+GMT; "
    "uid_tt=50b6527f1b5b6568ba7aecff1a360ae3ec60296105b9449514fb34bd02c4eab0; uid_tt_ss=50b6527f1b5b6568ba7aecff1a360ae3ec60296105b9449514fb34bd02c4eab0; "
    "sid_tt=5692b1702bfdde57ee60107e5f5b7310; sessionid=5692b1702bfdde57ee60107e5f5b7310; sessionid_ss=5692b1702bfdde57ee60107e5f5b7310; "
    "tt_session_tlb_tag=sttt%7C4%7CVpKxcCv93lfuYBB-X1tzEP_________p3mONJAj9ERB8VY_13w3X6mTQCZ7tjmSxdWhd5vYCaIA%3D; "
    "sid_ucp_v1=1.0.1-KDIyYTQ4YWZiNDY3NzdiNTFhNGFmYmUzZjM3MWI5ZGMwOWIyMTRmNGYKIQiOiKr2_ICHqWkQ5svIygYYswsgDDCkuMjKBjgIQBJIBBAEGgd1c2Vhc3Q1IiA1NjkyYjE3MDJiZmRkZTU3ZWU2MDEwN2U1ZjViNzMxMDJOCiD_yWW6h5Zvob7eXiEWv1t2jvrZvBscHDyz6V6E2DkP7RIg0Ex4HtLE4TZiUKROYkyQraxIuWfsWjMvMg4TmxYWrVcYBSIGdGlrdG9r; "
    "ssid_ucp_v1=1.0.1-KDIyYTQ4YWZiNDY3NzdiNTFhNGFmYmUzZjM3MWI5ZGMwOWIyMTRmNGYKIQiOiKr2_ICHqWkQ5svIygYYswsgDDCkuMjKBjgIQBJIBBAEGgd1c2Vhc3Q1IiA1NjkyYjE3MDJiZmRkZTU3ZWU2MDEwN2U1ZjViNzMxMDJOCiD_yWW6h5Zvob7eXiEWv1t2jvrZvBscHDyz6V6E2DkP7RIg0Ex4HtLE4TZiUKROYkyQraxIuWfsWjMvMg4TmxYWrVcYBSIGdGlrdG9r; "
    "store-idc=useast5; tt-target-idc-sign=HP1rITh0YnRCnDx9WwD8auM-yuXBfvBCnjAW7I_bkfB-Rrs3cOnuWgfd9pFRveJY2aWbyFABbqkvz6OKRibdVqnkepLhDX5E9GvF_pdslt454_OyoouYIC0p__IKfsmWBag4nsZo6peRno30gm_Q9F28onarztkcWv_H7fguTdCXyJPitOOb3iSxrMBfUBcVAIVO_7VawXZAgg3TOLs-yvM1WHtCbtZlz-dqm7SA7HbNesbwUTNujqHsb7vHQWV0lma46hRpikU-PZ7mQNlmCwjzryybSUPnvlbqnofm52agJfDeEHSQa6PbT5BcSGM339wIhfxetq7TRw7IBuzFxMkW4voAtjKkqn2K5RgkksYUvf9kI_Rb37FKmNfXE9xKU1KwPNCStARCwvvMtrOPAZu7jYDtiXl592W-etA1IsqDvLlq-XLI0EdqPmaDljmXqGRY0QMe-H4Y8uqYPdVE8dVNjYYRquxEs-XYtEFp45VQG_8dFazbn4b9GrYy3qkJ; "
    "_ga=GA1.1.38602867.1766991367; tt_ticket_guard_client_web_domain=2; _ga_BZBQ2QHQSP=GS2.1.s1766991366$o1$g1$t1766991509$j44$l0$h1650451058; "
    "lang_type=en; store-country-sign=MEIEDBGyvlsLo15Rl3zI8QQgf01H7fv2mf9ozJbKOxNFQa1_AMQlcU8jGh_3J0l4UPsEENlpdW_yPw8jo0uNTeWTrcg; "
    "ttwid=1%7CKdMyYGvUQLa-0WZkMOVNMsCEKXJKbdZjROygtrNy-C4%7C1770004675%7Cb1abc99b011d203a424ed5d4fc54be5722a7b3e9aead3f5dbb1c03b0cd1316fa; "
    "msToken=D6fDB8va-sxzA3C9q3tXZUXPr2baSWTk1NStv4Rau39xucdyb1W4fCFgbftodOHLLFs1bIUtik828PORuOR3QaMNOLAc4TjoCwWIHzIy31GBvdxU5eGU_7tXSPcRkUFIYWlX2eE="
)


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
        country_code: str = "US",
        timeout: float = 30.0,
    ) -> None:
        # Use anonymous defaults when not provided so unauthenticated access works
        self._cookie = (cookie or "").strip() or DEFAULT_ANONYMOUS_COOKIE
        self._user_sign = (user_sign or "").strip() or DEFAULT_ANONYMOUS_USER_SIGN
        self._web_id = (web_id or "").strip() or DEFAULT_ANONYMOUS_WEB_ID
        self._anonymous_user_id = DEFAULT_ANONYMOUS_USER_ID
        self._country_code = country_code.upper()
        self._timeout = timeout
        self._logger = get_logger(__name__)
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        return {
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,zh-HK;q=0.6,ko;q=0.5,fr;q=0.4,ru;q=0.3,tr;q=0.2,vi;q=0.1",
            "anonymous-user-id": self._anonymous_user_id,
            "cache-control": "no-cache",
            "lang": "en",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en",
            "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "timestamp": timestamp,
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
            "user-sign": self._user_sign,
            "web-id": self._web_id,
            "Cookie": self._cookie,
        }

    def is_configured(self) -> bool:
        return bool(self._cookie and self._user_sign and self._web_id)

    async def _request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resp = await self._client.request(
            method,
            url,
            params=params,
            headers=self._headers(),
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
        """Fetch one page of creator list. Returns API response with data.creators and data.pagination."""
        url = f"{self.BASE_URL}/creator/list"
        params = {
            "page": page,
            "limit": limit,
            "sort_by": "avg_views",
            "creator_country": self._country_code,
            "audience_country": self._country_code,
        }
        return await self._request("GET", url, params=params)

    async def fetch_sound_page(self, page: int = 1, limit: int = PAGE_SIZE) -> Dict[str, Any]:
        """Fetch one page of sound rank list. Returns API response with data.sound_list and data.pagination."""
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
        return await self._request("GET", url, params=params)

    async def fetch_hashtag_page(self, page: int = 1, limit: int = PAGE_SIZE) -> Dict[str, Any]:
        """Fetch one page of hashtag list. Returns API response with data.list and data.pagination."""
        url = f"{self.BASE_URL}/hashtag/list"
        params = {
            "page": page,
            "limit": limit,
            "period": 7,
            "country_code": self._country_code,
            "sort_by": "popular",
        }
        return await self._request("GET", url, params=params)

    async def fetch_all_creators(self, max_items: int = MAX_TREND_ITEMS) -> List[Dict[str, Any]]:
        """Paginate and return up to max_items creators."""
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
        """Paginate and return up to max_items sounds."""
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
        """Paginate and return up to max_items hashtags."""
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

"""Capture TikTok Creative Radar headers (Cookie, user-sign, web-id) via Playwright.

Called **inside the worker** at the start of each fetch_trends job (when
TIKTOK_ADS_SESSION_DIR is set): no separate script run is needed per fetch.
Worker runs this in-process → gets headers → then calls the Creative Radar API.

The standalone script (scripts/refresh_tiktok_creative_radar_headers.py) is only
for one-time creation of state.json (log in in a browser, run script with
--persist). After that, the worker loads state.json and captures headers here
before every fetch. Playwright is optional; if not installed or capture fails,
returns None and caller falls back to env vars or anonymous defaults.
"""

import asyncio
import os
from typing import Optional, Tuple

POPULAR_URL = "https://ads.tiktok.com/business/creativecenter/inspiration/popular/creator/pc/en"
API_PATTERN = "creative_radar_api/v1/popular_trend/"
CAPTURE_TIMEOUT = 90.0  # seconds per navigation attempt
NAVIGATION_TIMEOUT_MS = 60_000
RETRY_NAVIGATION = 1  # one retry if no request captured


def _get_header(request, name: str) -> Optional[str]:
    raw = request.headers.get(name)
    return (raw.strip() or None) if raw else None


async def capture_headers(session_dir: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """Capture Cookie, user-sign, web-id from Creative Radar API request.

    Uses Playwright (optional): launches headless Chromium, loads session from
    session_dir/state.json if present, navigates to Popular page, waits for
    creative_radar_api request and extracts the three headers.

    Returns (cookie, user_sign, web_id) or None if Playwright not installed,
    session_dir invalid, or capture failed (e.g. timeout, no request).
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return None

    if not session_dir or not os.path.isdir(session_dir):
        return None

    state_path = os.path.join(session_dir, "state.json")
    if not os.path.isfile(state_path):
        return None

    captured: dict = {"cookie": None, "user_sign": None, "web_id": None}
    done = asyncio.Event()

    def on_request(request):
        if API_PATTERN in request.url and not done.is_set():
            captured["cookie"] = _get_header(request, "cookie")
            captured["user_sign"] = _get_header(request, "user-sign")
            captured["web_id"] = _get_header(request, "web-id")
            if captured["cookie"] and captured["user_sign"] and captured["web_id"]:
                done.set()

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(storage_state=state_path)
                page = await context.new_page()
                page.on("request", on_request)

                for attempt in range(RETRY_NAVIGATION + 1):
                    await page.goto(POPULAR_URL, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
                    try:
                        await asyncio.wait_for(done.wait(), timeout=CAPTURE_TIMEOUT)
                    except asyncio.TimeoutError:
                        if attempt < RETRY_NAVIGATION:
                            await asyncio.sleep(5)
                            continue
                        return None
                    break

                cookie = captured.get("cookie") or ""
                user_sign = captured.get("user_sign") or ""
                web_id = captured.get("web_id") or ""
                if not (cookie and user_sign and web_id):
                    return None

                # Persist state again so session stays fresh for next run
                await context.storage_state(path=state_path)
                return (cookie, user_sign, web_id)
            finally:
                await browser.close()
    except Exception:
        return None

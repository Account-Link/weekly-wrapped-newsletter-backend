#!/usr/bin/env python3
"""Semi-automated capture of TikTok Creative Radar headers (Cookie, user-sign, web-id).

Opens a browser, navigates to Creative Center Popular page; when the page (or you
after logging in) triggers the creative_radar_api request, we capture the three
headers and print them as env vars or write to a file.

Usage:
  uv run python scripts/refresh_tiktok_creative_radar_headers.py
  uv run python scripts/refresh_tiktok_creative_radar_headers.py --output .env.radar
  uv run python scripts/refresh_tiktok_creative_radar_headers.py --headless  # only if already logged in elsewhere and session in storage

Requires: uv sync --extra tiktok-radar-refresh && uv run playwright install chromium
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

# Playwright is optional; fail with clear message if not installed
try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright not installed. Run: uv sync --extra tiktok-radar-refresh && uv run playwright install chromium", file=sys.stderr)
    sys.exit(1)

POPULAR_URL = "https://ads.tiktok.com/business/creativecenter/inspiration/popular/creator/pc/en"
API_PATTERN = "creative_radar_api/v1/popular_trend/"
WAIT_TIMEOUT_MS = 120_000  # 2 minutes per attempt
RETRY_NAVIGATION = 2  # retry navigation if no request captured


def _get_header(request, name: str) -> Optional[str]:
    raw = request.headers.get(name)
    return (raw.strip() or None) if raw else None


async def run(headed: bool, output_path: Optional[str], persist_storage: Optional[str]) -> bool:
    captured = {"cookie": None, "user_sign": None, "web_id": None}
    done = asyncio.Event()

    def on_request(request):
        if API_PATTERN in request.url and not done.is_set():
            captured["cookie"] = _get_header(request, "cookie")
            captured["user_sign"] = _get_header(request, "user-sign")
            captured["web_id"] = _get_header(request, "web-id")
            if captured["cookie"] and captured["user_sign"] and captured["web_id"]:
                done.set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        try:
            state_path = os.path.join(persist_storage, "state.json") if persist_storage else None
            if state_path and os.path.isfile(state_path):
                context = await browser.new_context(storage_state=state_path)
            else:
                context = await browser.new_context()
            await context.add_init_script("() => {}")
            page = await context.new_page()
            page.on("request", on_request)

            for attempt in range(RETRY_NAVIGATION + 1):
                if attempt > 0:
                    print("No Creative Radar request seen yet. If you need to log in, do it in the browser now.", flush=True)
                    await asyncio.sleep(10)
                print(f"Navigating to Creative Center Popular (attempt {attempt + 1})...", flush=True)
                await page.goto(POPULAR_URL, wait_until="domcontentloaded", timeout=60_000)
                try:
                    await asyncio.wait_for(done.wait(), timeout=WAIT_TIMEOUT_MS / 1000)
                except asyncio.TimeoutError:
                    if attempt < RETRY_NAVIGATION:
                        continue
                    print("Timeout: Creative Radar API request was not captured.", file=sys.stderr)
                    print("Make sure you are logged in at ads.tiktok.com and that Creative Center is accessible.", file=sys.stderr)
                    return False
                break

            cookie = captured.get("cookie") or ""
            user_sign = captured.get("user_sign") or ""
            web_id = captured.get("web_id") or ""
            if not (cookie and user_sign and web_id):
                print("Captured headers are incomplete.", file=sys.stderr)
                return False

            def escape_env(v: str) -> str:
                return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

            lines = [
                f'TIKTOK_ADS_COOKIE="{escape_env(cookie)}"',
                f'TIKTOK_ADS_USER_SIGN="{escape_env(user_sign)}"',
                f'TIKTOK_ADS_WEB_ID="{escape_env(web_id)}"',
            ]
            env_block = "\n".join(lines)

            if output_path:
                with open(output_path, "w") as f:
                    f.write("# TikTok Creative Radar headers (do not commit)\n")
                    f.write(env_block)
                    f.write("\n")
                print(f"Wrote {len(lines)} env lines to {output_path}", flush=True)
            else:
                print("\n# Add these to your .env or export:\n")
                print(env_block, flush=True)

            if persist_storage:
                os.makedirs(persist_storage, exist_ok=True)
                await context.storage_state(path=os.path.join(persist_storage, "state.json"))
                print(f"\nSession saved to {persist_storage} (use --persist next run for headless).", flush=True)

            return True
        finally:
            await browser.close()

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture TikTok Creative Radar headers from browser")
    parser.add_argument("--headless", action="store_true", help="Run browser headless (only works if session was saved with --persist)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Write env vars to this file instead of stdout")
    parser.add_argument("--persist", type=str, default=None, metavar="DIR", help="Save/load browser session to DIR for headless reuse")
    args = parser.parse_args()
    ok = asyncio.run(run(headed=not args.headless, output_path=args.output, persist_storage=args.persist))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

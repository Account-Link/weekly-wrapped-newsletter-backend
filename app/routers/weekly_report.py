"""Weekly report routes."""
import json
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Response, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from datetime import timedelta

from app.db import SessionLocal
from app.models import (
    AppUser,
    AppUserEmail,
    TikTokRadarHeaderConfig,
    WeeklyReport,
    WeeklyReportGlobal,
    WeeklyTrendCreator,
    WeeklyTrendHashtag,
    WeeklyTrendSound,
)
from app.schemas import (
    TopicItem,
    UnsubscribeRequest,
    UnsubscribeResponse,
    WeeklyReportResponse,
    WeeklyReportTestRequest,
    WeeklyReportTestResponse,
)
from app.services.job_queue import DBJobQueue
from app.settings import get_settings
from app.observability import get_logger

router = APIRouter()
job_queue = DBJobQueue()
settings = get_settings()
logger = get_logger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _latest_user_email(db: Session, app_user_id: str) -> Optional[str]:
    latest_email = (
        db.query(AppUserEmail)
        .filter(AppUserEmail.app_user_id == app_user_id)
        .order_by(AppUserEmail.created_at.desc())
        .first()
    )
    if latest_email and isinstance(latest_email.email, str):
        return latest_email.email
    return None


def _weekly_trends_ready(db: Session, global_report_id: int) -> bool:
    hashtag_count = db.query(WeeklyTrendHashtag).filter(WeeklyTrendHashtag.global_report_id == global_report_id).count()
    sound_count = db.query(WeeklyTrendSound).filter(WeeklyTrendSound.global_report_id == global_report_id).count()
    creator_count = db.query(WeeklyTrendCreator).filter(WeeklyTrendCreator.global_report_id == global_report_id).count()
    return hashtag_count > 0 and sound_count > 0 and creator_count > 0


def _to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def require_weekly_token(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_weekly_token: Optional[str] = Header(None, alias="X-Weekly-Token"),
):
    expected = settings.weekly_token.get_secret_value() if settings.weekly_token else None
    if not expected:
        raise HTTPException(status_code=404, detail="not_found")
    
    # Check Authorization header (Bearer token) or X-Weekly-Token header
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_weekly_token:
        token = x_weekly_token.strip()
    
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_weekly_token")
    return True


def _require_trends_page_password(password: str) -> None:
    expected = (
        settings.tiktok_trends_admin_password.get_secret_value()
        if settings.tiktok_trends_admin_password
        else None
    )
    if not expected:
        raise HTTPException(status_code=404, detail="not_found")
    provided = (password or "").strip()
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_password")


def _render_trends_login_page(error: Optional[str] = None) -> str:
    error_html = f"<div class='error'>{error}</div>" if error else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TikTok Trends Login</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; display: grid; place-items: center; min-height: 80vh; }}
    .card {{ width: 100%; max-width: 420px; border: 1px solid #e7e7e7; border-radius: 12px; padding: 20px; box-sizing: border-box; }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    p {{ margin: 0 0 14px; color: #666; }}
    label {{ display: block; margin-bottom: 6px; font-weight: 600; }}
    input {{ width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; }}
    button {{ margin-top: 12px; width: 100%; padding: 10px 14px; border: none; border-radius: 8px; cursor: pointer; background: #111; color: #fff; }}
    .error {{ margin-bottom: 10px; color: #b00020; font-size: 13px; }}
  </style>
</head>
<body>
  <form class="card" method="post" action="/admin/weekly-report/trends/page/login">
    <h1>TikTok Trends 管理页登录</h1>
    <p>请输入密码后继续。</p>
    {error_html}
    <label for="password">密码</label>
    <input id="password" name="password" type="password" required />
    <button type="submit">登录</button>
  </form>
</body>
</html>"""


def _render_trends_admin_page(weekly_token: str) -> str:
    escaped_token = weekly_token.replace("\\", "\\\\").replace("'", "\\'")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TikTok Trends Admin</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; max-width: 920px; }}
    h1 {{ margin-bottom: 8px; }}
    .hint {{ color: #555; margin-bottom: 16px; }}
    label {{ display: block; margin: 10px 0 6px; font-weight: 600; }}
    input, textarea {{ width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; }}
    textarea {{ min-height: 96px; resize: vertical; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .actions {{ margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }}
    button {{ padding: 10px 14px; border: none; border-radius: 8px; cursor: pointer; background: #111; color: #fff; }}
    button.secondary {{ background: #666; }}
    pre {{ margin-top: 16px; background: #f7f7f7; border: 1px solid #eee; border-radius: 8px; padding: 12px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>TikTok Trend Header 配置</h1>
  <div class="hint">填写 Cookie / user-sign / web-id，保存后可手动抓取 trends。全局分析时会复用已存在的 trends 数据。</div>

  <label>Creator curl（可自动解析）</label>
  <textarea id="creator_curl" placeholder="粘贴 creator/list 的完整 curl 命令"></textarea>
  <label>Sound curl（可自动解析）</label>
  <textarea id="sound_curl" placeholder="粘贴 sound/rank_list 的完整 curl 命令"></textarea>
  <label>Hashtag curl（可自动解析）</label>
  <textarea id="hashtag_curl" placeholder="粘贴 hashtag/list 的完整 curl 命令"></textarea>
  <div class="actions">
    <button id="btnParseCreatorCurl" class="secondary" type="button">解析 Creator curl</button>
    <button id="btnParseSoundCurl" class="secondary" type="button">解析 Sound curl</button>
    <button id="btnParseHashtagCurl" class="secondary" type="button">解析 Hashtag curl</button>
    <button id="btnParseAllCurl" class="secondary" type="button">解析全部并合并填充</button>
  </div>

  <label>Creator Cookie</label>
  <textarea id="creator_cookie" placeholder="creator cookie"></textarea>
  <div class="row">
    <div>
      <label>creator user-sign</label>
      <input id="creator_user_sign" placeholder="creator user-sign" />
    </div>
    <div>
      <label>creator web-id</label>
      <input id="creator_web_id" placeholder="creator web-id" />
    </div>
  </div>

  <label>Sound Cookie</label>
  <textarea id="sound_cookie" placeholder="sound cookie"></textarea>
  <div class="row">
    <div>
      <label>sound user-sign</label>
      <input id="sound_user_sign" placeholder="sound user-sign" />
    </div>
    <div>
      <label>sound web-id</label>
      <input id="sound_web_id" placeholder="sound web-id" />
    </div>
  </div>

  <label>Hashtag Cookie</label>
  <textarea id="hashtag_cookie" placeholder="hashtag cookie"></textarea>
  <div class="row">
    <div>
      <label>hashtag user-sign</label>
      <input id="hashtag_user_sign" placeholder="hashtag user-sign" />
    </div>
    <div>
      <label>hashtag web-id</label>
      <input id="hashtag_web_id" placeholder="hashtag web-id" />
    </div>
  </div>
  <label>country_code</label>
  <input id="country_code" value="US" />

  <div class="actions">
    <button id="btnLoad" class="secondary" type="button">加载当前配置</button>
    <button id="btnSave" type="button">保存配置</button>
    <button id="btnFetch" type="button">手动抓取本周 Trends</button>
  </div>

  <hr style="margin: 24px 0; border: none; border-top: 1px solid #eee;" />
  <h2 style="margin: 0 0 8px;">手动导入 Postman 趋势响应</h2>
  <div class="hint">每个框支持粘贴：单页 JSON、分页 JSON 数组、或多个 JSON 连续粘贴（按顺序自动合并）。</div>
  <label>Creator 响应（creator/list）</label>
  <textarea id="creator_response_raw" style="min-height: 170px;" placeholder="粘贴 creator/list 的 Postman JSON 响应"></textarea>
  <label>Sound 响应（sound/rank_list）</label>
  <textarea id="sound_response_raw" style="min-height: 170px;" placeholder="粘贴 sound/rank_list 的 Postman JSON 响应"></textarea>
  <label>Hashtag 响应（hashtag/list）</label>
  <textarea id="hashtag_response_raw" style="min-height: 170px;" placeholder="粘贴 hashtag/list 的 Postman JSON 响应"></textarea>
  <div class="actions">
    <button id="btnImportManual" type="button">解析并写入本周 Trends</button>
  </div>
  <pre id="out">Ready</pre>

  <script>
    const token = '{escaped_token}';
    const headers = () => ({{'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}});
    const out = document.getElementById('out');
    function setOut(v) {{ out.textContent = typeof v === 'string' ? v : JSON.stringify(v, null, 2); }}
    function unquote(s) {{
      if (!s) return '';
      const t = s.trim();
      if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) return t.slice(1, -1);
      return t;
    }}
    function parseCurlHeaders(curlText) {{
      const text = curlText || '';
      const headerMap = {{}};
      const headerRegex = /(?:^|\\s)(?:-H|--header)\\s+((?:'[^']*')|(?:\"[^\"]*\")|(?:[^\\s\\\\]+))/g;
      const cookieArgRegex = /(?:^|\\s)(?:-b|--cookie)\\s+((?:'[^']*')|(?:\"[^\"]*\")|(?:[^\\s\\\\]+))/;
      let m;
      while ((m = headerRegex.exec(text)) !== null) {{
        const raw = unquote(m[1]);
        const idx = raw.indexOf(':');
        if (idx <= 0) continue;
        const key = raw.slice(0, idx).trim().toLowerCase();
        const val = raw.slice(idx + 1).trim();
        if (key) headerMap[key] = val;
      }}
      const cookieArgMatch = text.match(cookieArgRegex);
      const cookieFromArg = cookieArgMatch ? unquote(cookieArgMatch[1]).trim() : '';
      let countryCode = '';
      const urlMatch = text.match(/https?:\\/\\/[^\\s'\"\\\\]+/);
      if (urlMatch && urlMatch[0]) {{
        try {{
          const u = new URL(urlMatch[0]);
          countryCode =
            u.searchParams.get('country_code') ||
            u.searchParams.get('audience_country') ||
            u.searchParams.get('creator_country') ||
            '';
        }} catch (_e) {{}}
      }}
      return {{
        cookie: headerMap['cookie'] || cookieFromArg || '',
        user_sign: headerMap['user-sign'] || '',
        web_id: headerMap['web-id'] || '',
        country_code: countryCode ? String(countryCode).toUpperCase() : '',
      }};
    }}
    function fillFromCurlText(curlText) {{
      const parsed = parseCurlHeaders(curlText);
      return parsed;
    }}
    function applyParsed(prefix, parsed) {{
      if (parsed.cookie) document.getElementById(prefix + '_cookie').value = parsed.cookie;
      if (parsed.user_sign) document.getElementById(prefix + '_user_sign').value = parsed.user_sign;
      if (parsed.web_id) document.getElementById(prefix + '_web_id').value = parsed.web_id;
      if (parsed.country_code) document.getElementById('country_code').value = parsed.country_code;
    }}
    function fillFromSingleCurl(curlId, prefix, label) {{
      const parsed = fillFromCurlText(document.getElementById(curlId).value);
      applyParsed(prefix, parsed);
      setOut({{
        parsed: true,
        source: label,
        cookie: Boolean(parsed.cookie),
        user_sign: Boolean(parsed.user_sign),
        web_id: Boolean(parsed.web_id),
        country_code: parsed.country_code || null,
      }});
    }}
    function fillFromAllCurl() {{
      const p1 = fillFromCurlText(document.getElementById('creator_curl').value);
      const p2 = fillFromCurlText(document.getElementById('sound_curl').value);
      const p3 = fillFromCurlText(document.getElementById('hashtag_curl').value);
      applyParsed('creator', p1);
      applyParsed('sound', p2);
      applyParsed('hashtag', p3);
      setOut({{
        parsed: true,
        source: 'all',
        creator: {{ cookie: Boolean(p1.cookie), user_sign: Boolean(p1.user_sign), web_id: Boolean(p1.web_id) }},
        sound: {{ cookie: Boolean(p2.cookie), user_sign: Boolean(p2.user_sign), web_id: Boolean(p2.web_id) }},
        hashtag: {{ cookie: Boolean(p3.cookie), user_sign: Boolean(p3.user_sign), web_id: Boolean(p3.web_id) }},
        country_code: p1.country_code || p2.country_code || p3.country_code || null,
      }});
    }}

    async function loadConfig() {{
      const resp = await fetch('/admin/weekly-report/trends/headers', {{ headers: headers() }});
      const data = await resp.json();
      if (data.creator_cookie) document.getElementById('creator_cookie').value = data.creator_cookie;
      if (data.creator_user_sign) document.getElementById('creator_user_sign').value = data.creator_user_sign;
      if (data.creator_web_id) document.getElementById('creator_web_id').value = data.creator_web_id;
      if (data.sound_cookie) document.getElementById('sound_cookie').value = data.sound_cookie;
      if (data.sound_user_sign) document.getElementById('sound_user_sign').value = data.sound_user_sign;
      if (data.sound_web_id) document.getElementById('sound_web_id').value = data.sound_web_id;
      if (data.hashtag_cookie) document.getElementById('hashtag_cookie').value = data.hashtag_cookie;
      if (data.hashtag_user_sign) document.getElementById('hashtag_user_sign').value = data.hashtag_user_sign;
      if (data.hashtag_web_id) document.getElementById('hashtag_web_id').value = data.hashtag_web_id;
      if (data.country_code) document.getElementById('country_code').value = data.country_code;
      setOut(data);
    }}

    async function saveConfig() {{
      const payload = {{
        creator_cookie: document.getElementById('creator_cookie').value.trim(),
        creator_user_sign: document.getElementById('creator_user_sign').value.trim(),
        creator_web_id: document.getElementById('creator_web_id').value.trim(),
        sound_cookie: document.getElementById('sound_cookie').value.trim(),
        sound_user_sign: document.getElementById('sound_user_sign').value.trim(),
        sound_web_id: document.getElementById('sound_web_id').value.trim(),
        hashtag_cookie: document.getElementById('hashtag_cookie').value.trim(),
        hashtag_user_sign: document.getElementById('hashtag_user_sign').value.trim(),
        hashtag_web_id: document.getElementById('hashtag_web_id').value.trim(),
        country_code: document.getElementById('country_code').value.trim() || 'US',
      }};
      const resp = await fetch('/admin/weekly-report/trends/headers', {{ method: 'PUT', headers: headers(), body: JSON.stringify(payload) }});
      setOut(await resp.json());
    }}

    async function fetchTrends() {{
      const resp = await fetch('/admin/weekly-report/trends/fetch', {{ method: 'POST', headers: headers() }});
      const data = await resp.json();
      setOut(data);
    }}

    async function importManualTrends() {{
      const payload = {{
        creator_response_raw: document.getElementById('creator_response_raw').value,
        sound_response_raw: document.getElementById('sound_response_raw').value,
        hashtag_response_raw: document.getElementById('hashtag_response_raw').value,
      }};
      const resp = await fetch('/admin/weekly-report/trends/manual-import', {{ method: 'POST', headers: headers(), body: JSON.stringify(payload) }});
      const data = await resp.json();
      setOut(data);
    }}

    document.getElementById('btnLoad').addEventListener('click', () => loadConfig().catch(e => setOut(String(e))));
    document.getElementById('btnSave').addEventListener('click', () => saveConfig().catch(e => setOut(String(e))));
    document.getElementById('btnFetch').addEventListener('click', () => fetchTrends().catch(e => setOut(String(e))));
    document.getElementById('btnParseCreatorCurl').addEventListener('click', () => fillFromSingleCurl('creator_curl', 'creator', 'creator'));
    document.getElementById('btnParseSoundCurl').addEventListener('click', () => fillFromSingleCurl('sound_curl', 'sound', 'sound'));
    document.getElementById('btnParseHashtagCurl').addEventListener('click', () => fillFromSingleCurl('hashtag_curl', 'hashtag', 'hashtag'));
    document.getElementById('btnParseAllCurl').addEventListener('click', () => fillFromAllCurl());
    document.getElementById('btnImportManual').addEventListener('click', () => importManualTrends().catch(e => setOut(String(e))));
    loadConfig().catch(() => {{}});
  </script>
</body>
</html>"""


class TikTokRadarHeadersUpsertRequest(BaseModel):
    creator_cookie: str = Field(..., min_length=1)
    creator_user_sign: str = Field(..., min_length=1)
    creator_web_id: str = Field(..., min_length=1)
    sound_cookie: str = Field(..., min_length=1)
    sound_user_sign: str = Field(..., min_length=1)
    sound_web_id: str = Field(..., min_length=1)
    hashtag_cookie: str = Field(..., min_length=1)
    hashtag_user_sign: str = Field(..., min_length=1)
    hashtag_web_id: str = Field(..., min_length=1)
    country_code: str = Field("US", min_length=2, max_length=8)


class TikTokRadarManualImportRequest(BaseModel):
    creator_response_raw: str = Field(..., min_length=2)
    sound_response_raw: str = Field(..., min_length=2)
    hashtag_response_raw: str = Field(..., min_length=2)
    global_report_id: Optional[int] = None
    replace_existing: bool = True


def _decode_json_segments(raw_text: str) -> List[Any]:
    text = (raw_text or "").strip()
    if not text:
        return []
    decoder = json.JSONDecoder()
    idx = 0
    values: List[Any] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        value, end = decoder.raw_decode(text, idx)
        values.append(value)
        idx = end
    return values


def _collect_page_payloads(value: Any) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("data"), dict):
            pages.append(value)
        for key in ("pages", "responses", "items", "results", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                for item in nested:
                    pages.extend(_collect_page_payloads(item))
    elif isinstance(value, list):
        for item in value:
            pages.extend(_collect_page_payloads(item))
    return pages


def _dedupe_key_for_item(trend_type: str, item: Dict[str, Any]) -> Optional[str]:
    if trend_type == "creator":
        return str(item.get("user_id") or item.get("tcm_id") or "").strip() or None
    if trend_type == "sound":
        return str(item.get("song_id") or item.get("clip_id") or item.get("title") or "").strip() or None
    return str(item.get("hashtag_id") or item.get("hashtag_name") or "").strip() or None


def _extract_trend_items(raw_text: str, trend_type: str) -> Dict[str, Any]:
    trend_key_map = {
        "creator": "creators",
        "sound": "sound_list",
        "hashtag": "list",
    }
    list_key = trend_key_map[trend_type]
    decoded_values = _decode_json_segments(raw_text)
    if not decoded_values:
        raise ValueError("empty_input")
    all_pages: List[Dict[str, Any]] = []
    for val in decoded_values:
        all_pages.extend(_collect_page_payloads(val))
    if not all_pages:
        raise ValueError("no_page_payloads_found")

    parsed_items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for page in all_pages:
        if page.get("code") not in (0, "0", None):
            continue
        data = page.get("data")
        if not isinstance(data, dict):
            continue
        rows = data.get(list_key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = _dedupe_key_for_item(trend_type, row)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            parsed_items.append(row)

    return {
        "pages": len(all_pages),
        "items": parsed_items[:100],
        "total_items": len(parsed_items),
    }


@router.get(
    "/weekly-report/{app_user_id}",
    response_model=WeeklyReportResponse,
    tags=["weekly-report"],
)
async def get_weekly_report(
    app_user_id: str, db: Session = Depends(get_db)
) -> WeeklyReportResponse:
    report = (
        db.query(WeeklyReport)
        .filter(WeeklyReport.app_user_id == app_user_id)
        .order_by(WeeklyReport.created_at.desc())
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="not_found")
    
    # Convert topics JSON to list of TopicItem if present
    topics = None
    if report.topics:
        try:
            topics = [TopicItem(**item) if isinstance(item, dict) else item for item in report.topics]
        except Exception:
            topics = None
    
    return WeeklyReportResponse(
        id=report.id,
        app_user_id=report.app_user_id,
        email_content=report.email_content,
        period_start=report.period_start,
        period_end=report.period_end,
        created_at=report.created_at,
        updated_at=report.updated_at,
        send_status=report.send_status,
        feeding_state=report.feeding_state,
        trend_name=report.trend_name,
        trend_type=report.trend_type,
        discovery_rank=report.discovery_rank,
        total_discoverers=report.total_discoverers,
        origin_niche_text=report.origin_niche_text,
        spread_end_text=report.spread_end_text,
        reach_start=report.reach_start,
        reach_end=report.reach_end,
        current_reach=report.current_reach,
        total_videos=report.total_videos,
        total_time=report.total_time,
        pre_total_time=report.pre_total_time,
        miles_scrolled=report.miles_scrolled,
        topics=topics,
        timezone=report.timezone,
        rabbit_hole_datetime=report.rabbit_hole_datetime,
        rabbit_hole_date=report.rabbit_hole_date,
        rabbit_hole_time=report.rabbit_hole_time,
        rabbit_hole_count=report.rabbit_hole_count,
        rabbit_hole_category=report.rabbit_hole_category,
        nudge_text=report.nudge_text,
    )


@router.post(
    "/weekly-report/unsubscribe",
    response_model=UnsubscribeResponse,
    tags=["weekly-report"],
)
async def unsubscribe_weekly_report(
    payload: UnsubscribeRequest, db: Session = Depends(get_db)
) -> UnsubscribeResponse:
    user = db.get(AppUser, payload.app_user_id)
    if user:
        user.weekly_report_unsubscribed = True
        user.updated_at = datetime.utcnow()
        db.add(user)
        db.commit()
    # Return success even if user not found (idempotent)
    return UnsubscribeResponse(success=True, message="Unsubscribed")


@router.post(
    "/weekly-report/resubscribe",
    response_model=UnsubscribeResponse,
    tags=["weekly-report"],
)
async def resubscribe_weekly_report(
    payload: UnsubscribeRequest, db: Session = Depends(get_db)
) -> UnsubscribeResponse:
    user = db.get(AppUser, payload.app_user_id)
    if user:
        user.weekly_report_unsubscribed = False
        user.updated_at = datetime.utcnow()
        db.add(user)
        db.commit()
    # Return success even if user not found (idempotent)
    return UnsubscribeResponse(success=True, message="Re-subscribed")


@router.post(
    "/admin/cron/weekly-report-analyze",
    status_code=200,
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def cron_weekly_report_analyze(db: Session = Depends(get_db)):
    """Legacy alias. Weekly report now runs through the batch pipeline only."""
    return await cron_weekly_report_batch(db)


@router.post(
    "/admin/cron/weekly-report-send",
    status_code=200,
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def cron_weekly_report_send(db: Session = Depends(get_db)):
    """Legacy alias. Weekly report now runs through the batch pipeline only."""
    return await cron_weekly_report_batch(db)


@router.post(
    "/admin/cron/weekly-report-batch",
    status_code=200,
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def cron_weekly_report_batch(db: Session = Depends(get_db)):
    """Trigger the batch weekly report processing flow.
    
    This endpoint initiates the complete batch processing pipeline:
    1. Creates a WeeklyReportGlobal record for this week
    2. Enqueues weekly_report_fetch_trends job which will:
       - Fetch TikTok Creative Radar top 100 (hashtag/sound/creator) and persist
       - Enqueue weekly_report_batch_fetch to fetch user data, then global analyze,
         per-user analyze, and email sending
    
    The entire flow runs asynchronously via job queue.
    
    Returns:
        JSON with global_report_id and fetch_trends_job_id
    """
    # Calculate current week boundaries
    now = datetime.utcnow()
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=7)

    # Get or create global report for this week
    global_report = (
        db.query(WeeklyReportGlobal)
        .filter(
            WeeklyReportGlobal.period_start == week_start,
            WeeklyReportGlobal.period_end == week_end,
        )
        .first()
    )

    if not global_report:
        global_report = WeeklyReportGlobal(
            period_start=week_start,
            period_end=week_end,
            status="pending",
        )
        db.add(global_report)
        db.commit()
        db.refresh(global_report)

    # If trends are already prepared for this week, skip fetch_trends and start from batch_fetch directly.
    trends_ready = _weekly_trends_ready(db, global_report.id)
    if trends_ready:
        start_job = job_queue.enqueue(
            db,
            task_name="weekly_report_batch_fetch",
            payload={"global_report_id": global_report.id},
            idempotency_key=f"weekly_report_batch_fetch:{global_report.id}",
        )
        start_task = "weekly_report_batch_fetch"
    else:
        start_job = job_queue.enqueue(
            db,
            task_name="weekly_report_fetch_trends",
            payload={"global_report_id": global_report.id},
            idempotency_key=f"weekly_report_fetch_trends:{global_report.id}",
        )
        start_task = "weekly_report_fetch_trends"

    logger.info(
        "cron.weekly_report_batch.triggered",
        extra={
            "event": "cron.weekly_report_batch.triggered",
            "global_report_id": global_report.id,
            "start_task": start_task,
            "start_job_id": start_job.id,
            "period_start": week_start.isoformat(),
            "period_end": week_end.isoformat(),
        },
    )

    return {
        "global_report_id": global_report.id,
        "start_task": start_task,
        "start_job_id": start_job.id,
        "fetch_trends_job_id": start_job.id if start_task == "weekly_report_fetch_trends" else None,
        "batch_fetch_job_id": start_job.id if start_task == "weekly_report_batch_fetch" else None,
        "period_start": week_start.isoformat() + "Z",
        "period_end": week_end.isoformat() + "Z",
        "status": global_report.status,
    }


@router.get(
    "/admin/weekly-report-global/{global_report_id}",
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def get_global_report_status(global_report_id: int, db: Session = Depends(get_db)):
    """Get the status of a global weekly report batch job.
    
    Returns:
        JSON with global report status and statistics
    """
    global_report = db.get(WeeklyReportGlobal, global_report_id)
    if not global_report:
        raise HTTPException(status_code=404, detail="not_found")
    
    # Count user reports by status
    fetch_pending = db.query(WeeklyReport).filter(
        WeeklyReport.global_report_id == global_report_id,
        WeeklyReport.fetch_status == "pending",
    ).count()
    fetch_fetching = db.query(WeeklyReport).filter(
        WeeklyReport.global_report_id == global_report_id,
        WeeklyReport.fetch_status == "fetching",
    ).count()
    fetch_fetched = db.query(WeeklyReport).filter(
        WeeklyReport.global_report_id == global_report_id,
        WeeklyReport.fetch_status == "fetched",
    ).count()
    fetch_failed = db.query(WeeklyReport).filter(
        WeeklyReport.global_report_id == global_report_id,
        WeeklyReport.fetch_status == "failed",
    ).count()
    
    analyze_pending = db.query(WeeklyReport).filter(
        WeeklyReport.global_report_id == global_report_id,
        WeeklyReport.analyze_status == "pending",
    ).count()
    analyze_analyzing = db.query(WeeklyReport).filter(
        WeeklyReport.global_report_id == global_report_id,
        WeeklyReport.analyze_status == "analyzing",
    ).count()
    analyze_analyzed = db.query(WeeklyReport).filter(
        WeeklyReport.global_report_id == global_report_id,
        WeeklyReport.analyze_status == "analyzed",
    ).count()
    analyze_failed = db.query(WeeklyReport).filter(
        WeeklyReport.global_report_id == global_report_id,
        WeeklyReport.analyze_status == "failed",
    ).count()
    
    return {
        "id": global_report.id,
        "period_start": global_report.period_start.isoformat() + "Z" if global_report.period_start else None,
        "period_end": global_report.period_end.isoformat() + "Z" if global_report.period_end else None,
        "status": global_report.status,
        "total_users": global_report.total_users,
        "total_videos": global_report.total_videos,
        "total_watch_hours": global_report.total_watch_hours,
        "analysis_result": global_report.analysis_result,
        "created_at": global_report.created_at.isoformat() + "Z" if global_report.created_at else None,
        "updated_at": global_report.updated_at.isoformat() + "Z" if global_report.updated_at else None,
        "user_reports": {
            "fetch": {
                "pending": fetch_pending,
                "fetching": fetch_fetching,
                "fetched": fetch_fetched,
                "failed": fetch_failed,
            },
            "analyze": {
                "pending": analyze_pending,
                "analyzing": analyze_analyzing,
                "analyzed": analyze_analyzed,
                "failed": analyze_failed,
            },
        },
    }


@router.post(
    "/admin/test/weekly-report/{app_user_id}",
    response_model=WeeklyReportTestResponse,
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
)
async def admin_test_weekly_report(
    app_user_id: str,
    payload: WeeklyReportTestRequest,
    db: Session = Depends(get_db),
) -> WeeklyReportTestResponse:
    """Test endpoint to run weekly report pipeline for a single user.

    Only batch pipeline is supported:
    fetch_trends -> user_fetch -> global_analyze -> user_analyze -> batch_send.
    `payload.use_batch_flow` is kept for backward compatibility and ignored.
    """
    user = db.get(AppUser, app_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")
    if not user.latest_sec_user_id:
        raise HTTPException(status_code=400, detail="sec_user_id_required")

    email_address: Optional[str] = None
    if payload.send_email:
        email_address = payload.email or _latest_user_email(db, app_user_id)
        if not email_address:
            raise HTTPException(status_code=400, detail="email_required_for_sending")

    fetch_trends_job_id: Optional[str] = None
    analyze_job_id: str
    send_job_id: Optional[str] = None

    global_report_id_for_response: Optional[int] = None
    # Full pipeline: create global report -> enqueue fetch_trends (limit to this user)
    if (payload.period_start is None) ^ (payload.period_end is None):
        raise HTTPException(status_code=400, detail="period_start_and_period_end_must_be_provided_together")
    if payload.period_start is not None and payload.period_end is not None:
        week_start = _to_naive_utc(payload.period_start).replace(microsecond=0)
        week_end = _to_naive_utc(payload.period_end).replace(microsecond=0)
        if week_end <= week_start:
            raise HTTPException(status_code=400, detail="invalid_period_range")
    else:
        now = datetime.utcnow()
        week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)
    global_report: Optional[WeeklyReportGlobal] = None
    if not payload.force_new_global_report:
        global_report = (
            db.query(WeeklyReportGlobal)
            .filter(
                WeeklyReportGlobal.period_start == week_start,
                WeeklyReportGlobal.period_end == week_end,
            )
            .first()
        )
    if not global_report:
        global_report = WeeklyReportGlobal(
            period_start=week_start,
            period_end=week_end,
            status="pending",
        )
        db.add(global_report)
        db.commit()
        db.refresh(global_report)
    global_report_id_for_response = global_report.id

    trends_ready = _weekly_trends_ready(db, global_report.id)
    pipeline_payload: Dict[str, Any] = {
        "global_report_id": global_report.id,
        "limit_to_app_user_ids": [app_user_id],
    }
    if payload.send_email and email_address:
        pipeline_payload["email_override"] = email_address

    if trends_ready:
        start_job = job_queue.enqueue(
            db,
            task_name="weekly_report_batch_fetch",
            payload=pipeline_payload,
            idempotency_key=f"weekly_report_batch_fetch_test:{global_report.id}:{app_user_id}:{datetime.utcnow().isoformat()}",
        )
        fetch_trends_job_id = None
        analyze_job_id = start_job.id
        start_task = "weekly_report_batch_fetch"
    else:
        fetch_trends_job = job_queue.enqueue(
            db,
            task_name="weekly_report_fetch_trends",
            payload=pipeline_payload,
            idempotency_key=f"weekly_report_fetch_trends_test:{global_report.id}:{app_user_id}:{datetime.utcnow().isoformat()}",
        )
        fetch_trends_job_id = fetch_trends_job.id
        analyze_job_id = fetch_trends_job_id  # pipeline starts with fetch_trends
        start_task = "weekly_report_fetch_trends"

    logger.info(
        "admin.test.weekly_report.batch_flow.enqueued",
        extra={
            "event": "admin.test.weekly_report.batch_flow.enqueued",
            "app_user_id": app_user_id,
            "global_report_id": global_report.id,
            "start_task": start_task,
            "fetch_trends_job_id": fetch_trends_job_id,
            "start_job_id": analyze_job_id,
            "force_new_global_report": payload.force_new_global_report,
        },
    )

    report = (
        db.query(WeeklyReport)
        .filter(WeeklyReport.app_user_id == app_user_id)
        .order_by(WeeklyReport.created_at.desc())
        .first()
    )
    if not report:
        report = WeeklyReport(
            app_user_id=app_user_id,
            send_status="pending",
        )
        db.add(report)
        db.commit()
        db.refresh(report)
    
    # Convert topics JSON to list of TopicItem if present
    topics = None
    if report.topics:
        try:
            topics = [TopicItem(**item) if isinstance(item, dict) else item for item in report.topics]
        except Exception:
            topics = None
    
    report_response = WeeklyReportResponse(
        id=report.id,
        app_user_id=report.app_user_id,
        email_content=report.email_content,
        period_start=report.period_start,
        period_end=report.period_end,
        created_at=report.created_at,
        updated_at=report.updated_at,
        send_status=report.send_status,
        feeding_state=report.feeding_state,
        trend_name=report.trend_name,
        trend_type=report.trend_type,
        discovery_rank=report.discovery_rank,
        total_discoverers=report.total_discoverers,
        origin_niche_text=report.origin_niche_text,
        spread_end_text=report.spread_end_text,
        reach_start=report.reach_start,
        reach_end=report.reach_end,
        current_reach=report.current_reach,
        total_videos=report.total_videos,
        total_time=report.total_time,
        pre_total_time=report.pre_total_time,
        miles_scrolled=report.miles_scrolled,
        topics=topics,
        timezone=report.timezone,
        rabbit_hole_datetime=report.rabbit_hole_datetime,
        rabbit_hole_date=report.rabbit_hole_date,
        rabbit_hole_time=report.rabbit_hole_time,
        rabbit_hole_count=report.rabbit_hole_count,
        rabbit_hole_category=report.rabbit_hole_category,
        nudge_text=report.nudge_text,
    )
    
    return WeeklyReportTestResponse(
        app_user_id=app_user_id,
        report_id=report.id,
        analyze_job_id=analyze_job_id,
        send_job_id=send_job_id,
        email_sent=False,  # Jobs are asynchronous, email not sent yet
        email_address=email_address,
        report=report_response,
        use_batch_flow=True,
        fetch_trends_job_id=fetch_trends_job_id,
        global_report_id=global_report_id_for_response,
    )


@router.get(
    "/admin/weekly-report/trends/headers",
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def get_tiktok_trends_headers(db: Session = Depends(get_db)):
    cfg = (
        db.query(TikTokRadarHeaderConfig)
        .order_by(TikTokRadarHeaderConfig.updated_at.desc(), TikTokRadarHeaderConfig.id.desc())
        .first()
    )
    if not cfg:
        return {"configured": False}
    creator_ok = bool((cfg.creator_cookie or "").strip() and (cfg.creator_user_sign or "").strip() and (cfg.creator_web_id or "").strip())
    sound_ok = bool((cfg.sound_cookie or "").strip() and (cfg.sound_user_sign or "").strip() and (cfg.sound_web_id or "").strip())
    hashtag_ok = bool((cfg.hashtag_cookie or "").strip() and (cfg.hashtag_user_sign or "").strip() and (cfg.hashtag_web_id or "").strip())
    return {
        "configured": bool(creator_ok and sound_ok and hashtag_ok),
        "id": cfg.id,
        "cookie": cfg.cookie,
        "user_sign": cfg.user_sign,
        "web_id": cfg.web_id,
        "creator_cookie": cfg.creator_cookie,
        "creator_user_sign": cfg.creator_user_sign,
        "creator_web_id": cfg.creator_web_id,
        "sound_cookie": cfg.sound_cookie,
        "sound_user_sign": cfg.sound_user_sign,
        "sound_web_id": cfg.sound_web_id,
        "hashtag_cookie": cfg.hashtag_cookie,
        "hashtag_user_sign": cfg.hashtag_user_sign,
        "hashtag_web_id": cfg.hashtag_web_id,
        "country_code": cfg.country_code or "US",
        "updated_by": cfg.updated_by,
        "updated_at": cfg.updated_at.isoformat() + "Z" if cfg.updated_at else None,
    }


@router.put(
    "/admin/weekly-report/trends/headers",
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def upsert_tiktok_trends_headers(payload: TikTokRadarHeadersUpsertRequest, db: Session = Depends(get_db)):
    cfg = (
        db.query(TikTokRadarHeaderConfig)
        .order_by(TikTokRadarHeaderConfig.updated_at.desc(), TikTokRadarHeaderConfig.id.desc())
        .first()
    )
    if not cfg:
        cfg = TikTokRadarHeaderConfig()
        db.add(cfg)
        db.flush()

    cfg.creator_cookie = payload.creator_cookie.strip()
    cfg.creator_user_sign = payload.creator_user_sign.strip()
    cfg.creator_web_id = payload.creator_web_id.strip()
    cfg.sound_cookie = payload.sound_cookie.strip()
    cfg.sound_user_sign = payload.sound_user_sign.strip()
    cfg.sound_web_id = payload.sound_web_id.strip()
    cfg.hashtag_cookie = payload.hashtag_cookie.strip()
    cfg.hashtag_user_sign = payload.hashtag_user_sign.strip()
    cfg.hashtag_web_id = payload.hashtag_web_id.strip()

    # Legacy fallback fields: keep aligned with creator for backward compatibility.
    cfg.cookie = cfg.creator_cookie
    cfg.user_sign = cfg.creator_user_sign
    cfg.web_id = cfg.creator_web_id
    cfg.country_code = payload.country_code.strip().upper() or "US"
    cfg.updated_by = "admin_api"

    # Keep a single active row to avoid ambiguity.
    db.query(TikTokRadarHeaderConfig).filter(TikTokRadarHeaderConfig.id != cfg.id).delete(synchronize_session=False)
    db.commit()
    db.refresh(cfg)
    return {
        "ok": True,
        "id": cfg.id,
        "updated_at": cfg.updated_at.isoformat() + "Z" if cfg.updated_at else None,
    }


@router.post(
    "/admin/weekly-report/trends/fetch",
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def manual_fetch_tiktok_trends(db: Session = Depends(get_db)):
    now = datetime.utcnow()
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=7)
    global_report = (
        db.query(WeeklyReportGlobal)
        .filter(
            WeeklyReportGlobal.period_start == week_start,
            WeeklyReportGlobal.period_end == week_end,
        )
        .first()
    )
    if not global_report:
        global_report = WeeklyReportGlobal(
            period_start=week_start,
            period_end=week_end,
            status="pending",
        )
        db.add(global_report)
        db.commit()
        db.refresh(global_report)

    job = job_queue.enqueue(
        db,
        task_name="weekly_report_fetch_trends",
        payload={
            "global_report_id": global_report.id,
            "force_refresh_trends": True,
            "skip_batch_fetch": True,
        },
        idempotency_key=f"weekly_report_fetch_trends_manual:{global_report.id}",
        force_new=True,
    )
    return {
        "ok": True,
        "global_report_id": global_report.id,
        "fetch_trends_job_id": job.id,
        "period_start": week_start.isoformat() + "Z",
        "period_end": week_end.isoformat() + "Z",
    }


@router.post(
    "/admin/weekly-report/trends/manual-import",
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def manual_import_tiktok_trends(payload: TikTokRadarManualImportRequest, db: Session = Depends(get_db)):
    try:
        creator_result = _extract_trend_items(payload.creator_response_raw, "creator")
        sound_result = _extract_trend_items(payload.sound_response_raw, "sound")
        hashtag_result = _extract_trend_items(payload.hashtag_response_raw, "hashtag")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid_trend_payload:{exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid_trend_payload:{str(exc)}") from exc

    if not creator_result["items"] or not sound_result["items"] or not hashtag_result["items"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_trend_items",
                "creator_items": len(creator_result["items"]),
                "sound_items": len(sound_result["items"]),
                "hashtag_items": len(hashtag_result["items"]),
            },
        )

    if payload.global_report_id:
        global_report = db.get(WeeklyReportGlobal, int(payload.global_report_id))
        if not global_report:
            raise HTTPException(status_code=404, detail="global_report_not_found")
    else:
        now = datetime.utcnow()
        week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)
        global_report = (
            db.query(WeeklyReportGlobal)
            .filter(
                WeeklyReportGlobal.period_start == week_start,
                WeeklyReportGlobal.period_end == week_end,
            )
            .first()
        )
        if not global_report:
            global_report = WeeklyReportGlobal(
                period_start=week_start,
                period_end=week_end,
                status="pending",
            )
            db.add(global_report)
            db.commit()
            db.refresh(global_report)

    global_report_id = global_report.id
    period_start = global_report.period_start
    period_end = global_report.period_end
    if payload.replace_existing:
        db.query(WeeklyTrendHashtag).filter(WeeklyTrendHashtag.global_report_id == global_report_id).delete()
        db.query(WeeklyTrendSound).filter(WeeklyTrendSound.global_report_id == global_report_id).delete()
        db.query(WeeklyTrendCreator).filter(WeeklyTrendCreator.global_report_id == global_report_id).delete()

    hashtags = hashtag_result["items"]
    sounds = sound_result["items"]
    creators = creator_result["items"]

    for rank, h in enumerate(hashtags, start=1):
        country_info = h.get("country_info") or {}
        country_code_val = country_info.get("id") if isinstance(country_info, dict) else h.get("country_code")
        row = WeeklyTrendHashtag(
            global_report_id=global_report_id,
            period_start=period_start,
            period_end=period_end,
            rank=rank,
            hashtag_id=str(h.get("hashtag_id") or ""),
            hashtag_name=h.get("hashtag_name"),
            country_code=country_code_val or h.get("country_code"),
            publish_cnt=int(h.get("publish_cnt") or 0) if h.get("publish_cnt") is not None else None,
            video_views=int(h.get("video_views") or 0) if h.get("video_views") is not None else None,
            rank_diff=int(h.get("rank_diff")) if h.get("rank_diff") is not None else None,
            rank_diff_type=int(h.get("rank_diff_type")) if h.get("rank_diff_type") is not None else None,
            trend=h.get("trend"),
            industry_info=h.get("industry_info"),
        )
        db.add(row)

    for rank, s in enumerate(sounds, start=1):
        row = WeeklyTrendSound(
            global_report_id=global_report_id,
            period_start=period_start,
            period_end=period_end,
            rank=rank,
            clip_id=str(s.get("clip_id") or ""),
            song_id=str(s.get("song_id") or ""),
            title=s.get("title"),
            author=s.get("author"),
            country_code=s.get("country_code"),
            duration=int(s.get("duration")) if s.get("duration") is not None else None,
            link=s.get("link"),
            trend=s.get("trend"),
        )
        db.add(row)

    for rank, c in enumerate(creators, start=1):
        row = WeeklyTrendCreator(
            global_report_id=global_report_id,
            period_start=period_start,
            period_end=period_end,
            rank=rank,
            tcm_id=str(c.get("tcm_id") or ""),
            user_id=str(c.get("user_id") or ""),
            nick_name=c.get("nick_name"),
            avatar_url=c.get("avatar_url"),
            country_code=c.get("country_code"),
            follower_cnt=int(c.get("follower_cnt")) if c.get("follower_cnt") is not None else None,
            liked_cnt=int(c.get("liked_cnt")) if c.get("liked_cnt") is not None else None,
            tt_link=c.get("tt_link"),
            items=c.get("items"),
        )
        db.add(row)

    db.commit()
    return {
        "ok": True,
        "global_report_id": global_report_id,
        "saved": {
            "creator_count": len(creators),
            "sound_count": len(sounds),
            "hashtag_count": len(hashtags),
        },
        "parsed": {
            "creator_pages": creator_result["pages"],
            "creator_total_items": creator_result["total_items"],
            "sound_pages": sound_result["pages"],
            "sound_total_items": sound_result["total_items"],
            "hashtag_pages": hashtag_result["pages"],
            "hashtag_total_items": hashtag_result["total_items"],
        },
        "replace_existing": bool(payload.replace_existing),
    }


@router.get(
    "/admin/weekly-report/trends/stats/{global_report_id}",
    dependencies=[Depends(require_weekly_token)],
    tags=["weekly-report", "admin"],
    include_in_schema=False,
)
async def get_tiktok_trends_stats(global_report_id: int, db: Session = Depends(get_db)):
    hashtag_count = db.query(WeeklyTrendHashtag).filter(WeeklyTrendHashtag.global_report_id == global_report_id).count()
    sound_count = db.query(WeeklyTrendSound).filter(WeeklyTrendSound.global_report_id == global_report_id).count()
    creator_count = db.query(WeeklyTrendCreator).filter(WeeklyTrendCreator.global_report_id == global_report_id).count()
    return {
        "global_report_id": global_report_id,
        "hashtag_count": hashtag_count,
        "sound_count": sound_count,
        "creator_count": creator_count,
        "ready": hashtag_count > 0 and sound_count > 0 and creator_count > 0,
    }


@router.get(
    "/admin/weekly-report/trends/page",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def tiktok_trends_admin_page():
    return HTMLResponse(_render_trends_login_page())


@router.post(
    "/admin/weekly-report/trends/page/login",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def tiktok_trends_admin_page_login(password: str = Form(...)):
    try:
        _require_trends_page_password(password)
    except HTTPException:
        return HTMLResponse(_render_trends_login_page(error="密码错误，请重试。"), status_code=401)

    weekly_token = settings.weekly_token.get_secret_value() if settings.weekly_token else None
    if not weekly_token:
        raise HTTPException(status_code=404, detail="not_found")
    return HTMLResponse(_render_trends_admin_page(weekly_token))

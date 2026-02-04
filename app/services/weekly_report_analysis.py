import math
import os
import re
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


WEEKLY_TOPIC_STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "this", "that", "from", "your", "you", "are",
    "was", "were", "into", "how", "why", "what", "when", "where", "best", "new",
    "tips", "guide", "video", "videos", "shorts", "tiktok", "part", "episode",
}


def _safe_zone(tz_name: Optional[str]) -> ZoneInfo:
    if not tz_name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _normalize_match_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _compact_match_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_match_text(value))


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    with suppress(Exception):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _parse_watched_at_value(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        with suppress(Exception):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    text = _safe_str(value)
    if not text:
        return None
    if text.isdigit():
        return _parse_watched_at_value(int(text))
    return _parse_iso_datetime(text)


def _normalize_topic_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.lstrip("#")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    words = [w for w in text.split(" ") if w]
    return " ".join(w.capitalize() for w in words[:4])


def _topic_key(value: Any) -> str:
    return _compact_match_text(_normalize_topic_label(value))


def _title_topic_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    words = [w.lower() for w in text.split() if len(w) >= 3]
    words = [w for w in words if w not in WEEKLY_TOPIC_STOPWORDS]
    if len(words) < 2:
        return ""
    return " ".join(w.capitalize() for w in words[:4])


def _extract_topic_candidates(item: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen_keys: Set[str] = set()
    hashtags = item.get("hashtags")
    if isinstance(hashtags, list):
        raw_tags = hashtags
    elif isinstance(hashtags, str):
        raw_tags = re.split(r"[\s,]+", hashtags)
    else:
        raw_tags = []
    for raw in raw_tags[:6]:
        label = _normalize_topic_label(raw)
        key = _topic_key(label)
        if key and key not in seen_keys and len(key) >= 4:
            out.append((key, label))
            seen_keys.add(key)

    music = item.get("music")
    music_candidates: List[Any] = []
    if isinstance(music, dict):
        music_candidates.extend([music.get("title"), music.get("author")])
    else:
        music_candidates.append(music)

    for raw in (item.get("title"), item.get("description"), *music_candidates):
        label = _title_topic_label(raw)
        key = _topic_key(label)
        if key and key not in seen_keys and len(key) >= 4:
            out.append((key, label))
            seen_keys.add(key)
    return out


def _build_topic_counts(sample_items: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, str]]:
    counts: Dict[str, int] = {}
    labels: Dict[str, str] = {}
    for item in sample_items:
        if not isinstance(item, dict):
            continue
        candidates = _extract_topic_candidates(item)
        if not candidates:
            continue
        key, label = candidates[0]
        counts[key] = counts.get(key, 0) + 1
        labels.setdefault(key, label)
    return counts, labels


def compute_content_diversity_score(sample_items: List[Dict[str, Any]]) -> int:
    counts, _ = _build_topic_counts(sample_items)
    total = sum(counts.values())
    if total <= 0:
        return 0
    unique = len(counts)
    if unique <= 1:
        return 5
    entropy = 0.0
    for v in counts.values():
        p = v / total
        if p > 0:
            entropy -= p * math.log(p)
    entropy_norm = entropy / math.log(unique) if unique > 1 else 0.0
    unique_factor = min(1.0, unique / 12.0)
    score = (entropy_norm * 0.65 + unique_factor * 0.35) * 100.0
    return max(0, min(100, int(round(score))))


def derive_new_topics(
    weekly_items: List[Dict[str, Any]],
    baseline_items: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    week_counts, week_labels = _build_topic_counts(weekly_items)
    base_counts, _base_labels = _build_topic_counts(baseline_items)
    if not week_counts:
        return []
    ranked: List[Tuple[str, int, float]] = []
    for key, week_count in week_counts.items():
        baseline_per_week = (base_counts.get(key, 0) / 4.0) if base_counts else 0.0
        ratio = week_count / max(0.2, baseline_per_week)
        if week_count >= 3 and ratio >= 5.0:
            ranked.append((key, week_count, ratio))
    ranked.sort(key=lambda row: (-row[1], -row[2], row[0]))
    topics: List[Dict[str, str]] = []
    for key, _count, _ratio in ranked[:3]:
        label = week_labels.get(key) or _normalize_topic_label(key)
        if not label:
            continue
        topics.append({"topic": label, "pic_url": ""})
    if topics:
        return topics

    # Fallback for sparse histories: return the most watched weekly topics.
    fallback_ranked = sorted(week_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    for key, count in fallback_ranked:
        if count < 2:
            continue
        label = week_labels.get(key) or _normalize_topic_label(key)
        if not label:
            continue
        topics.append({"topic": label, "pic_url": ""})
        if len(topics) >= 3:
            break
    return topics


def derive_rabbit_hole(
    sample_items: List[Dict[str, Any]],
    *,
    tz_name: Optional[str],
) -> Dict[str, Any]:
    streak_threshold = max(1, int(os.getenv("WEEKLY_RABBIT_HOLE_MIN_STREAK", "20")))
    rows: List[Tuple[datetime, str, str]] = []
    for item in sample_items:
        if not isinstance(item, dict):
            continue
        watched_at = _parse_watched_at_value(item.get("watched_at") or item.get("watchedAt"))
        if watched_at is None:
            continue
        candidates = _extract_topic_candidates(item)
        if not candidates:
            continue
        key, label = candidates[0]
        rows.append((watched_at, key, label))
    if not rows:
        return {"count": 0}
    rows.sort(key=lambda row: row[0])
    best_count = 0
    best_key = ""
    best_label = ""
    best_start: Optional[datetime] = None
    cur_key = ""
    cur_label = ""
    cur_count = 0
    cur_start: Optional[datetime] = None
    for watched_at, key, label in rows:
        if key == cur_key:
            cur_count += 1
        else:
            cur_key = key
            cur_label = label
            cur_count = 1
            cur_start = watched_at
        if cur_count > best_count:
            best_count = cur_count
            best_key = cur_key
            best_label = cur_label
            best_start = cur_start
    if best_count < streak_threshold or not best_start or not best_key:
        return {"count": 0}
    local_dt = best_start.astimezone(_safe_zone(tz_name))
    local_time = local_dt.strftime("%I:%M %p").lstrip("0")
    return {
        "count": best_count,
        "category": best_label,
        "start_at": best_start,
        "day": local_dt.strftime("%A"),
        "time": local_time,
    }


def extract_brainrot_pct(summary: Optional[Dict[str, Any]]) -> float:
    if not isinstance(summary, dict):
        return 0.0
    brainrot = summary.get("brainrot")
    if not isinstance(brainrot, dict):
        return 0.0
    raw = brainrot.get("raw")
    with suppress(Exception):
        val = float(raw)
        if val <= 1.0:
            val *= 100.0
        return max(0.0, min(100.0, val))
    return 0.0


def derive_feedling_state(
    *,
    trend_variant: Optional[str],
    diversity_score: int,
    new_topic_count: int,
    total_time: int,
    pre_total_time: Optional[int],
    rabbit_hole_count: int,
    miles_scrolled: int,
    brainrot_pct: float,
) -> str:
    if trend_variant == "early":
        return "excited"
    if diversity_score > 60 or new_topic_count >= 3:
        return "curious"
    if pre_total_time and total_time < pre_total_time and rabbit_hole_count <= 100:
        return "cozy"
    if rabbit_hole_count > 100 or miles_scrolled > 26:
        return "sleepy"
    if brainrot_pct > 20.0:
        return "dizzy"
    return "cozy"


def derive_nudge_text(
    *,
    rabbit_hole_count: int,
    rabbit_hole_time: Optional[str],
    miles_scrolled: int,
    brainrot_pct: float,
    total_time: int,
    pre_total_time: Optional[int],
) -> str:
    if rabbit_hole_count > 100:
        if rabbit_hole_time:
            m = re.match(r"(\d{1,2})", rabbit_hole_time)
            if m:
                hour = int(m.group(1))
                suffix = "AM" if "AM" in rabbit_hole_time.upper() else "PM"
                return f"Try putting your phone down before {hour} {suffix} this week"
        return "When you notice the drift, try searching for something specific"
    if miles_scrolled > 26:
        return "Try setting a scroll limit for yourself"
    if brainrot_pct > 25.0:
        return 'Try using "Not Interested" on a few videos this week'
    if pre_total_time and pre_total_time > 0 and total_time > int(pre_total_time * 1.3):
        return "Try setting a scroll limit for yourself"
    return "Your Feedling had a balanced week. Keep it up!"

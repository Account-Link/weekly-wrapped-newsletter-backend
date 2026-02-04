"""LLM prompt templates for wrapped analysis - IMPROVED VERSION.

Key improvements over original prompts.py:
1. Person-centric framing - "Who is this?" not "Analyze the data"
2. Concrete calibration anchors - Real content examples at score tiers
3. Permission for extremes - "Genuinely chaotic feeds deserve 85+"
4. Vocabulary bans - Explicit lists of forbidden meta-words
5. Logic examples, not output examples - Show reasoning, not answers
6. Voice injection - "As a friend" shifts register
7. Contrastive examples - Bad vs Good to calibrate quality
8. Tighter constraints - "Exactly 2 words" not "1-4 words"
"""

import hashlib
from contextlib import suppress
from typing import Any, Dict, List, Optional

PERSONALITY_PROMPT = (
    "Who is this person? Give them a character name in exactly 2 words.\n\n"
    "Format: [Adjective] [Noun] in Title Case.\n\n"
    "Do NOT use platform words like scroll, feed, viewer, consumer, content, browser, dopamine, or watcher.\n"
    "Describe who they ARE, not what they do on the app. And remember everybody on TikTok has some level of brainrot or dopamine addiction to short form, so that's not unique, the personality needs to be unique\n\n"
    "Respond with ONLY the 2-word name. No explanation, no markdown, no quotes."
)

PERSONALITY_EXPLANATION_PROMPT = (
    "Why does this name fit? One sentence, max 80 characters.\n\n"
    "Rules:\n"
    "- Start with evidence, not This person or They, say You, you are explaining to the person directly why they got that personality type\n"
    "- Name specific content you noticed\n"
    "- Do NOT use generic words like scroll, feed, viewer, consumer, content, browser, dopamine, or watcher, algorithm, viral, hashtags, trending, repetitive, engagement.\n"
    "- No markdown, no headers, no formatting\n\n"
    "Respond with ONLY the sentence. Nothing else."
)

NICHE_JOURNEY_PROMPT = (
    "Name 5 distinct interest areas from this watch history.\n\n"
    "Return ONLY valid JSON: a JSON array of exactly 5 strings.\n"
    "Example:\n"
    "[\"Area 1\",\"Area 2\",\"Area 3\",\"Area 4\",\"Area 5\"]\n\n"
    "Rules:\n"
    "- Each string is 2-4 words, Title Case\n"
    "- Be specific (no generic labels like \"Comedy\" or \"Music\")\n"
    "- No markdown, no quotes around the whole response, no extra keys"
)

TOP_NICHES_PROMPT = (
    "Name their top 2 niche interests. Niches are SPECIFIC.\n\n"
    "Too broad: Music or Comedy\n"
    "Just right: Pedal Steel Guitar or Sink Humor\n\n"
    "Never use: algorithm, viral, hashtags, trending, repetitive, engagement.\n\n"
    "{\"top_niches\": [\"Niche1\", \"Niche2\"], \"top_niche_percentile\": \"top 2.5%\"}\n\n"
    "Each niche: 1-3 words max. No ampersands.\n"
    "Percentile rules:\n"
    "- MUST be within top 0.1% to top 3% (never higher than top 3%)\n"
    "- Include a numeric value + % sign (e.g. \"top 0.25%\", \"top 2%\")\n"
    "- Use digits (no words like \"three percent\")\n\n"
    "Return ONLY valid JSON. No markdown, no extra keys, no extra text."
)

BRAINROT_SCORE_PROMPT = (
    "How brainrotted is this feed? Rate 0-100.\n\n"
    "Calibration:\n"
    "- 15: Curated documentaries, tutorials, specific hobbies\n"
    "- 45: Mix of intentional interests and algorithmic drift\n"
    "- 75: Mostly viral trends and rabbit holes\n"
    "- 95: Pure chaos, no coherent theme, random viral clips\n\n"
    "Use the FULL range. Genuinely chaotic feeds deserve 85+.\n"
    "Genuinely curated feeds deserve 20-.\n\n"
    "Respond with ONLY the integer. No explanation."
)

BRAINROT_EXPLANATION_PROMPT = (
    "As a friend who just saw their history, describe their scroll habits in one sentence to them.\n\n"
    "Max 80 characters. Name specific content, not patterns.\n"
    "Never use: algorithm, viral, hashtags, trending, repetitive, engagement.\n\n"
    "Sentence shuld start with You"
    "Respond with ONLY the sentence. No markdown."
)

KEYWORD_2026_PROMPT = (
    "What mantra does this person need for 2026?\n\n"
    "This is about TRAJECTORY, not description. Do NOT echo content themes.\n\n"
    "The logic: fitness obsession -> Rest Is Power (not Keep Grinding)\n"
    "The logic: chaos consumption -> Find Your Anchor (not Embrace Chaos)\n\n"
    "2-3 words, Title Case.\n\n"
    "Respond with ONLY the mantra. No explanation, no analysis, no markdown."
)

ROAST_THUMB_PROMPT = (
    "Write a playful one-liner roast about how much the user's thumb has scrolled, given the total videos/time watched.\n"
    "Return ONLY one line in the format: <clause> — <clause>\n"
    "Constraints:\n"
    "- total length <= 100 characters\n"
    "- the text after the em dash (—) must be <= 60 characters\n"
    "- no emojis, quotes, braces, or line breaks"
)

ACCESSORY_SET_PROMPT = (
    "Pick a cat and one accessory internal_name for each slot (head/body/other) based on the user's watch patterns.\n"
    "You MUST choose values EXACTLY from the allowed lists provided.\n"
    "Important: internal_name values may contain underscores; do NOT modify them. Keep them EXACTLY how they are, like Head_Vlog_MessyBun\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    "{\"cat_name\":\"...\",\"head\":{\"internal_name\":\"...\",\"reason\":\"...\"},\"body\":{...},\"other\":{...}}\n"
    "Constraints:\n"
    "- reason must be EXACTLY 3 words\n"
    "- each word is 1-9 characters (letters/numbers only)\n"
    "- no emojis, punctuation, or line breaks inside the reason text\n"
    "No extra keys, no extra text, no markdown."
)

ACCESSORY_REASONS_PROMPT = (
    "Write a short reason for each chosen accessory internal_name based on the user's watch patterns.\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    "{\"head_reason\":\"...\",\"body_reason\":\"...\",\"other_reason\":\"...\"}\n"
    "Constraints:\n"
    "- each reason is EXACTLY 3 words\n"
    "- Do NOT use platform words like scroll, feed, viewer, consumer, content, browser, dopamine, or watcher, algorithm, viral, hashtags, trending, repetitive, engagement.\n"
    "- each word is 1-9 characters (letters/numbers only)\n"
    "- no emojis, punctuation, underscores, or line breaks inside the reason text\n"
    "No extra keys, no extra text, no markdown."
)

# ============================================================================
# Weekly Report Prompts
# ============================================================================

WEEKLY_FEEDING_STATE_PROMPT = (
    "Based on this week's watch history, determine the user's viewing state.\n\n"
    "Choose EXACTLY ONE from: curious | excited | cozy | sleepy | dizzy\n\n"
    "State definitions:\n"
    "- curious: actively exploring new topics, trying different content types\n"
    "- excited: engaged with trending content, popular creators, current events\n"
    "- cozy: comfort content, familiar creators, rewatching favorites\n"
    "- sleepy: passive scrolling, low engagement, background viewing\n"
    "- dizzy: chaotic browsing, no clear pattern, random content jumps\n\n"
    "Respond with ONLY the state word in lowercase. No explanation, no punctuation."
)

WEEKLY_TOPICS_PROMPT = (
    "Extract the top 3 topics from this week's watch history.\n\n"
    "Return ONLY valid JSON: an array of objects with 'topic' key.\n"
    "Example: [{\"topic\": \"Korean Street Food\"}, {\"topic\": \"Home Renovation\"}, {\"topic\": \"Cat Videos\"}]\n\n"
    "Rules:\n"
    "- Each topic is 2-4 words, Title Case\n"
    "- Be specific (not just 'Comedy' or 'Music')\n"
    "- Topics should reflect what they actually watched this week\n"
    "- No markdown, no extra keys, no extra text"
)

WEEKLY_RABBIT_HOLE_PROMPT = (
    "Did the user fall into any 'rabbit holes' this week? (binge-watching one specific topic)\n\n"
    "Return ONLY valid JSON:\n"
    "If rabbit hole detected: {\"category\": \"topic name\", \"count\": estimated_video_count}\n"
    "If no rabbit hole: {\"category\": null, \"count\": 0}\n\n"
    "A rabbit hole means: watched 20+ videos on one narrow topic in a short time.\n"
    "The category should be specific (e.g., 'ASMR Soap Cutting' not just 'ASMR').\n\n"
    "No markdown, no extra keys, no explanation."
)

WEEKLY_NUDGE_PROMPT = (
    "Write a friendly one-liner about the user's viewing habits this week.\n\n"
    "Max 60 characters. Be specific to what they watched, not generic.\n"
    "Tone: playful, like a friend commenting on their habits.\n\n"
    "Examples of good nudges:\n"
    "- 'Your cat videos peaked on Tuesday night'\n"
    "- 'That cooking phase hit different this week'\n"
    "- 'You found a new favorite creator, huh?'\n\n"
    "Never use: algorithm, viral, trending, engagement, dopamine.\n\n"
    "Respond with ONLY the sentence. No quotes, no markdown."
)


def _stable_pick(seed: str, options: List[str], default: str) -> str:
    if not options:
        return default
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    with suppress(Exception):
        idx = int(digest[:8], 16) % len(options)
        return options[idx]
    return options[0]


def accessory_fallback_reason(
    *,
    seed: str,
    slot: str,
    internal_name: str,
    hints: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Single source of truth for accessory "reason" strings.

    Requirements:
    - Exactly 3 words
    - Only letters/numbers/spaces
    """
    night_templates = [
        "Late Night Loop",
        "Midnight Doom Scroll",
        "After Dark Binge",
        "Night Owl Mode",
        "Bedtime Scroll Spiral",
    ]
    early_templates = [
        "Early Bird Scroll",
        "Morning Feed Ritual",
        "Sunrise Scroll Mode",
        "Coffee Then Scroll",
        "Fresh Start Vibes",
    ]
    day_templates = [
        "Lunch Break Scroll",
        "Workday Feed Check",
        "Afternoon Scroll Drift",
        "Daytime Trend Hunt",
        "Between Tasks Scroll",
    ]
    music_templates = [
        "Music First Mood",
        "Beat Drop Repeat",
        "Audio On Always",
        "Song Loop Season",
    ]
    creator_templates = [
        "Creator Stan Energy",
        "Fave Creator Loop",
        "Creator Rabbit Hole",
        "Account Check Ritual",
    ]
    niche_templates = [
        "Deep Niche Dive",
        "Niche Hop Energy",
        "Hyper Niche Focus",
        "Deep Dive Energy",
    ]
    algo_templates = [
        "ForYou Page Match",
        "Feed Taste Locked",
        "Your Feed Approved",
    ]
    generic_templates = [
        "Hyper Focus Mode",
        "Chaotic Scroll Era",
        "Cozy Core Vibes",
        "Main Char Arc",
        "Save Share Repeat",
        "Quiet Lurk Vibes",
    ]

    hints = hints or {}
    scroll_time = hints.get("scroll_time")
    scroll_title = str((scroll_time or {}).get("title") or "") if isinstance(scroll_time, dict) else ""
    peak_hour_int: Optional[int] = None
    night_pct_num: Optional[float] = None
    with suppress(Exception):
        peak_hour_int = int(hints.get("peak_hour"))
    with suppress(Exception):
        night_pct_num = float(hints.get("night_pct"))

    is_night = (
        ("Night" in scroll_title)
        or ("3AM" in scroll_title)
        or ("3Am" in scroll_title)
        or (peak_hour_int is not None and (peak_hour_int >= 22 or peak_hour_int <= 3))
        or (night_pct_num is not None and night_pct_num >= 20.0)
    )
    is_early = ("Early" in scroll_title) or (peak_hour_int is not None and 4 <= peak_hour_int <= 9)

    candidates: List[str] = []
    if is_night:
        candidates.extend(night_templates)
    elif is_early:
        candidates.extend(early_templates)
    else:
        candidates.extend(day_templates)

    top_music = hints.get("top_music")
    if isinstance(top_music, dict):
        with suppress(Exception):
            if int(top_music.get("count") or 0) > 0:
                candidates.extend(music_templates)
    top_creators = hints.get("top_creators")
    if isinstance(top_creators, list) and top_creators:
        candidates.extend(creator_templates)
    top_niches = hints.get("top_niches")
    if isinstance(top_niches, list) and top_niches:
        candidates.extend(niche_templates)

    candidates.extend(algo_templates)
    candidates.extend(generic_templates)

    return _stable_pick(f"{seed}:{slot}:{internal_name}", candidates, "Deep Dive Energy")

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

PERSONALITY_PROMPT = (
    "Who is this person? Give them a character name in exactly 2 words.\n\n"
    "Format: [Adjective] [Noun] in Title Case.\n\n"
    "Do NOT use platform words like scroll, feed, viewer, consumer, content, browser, or watcher.\n"
    "Describe who they ARE, not what they do on the app.\n\n"
    "Respond with ONLY the 2-word name. No explanation, no markdown, no quotes."
)

PERSONALITY_EXPLANATION_PROMPT = (
    "Why does this name fit? One sentence, max 80 characters.\n\n"
    "Rules:\n"
    "- Start with evidence, not This person or They\n"
    "- Name specific content you noticed\n"
    "- No markdown, no headers, no formatting\n\n"
    "Respond with ONLY the sentence. Nothing else."
)

NICHE_JOURNEY_PROMPT = (
    "Name 5 distinct interest areas from this watch history.\n\n"
    "{\"themes\": [\"Area 1\", \"Area 2\", \"Area 3\", \"Area 4\", \"Area 5\"]}\n\n"
    "Each area: 2-4 words, Title Case. Be specific, not generic.\n"
    "Raw JSON only. Start with {, end with }. No markdown."
)

TOP_NICHES_PROMPT = (
    "Name their top 2 niche interests. Niches are SPECIFIC.\n\n"
    "Too broad: Music or Comedy\n"
    "Just right: Pedal Steel Guitar or Sink Humor\n\n"
    "{\"top_niches\": [\"Niche1\", \"Niche2\"], \"top_niche_percentile\": \"top X%\"}\n\n"
    "Each niche: 1-3 words max. No ampersands.\n"
    "Percentile: 1%=extremely obscure, 5%=dedicated community, 15%=somewhat niche, 25%=common interest.\n\n"
    "Raw JSON only. Start with {, end with }. No markdown."
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
    "As a friend who just saw their history, describe their scroll habits in one sentence.\n\n"
    "Max 80 characters. Name specific content, not patterns.\n"
    "Never use: algorithm, viral, hashtags, trending, repetitive, engagement.\n\n"
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

# Unchanged from original - accessory prompts work well as-is
ACCESSORY_SET_PROMPT = (
    "Pick a cat and one accessory internal_name for each slot (head/body/other) based on the user's watch patterns.\n"
    "You MUST choose values EXACTLY from the allowed lists provided.\n"
    "Important: internal_name values may contain underscores; do NOT modify them. Keep them EXACTLY how they are, like Head_Vlog_MessyBun\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    "{\"cat_name\":\"...\",\"head\":{\"internal_name\":\"...\",\"reason\":\"...\"},\"body\":{...},\"other\":{...}}\n"
    "Constraints:\n"
    "- reason must be EXACTLY 3 words\n"
    "- each word is 1-9 characters (letters/numbers only)\n"
    "- no emojis, quotes, punctuation, or line breaks\n"
    "No extra keys, no extra text, no markdown."
)

ACCESSORY_REASONS_PROMPT = (
    "Write a short reason for each chosen accessory internal_name based on the user's watch patterns.\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    "{\"head_reason\":\"...\",\"body_reason\":\"...\",\"other_reason\":\"...\"}\n"
    "Constraints:\n"
    "- each reason is EXACTLY 3 words\n"
    "- each word is 1-9 characters (letters/numbers only)\n"
    "- no emojis, quotes, punctuation, underscores, or line breaks\n"
    "No extra keys, no extra text, no markdown."
)

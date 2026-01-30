import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .prompts import accessory_fallback_reason

ItemRow = Dict[str, str]

_ITEMS_CACHE: Optional[List[ItemRow]] = None
_ITEM_ID_TO_INTERNAL_NAME: Optional[Dict[str, str]] = None
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "items.csv"


def load_items(path: Optional[Path] = None) -> List[ItemRow]:
    """Load items.csv once and cache it."""
    global _ITEMS_CACHE
    global _ITEM_ID_TO_INTERNAL_NAME
    if _ITEMS_CACHE is not None:
        return _ITEMS_CACHE
    target = path or _DEFAULT_PATH
    rows: List[ItemRow] = []
    with target.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    _ITEMS_CACHE = rows
    _ITEM_ID_TO_INTERNAL_NAME = None
    return rows


def _split_slot(internal_name: str) -> str:
    # Names look like "Head_Vlog_MessyBun"; slot is the first token.
    return internal_name.split("_", 1)[0]


def build_accessory_lookup(items: List[ItemRow]) -> Dict[str, List[ItemRow]]:
    grouped: Dict[str, List[ItemRow]] = {"Head": [], "Body": [], "Other": []}
    for row in items:
        slot = _split_slot(row["internal_name"])
        if slot in grouped:
            grouped[slot].append(row)
    return grouped


def pick_from_series(
    grouped: Dict[str, List[ItemRow]], set_series: Optional[str] = None
) -> Tuple[ItemRow, ItemRow, ItemRow]:
    """Pick a head/body/other trio. Prefer matching set_series if available."""
    def pick(slot: str) -> ItemRow:
        candidates = grouped.get(slot, [])
        if set_series:
            filtered = [row for row in candidates if row.get("set_series") == set_series]
            if filtered:
                return random.choice(filtered)
        return random.choice(candidates) if candidates else {"internal_name": "unknown"}

    return pick("Head"), pick("Body"), pick("Other")


def select_accessory_set(
    preferred_series: Optional[str] = None, items_path: Optional[Path] = None
) -> Dict[str, Dict[str, str]]:
    """Return head/body/other internal names chosen from items.csv."""
    items = load_items(items_path)
    grouped = build_accessory_lookup(items)
    head, body, other = pick_from_series(grouped, preferred_series)

    # Reasons here are only a UI-safe placeholder for callers that need a complete AccessorySet.
    # The wrapped analysis worker overwrites reasons based on user-specific context.
    seed = f"accessories:{random.getrandbits(64)}"
    head_name = head.get("internal_name", "unknown")
    body_name = body.get("internal_name", "unknown")
    other_name = other.get("internal_name", "unknown")
    return {
        "head": {"internal_name": head_name, "reason": accessory_fallback_reason(seed=seed, slot="head", internal_name=head_name)},
        "body": {"internal_name": body_name, "reason": accessory_fallback_reason(seed=seed, slot="body", internal_name=body_name)},
        "other": {"internal_name": other_name, "reason": accessory_fallback_reason(seed=seed, slot="other", internal_name=other_name)},
    }


def internal_name_for_item_id(item_id: str, items_path: Optional[Path] = None) -> Optional[str]:
    global _ITEM_ID_TO_INTERNAL_NAME
    if not item_id:
        return None
    if _ITEM_ID_TO_INTERNAL_NAME is None:
        items = load_items(items_path)
        mapping: Dict[str, str] = {}
        for row in items:
            rid = row.get("item_id")
            name = row.get("internal_name")
            if rid and name:
                mapping[str(rid)] = str(name)
        _ITEM_ID_TO_INTERNAL_NAME = mapping
    return _ITEM_ID_TO_INTERNAL_NAME.get(str(item_id))

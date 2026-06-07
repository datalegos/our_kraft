"""
Deduplicate normalized records on ``(type, value)`` (case-insensitive value).
"""

from __future__ import annotations

from models.schemas import NormalizedIOC


def deduplicate(records: list[NormalizedIOC]) -> tuple[list[NormalizedIOC], int]:
    """
    Keep first occurrence of each (type, value); merge tags from duplicates into the kept row.

    Returns ``(deduped_list, number_of_duplicates_removed)``.
    """
    seen: dict[tuple[str, str], NormalizedIOC] = {}
    order: list[tuple[str, str]] = []
    dup_count = 0

    for r in records:
        t = (r.get("type") or "").strip().lower()
        v = (r.get("value") or "").strip().lower()
        if not t or not v:
            continue
        key = (t, v)
        if key not in seen:
            seen[key] = r
            order.append(key)
        else:
            dup_count += 1
            keeper = seen[key]
            extra_tags = r.get("tags") or []
            if extra_tags:
                merged = list(dict.fromkeys((keeper.get("tags") or []) + list(extra_tags)))
                keeper["tags"] = merged

    return [seen[k] for k in order], dup_count

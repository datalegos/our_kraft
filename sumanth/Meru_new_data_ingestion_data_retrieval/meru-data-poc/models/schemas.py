"""
Unified normalized IOC schema (JSON-serializable dicts).
"""

from __future__ import annotations

from typing import Any, TypedDict


class ContextDict(TypedDict, total=False):
    """Optional campaign / actor / free-text context."""

    campaign: str
    actor: str
    description: str


class NormalizedIOC(TypedDict, total=False):
    """One observable in the canonical pipeline output."""

    source: str
    type: str
    value: str
    malware_family: str
    threat_type: str
    tags: list[str]
    confidence: str
    first_seen: str
    last_seen: str
    context: ContextDict


def empty_context() -> ContextDict:
    """Default empty context object."""
    return {"campaign": "", "actor": "", "description": ""}


def new_record(**kwargs: Any) -> NormalizedIOC:
    """Build a normalized record with safe defaults for missing fields."""
    ctx = kwargs.pop("context", None)
    if ctx is None:
        ctx = empty_context()
    rec: NormalizedIOC = {
        "source": str(kwargs.get("source", "")),
        "type": str(kwargs.get("type", "")),
        "value": str(kwargs.get("value", "")),
        "malware_family": str(kwargs.get("malware_family", "")),
        "threat_type": str(kwargs.get("threat_type", "")),
        "tags": list(kwargs.get("tags", []) or []),
        "confidence": str(kwargs.get("confidence", "")),
        "first_seen": str(kwargs.get("first_seen", "")),
        "last_seen": str(kwargs.get("last_seen", "")),
        "context": ctx,
    }
    return rec

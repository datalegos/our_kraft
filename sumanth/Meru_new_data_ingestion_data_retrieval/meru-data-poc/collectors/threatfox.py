"""
Abuse.ch ThreatFox: POST ``{"query": "get_iocs", "days": N}``.

Auth: HTTP header ``Auth-Key`` (optional).
"""

from __future__ import annotations

from typing import Any

from core.config import Settings
from core.http_client import HttpClient, abuse_ch_headers, abuse_ch_json_body
from core.logger import get_logger

logger = get_logger(__name__)


def collect(settings: Settings, http: HttpClient) -> list[dict[str, Any]]:
    """Return ThreatFox ``data`` IoC rows."""
    headers = abuse_ch_headers(settings.threatfox_auth_key)
    body = abuse_ch_json_body({"query": "get_iocs", "days": settings.threatfox_days}, settings.threatfox_auth_key)
    data = http.post_json(settings.threatfox_api_url, body, headers=headers)
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        logger.warning("ThreatFox: unexpected response (query_status=%s)", data.get("query_status"))
        return []
    logger.info("ThreatFox collected %s IoCs", len(rows))
    return [r for r in rows if isinstance(r, dict)]

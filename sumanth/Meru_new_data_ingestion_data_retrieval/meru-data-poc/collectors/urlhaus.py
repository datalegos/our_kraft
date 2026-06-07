"""
Abuse.ch URLhaus: POST ``{"query": "get_urls", "limit": N}``.

Auth: HTTP header ``Auth-Key`` (optional but recommended).
"""

from __future__ import annotations

from typing import Any

from core.config import Settings
from core.http_client import HttpClient, abuse_ch_headers, abuse_ch_json_body
from core.logger import get_logger

logger = get_logger(__name__)


def collect(settings: Settings, http: HttpClient) -> list[dict[str, Any]]:
    """Return URLhaus ``urls`` entries (each includes url, host, payloads, etc.)."""
    headers = abuse_ch_headers(settings.urlhaus_auth_key)
    body = abuse_ch_json_body({"query": "get_urls", "limit": settings.urlhaus_limit}, settings.urlhaus_auth_key)
    data = http.post_json(settings.urlhaus_api_url, body, headers=headers)
    urls = data.get("urls") if isinstance(data, dict) else None
    if not isinstance(urls, list):
        logger.warning("URLhaus: unexpected response (query_status=%s)", data.get("query_status"))
        return []
    logger.info("URLhaus collected %s url rows", len(urls))
    return [u for u in urls if isinstance(u, dict)]

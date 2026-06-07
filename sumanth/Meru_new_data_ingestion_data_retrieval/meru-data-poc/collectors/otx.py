"""
AlienVault OTX: ``/pulses/subscribed`` plus per-pulse indicators.

Auth: header ``X-OTX-API-KEY``.
"""

from __future__ import annotations

from typing import Any

from core.config import Settings
from core.http_client import HttpClient
from core.logger import get_logger

logger = get_logger(__name__)


def collect(settings: Settings, http: HttpClient) -> list[dict[str, Any]]:
    """
    Fetch subscribed pulses (with search fallback) and flatten pulse + indicator pairs.

    Each returned row has ``pulse`` and ``indicator`` dicts aligned with normalize expectations.
    """
    if not settings.otx_api_key.strip():
        logger.warning("OTX skipped: OTX_API_KEY empty")
        return []

    base = settings.otx_base_url.rstrip("/")
    headers = {"X-OTX-API-KEY": settings.otx_api_key}

    sub = http.get_json(f"{base}/pulses/subscribed", headers=headers)
    pulses = sub.get("results") if isinstance(sub, dict) else None
    if not isinstance(pulses, list) or not pulses:
        logger.info("OTX: no subscribed pulses, using search fallback")
        q = "tag:malware"
        sub = http.get_json(f"{base}/search/pulses", headers=headers, params={"q": q, "page": 1})
        pulses = sub.get("results") if isinstance(sub, dict) else []
        if not isinstance(pulses, list):
            pulses = []

    out: list[dict[str, Any]] = []
    for pulse in pulses[: settings.otx_pulse_limit]:
        if not isinstance(pulse, dict):
            continue
        pid = pulse.get("id")
        if pid is None:
            continue
        ind_resp = http.get_json(f"{base}/pulses/{pid}/indicators", headers=headers)
        indicators = ind_resp.get("results") if isinstance(ind_resp, dict) else []
        if not isinstance(indicators, list):
            indicators = []

        pulse_meta = {
            "pulse_id": pulse.get("id"),
            "name": pulse.get("name"),
            "description": pulse.get("description"),
            "tags": pulse.get("tags"),
            "adversary": pulse.get("adversary"),
            "malware_families": pulse.get("malware_families"),
        }

        for ind in indicators:
            if not isinstance(ind, dict):
                continue
            out.append(
                {
                    "pulse": pulse_meta,
                    "indicator": {
                        "indicator": ind.get("indicator"),
                        "type": ind.get("type"),
                        "role": ind.get("role"),
                        "is_active": ind.get("is_active"),
                        "created": ind.get("created"),
                    },
                }
            )

    logger.info("OTX collected %s pulse-indicator rows", len(out))
    return out

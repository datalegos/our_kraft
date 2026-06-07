"""
HTTP client with retries, timeouts, and optional Abuse.ch ``Auth-Key`` header support.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import requests
from requests import Response

from core.config import Settings
from core.logger import get_logger

logger = get_logger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpClient:
    """Thin ``requests.Session`` wrapper with retry on transient errors."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": settings.http_user_agent,
                "Accept": "application/json",
            }
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Response:
        merged = dict(self._session.headers)
        if headers:
            merged.update(headers)
        last_exc: Exception | None = None
        for attempt in range(1, self._s.max_retries + 1):
            try:
                resp = self._session.request(
                    method,
                    url,
                    headers=merged,
                    json=json_body,
                    params=dict(params) if params else None,
                    timeout=self._s.request_timeout,
                )
                if resp.status_code in RETRY_STATUS and attempt < self._s.max_retries:
                    wait = self._s.backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        "HTTP %s %s — retry %s/%s in %.1fs",
                        resp.status_code,
                        url,
                        attempt,
                        self._s.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                if attempt < self._s.max_retries:
                    wait = self._s.backoff_factor * (2 ** (attempt - 1))
                    logger.warning("HTTP error %s — retry %s in %.1fs", e, attempt, wait)
                    time.sleep(wait)
                else:
                    raise
        if last_exc:
            raise last_exc
        raise RuntimeError("request failed without response")

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        """GET and parse JSON."""
        r = self._request("GET", url, headers=headers, params=params)
        return r.json()

    def post_json(
        self,
        url: str,
        body: dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """POST JSON body; sets ``Content-Type: application/json``."""
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        r = self._request("POST", url, json_body=body, headers=h)
        return r.json()

    def get_text(self, url: str, *, headers: Mapping[str, str] | None = None) -> str:
        """GET and return decoded text (for HTML)."""
        h = dict(headers or {})
        h.setdefault("Accept", "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8")
        r = self._request("GET", url, headers=h)
        return r.text


def abuse_ch_headers(auth_key: str) -> dict[str, str]:
    """Build headers for Abuse.ch APIs per spec: ``Auth-Key`` HTTP header."""
    h: dict[str, str] = {}
    if auth_key.strip():
        h["Auth-Key"] = auth_key.strip()
    return h


def abuse_ch_json_body(base: dict[str, Any], auth_key: str) -> dict[str, Any]:
    """
    Merge ``Auth-Key`` into POST JSON when configured.

    Abuse.ch accepts the key in the JSON body; some networks also expect the header.
    This project sends **both** when a key is present.
    """
    if not auth_key.strip():
        return base
    return {**base, "Auth-Key": auth_key.strip()}

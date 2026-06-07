"""
NJCCIC public alerts: HTML fetch + BeautifulSoup + regex extraction.

No authentication. Many deployments return an Incapsula shell (no content) to non-browser clients.
"""

from __future__ import annotations

import re
import time
from typing import Any

from core.config import Settings
from core.http_client import HttpClient
from core.logger import get_logger

logger = get_logger(__name__)

_MALWARE_HINTS = (
    "emotet",
    "qakbot",
    "quakbot",
    "cobalt strike",
    "lockbit",
    "blackcat",
    "alphv",
    "ryuk",
    "trickbot",
    "bazarloader",
    "icedid",
)


def _malware_hits(text: str) -> list[str]:
    low = text.lower()
    return [name for name in _MALWARE_HINTS if name in low]


def collect(settings: Settings, http: HttpClient) -> list[dict[str, Any]]:
    """
    Fetch alerts page and emit structured documents for :mod:`pipeline.normalize`.

    Each document: title, date, summary, full_text, mitigation, malware_names.
    """
    time.sleep(1)
    url = settings.njccic_base_url
    html = http.get_text(url)

    if "_Incapsula_" in html or "Incapsula" in html:
        logger.warning("NJCCIC: Incapsula challenge page — no alert HTML to parse")
        return [
            {
                "title": "WAF challenge (no HTML content)",
                "date": "",
                "summary": "Server returned Incapsula JavaScript challenge; use browser automation or an official feed if available.",
                "full_text": "",
                "mitigation": "",
                "malware_names": [],
                "blocked_by": "incapsula",
            }
        ]

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 required for NJCCIC collector")
        return []

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    docs: list[dict[str, Any]] = []

    # Prefer article-like blocks; fall back to link-driven sections
    articles = soup.find_all("article")
    if not articles:
        articles = soup.find_all("div", class_=re.compile(r"teaser|card|alert|news", re.I))

    if articles:
        for art in articles[:40]:
            title_el = art.find(["h1", "h2", "h3", "h4", "a"])
            title = title_el.get_text(separator=" ", strip=True) if title_el else ""
            if not title or len(title) < 4:
                continue
            full_text = art.get_text(separator="\n", strip=True)
            summary = full_text[:800]
            date_el = art.find(class_=re.compile(r"date|time|published", re.I))
            date_s = date_el.get_text(strip=True) if date_el else ""
            mit = ""
            for lab in art.find_all(string=re.compile(r"mitigat|recommend|action", re.I)):
                parent = getattr(lab, "parent", None)
                if parent:
                    mit = parent.get_text(separator=" ", strip=True)[:2000]
                    break
            blob = f"{title}\n{full_text}"
            docs.append(
                {
                    "title": title,
                    "date": date_s,
                    "summary": summary,
                    "full_text": full_text,
                    "mitigation": mit,
                    "malware_names": _malware_hits(blob),
                }
            )
    else:
        # Fallback: significant links on threat pages
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            t = a.get_text(separator=" ", strip=True)
            href = str(a.get("href", ""))
            if not t or len(t) < 12:
                continue
            if not any(x in href.lower() for x in ("/threat-center/", "alert", "advis")):
                continue
            if t in seen:
                continue
            seen.add(t)
            parent = a.find_parent(["article", "li", "div"]) or a
            block = parent.get_text(separator="\n", strip=True)
            docs.append(
                {
                    "title": t[:300],
                    "date": "",
                    "summary": block[:800],
                    "full_text": block,
                    "mitigation": "",
                    "malware_names": _malware_hits(block),
                }
            )
            if len(docs) >= 25:
                break

    logger.info("NJCCIC collected %s alert documents", len(docs))
    return docs

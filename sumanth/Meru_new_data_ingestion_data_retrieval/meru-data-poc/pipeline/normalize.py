"""
Map per-collector raw payloads into :class:`models.schemas.NormalizedIOC` rows.
"""

from __future__ import annotations

import re
from typing import Any

from models.schemas import ContextDict, NormalizedIOC, empty_context, new_record

SOURCE_RELIABILITY: dict[str, str] = {
    "otx": "medium",
    "urlhaus": "high",
    "malwarebazaar": "high",
    "threatfox": "high",
    "njccic": "medium",
}

_OTX_TYPE_MAP: dict[str, str] = {
    "IPv4": "ip",
    "IPv6": "ip",
    "domain": "domain",
    "hostname": "domain",
    "URL": "url",
    "FileHash-MD5": "hash",
    "FileHash-SHA1": "hash",
    "FileHash-SHA256": "hash",
}


def _tags(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            if x is None:
                continue
            if isinstance(x, dict):
                n = x.get("name")
                if n is not None:
                    out.append(str(n))
            else:
                out.append(str(x))
        return out
    return [str(v)]


def normalize_batch(source: str, raw_items: list[dict[str, Any]]) -> list[NormalizedIOC]:
    """Dispatch normalization by logical source name."""
    if source == "otx":
        return _norm_otx(raw_items)
    if source == "urlhaus":
        return _norm_urlhaus(raw_items)
    if source == "malwarebazaar":
        return _norm_malwarebazaar(raw_items)
    if source == "threatfox":
        return _norm_threatfox(raw_items)
    if source == "njccic":
        return _norm_njccic(raw_items)
    return []


def _norm_otx(items: list[dict[str, Any]]) -> list[NormalizedIOC]:
    out: list[NormalizedIOC] = []
    rel = SOURCE_RELIABILITY["otx"]
    for row in items:
        pulse = row.get("pulse") or {}
        ind = row.get("indicator") or {}
        ptype = str(ind.get("type") or "")
        utype = _OTX_TYPE_MAP.get(ptype, "hash" if "hash" in ptype.lower() else "domain")
        val = str(ind.get("indicator") or "").strip()
        if not val:
            continue
        ctx: ContextDict = {
            "campaign": str(pulse.get("name") or ""),
            "actor": str(pulse.get("adversary") or ""),
            "description": str(pulse.get("description") or "")[:4000],
        }
        tags = _tags(pulse.get("tags"))
        mf_raw = pulse.get("malware_families")
        if isinstance(mf_raw, list):
            mf_s = ", ".join(str(x) for x in mf_raw if x)[:500]
            for mf in mf_raw:
                t = str(mf).strip()
                if t and t not in tags:
                    tags.append(t)
        else:
            mf_s = str(mf_raw or "")[:500]
        out.append(
            new_record(
                source="otx",
                type=utype,
                value=val,
                malware_family=mf_s,
                threat_type="pulse_indicator",
                tags=tags,
                confidence=rel,
                first_seen=str(ind.get("created") or ""),
                last_seen="",
                context=ctx,
            )
        )
    return out


def _norm_urlhaus(rows: list[dict[str, Any]]) -> list[NormalizedIOC]:
    out: list[NormalizedIOC] = []
    rel = SOURCE_RELIABILITY["urlhaus"]
    for u in rows:
        url = str(u.get("url") or "").strip()
        url_snip = url[:200] if url else ""
        if url:
            ctx: ContextDict = {
                "campaign": "",
                "actor": "",
                "description": f"host={u.get('urlhaus_reference','')} status={u.get('url_status','')}",
            }
            out.append(
                new_record(
                    source="urlhaus",
                    type="url",
                    value=url,
                    malware_family="",
                    threat_type="malware_url",
                    tags=_tags(u.get("tags")),
                    confidence=rel,
                    first_seen=str(u.get("date_added") or ""),
                    last_seen="",
                    context=ctx,
                )
            )
        for p in u.get("payloads") or []:
            if not isinstance(p, dict):
                continue
            for hfield in ("sha256", "md5"):
                hv = str(p.get(hfield) or "").strip()
                if len(hv) >= 32:
                    out.append(
                        new_record(
                            source="urlhaus",
                            type="hash",
                            value=hv.lower(),
                            malware_family=str(p.get("signature") or ""),
                            threat_type="payload",
                            tags=_tags(p.get("file_type")),
                            confidence=rel,
                            first_seen=str(u.get("date_added") or ""),
                            last_seen="",
                            context=empty_context()
                            | {
                                "description": f"payload {hfield} from urlhaus url {url_snip}",
                            },
                        )
                    )
    return out


def _norm_malwarebazaar(rows: list[dict[str, Any]]) -> list[NormalizedIOC]:
    out: list[NormalizedIOC] = []
    rel = SOURCE_RELIABILITY["malwarebazaar"]
    for s in rows:
        sha256 = str(s.get("sha256") or "").strip()
        if sha256:
            out.append(
                new_record(
                    source="malwarebazaar",
                    type="hash",
                    value=sha256.lower(),
                    malware_family=str(s.get("signature") or ""),
                    threat_type="sample",
                    tags=_tags(s.get("tags")) + _tags(s.get("delivery_method")),
                    confidence=rel,
                    first_seen=str(s.get("first_seen") or ""),
                    last_seen="",
                    context={
                        "campaign": "",
                        "actor": "",
                        "description": f"file_name={s.get('file_name','')} type={s.get('file_type','')}",
                    },
                )
            )
        for alt in ("md5", "sha1"):
            hv = str(s.get(alt) or "").strip()
            if hv and len(hv) >= 32:
                out.append(
                    new_record(
                        source="malwarebazaar",
                        type="hash",
                        value=hv.lower(),
                        malware_family=str(s.get("signature") or ""),
                        threat_type=f"sample_{alt}",
                        tags=_tags(s.get("tags")),
                        confidence=rel,
                        first_seen=str(s.get("first_seen") or ""),
                        last_seen="",
                        context={"campaign": "", "actor": "", "description": f"related_{alt}_for_sha256"},
                    )
                )
    return out


def _norm_threatfox(rows: list[dict[str, Any]]) -> list[NormalizedIOC]:
    out: list[NormalizedIOC] = []
    rel = SOURCE_RELIABILITY["threatfox"]
    type_map = {
        "ip:port": "ip",
        "domain": "domain",
        "url": "url",
        "md5": "hash",
        "sha256": "hash",
    }
    for row in rows:
        ioc = str(row.get("ioc") or "").strip()
        if not ioc:
            continue
        it = str(row.get("ioc_type") or "").lower()
        ut = type_map.get(
            it,
            "domain" if "." in ioc and "://" not in ioc else "url" if ioc.startswith("http") else "hash",
        )
        out.append(
            new_record(
                source="threatfox",
                type=ut,
                value=ioc,
                malware_family=str(row.get("malware") or row.get("malware_printable") or ""),
                threat_type=str(row.get("threat_type") or "ioc"),
                tags=_tags(row.get("tags")),
                confidence=str(row.get("confidence_level") or rel),
                first_seen=str(row.get("first_seen") or ""),
                last_seen=str(row.get("last_seen") or ""),
                context={"campaign": "", "actor": "", "description": str(row.get("malware_alias") or "")},
            )
        )
    return out


_CVE_RE = re.compile(r"\bCVE-\d{4}-\d+\b", re.I)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
_DOMAIN_LIKE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}\b",
    re.I,
)
_SHA256_RE = re.compile(r"\b[a-f0-9]{64}\b", re.I)
_MD5_RE = re.compile(r"\b[a-f0-9]{32}\b", re.I)


def _norm_njccic(items: list[dict[str, Any]]) -> list[NormalizedIOC]:
    """Each item is one scraped alert document with text fields."""
    out: list[NormalizedIOC] = []
    rel = SOURCE_RELIABILITY["njccic"]
    for doc in items:
        title = str(doc.get("title") or "")
        body = str(doc.get("full_text") or "")
        summary = str(doc.get("summary") or "")
        mitigation = str(doc.get("mitigation") or "")
        malware_names = doc.get("malware_names") or []
        if not isinstance(malware_names, list):
            malware_names = []
        mjoin = ", ".join(str(m) for m in malware_names if m)
        blob = f"{title}\n{summary}\n{body}\n{mitigation}"

        base_ctx: ContextDict = {
            "campaign": title[:500],
            "actor": "",
            "description": (summary or body)[:4000],
        }
        if mitigation.strip():
            base_ctx["description"] = (base_ctx.get("description") or "") + "\nMitigation: " + mitigation[:2000]

        def add(ioc_type: str, value: str, tag_extra: str = "") -> None:
            value = value.strip()
            if not value:
                return
            tags = ["njccic", "advisory"]
            if tag_extra:
                tags.append(tag_extra)
            out.append(
                new_record(
                    source="njccic",
                    type=ioc_type,
                    value=value,
                    malware_family=mjoin,
                    threat_type="advisory",
                    tags=tags,
                    confidence=rel,
                    first_seen=str(doc.get("date") or ""),
                    last_seen="",
                    context=dict(base_ctx),
                )
            )

        for m in _CVE_RE.finditer(blob):
            cve = m.group(0).upper()
            add("url", f"https://nvd.nist.gov/vuln/detail/{cve}", "cve")

        for u in sorted(set(_URL_RE.findall(blob))):
            if "nvd.nist.gov" in u.lower():
                continue
            add("url", u.rstrip(").,;]"), "extracted_url")

        for ip in sorted(set(_IPV4_RE.findall(blob))):
            add("ip", ip, "extracted_ip")

        for dom in sorted(set(_DOMAIN_LIKE.findall(blob))):
            if dom.lower().startswith("cve-"):
                continue
            if any(x in dom.lower() for x in ("nvd.nist.gov", "microsoft.com", "cyber.nj.gov")):
                continue
            add("domain", dom.lower(), "extracted_domain")

        for hx in sorted(set(_SHA256_RE.findall(blob))):
            add("hash", hx.lower(), "sha256")

        for hx in sorted(set(_MD5_RE.findall(blob))):
            add("hash", hx.lower(), "md5")

    return out

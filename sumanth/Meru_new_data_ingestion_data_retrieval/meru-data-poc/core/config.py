"""
Load settings from ``meru-data-poc/.env`` (``python-dotenv`` / ``dotenv_values`` only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

_ENV_NAME = ".env"
_ROOT = Path(__file__).resolve().parent.parent


def env_path() -> Path:
    return _ROOT / _ENV_NAME


def _str(v: Mapping[str, str | None], key: str, default: str = "") -> str:
    raw = v.get(key)
    return default if raw is None else str(raw).strip()


def _int(v: Mapping[str, str | None], key: str, default: int) -> int:
    raw = v.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Application settings from ``.env``."""

    otx_api_key: str
    otx_base_url: str
    otx_pulse_limit: int

    urlhaus_api_url: str
    urlhaus_auth_key: str
    urlhaus_limit: int

    malwarebazaar_api_url: str
    malwarebazaar_auth_key: str
    malwarebazaar_selector: int

    threatfox_api_url: str
    threatfox_auth_key: str
    threatfox_days: int

    njccic_base_url: str

    output_dir: Path
    request_timeout: int
    max_retries: int
    backoff_factor: float
    http_user_agent: str
    log_level: str

    mongodb_uri: str


def load_settings() -> Settings:
    path = env_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Copy .env.example to .env and configure collectors."
        )
    v = dict(dotenv_values(path, encoding="utf-8-sig"))
    otx_base = _str(v, "OTX_BASE_URL") or _str(v, "OTX_API_BASE_URL", "https://otx.alienvault.com/api/v1")
    mb_sel = _int(v, "MALWAREBAZAAR_SELECTOR", 0)
    if mb_sel <= 0:
        mb_sel = _int(v, "MALWAREBAZAAR_QUERY_LIMIT", 100)

    return Settings(
        otx_api_key=_str(v, "OTX_API_KEY"),
        otx_base_url=otx_base.rstrip("/"),
        otx_pulse_limit=_int(v, "OTX_PULSE_LIMIT", 25),
        urlhaus_api_url=_str(v, "URLHAUS_API_URL", "https://urlhaus-api.abuse.ch/v1/").rstrip("/") + "/",
        urlhaus_auth_key=_str(v, "URLHAUS_AUTH_KEY"),
        urlhaus_limit=_int(v, "URLHAUS_LIMIT", 100),
        malwarebazaar_api_url=_str(v, "MALWAREBAZAAR_API_URL", "https://mb-api.abuse.ch/api/v1/").rstrip("/") + "/",
        malwarebazaar_auth_key=_str(v, "MALWAREBAZAAR_AUTH_KEY"),
        malwarebazaar_selector=mb_sel,
        threatfox_api_url=_str(v, "THREATFOX_API_URL", "https://threatfox-api.abuse.ch/api/v1/").rstrip("/") + "/",
        threatfox_auth_key=_str(v, "THREATFOX_AUTH_KEY"),
        threatfox_days=_int(v, "THREATFOX_DAYS", 7),
        njccic_base_url=_str(
            v,
            "NJCCIC_BASE_URL",
            "https://www.cyber.nj.gov/threat-center/alerts-advisories",
        ),
        output_dir=(_ROOT / _str(v, "OUTPUT_DIR", "output")).resolve(),
        request_timeout=_int(v, "REQUEST_TIMEOUT", 45),
        max_retries=_int(v, "HTTP_MAX_RETRIES", 3),
        backoff_factor=float(_str(v, "HTTP_BACKOFF_FACTOR", "0.8") or "0.8"),
        http_user_agent=_str(
            v,
            "HTTP_USER_AGENT",
            "meru-threat-collector/1.0 (research)",
        ),
        log_level=_str(v, "LOG_LEVEL", "INFO").upper(),
        mongodb_uri=_str(v, "MONGODB_URI"),
    )

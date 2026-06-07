#!/usr/bin/env python3
"""
Threat Intelligence Data Collector — run from the ``meru-data-poc`` directory:

    python main.py
    python main.py --sources otx,urlhaus,malwarebazaar,threatfox
    python main.py --sources njccic

Requires ``.env`` next to this file (see ``.env.example``).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from collectors import malwarebazaar, njccic, otx, threatfox, urlhaus
from core.config import Settings, env_path, load_settings
from core.http_client import HttpClient
from core.logger import get_logger, setup_logging
from models.schemas import NormalizedIOC
from pipeline.deduplicate import deduplicate
from pipeline.normalize import normalize_batch

logger = get_logger(__name__)

CollectorFn = Callable[[Settings, HttpClient], list[dict[str, Any]]]

COLLECTORS: dict[str, tuple[str, CollectorFn]] = {
    "otx": ("otx", otx.collect),
    "urlhaus": ("urlhaus", urlhaus.collect),
    "malwarebazaar": ("malwarebazaar", malwarebazaar.collect),
    "threatfox": ("threatfox", threatfox.collect),
    "njccic": ("njccic", njccic.collect),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _maybe_mongodb(uri: str, document: dict[str, Any]) -> None:
    if not uri.strip():
        return
    try:
        from pymongo import MongoClient
    except ImportError:
        logger.warning("MONGODB_URI set but pymongo not installed; skip DB write")
        return
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        db = client.get_default_database()
        db["ti_ingestion"].insert_one(document)
        logger.info("Wrote ingestion document to MongoDB")
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB write failed: %s", exc)


def run(settings: Settings, only: list[str] | None) -> int:
    """Execute selected collectors, normalize, dedupe, persist."""
    http = HttpClient(settings)

    names = only if only else list(COLLECTORS.keys())
    fetched_by_source: dict[str, int] = {}
    raw_by_norm_source: dict[str, list[dict[str, Any]]] = {
        "otx": [],
        "urlhaus": [],
        "malwarebazaar": [],
        "threatfox": [],
        "njccic": [],
    }
    errors = 0

    for key in names:
        if key not in COLLECTORS:
            logger.error("Unknown source %r (choose: %s)", key, ", ".join(sorted(COLLECTORS)))
            errors += 1
            continue
        norm_name, fn = COLLECTORS[key]
        try:
            rows = fn(settings, http)
            fetched_by_source[key] = len(rows)
            raw_by_norm_source[norm_name].extend(rows)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s collector failed: %s", key, exc)
            errors += 1

    normalized: list[NormalizedIOC] = []
    for src, rows in raw_by_norm_source.items():
        if not rows:
            continue
        normalized.extend(normalize_batch(src, rows))

    before_dedupe = len(normalized)
    deduped, dups_removed = deduplicate(normalized)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = settings.output_dir / f"ingestion_{ts}.json"
    payload = {
        "ingested_at_utc": ts,
        "stats": {
            "records_fetched_by_collector": fetched_by_source,
            "records_normalized": before_dedupe,
            "duplicates_removed": dups_removed,
            "records_after_dedupe": len(deduped),
        },
        "records": deduped,
    }
    _write_json(out_path, payload)
    logger.info(
        "Ingestion complete -> %s | normalized=%s deduped=%s removed=%s",
        out_path,
        before_dedupe,
        len(deduped),
        dups_removed,
    )

    _maybe_mongodb(
        settings.mongodb_uri,
        {"ingested_at_utc": ts, "stats": payload["stats"], "records": deduped},
    )

    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Threat intelligence collector + normalizer")
    parser.add_argument(
        "--sources",
        default="",
        help="Comma-separated collectors (default: all). E.g. otx,urlhaus,threatfox",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    setup_logging(settings)
    logger.info("Configuration: %s", env_path())

    only = [s.strip() for s in args.sources.split(",") if s.strip()] or None
    return run(settings, only)


if __name__ == "__main__":
    sys.exit(main())

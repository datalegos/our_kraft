#!/usr/bin/env python3
"""
=============================================================
  ORBIT - NJSecure  |  Package Change Alert Watcher
=============================================================
  Purpose  : Polls the Wazuh Indexer (OpenSearch) every X
             seconds for package install/update/remove alerts
             and writes each new event to a JSON file.

  Confirmed working with:
    - Wazuh 4.14.2 Docker
    - Windows Agent (sumanth)
    - Rule IDs: 100401 (install), 100402 (update), 100403 (remove)
    - Real field path: data.program  (NOT data.package)

  Run this script on your SEPARATE machine.

  Install dependency first:
    pip install requests

  Wazuh Indexer API Docs:
    https://documentation.wazuh.com/current/user-manual/wazuh-indexer/index.html
=============================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import json
import time
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Third-Party (pip install requests) ────────────────────────────────────────
import requests
import urllib3

# Suppress self-signed SSL certificate warnings from Wazuh
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ==============================================================================
#  SECTION 1 — CONFIGURATION
#  ➜ All configuration is loaded from config.py
# ==============================================================================

try:
    import config
except ImportError:
    print("=" * 65)
    print("  ERROR: config.py not found!")
    print("=" * 65)
    print()
    print("  Please create a config.py file with your Wazuh settings.")
    print("  You can copy config.example.py as a starting point:")
    print()
    print("    copy config.example.py config.py")
    print()
    print("  Then edit config.py with your actual values.")
    print()
    sys.exit(1)

# Load all configuration from config.py
INDEXER_HOST          = getattr(config, 'INDEXER_HOST', None)
INDEXER_PORT          = getattr(config, 'INDEXER_PORT', 9200)
INDEXER_USER          = getattr(config, 'INDEXER_USER', 'admin')
INDEXER_PASSWORD      = getattr(config, 'INDEXER_PASSWORD', None)
POLL_INTERVAL_SECONDS = getattr(config, 'POLL_INTERVAL_SECONDS', 60)
OUTPUT_FILE           = Path(getattr(config, 'OUTPUT_FILE', 'orbit_package_changes.json'))
STATE_FILE            = Path("orbit_last_check.txt")
RULE_IDS              = getattr(config, 'RULE_IDS_TO_WATCH', ["100401", "100402", "100403"])
ALERTS_INDEX          = getattr(config, 'WAZUH_ALERTS_INDEX', 'wazuh-alerts-*')

# Validate required configuration
if not INDEXER_HOST:
    print("=" * 65)
    print("  ERROR: INDEXER_HOST not configured!")
    print("=" * 65)
    print()
    print("  Please set INDEXER_HOST in config.py")
    print("  Example: INDEXER_HOST = '192.168.1.100'")
    print()
    sys.exit(1)

if not INDEXER_PASSWORD:
    print("=" * 65)
    print("  ERROR: INDEXER_PASSWORD not configured!")
    print("=" * 65)
    print()
    print("  Please set INDEXER_PASSWORD in config.py")
    print("  Example: INDEXER_PASSWORD = 'YourSecurePassword'")
    print()
    sys.exit(1)


# ==============================================================================
#  SECTION 2 — STATE MANAGEMENT
#  Tracks "when did we last check?" so we never re-process old alerts
# ==============================================================================

def get_last_check_time() -> str:
    """
    Read the timestamp from our state file.

    The state file contains ONE line — an ISO 8601 timestamp
    like: 2026-02-23T17:44:52.000Z

    If this is the first time running (no state file exists yet),
    we look back 10 minutes to catch any very recent alerts.

    Returns a timestamp string the Wazuh Indexer understands.
    """
    if STATE_FILE.exists():
        saved = STATE_FILE.read_text(encoding="utf-8").strip()
        if saved:
            return saved

    # First run — look back 10 minutes
    ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
    return ten_minutes_ago.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def save_last_check_time(timestamp: str):
    """
    Save the timestamp of the newest alert we just processed.
    Called after every successful poll that found alerts.

    This prevents re-processing the same alerts on the next poll.
    """
    STATE_FILE.write_text(timestamp, encoding="utf-8")


# ==============================================================================
#  SECTION 3 — WAZUH INDEXER QUERY
#  Searches the wazuh-alerts-* index for new package alerts
# ==============================================================================

def build_search_query(since_timestamp: str) -> dict:
    """
    Build the OpenSearch query to find package change alerts.

    This is like SQL:
    SELECT * FROM wazuh_alerts
    WHERE rule.id IN ('100401', '100402', '100403')
      AND @timestamp > since_timestamp
    ORDER BY @timestamp ASC
    LIMIT 1000

    The "bool/must" means ALL conditions must be true (like AND).
    The "terms" matches any of the rule IDs in the list.
    The "range/gt" means "greater than" the last check time.

    OpenSearch Query DSL Docs:
    https://opensearch.org/docs/latest/query-dsl/
    """
    return {
        "size": 1000,
        "sort": [
            {"@timestamp": {"order": "asc"}}   # Process oldest first
        ],
        "query": {
            "bool": {
                "must": [
                    {
                        # Match any of our custom package rule IDs
                        "terms": {
                            "rule.id": RULE_IDS
                        }
                    },
                    {
                        # Only alerts newer than our last check
                        "range": {
                            "@timestamp": {
                                "gt": since_timestamp
                            }
                        }
                    }
                ]
            }
        }
    }


def fetch_alerts(base_url: str, auth: tuple, since: str) -> list:
    """
    Call the Wazuh Indexer _search API and return raw alert documents.

    The Indexer API endpoint for searching is:
    POST https://<host>:9200/<index>/_search

    Returns a list of raw OpenSearch documents.
    Each document has "_id" and "_source" keys.
    "_source" contains the actual alert data.

    Returns empty list if something goes wrong (we log the error).
    """
    url      = f"{base_url}/{ALERTS_INDEX}/_search"
    query    = build_search_query(since)

    try:
        response = requests.post(
            url,
            auth=auth,
            json=query,
            verify=False,   # Self-signed cert — safe on internal network
            timeout=30
        )
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to Wazuh Indexer at {base_url}")
        print(f"    Check: Is Docker running? Is port {INDEXER_PORT} reachable?")
        return []
    except requests.exceptions.Timeout:
        print(f"  ✗ Connection timed out after 30s.")
        return []

    if response.status_code == 401:
        print(f"  ✗ Authentication failed (HTTP 401).")
        print(f"    Check: INDEXER_USER and INDEXER_PASSWORD in Section 1.")
        return []

    if response.status_code != 200:
        print(f"  ✗ Indexer returned HTTP {response.status_code}")
        print(f"    Response: {response.text[:300]}")
        return []

    result = response.json()
    return result.get("hits", {}).get("hits", [])


# ==============================================================================
#  SECTION 4 — ALERT PARSING
#  Extracts package details from raw alert using CONFIRMED real field paths
#  Field path confirmed from your real alert: data.program (not data.package)
# ==============================================================================

def parse_alert(raw: dict) -> dict:
    """
    Extract package change details from a raw Wazuh alert.

    Your CONFIRMED real alert structure (from your logs):
    {
      "timestamp": "2026-02-23T17:44:52.891+0000",
      "rule": {
        "id": "100401",
        "level": 7,
        "description": "New package installed:  version ."
      },
      "agent": {
        "id": "001",
        "name": "sumanth",
        "ip": "172.27.112.1"
      },
      "data": {
        "type": "dbsync_packages",
        "operation_type": "INSERTED",
        "program": {                        <-- Windows uses "program" NOT "package"
          "name": "7-Zip 26.00 (x64)",
          "version": "26.00",
          "vendor": "Igor Pavlov",
          "architecture": "x86_64",
          "format": "win",
          "install_time": "2026/02/23 17:44:03",
          "location": "C:\\Program Files\\7-Zip\\",
          "size": "0",
          "description": " "
        }
      }
    }

    Returns a clean flat dictionary with all the important fields.
    """
    # "_source" is the OpenSearch wrapper — the actual alert is inside it
    source  = raw.get("_source", {})

    # Navigate the nested structure safely using .get()
    # .get("key", {}) means: if "key" doesn't exist, use empty dict
    # This prevents KeyError crashes if a field is missing
    agent   = source.get("agent",  {})
    rule    = source.get("rule",   {})
    data    = source.get("data",   {})

    # ── Windows uses "program", Linux uses "package" ──────────────────────────
    # We confirmed from YOUR real alert that Windows sends "program"
    program = data.get("program", {})

    # ── Map rule ID to human-readable action ──────────────────────────────────
    rule_id    = str(rule.get("id", ""))
    action_map = {
        "100401": "INSTALLED",
        "100402": "UPDATED",
        "100403": "REMOVED"
    }
    action = action_map.get(rule_id, data.get("operation_type", "UNKNOWN"))

    # ── Build our clean output event ──────────────────────────────────────────
    return {
        # When did this happen?
        "event_timestamp":     source.get("@timestamp", ""),

        # Which machine?
        "agent_id":            agent.get("id",   ""),
        "agent_name":          agent.get("name", ""),
        "agent_ip":            agent.get("ip",   ""),

        # What happened? (INSTALLED / UPDATED / REMOVED)
        "action":              action,

        # Which program? (using confirmed field path: data.program)
        "program_name":        program.get("name",         ""),
        "program_version":     program.get("version",      ""),
        "program_vendor":      program.get("vendor",       ""),
        "program_architecture":program.get("architecture", ""),
        "program_format":      program.get("format",       ""),
        "program_location":    program.get("location",     ""),
        "program_install_time":program.get("install_time", ""),
        "program_size":        program.get("size",         ""),
        "program_description": program.get("description",  ""),

        # Rule metadata (useful for debugging)
        "rule_id":             rule_id,
        "rule_level":          rule.get("level",       ""),
        "rule_description":    rule.get("description", ""),

        # Internal tracking
        "wazuh_alert_id":      raw.get("_id", ""),
        "captured_by_orbit_at":datetime.now(timezone.utc).isoformat()
    }


# ==============================================================================
#  SECTION 5 — JSON FILE MANAGEMENT
#  Maintains a single growing JSON array file
#  Structure: [ {event1}, {event2}, {event3}, ... ]
# ==============================================================================

def load_existing_events() -> list:
    """
    Load existing events from the output file.
    Returns empty list if file does not exist or is empty.
    """
    if not OUTPUT_FILE.exists():
        return []

    try:
        content = OUTPUT_FILE.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  ! Warning: Could not read {OUTPUT_FILE}: {e}")
        print(f"    Starting with empty file.")
        return []


def append_events(new_events: list):
    """
    Append new package change events to the JSON output file.

    Reads existing events → adds new ones → writes everything back.
    The result is always a valid JSON array.
    """
    existing   = load_existing_events()
    combined   = existing + new_events

    OUTPUT_FILE.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"  ✓ Written {len(new_events)} new event(s) → {OUTPUT_FILE.resolve()}")
    print(f"    Total events in file: {len(combined)}")


# ==============================================================================
#  SECTION 6 — MAIN POLLING LOOP
# ==============================================================================

def main():
    print("=" * 65)
    print("  ORBIT - NJSecure | Package Change Alert Watcher")
    print("  Wazuh 4.14.2 Docker | Windows Agent | Syscollector")
    print("=" * 65)
    print(f"  Indexer  : https://{INDEXER_HOST}:{INDEXER_PORT}")
    print(f"  Rules    : {RULE_IDS}")
    print(f"  Output   : {OUTPUT_FILE.resolve()}")
    print(f"  Interval : Every {POLL_INTERVAL_SECONDS}s")
    print("=" * 65)
    print("  Press CTRL+C to stop.\n")

    base_url = f"https://{INDEXER_HOST}:{INDEXER_PORT}"
    auth     = (INDEXER_USER, INDEXER_PASSWORD)

    # ── Main loop — runs forever until CTRL+C ─────────────────────────────────
    while True:
        now = datetime.now(timezone.utc)
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')} UTC] Polling Wazuh Indexer...")

        # Get timestamp of last successful check
        since = get_last_check_time()
        print(f"  Looking for alerts since: {since}")

        # Query the Indexer
        raw_alerts = fetch_alerts(base_url, auth, since)

        if not raw_alerts:
            print(f"  → No new package alerts found.")
        else:
            print(f"  → {len(raw_alerts)} new alert(s) found!")

            # Parse each alert into a clean event dict
            parsed = []
            for raw in raw_alerts:
                event = parse_alert(raw)
                parsed.append(event)

                # Print a one-line summary to the terminal
                emoji = {"INSTALLED": "📦", "UPDATED": "🔄", "REMOVED": "🗑️"}
                print(
                    f"    {emoji.get(event['action'], '❓')} "
                    f"[{event['action']}] "
                    f"{event['program_name']} "
                    f"v{event['program_version']} | "
                    f"Agent: {event['agent_name']} ({event['agent_ip']}) | "
                    f"At: {event['event_timestamp']}"
                )

            # Write to JSON file
            append_events(parsed)

            # Update state — use timestamp of the NEWEST alert we just processed
            # This ensures next poll only fetches alerts AFTER this one
            newest_timestamp = raw_alerts[-1]["_source"].get("@timestamp", "")
            if newest_timestamp:
                save_last_check_time(newest_timestamp)
                print(f"  State updated → next poll checks after: {newest_timestamp}")

        print(f"  Sleeping {POLL_INTERVAL_SECONDS}s...\n")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Watcher stopped. Goodbye!")
        sys.exit(0)
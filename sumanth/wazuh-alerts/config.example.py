# ==============================================================================
#  ORBIT - Package Alert Watcher Configuration
# ==============================================================================
# INSTRUCTIONS:
# 1. Copy this file: copy config.example.py config.py
# 2. Edit config.py with your actual Wazuh server details
# 3. Never commit config.py to git (it's in .gitignore)

# ==============================================================================
#  REQUIRED SETTINGS
# ==============================================================================

# ── Wazuh Indexer Connection ──────────────────────────────────────────────────
# The Wazuh Indexer is the OpenSearch instance inside your Docker setup.
# It listens on port 9200 by default.

INDEXER_HOST     = "192.168.1.100"           # REQUIRED: Your Wazuh server IP address
INDEXER_PORT     = 9200                       # Default Wazuh Indexer port
INDEXER_USER     = "admin"                    # Default indexer username
INDEXER_PASSWORD = "SecurePassword123"        # REQUIRED: Your indexer password

# ==============================================================================
#  OPTIONAL SETTINGS (with sensible defaults)
# ==============================================================================

# ── Polling Interval ──────────────────────────────────────────────────────────
# How many seconds to wait between each check for new alerts.
# 30 seconds is a good balance — not too frequent, not too slow.
POLL_INTERVAL_SECONDS = 30

# ── Output File ───────────────────────────────────────────────────────────────
# All detected package change events will be written to this file.
# The file grows over time — each new event is appended.
OUTPUT_FILE = "orbit_package_changes.json"

# ── Rule IDs to Watch ─────────────────────────────────────────────────────────
# These match the custom rules you added to local_rules.xml on the Wazuh Manager.
# 100401 = package installed
# 100402 = package updated
# 100403 = package removed
RULE_IDS_TO_WATCH = ["100401", "100402", "100403"]

# ── Index Pattern ─────────────────────────────────────────────────────────────
# Wazuh stores alerts in daily indices named like: wazuh-alerts-4.x-2026.02.23
# The wildcard (*) matches all of them at once.
WAZUH_ALERTS_INDEX = "wazuh-alerts-*"
